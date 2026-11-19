"""Launch only bounded GPU pilots for the 15 case/method combinations."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from build_mixed_fl import build, METHODS
sys.path.insert(0, str(Path(__file__).parent/"cloud_dispatch"))
import dispatcher as d

ROOT = Path(__file__).resolve().parents[1]


def main():
    inventory = json.loads((ROOT/"runs/accounts-results-check.json").read_text())
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(inventory["checked_at"])).total_seconds() < 1800
    snapshots = inventory["accounts"]
    path = ROOT/"runs/mixed128-pilot-launch.json"
    ledger = json.loads(path.read_text()) if path.exists() else []
    for case in ("case1", "case2", "case3"):
        for method in METHODS:
            if any(row["case"] == case and row["method"] == method for row in ledger): continue
            # Two-hour job timeout + setup/quota margin, at most one new pilot per account.
            used = {row["owner"] for row in ledger}
            candidates = [o for o, s in snapshots.items() if o not in used and s.get("active", 99) < 2 and s.get("free_hours", 0) >= 3.]
            if not candidates:
                print("No verified pilot slot/quota", flush=True); return
            owner = max(candidates, key=lambda o: snapshots[o]["free_hours"])
            token = (Path.home()/".kaggle/accounts"/f"{owner}.mcp-token").read_text().strip()
            snap = d.checked_inventory(owner, token, snapshots[owner].get("records", {}))
            if snap["active"] >= 2 or snap["free_hours"] < 3.:
                snapshots[owner]["active"] = 99
                continue
            entry = build(case, method, 42, owner, pilot=True)
            entry.update(ref=f"{owner}/{entry['slug']}", state="reserved", time=datetime.now(timezone.utc).isoformat())
            ledger.append(entry); path.write_text(json.dumps(ledger, indent=2))
            try:
                api = snap["api"]
                reply = api.kernels_push(entry["directory"], timeout="7200")
                if not d.accepted(reply, entry["ref"]): raise RuntimeError("Push rejected")
                entry.update(version=d.confirm_submission(api, entry["ref"], Path(entry["directory"])), state="submitted")
                entry["status"] = d.read_call(lambda: api.kernels_status(entry["ref"])).status.name
            except Exception as exc:
                entry.update(state="review", error=type(exc).__name__)
            path.write_text(json.dumps(ledger, indent=2))
            print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
