"""Push checkpoint-resume notebooks for incomplete aligned CNN-BiGRU runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent / "cloud_dispatch"))
import dispatcher as d

ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    ("minhiw", "cnn-bigru-attention-fl-apfed-s41-a2", "apfed", 41, 42, 3.2),
    ("phamquang0606", "cnn-bigru-attention-fl-apfed-s42-a2", "apfed", 42, 40, 3.5),
    ("dangthiet", "cnn-bigru-attention-fl-apfed-s43-a2", "apfed", 43, 41, 3.5),
    ("hunhthanhhong", "cnn-bigru-attention-fl-pfedes-s41-a2", "pfedes", 41, 31, 5.5),
    ("huynhkyy", "cnn-bigru-attention-fl-pfedes-s42-a2", "pfedes", 42, 34, 5.0),
    ("hiulhuy", "cnn-bigru-attention-fl-pfedes-s43-a2", "pfedes", 43, 33, 5.2),
    ("iamabighotdog", "cnn-bigru-attention-fl-nilm-s41-a2", "nilm", 41, 29, 5.8),
    ("hoangdv1711", "cnn-bigru-attention-fl-nilm-s42-a2", "nilm", 42, 30, 5.5),
    ("phamquang0606", "cnn-bigru-attention-fl-nilm-s43-a2", "nilm", 43, 29, 5.8),
    ("minhiw", "cnn-bigru-attention-fl-perfedskd-s41-a2", "perfedskd", 41, 45, 2.4),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner")
    ap.add_argument("--exclude-owner", action="append", default=[])
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args()
    inventory_path = ROOT / "runs/aligned-v2-inventory.json"
    assert datetime.now().timestamp() - inventory_path.stat().st_mtime < 900, "stale inventory"
    snapshots = json.loads(inventory_path.read_text(encoding="utf-8"))
    ledger_path = ROOT / "runs/aligned-v2-resume-launch.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else []
    submitted = {entry["slug"] for entry in ledger if entry["state"] in ("reserved", "submitted")}

    def save():
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    count = 0
    for owner, run, method, seed, from_round, hours in JOBS:
        slug = run + "-r2"
        if count >= a.limit or (a.owner and owner != a.owner) or owner in a.exclude_owner or slug in submitted:
            continue
        snap = snapshots[owner]
        if snap["active"] >= 2 or snap["free_hours"] < hours + 0.5:
            continue
        token = (Path.home() / ".kaggle/accounts" / f"{owner}.mcp-token").read_text().strip()
        api = d.make_api(token)
        if d.quota(api) < hours + 0.5:
            continue
        command = [sys.executable, "scripts/build_aligned_resume.py", "--owner", owner,
                   "--parent", f"{owner}/{run}", "--slug", slug,
                   "--backbone", "cnn-bigru-attention", "--method", method,
                   "--seed", str(seed), "--from-round", str(from_round),
                   "--max-hours", str(hours)]
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        folder = ROOT / "notebooks" / slug
        entry = {"slug": slug, "ref": f"{owner}/{slug}", "parent": f"{owner}/{run}",
                 "method": method, "seed": seed, "from_round": from_round,
                 "max_hours": hours, "state": "reserved",
                 "time": datetime.now(timezone.utc).isoformat(),
                 "sha256": hashlib.sha256((folder / f"{slug}.ipynb").read_bytes()).hexdigest()}
        ledger.append(entry)
        save()
        try:
            reply = api.kernels_push(str(folder), timeout="28800")
            if not d.accepted(reply, entry["ref"]):
                raise RuntimeError("push rejected")
            entry.update(version=d.confirm_submission(api, entry["ref"], folder), state="submitted")
            entry["status"] = d.read_call(lambda: api.kernels_status(entry["ref"])).status.name
            snap["active"] += 1
            snap["free_hours"] -= hours + 0.5
            submitted.add(slug)
            count += 1
        except Exception as exc:
            entry.update(state="review", error=type(exc).__name__)
            snap["active"] = 2
        save()
        print(json.dumps(entry), flush=True)
    print("NEW_SUBMITTED", count, flush=True)


if __name__ == "__main__":
    main()
