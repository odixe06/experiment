"""Fill currently free Kaggle slots with aligned-v2 MLP/CNN-BiGRU jobs."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent / "cloud_dispatch"))
import dispatcher as d

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ("mlp", "cnn-bigru-attention")
DISABLED_ACCOUNTS = set()  # iamabighotdog re-enabled after verified 2xT4 probe on 2026-11-15.
METHODS = ("fedavg", "fedprox", "apfed", "fedproto", "fedtgp", "tinyproto",
           "fd_ids", "afpha", "lwfednids", "pfedes", "nilm", "perfedskd",
           "v2x_htfl", "canperfl")
SEEDS = (41, 42, 43)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=22)
    parser.add_argument("--owner", choices=None)
    parser.add_argument("--exclude-owner", action="append", default=[])
    args = parser.parse_args()
    inventory_path = ROOT / "runs/aligned-v2-inventory.json"
    assert datetime.now().timestamp() - inventory_path.stat().st_mtime < 900, "stale inventory"
    snapshots = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert len(snapshots) == 21
    if args.owner and args.owner not in snapshots:
        parser.error(f"owner not present in inventory: {args.owner}")
    records = {ref: value for snap in snapshots.values() for ref, value in snap["records"].items()}
    ledger_path = ROOT / "runs/aligned-v2-backbones-launch.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else []

    def save():
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    remote_slugs = {ref.split("/", 1)[1] for ref, value in records.items()
                    if value["status"] != "ERROR"}
    # Remote inventory is authoritative once refreshed. Submissions newer than this snapshot must
    # still block locally, otherwise a second launcher invocation can duplicate an accepted write.
    inventory_time = datetime.fromtimestamp(inventory_path.stat().st_mtime, timezone.utc)
    recorded_slugs = {
        entry["slug"] for entry in ledger
        if entry["state"] == "reserved"
        or (entry["state"] == "submitted"
            and datetime.fromisoformat(entry["time"]) > inventory_time)
    }
    # Complete FedAvg early, while alternating model and seed. CANPerFL remains dependency-gated.
    jobs = []
    for method in METHODS:
        for seed in SEEDS:
            for backbone in BACKBONES:
                jobs.append((backbone, method, seed))

    submitted = 0
    for backbone, method, seed in jobs:
        if submitted >= args.limit:
            break
        slug = f'{backbone}-fl-{method.replace("_", "-")}-s{seed}-a2'
        if slug in remote_slugs or slug in recorded_slugs:
            continue
        parent = None
        candidates = [args.owner] if args.owner else list(snapshots)
        if method == "canperfl":
            parent_slug = f'{backbone}-fl-fedavg-s{seed}-a2'
            hits = [ref for ref, value in records.items()
                    if ref.split("/", 1)[1] == parent_slug and value["status"] == "COMPLETE"]
            if len(hits) != 1:
                continue
            parent = hits[0]
            parent_owner = parent.split("/", 1)[0]
            if args.owner and args.owner != parent_owner:
                continue
            api_parent = d.make_api((Path.home() / ".kaggle/accounts" /
                                     f"{parent_owner}.mcp-token").read_text().strip())
            parent_version = d.kernel_metadata(api_parent, parent).current_version_number
            if not parent_version or not d.parent_ready(
                    api_parent, parent, parent_version, {"seed": seed, "backbone": backbone}):
                continue
            candidates = [parent_owner]
        session_hours = 5 if method == "canperfl" else 7
        required_hours = session_hours + 0.5
        candidates = [owner for owner in candidates
                      if owner not in DISABLED_ACCOUNTS
                      and owner not in args.exclude_owner
                      and snapshots[owner]["active"] < 2 and snapshots[owner]["free_hours"] >= required_hours]
        if not candidates:
            continue
        owner = max(candidates, key=lambda item: snapshots[item]["free_hours"])
        snap = snapshots[owner]
        token = (Path.home() / ".kaggle/accounts" / f"{owner}.mcp-token").read_text().strip()
        api = d.make_api(token)
        snap["free_hours"] = min(snap["free_hours"], d.quota(api))
        if snap["free_hours"] < required_hours:
            continue
        command = [sys.executable, "scripts/build_tabm_fl.py", "--owner", owner,
                   "--backbone", backbone, "--method", method, "--seed", str(seed),
                   "--slug", slug, "--max-hours", str(session_hours)]
        if parent:
            command += ["--parent", parent]
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        folder = ROOT / "notebooks" / slug
        entry = {"slug": slug, "ref": f"{owner}/{slug}", "backbone": backbone,
                 "method": method, "seed": seed, "state": "reserved", "parent": parent,
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
            snap["free_hours"] -= session_hours + 0.5
            remote_slugs.add(slug)
            submitted += 1
        except Exception as exc:
            entry.update(state="review", error=type(exc).__name__)
            snap["active"] = 2
        save()
        print(json.dumps(entry), flush=True)
    print(f"NEW_SUBMITTED {submitted}", flush=True)


if __name__ == "__main__":
    main()
