"""Read-only status, completion marker and final-metric check for aligned MLP/CNN-BiGRU jobs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "cloud_dispatch"))
import dispatcher as d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/aligned-v2-backbones-live"
VIEW = {m: ("global" if m in ("fedavg", "fedprox", "afpha", "fd_ids", "lwfednids") else
            "client_proto" if m in ("tinyproto", "fedtgp", "fedproto") else "client")
        for m in ("fedavg", "fedprox", "afpha", "fd_ids", "apfed", "lwfednids",
                  "canperfl", "tinyproto", "pfedes", "nilm", "fedtgp", "perfedskd",
                  "fedproto", "v2x_htfl")}


def check(job):
    owner = job["ref"].split("/", 1)[0]
    token = (Path.home() / ".kaggle/accounts" / f"{owner}.mcp-token").read_text().strip()
    api = d.make_api(token)
    status = d.read_call(lambda: api.kernels_status(job["ref"])).status.name
    row = {key: job[key] for key in ("ref", "version", "slug", "backbone", "method", "seed")}
    row["status"] = status
    if status == "COMPLETE":
        folder = OUT / job["slug"]
        api.kernels_output(f'{job["ref"]}/{job["version"]}', str(folder),
                           file_pattern=r"^(RUN_STATUS|SMOKE_PASS|experiment_manifest)\.json$",
                           force=True, quiet=True)
        run_status = json.loads((folder / "RUN_STATUS.json").read_text(encoding="utf-8"))
        manifest = json.loads((folder / "experiment_manifest.json").read_text(encoding="utf-8"))
        final = run_status.get("final", {})
        view = VIEW[job["method"]]
        row.update(smoke=(folder / "SMOKE_PASS.json").exists(),
                   complete=run_status.get("complete"),
                   last_round=run_status.get("last_round"),
                   target_round=run_status.get("target_round"),
                   manifest_method=manifest.get("method"), manifest_seed=manifest.get("seed"),
                   skipped_frac=final.get("skipped_frac"), applied_steps=final.get("applied_steps"),
                   view=view, f1_macro=final.get(f"{view}/f1_macro"),
                   accuracy=final.get(f"{view}/accuracy"))
    return row


def safe_check(job):
    try:
        return check(job)
    except Exception as exc:
        return {key: job[key] for key in ("ref", "version", "slug", "backbone", "method", "seed")} | {
            "status": "UNVERIFIED", "error_type": type(exc).__name__,
            "http": getattr(getattr(exc, "response", None), "status_code", None),
        }


def main():
    ledger = json.loads((ROOT / "runs/aligned-v2-backbones-launch.json").read_text(encoding="utf-8"))
    jobs = list({job["slug"]: job for job in ledger if job["state"] == "submitted"}.values())
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(safe_check, jobs))
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
    (ROOT / "runs/aligned-v2-backbones-live.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(Counter(row["status"] for row in rows))), flush=True)
    completed = [row for row in rows if row["status"] == "COMPLETE"]
    gates = [row for row in completed if row.get("smoke") and row.get("complete")
             and row.get("last_round") == row.get("target_round")
             and (row.get("skipped_frac") is None or row["skipped_frac"] < 0.01)]
    print(f"COMPLETION_GATE {len(gates)}/{len(completed)}", flush=True)
    for row in rows:
        print(row["ref"], row["status"], row.get("last_round", "-"),
              row.get("f1_macro", "-"), flush=True)


if __name__ == "__main__":
    main()
