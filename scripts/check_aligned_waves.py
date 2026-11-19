"""Read-only status and completion-marker check for aligned-v2 waves."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "cloud_dispatch"))
import dispatcher as d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/aligned-v2-live"
VIEW = {m: ("global" if m in ("fedavg", "fedprox", "afpha", "fd_ids", "lwfednids") else
            "client_proto" if m in ("tinyproto", "fedtgp", "fedproto") else "client")
        for m in ("fedavg", "fedprox", "afpha", "fd_ids", "apfed", "lwfednids",
                  "canperfl", "tinyproto", "pfedes", "nilm", "fedtgp", "perfedskd",
                  "fedproto", "v2x_htfl")}


def check(job):
    owner = job["ref"].split("/")[0]
    token = (Path.home() / ".kaggle/accounts" / f"{owner}.mcp-token").read_text().strip()
    api = d.make_api(token)
    status = d.read_call(lambda: api.kernels_status(job["ref"])).status.name
    row = {"ref": job["ref"], "version": job["version"], "status": status}
    if status == "COMPLETE":
        folder = OUT / job["slug"]
        api.kernels_output(
            f'{job["ref"]}/{job["version"]}', str(folder),
            file_pattern=r"^(RUN_STATUS|SMOKE_PASS|experiment_manifest)\.json$",
            force=True, quiet=True,
        )
        run_status = json.loads((folder / "RUN_STATUS.json").read_text(encoding="utf-8"))
        manifest = json.loads((folder / "experiment_manifest.json").read_text(encoding="utf-8"))
        final = run_status.get("final", {})
        view = VIEW[manifest["method"]]
        row.update(
            smoke=(folder / "SMOKE_PASS.json").exists(),
            complete=run_status.get("complete"),
            last_round=run_status.get("last_round"),
            target_round=run_status.get("target_round"),
            method=manifest.get("method"),
            seed=manifest.get("seed"),
            backbone=manifest.get("backbone"),
            view=view,
            f1_macro=final.get(f"{view}/f1_macro"),
            accuracy=final.get(f"{view}/accuracy"),
        )
    return row


def safe_check(job):
    try:
        return check(job)
    except Exception as exc:
        return {"ref": job["ref"], "version": job.get("version"), "slug": job["slug"],
                "status": "UNVERIFIED", "error_type": type(exc).__name__,
                "http": getattr(getattr(exc, "response", None), "status_code", None)}


def main():
    jobs = {}
    for wave in (1, 2, 3, 4):
        path = ROOT / f"runs/aligned-v2-launch-wave{wave}.json"
        for job in json.loads(path.read_text(encoding="utf-8")):
            if job["state"] == "submitted":
                jobs[job["slug"]] = dict(job, wave=wave)
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(safe_check, jobs.values()))
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
    (ROOT / "runs/aligned-v2-live.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(Counter(r["status"] for r in rows))))
    for row in rows:
        print(row["ref"], row["status"], row.get("last_round", "-"), row.get("smoke", "-"))


if __name__ == "__main__":
    main()
