"""Read-only exact-version pilot status and bounded output collection."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent/"cloud_dispatch"))
import dispatcher as d

ROOT = Path(__file__).resolve().parents[1]


def main():
    jobs = json.loads((ROOT/"runs/mixed128-pilot-launch.json").read_text())
    results = []
    for job in jobs:
        if job["state"] != "submitted": continue
        row = {k: job[k] for k in ("case", "method", "seed", "ref", "version")}
        owner = job["owner"]
        api = d.make_api((Path.home()/".kaggle/accounts"/f"{owner}.mcp-token").read_text().strip())
        try:
            meta = d.kernel_metadata(api, job["ref"])
            assert meta.current_version_number == job["version"], "Latest version changed"
            row["status"] = d.read_call(lambda: api.kernels_status(job["ref"])).status.name
            if row["status"] in ("COMPLETE", "ERROR"):
                dest = ROOT/"runs/mixed128-pilots"/job["slug"]
                dest.mkdir(parents=True, exist_ok=True)
                api.kernels_output(job["ref"]+"/"+str(job["version"]), str(dest),
                    file_pattern=r"(?:SMOKE_PASS\.json|source_manifest\.json|experiment_manifest\.json|rounds\.jsonl|cms_r002\.npz|.*\.log)$", force=True, quiet=True)
                marker = dest/"SMOKE_PASS.json"
                if marker.exists():
                    smoke = json.loads(marker.read_text())
                    assert all(smoke[k] == job[k] for k in ("case", "method", "seed"))
                    expected = json.loads((ROOT/"experiments/mixed-model-v1/manifest.json").read_text())["cases"][job["case"]]["sha256"]
                    assert smoke["assignment_sha256"] == expected
                    history = [json.loads(line) for line in (dest/"smoke/rounds.jsonl").read_text().splitlines()]
                    assert [x["round"] for x in history] == [1, 2]
                    assert all(x["applied_steps"] > 0 and x["skipped_frac"] < .01 for x in history)
                    experiment = json.loads((dest/"experiment_manifest.json").read_text())
                    hashes = json.loads((dest/"source_manifest.json").read_text())
                    assert hashes == experiment["source_hashes"]
                    sys.path.insert(0, str(ROOT))
                    from core import metrics
                    from core.model_assignment import load_assignment, CASES
                    mapping = load_assignment(ROOT/"experiments/mixed-model-v1", job["case"])
                    with np.load(dest/"smoke/cms_r002.npz") as arrays:
                        ids = arrays['ids']
                        assert len(ids) == len(set(ids)) == 30
                        assert {mapping[int(cid)] for cid in ids} == set(dict(CASES[job['case']]))
                        for view in arrays.files:
                            if view == 'ids': continue
                            cm = arrays[view]
                            assert cm.shape == (30,16,16) and (cm >= 0).all()
                            assert ((cm.sum((1,2)) > 0) & (cm.sum((1,2)) <= 128)).all()
                            summary = metrics.summary(cm)
                            for key in metrics.TEN:
                                assert abs(summary[key]-history[-1][f'{view}/{key}']) < 1e-10
                    row["gpu_smoke"] = "PASS"
                    row["cm_metric_audit"] = "PASS"
                    row["max_skip"] = max(x["skipped_frac"] for x in history)
                    row["train_seconds_r2"] = history[-1]["t_train"]
                elif row["status"] == "COMPLETE":
                    row["gpu_smoke"] = "FAIL_missing_marker"
                if row["status"] == "ERROR":
                    for path in dest.rglob("*.log"):
                        row["log_tail"] = path.read_text(errors="replace")[-4000:]
        except Exception as exc:
            row["check_error"] = type(exc).__name__
        results.append(row)
        print(json.dumps(row), flush=True)
    (ROOT/"runs/mixed128-pilots-live.json").write_text(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(), rows=results), indent=2))


if __name__ == "__main__":
    main()
