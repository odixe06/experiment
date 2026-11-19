"""Audit pulled artifacts from one TabM single-backbone FL run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import metrics


EXPECTED_TEST_ROWS = 9_347_679


def close(a, b, tol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def audit(root: Path, require_complete=True):
    root = Path(root)
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    source = json.loads((root / "source_snapshot.json").read_text())
    source_hashes = json.loads((root / "source_manifest.json").read_text())
    got_hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()}
    assert got_hashes == source_hashes

    run = root / f"tabm-fl-{manifest['method'].replace('_', '-')}-s{manifest['seed']}"
    # The seed-41 FedAvg probe deliberately kept its historical slug.
    if not run.is_dir():
        candidates = [p for p in root.iterdir()
                      if p.is_dir() and p.name != "smoke" and (p / "config.json").exists()]
        assert len(candidates) == 1, candidates
        run = candidates[0]
    cfg = json.loads((run / "config.json").read_text())
    rows = [json.loads(line) for line in (run / "rounds.jsonl").read_text().splitlines() if line.strip()]
    target = int(manifest["rounds"])
    expected_rounds = list(range(1, target + 1)) if require_complete else list(range(1, len(rows) + 1))
    assert [row["round"] for row in rows] == expected_rounds
    assert cfg["method"] == manifest["method"]
    assert cfg["hp"]["seed"] == manifest["seed"]
    assert cfg["cfg"]["model"] == "tabm" and cfg["n_params"] == manifest["parameters"] == 6548
    assert cfg["clients"] == 5343 and cfg["n_buffers"] == 0

    metric_keys = metrics.TEN + ["client_f1_macro_mean", "client_f1_macro_p10",
                                 "client_f1_macro_median", "client_f1_macro_min"]
    views = None
    total_cm_rows = 0
    for row in rows:
        assert all(math.isfinite(float(v)) for v in row.values() if isinstance(v, (int, float)))
        assert row.get("skipped_frac", 0) < 0.01
        path = run / f"cms_r{row['round']:03d}.npz"
        assert path.stat().st_size > 0
        with np.load(path, allow_pickle=False) as payload:
            ids = payload["ids"]
            assert ids.shape == (5343,) and len(np.unique(ids)) == 5343
            current_views = sorted(set(payload.files) - {"ids"})
            views = current_views if views is None else views
            assert current_views == views
            for view in views:
                cms = payload[view]
                assert cms.shape == (5343, 16, 16) and np.issubdtype(cms.dtype, np.integer)
                assert int(cms.sum()) == EXPECTED_TEST_ROWS
                total_cm_rows += int(cms.sum())
                summary = metrics.summary(cms)
                for key in metric_keys:
                    assert close(summary[key], row[f"{view}/{key}"]), (row["round"], view, key)
                assert np.allclose(summary["f1_per_class"], row[f"{view}/f1_per_class"], rtol=1e-12, atol=1e-12)

    model_files = sorted((run / "models").glob("r*.pt"))
    assert len(model_files) == len(rows) and all(p.stat().st_size > 0 for p in model_files)
    assert (run / "state.pt").stat().st_size > 0 and (run / "state_rank1.pt").stat().st_size > 0
    status = json.loads((root / "RUN_STATUS.json").read_text())
    assert status["method"] == manifest["method"] and status["seed"] == manifest["seed"]
    assert status["last_round"] == rows[-1]["round"] and bool(status["complete"]) == (len(rows) == target)
    assert status["all_round_metrics"]

    report = {
        "pass": True,
        "scope": "source hashes; config; contiguous rounds; all CM shapes/counts; recomputed 10 metrics, client distribution and class F1; finite/skipped guard; artifact inventory",
        "weights_scope": "all checkpoint files present and non-empty; tensor deserialization/strict model reconstruction not repeated locally",
        "method": manifest["method"], "seed": manifest["seed"], "rounds": len(rows),
        "target_rounds": target, "views": views, "test_rows_per_view_round": EXPECTED_TEST_ROWS,
        "cm_rows_checked": total_cm_rows, "model_checkpoints": len(model_files),
        "final": rows[-1],
    }
    (root / "LOCAL_AUDIT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("pass", "method", "seed", "rounds", "views",
                                              "cm_rows_checked", "model_checkpoints")}, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    audit(args.root, not args.partial)
