"""Client/server FLOPs of the MobiHet L4 ablations (docs/research/mobihet-l4-ablation.md) next to L4, per scenario.

Same traces and convention as scripts/compute/report_detail.py (schema-3 counts, panel 200, group_rows 4096, seed 42, 50 rounds
projected from r1 + 49 x r2, formula + arithmetic estimate, evaluation excluded). Writes experiments/ablation-l4/<scenario>/
compute.csv: one row for L4 and one per arm whose trace exists (scripts/compute/run_queue.py --ablation, or the Kaggle count).

    python scripts/compute/report_ablation.py [trace dirs ...]   (default runs/compute-detailed-v3)
"""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/compute"))
from report_detail import CONVENTIONS, SIDES, TOP, atomic_csv, projection, trace  # noqa: E402

PREFIX = {"dagsnet": "dags-base", "bert": "bert", "tabm": "tabm", "sae": "sae", "mlp": "mlp", "cnn-bigru": "cnn-bigru",
          "mixed-case1": "mixed-case1", "mixed-case2": "mixed-case2", "mixed-case3": "mixed-case3"}
ARMS = ("mobihet_l4", "abl-a1", "abl-a2", "abl-a3", "abl-a4")
SPLIT = ("client", "server", "bootstrap", "other", "unattributed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", type=Path, nargs="*", default=[ROOT / "runs/compute-detailed-v3"])
    args = ap.parse_args()
    for scen, prefix in PREFIX.items():
        rows = []
        for arm in ARMS:
            paths = sorted({p for d in args.dirs for p in d.rglob(f"{prefix}-{arm}.count.json")})
            if not paths:
                continue
            assert len(paths) == 1, (scen, arm, paths)
            got = trace(paths[0])
            if got is None:
                continue
            raw, counts = got
            est = projection(counts, 50)
            side = {s: sum(est[s, c] for c in CONVENTIONS[:2]) for s in SIDES}
            total = sum(v for s, v in side.items() if s != "evaluation")
            rows.append({"arm": "l4" if arm == "mobihet_l4" else arm[4:], **{f"{s}_tops": side[s] / TOP for s in SPLIT},
                         "total_tops": total / TOP, "panel_ids_sha256": raw["rows"]["panel_ids_sha256"],
                         "source": paths[0].resolve().relative_to(ROOT) if paths[0].resolve().is_relative_to(ROOT) else paths[0],
                         "trace_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest()})
        if not rows:
            continue
        assert len({r["panel_ids_sha256"] for r in rows}) == 1, f"{scen}: input panels differ"
        base = next((r["total_tops"] for r in rows if r["arm"] == "l4"), None)
        out = ROOT / "experiments/ablation-l4" / scen / "compute.csv"
        cols = ["arm", *(f"{s}_tops" for s in SPLIT), "total_tops", "x_l4", "rounds", "panel", "trace_sha256", "source"]
        with atomic_csv(out) as stream:
            w = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({**{k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in r.items()},
                            "x_l4": f"{r['total_tops'] / base:.4f}" if base else "", "rounds": 50, "panel": 200})
        print(scen, ", ".join(f"{r['arm']} {r['total_tops']:.4g}" for r in rows), "Tops")


if __name__ == "__main__":
    main()
