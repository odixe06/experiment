"""Replace the `mobihet` rows of one cell with a rerun's exact ref (owner 28/11/2026: old rows are dropped, not archived).

The 27/11 reruns replace MobiHet cells whose sessions lost their checkpoints. For each exact ref this re-checks the pull:
ablation-audit.json of that ref PASS at round 50 (arm `a3` for the legacy `*-abl-a3-*` slugs, `none` for `*-mh-*`), scenario
and seed; then recomputes the ten metrics, client mean/p10 and per-class F1 from the round-50 confusion matrices and the bytes
per vehicle from config.json + rounds.jsonl. Only then it drops every `mobihet` row of that seed in
experiments/<scenario>/results.csv and experiments/ablation/<scenario>/{results,cost}.csv and writes the new rows.

    python scripts/replace_mobihet_rows.py OWNER/SLUG/VERSION ... [--write]
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import TEN, summary  # noqa: E402

PULLS = {"a3": ROOT / "runs/ablation-l4/pulls", "none": ROOT / "runs/ablation-mh/pulls"}
CLIENTS = 5343
KEYS = TEN + ["client_f1_macro_mean", "client_f1_macro_p10"]


def rows_for(ref):
    owner, slug, _ = ref.split("/")
    arm = "a3" if "-abl-a3-" in slug else "none"
    pull = PULLS[arm] / owner / slug
    rec = json.loads((pull / "ablation-audit.json").read_text())
    assert rec["ref"] == ref and rec["checks"]["pass"] is True and rec["last_round"] == 50 and rec["arm"] == arm, ref
    scen, seed = rec["scenario"], rec["seed"]
    man = json.loads((pull / "experiment_manifest.json").read_text())
    run = pull / man["run"]
    view = "client" if scen.startswith("mixed") else "private"
    with np.load(run / "cms_r050.npz", allow_pickle=False) as z:
        cm = z[view]
    assert cm.shape == (CLIENTS, 16, 16), ref
    s = summary(cm)
    assert all(abs(s[k] - rec["endpoint"][k]) < 1e-9 for k in KEYS), ref
    cfg = json.loads((run / "config.json").read_text())
    rounds = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines() if x.strip()]
    assert [r["round"] for r in rounds] == list(range(1, 51)), ref
    boot = cfg["payload_bytes"]["bootstrap"]
    up, down = sum(r["bytes_up"] for r in rounds), sum(r["bytes_down"] for r in rounds)
    total = boot["up"] + boot["down"] + up + down
    src = f"kaggle {ref} (MobiHet rerun 27/11; scripts/audit_ablation.py + scripts/replace_mobihet_rows.py)"
    ten = [f"{s[k]:.6f}" for k in TEN]
    return scen, seed, {
        ROOT / "experiments" / scen / "results.csv": ["mobihet", str(seed), "client", "50", *ten, "audited", src],
        ROOT / "experiments/ablation" / scen / "results.csv":
            ["mobihet", str(seed), view, "50", *ten, f"{s['client_f1_macro_mean']:.6f}", f"{s['client_f1_macro_p10']:.6f}",
             json.dumps([round(float(v), 6) for v in s["f1_per_class"]]), "audited", src],
        ROOT / "experiments/ablation" / scen / "cost.csv":
            ["mobihet", str(seed), f"{boot['up']:.1f}", f"{boot['down']:.1f}", f"{up:.1f}", f"{down:.1f}", f"{total:.1f}",
             f"{total * CLIENTS:.0f}", "audited", src],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    plans = [rows_for(ref) for ref in a.refs]
    for scen, seed, files in plans:
        for path, new in files.items():
            text = path.read_text()
            lines = text.splitlines(keepends=True)
            keep = [l for l in lines if next(csv.reader([l]))[:2] != ["mobihet", str(seed)]]
            old = [next(csv.reader([l]))[-1] for l in lines if l not in keep]
            print(path.relative_to(ROOT), seed, "drop", old, "->", new[-1].split()[1])
            if a.write:
                buf = io.StringIO()
                csv.writer(buf, lineterminator="\n").writerow(new)
                path.write_text("".join(keep) + buf.getvalue())
    print("written" if a.write else "dry run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
