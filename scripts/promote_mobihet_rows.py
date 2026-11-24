"""Promote the audited R50 runs of the former L4 ablation a3 (L4 without logit adjustment) to MobiHet rows (owner 24/11/2026).

MobiHet = L4 with la_mode=off, and a3 zeroed la_shift after the bootstrap: the same computation (tests/test_mobihet_ablation.py
test_mobihet_is_l4_without_la, bit-exact). For every a3 row of experiments/ablation-l4/<scenario>/{results,cost}.csv this script
re-checks the pulled session instead of trusting the CSV:

  * ablation-audit.json of the exact ref PASS at round 50 (arm a3, scenario, seed);
  * the ten metrics, client mean/p10 and per-class F1 recomputed from the round-50 confusion matrices (core.metrics.summary)
    equal the CSV row (6 decimals);
  * bytes per vehicle recomputed from config.json payload_bytes + rounds.jsonl (all 50 rounds; continued sessions carry the
    parent rows) equal the cost row.

Then it writes `mobihet` rows: experiments/<scenario>/results.csv (view client, as the other methods) and
experiments/ablation/<scenario>/{results,cost}.csv (control of the MobiHet ablations), plus a receipt.

    python scripts/promote_mobihet_rows.py [--write]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import TEN, summary  # noqa: E402
from scripts.audit_l4 import append_csv_row  # noqa: E402

OLD = ROOT / "experiments/ablation-l4"
NEW = ROOT / "experiments/ablation"
PULLS = ROOT / "runs/ablation-l4/pulls"
CLIENTS = 5343


def ref_of(source):
    assert source.startswith("kaggle "), source
    return source.split()[1]


def check(scen, row, cost):
    ref = ref_of(row["source"])
    owner, slug, _ = ref.split("/")
    pull = PULLS / owner / slug
    rec = json.loads((pull / "ablation-audit.json").read_text())
    out = {"ref": ref, "scenario": scen, "seed": int(row["seed"])}
    out["receipt_pass"] = (rec["ref"] == ref and rec["checks"]["pass"] is True and rec["last_round"] == 50 and
                           rec["arm"] == "a3" and rec["scenario"] == scen and rec["seed"] == int(row["seed"]))
    man = json.loads((pull / "experiment_manifest.json").read_text())
    run = pull / man["run"]
    view = row["view"]
    with np.load(run / "cms_r050.npz", allow_pickle=False) as z:
        cm = z[view]
    assert cm.shape == (CLIENTS, 16, 16)
    s = summary(cm)
    keys = TEN + ["client_f1_macro_mean", "client_f1_macro_p10"]
    out["metrics_eq_csv"] = all(f"{s[k]:.6f}" == row[k] for k in keys) and \
        [round(float(v), 6) for v in s["f1_per_class"]] == json.loads(row["f1_per_class"])
    cfg = json.loads((run / "config.json").read_text())
    rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines() if x.strip()]
    assert [r["round"] for r in rows] == list(range(1, 51)), ref
    boot = cfg["payload_bytes"]["bootstrap"]
    up, down = sum(r["bytes_up"] for r in rows), sum(r["bytes_down"] for r in rows)
    b = {"bootstrap_up_B": boot["up"], "bootstrap_down_B": boot["down"], "up_50r_B": up, "down_50r_B": down,
         "total_B_per_vehicle": boot["up"] + boot["down"] + up + down}
    out["bytes"] = b
    out["bytes_eq_csv"] = all(f"{b[k]:.1f}" == cost[k] for k in b) and f"{b['total_B_per_vehicle'] * CLIENTS:.0f}" == \
        cost["total_B_network"] and cost["source"] == row["source"]
    out["endpoint"] = {k: float(s[k]) for k in keys}
    out["pass"] = out["receipt_pass"] and out["metrics_eq_csv"] and out["bytes_eq_csv"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    done = []
    for folder in sorted(p for p in OLD.iterdir() if (p / "results.csv").is_file()):
        scen = folder.name
        res = [r for r in csv.DictReader((folder / "results.csv").open()) if r["arm"] == "a3"]
        cost = {r["seed"]: r for r in csv.DictReader((folder / "cost.csv").open()) if r["arm"] == "a3"}
        for r in sorted(res, key=lambda r: r["seed"]):
            c = check(scen, r, cost[r["seed"]])
            c["row"], c["cost_row"] = r, cost[r["seed"]]
            done.append(c)
            print(scen, r["seed"], c["ref"], "PASS" if c["pass"] else "FAIL",
                  {k: c[k] for k in ("receipt_pass", "metrics_eq_csv", "bytes_eq_csv") if not c[k]},
                  "macro %.6f" % c["endpoint"]["f1_macro"], "B %.1f" % c["bytes"]["total_B_per_vehicle"])
    if not all(c["pass"] for c in done):
        print("FAIL: nothing written")
        return 1
    if a.write:
        for c in done:
            r, k, scen = c["row"], c["cost_row"], c["scenario"]
            src = f"kaggle {c['ref']} (former L4 ablation a3 = MobiHet; scripts/audit_ablation.py + scripts/promote_mobihet_rows.py)"
            append_csv_row(ROOT / "experiments" / scen / "results.csv",
                           ["mobihet", r["seed"], "client", "50", *(r[t] for t in TEN), "audited", src])
            (NEW / scen).mkdir(parents=True, exist_ok=True)
            for name, header in (("results.csv", list(r)), ("cost.csv", list(k))):
                path = NEW / scen / name
                if not path.exists():
                    path.write_text(",".join(["arm", *header[1:]]) + "\n")
            append_csv_row(NEW / scen / "results.csv", ["mobihet", *(r[h] for h in list(r)[1:-1]), src])
            append_csv_row(NEW / scen / "cost.csv", ["mobihet", *(k[h] for h in list(k)[1:-1]), src])
        out = ROOT / "runs/mobihet-promote/receipt.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([{x: c[x] for x in c if x not in ("row", "cost_row")} for c in done], indent=1))
        print("written", len(done), "rows; receipt", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
