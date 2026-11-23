"""Audit one pulled MobiHet L4 ablation session (docs/research/mobihet-l4-ablation.md) and record it at r50.

    python scripts/audit_ablation.py <pull dir> --notebook <executed .ipynb> --ref <owner/slug/version>
                                     [--compare <L4 run dir>] [--record]

Single-backbone (scripts/build_l4_fl.py --ablation), FeatureBERT (scripts/build_bert_l4_ablation.py) and mixed
(scripts/build_mixed_l4.py --ablation) sessions. Checks: executed source == the notebook's embedded source == its manifest;
method, arm, seed and hp == PROTOCOL + L4_HP + seed + the arm's hp; rounds 1..last contiguous, cosine LR of 50 rounds,
finite loss, skipped steps < 1 % (per backbone for mixed), every client applied; every round's confusion matrices cover the
9,347,679 test rows and their ten metrics equal the logged ones; the arm's byte contract (a1: bootstrap 64 + 64 B and 0 B per
round, a4: 1024 B each way). --compare: rounds 1..last and their confusion matrices equal an L4 run (control arm `none`).
Writes ablation-audit.json in the pull dir; --record (r50 only) appends results.csv and cost.csv rows in
experiments/ablation-l4/<scenario>/ (FLOPs: scripts/compute/report_ablation.py -> compute.csv).
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.fl import cosine_lr  # noqa: E402
from core.metrics import TEN, summary  # noqa: E402
from methods.mobihet_ablation import HP, RAW_Q_BYTES  # noqa: E402
from scripts.audit_l4 import append_csv_row  # noqa: E402
from scripts.build_l4_fl import L4_HP  # noqa: E402

TEST_ROWS, CLIENTS = 9347679, 5343
SCENARIO = {"DAGSNet": "dagsnet", "TabM": "tabm", "SAE": "sae", "MLP": "mlp", "CNN-BiGRU-Attention": "cnn-bigru",
            "FeatureBERT": "bert"}                                       # manifest "backbone" -> experiments/ablation-l4/<dir>


def read(path):
    return json.loads(Path(path).read_text())


def value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pull", type=Path)
    ap.add_argument("--notebook", required=True, type=Path)
    ap.add_argument("--ref", required=True, help="exact owner/slug/version")
    ap.add_argument("--compare", type=Path, help="L4 run directory of the same seed (control arm)")
    ap.add_argument("--parent", type=Path, help="run directory of the audited parent session (session > 1): its rounds are a prefix")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    p = a.pull
    man, source, hashes = (read(p / n) for n in ("experiment_manifest.json", "source_snapshot.json", "source_manifest.json"))
    arm, seed = man["ablation"], man["seed"]
    assert arm in HP and seed in (41, 42, 43) and man["rounds"] == 50
    mixed = man["experiment"].startswith("mixed-")
    scen = f"mixed-{man['case']}" if mixed else SCENARIO[man["backbone"]]
    chk = {"source_eq_manifest": hashes == {k: hashlib.sha256(v.encode()).hexdigest() for k, v in source.items()}}
    cell = next("".join(c["source"]) for c in read(a.notebook)["cells"] if "".join(c["source"]).startswith("source = "))
    chk["source_eq_notebook"] = source == ast.literal_eval(cell.splitlines()[0].removeprefix("source = "))
    chk["ablation_module"] = "methods/mobihet_ablation.py" in source
    run = p / man["run"]
    cfg = read(run / "config.json")
    want = {k: value(v) for k, v in (s.split("=", 1) for s in L4_HP + [f"seed={seed}"] + HP[arm])}
    chk["hp"] = all(cfg["hp"].get(k) == v for k, v in want.items()) and cfg["hp"]["batch"] == 64
    chk["method"] = (cfg["method"] == "mobihet_l4" and man["method"] == "mobihet_ablation_mixed") if mixed else \
        (cfg["method"] == man["method"] == "mobihet_ablation")
    chk["compiled_2gpu"] = cfg.get("backend") == "compiled" and (cfg.get("world", 2) == 2)
    rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines() if x.strip()]
    last = rows[-1]["round"]
    chk["rounds_contiguous"] = [r["round"] for r in rows] == list(range(1, last + 1)) and last <= man["stop_after"]
    chk["lr_cosine50"] = all(abs(r["lr"] - cosine_lr(r["round"], 50, 1e-3, 1e-5)) < 1e-14 for r in rows)
    chk["finite_loss"] = all(math.isfinite(r["train_loss"]) for r in rows)
    if mixed:
        models = sorted(cfg["groups"])
        chk["health"] = all(r[f"health/{m}/finite"] and r[f"health/{m}/applied_steps"] > 0 and
                            r[f"health/{m}/skipped_frac"] < .01 for r in rows for m in models)
    else:
        chk["health"] = all(r["skipped_frac"] < .01 and r.get("clients_without_applied", 0) == 0 for r in rows)
    view = "client" if mixed else "private"
    ids_ref, cm_ok, ten_ok = None, True, True
    for r in rows:
        with np.load(run / f"cms_r{r['round']:03d}.npz", allow_pickle=False) as z:
            cm, ids = z[view], z["ids"]
        ids_ref = ids if ids_ref is None else ids_ref
        cm_ok &= (cm.shape == (CLIENTS, 16, 16) and int(cm.sum()) == TEST_ROWS and np.array_equal(ids, ids_ref)
                  and len(set(ids.tolist())) == CLIENTS)
        s = summary(cm)
        ten_ok &= all(abs(s[k] - r[f"{view}/{k}"]) < 1e-9 for k in TEN)
    chk["cm_rows_every_round"], chk["ten_from_cm_every_round"] = bool(cm_ok), bool(ten_ok)
    boot = cfg["payload_bytes"]["bootstrap"]
    up, down = sum(r["bytes_up"] for r in rows), sum(r["bytes_down"] for r in rows)
    if arm == "a1":
        chk["bytes_contract"] = boot == {"up": 64, "down": 64} and up == 0 and down == 0
    elif arm == "a4":
        chk["bytes_contract"] = all(r["bytes_up"] == RAW_Q_BYTES and r["bytes_down"] == RAW_Q_BYTES for r in rows)
    else:
        chk["bytes_contract"] = boot["up"] > 0 and all(0 < r["bytes_up"] < RAW_Q_BYTES for r in rows)
    if a.compare:
        ref = [json.loads(x) for x in (a.compare / "rounds.jsonl").read_text().splitlines() if x.strip()][:last]
        same, cms = True, True
        for x, y in zip(rows, ref):
            same &= all(x[f"{view}/{k}"] == y[f"{view}/{k}"] for k in TEN) and x["bytes_up"] == y["bytes_up"]
            with np.load(run / f"cms_r{x['round']:03d}.npz") as z1, np.load(a.compare / f"cms_r{x['round']:03d}.npz") as z2:
                cms &= np.array_equal(z1[view], z2[view])
        chk["eq_l4_metrics"], chk["eq_l4_cms"] = bool(same) and len(ref) == last, bool(cms)
    if man["session"] > 1:
        assert a.parent, "--parent is required for a continued session"
        par = [json.loads(x) for x in (a.parent / "rounds.jsonl").read_text().splitlines() if x.strip()]
        k = len(par)
        same = 0 < k < last and rows[:k] == par
        for r in range(1, k + 1):
            with np.load(run / f"cms_r{r:03d}.npz") as z1, np.load(a.parent / f"cms_r{r:03d}.npz") as z2:
                same &= np.array_equal(z1[view], z2[view])
        chk["parent_prefix"] = bool(same)
        chk["source_eq_parent"] = hashes == read(a.parent.parent / "source_manifest.json")
    chk["pass"] = all(chk.values())
    with np.load(run / f"cms_r{last:03d}.npz", allow_pickle=False) as z:
        end = summary(z[view])
    receipt = {"ref": a.ref, "scenario": scen, "arm": arm, "seed": seed, "last_round": last, "checks": chk,
               "endpoint": {k: float(end[k]) for k in TEN + ["client_f1_macro_mean", "client_f1_macro_p10"]},
               "f1_per_class": [float(v) for v in end["f1_per_class"]],
               "bytes_per_vehicle": {"bootstrap_up": boot["up"], "bootstrap_down": boot["down"], "up": up, "down": down,
                                     "total": boot["up"] + boot["down"] + up + down},
               "parent": str(a.parent) if a.parent else None,
               "t_train_hours": sum(r["t_train"] for r in rows) / 3600,
               "source_manifest_sha256": hashlib.sha256((p / "source_manifest.json").read_bytes()).hexdigest()}
    (p / "ablation-audit.json").write_text(json.dumps(receipt, indent=1))
    print(a.ref, scen, arm, seed, "r%d" % last, "PASS" if chk["pass"] else "FAIL", {k: v for k, v in chk.items() if not v},
          "macro %.6f" % end["f1_macro"])
    if a.record:
        if last != 50 or not chk["pass"]:
            ap.error("--record needs a passing r50 audit")
        folder = ROOT / "experiments/ablation-l4" / scen
        e = receipt["endpoint"]
        src = f"kaggle {a.ref} (scripts/audit_ablation.py: 50 CM -> 10 metrics)"
        append_csv_row(folder / "results.csv", [arm, str(seed), view, "50", *(f"{e[k]:.6f}" for k in TEN),
                                                f"{e['client_f1_macro_mean']:.6f}", f"{e['client_f1_macro_p10']:.6f}",
                                                json.dumps([round(v, 6) for v in receipt["f1_per_class"]]), "audited", src])
        b = receipt["bytes_per_vehicle"]
        append_csv_row(folder / "cost.csv", [arm, str(seed), f"{b['bootstrap_up']:.1f}", f"{b['bootstrap_down']:.1f}",
                                             f"{b['up']:.1f}", f"{b['down']:.1f}", f"{b['total']:.1f}",
                                             f"{b['total'] * CLIENTS:.0f}", "audited", src])
        print("recorded", folder)
    return 0 if chk["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
