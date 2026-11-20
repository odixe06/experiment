"""Audit one pulled MobiHet L4 run (scripts/build_l4_fl.py) and print its results.csv row.

    python scripts/audit_l4.py <pull dir> <notebook dir> [--csv experiments/<scenario>/results.csv]

Checks: rounds 1..50 exact; 5,343 clients; method/backbone/hp == the locked L4 hp + seed; executed source == the notebook's embedded
source and methods/mobihet_budget.py == ../mobihet (a4147a1b...); finite loss, skipped <= 1 %, every client applied; LR == the audited
DAGSNet L4 chain (../mobihet); r50 CM covers the 9,347,679 test rows and its ten metrics equal the logged ones. Writes audit.json in
the pull dir; --csv appends the row (view "client": each vehicle's own model on its own test rows) when every check passes.
"""
import argparse
import ast
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from core.metrics import TEN, summary  # noqa: E402
from build_l4_fl import L4_HP  # noqa: E402

BUDGET_SHA = "a4147a1b576f9579d6c424fd771115548d57adee8042bbd8c7a20fccc91f7b5a"   # ../mobihet methods/mobihet_budget.py
TEST_ROWS = 9347679
REF_LR = sorted(glob.glob("/home/odixe/project/mobihet/artifacts/fbl-r50/mh-fbl-l4-top-s42-r50-h2/**/rounds.jsonl", recursive=True))


def value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pull")
    ap.add_argument("notebook")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    pull, nbdir = Path(a.pull), Path(a.notebook)
    man = json.loads((pull / "experiment_manifest.json").read_text())
    run = pull / man["run"]
    rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines() if x.strip()]
    cfg = json.loads((run / "config.json").read_text())
    seed = man["seed"]
    hp = {k: value(v) for k, v in (s.split("=", 1) for s in L4_HP + [f"seed={seed}"])}
    nb = json.loads(next(nbdir.glob("*.ipynb")).read_text())
    embedded = ast.literal_eval("".join(nb["cells"][2]["source"]).split("\n")[0][len("source = "):])
    emb_hash = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in embedded.items()}
    executed = json.loads((pull / "source_manifest.json").read_text())
    ref = [json.loads(x) for x in open(REF_LR[0]).read().splitlines() if x.strip()]
    chk = {
        "rounds_exact": [r["round"] for r in rows] == list(range(1, 51)),
        "clients": cfg["clients"] == 5343,
        "method": cfg["method"] == "mobihet_l4" and man["method"] == "mobihet_l4",
        "backbone": cfg["cfg"].get("model") == man["experiment"].split("-")[2] or
                    (cfg["cfg"].get("model") == "cnn-bigru-attention" and "cnn-bigru" in man["experiment"]),
        "hp_eq_l4": all(cfg["hp"].get(k) == v for k, v in hp.items()),
        "source_eq_notebook": executed == emb_hash,
        "budget_sha_eq_mobihet": emb_hash["methods/mobihet_budget.py"] == BUDGET_SHA,
        "finite_loss": all(math.isfinite(r["train_loss"]) for r in rows),
        "skip_le_1pct": all(r["skipped_frac"] <= .01 for r in rows),
        "clients_applied": all(r.get("clients_without_applied", 0) == 0 for r in rows),
        "lr_eq_dagsnet_l4": len(ref) == 50 and all(abs(x["lr"] - y["lr"]) < 1e-12 for x, y in zip(rows, ref)),
    }
    with np.load(run / "cms_r050.npz") as z:
        cm, ids = z["private"], z["ids"]
    ten = summary(cm)
    chk["cm_rows_eq_test"] = int(cm.sum()) == TEST_ROWS and len(ids) == 5343 == len(set(ids.tolist()))
    chk["ten_from_cm_eq_log"] = all(abs(ten[k] - rows[-1]["private/" + k]) < 1e-9 for k in TEN)
    chk["pass"] = all(chk.values())
    boot = cfg["payload_bytes"]["bootstrap"]
    total = boot["up"] + boot["down"] + sum(r["bytes_up"] + r["bytes_down"] for r in rows)
    res = {"checks": chk, "endpoint": {k: ten[k] for k in TEN + ["client_f1_macro_mean", "client_f1_macro_p10"]},
           "total_bytes_per_vehicle": total, "t_train_hours": sum(r["t_train"] for r in rows) / 3600,
           "real_frac_r50": rows[-1].get("real_frac"), "kernel": man.get("run")}
    (pull / "audit.json").write_text(json.dumps(res, indent=1))
    print(pull.name, "PASS" if chk["pass"] else "FAIL", {k: v for k, v in chk.items() if not v},
          "macro %.5f bytes %.0f train %.2f h" % (ten["f1_macro"], total, res["t_train_hours"]))
    if a.csv and chk["pass"]:
        ref_k = json.loads((nbdir / "kernel-metadata.json").read_text())["id"]
        row = ",".join(["mobihet_l4", str(seed), "client", "50", *(f"{ten[k]:.6f}" for k in TEN), "audited",
                        f'"kaggle {ref_k} (scripts/audit_l4.py: CM r50 -> 10 metrics, hp/source/rounds checked)"'])
        existing = Path(a.csv).read_text()
        assert f"mobihet_l4,{seed}," not in existing, "row already recorded"
        with open(a.csv, "a") as f:
            f.write(row + "\n")
        print("appended", a.csv)


if __name__ == "__main__":
    main()
