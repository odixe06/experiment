"""FLOP tables for every scenario -> experiments/<scenario>/compute.csv + the generated block of docs/research/compute.md.

    python scripts/compute/report.py [COUNT_DIR ...]
    (default: ../mobihet/runs/total-compute-20261119 for DAGSNet/FeatureBERT, runs/compute-20261120 for the rest)

Ops = mm FLOPs + one op per element of every other aten op (scripts/compute/op_counter.py), forward + backward, eager count run,
panel 200 vehicles, round 2 = steady state. Total FL = bootstrap + 50 x (train + round_other); eval reported apart.
CANPerFL = FedAvg 50-round total + its bootstrap + 10 x (train + round_other) of the fine-tuning epochs.
FeatureBERT counts at group_rows 2048 (eager OOM at 4096) are rescaled to the 4096 engine's executed rows from the time run.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRS = [Path(p) for p in sys.argv[1:]] or [ROOT.parent / "mobihet/runs/total-compute-20261119", ROOT / "runs/compute-20261120"]
SPEC = json.loads((ROOT / "experiments/scenarios.json").read_text())
T = 1e12


def scen_method(name):
    if name.startswith("bert-mh") or name.startswith("dags-mh") or name == "dags-fedavg" or name.endswith("-init"):
        return None, None
    m = re.match(r"(dags-base|bert|tabm|sae|mlp|cnn-bigru|mixed-case[123])-(.+)$", name)
    if not m:
        return None, None
    return {"dags-base": "dagsnet"}.get(m.group(1), m.group(1)), m.group(2)


def phases(c):
    out = {}
    for r in c["ops"]:
        if r["phase"].startswith("r1."):
            continue
        ph = r["phase"].split(".")[-1] if r["phase"].startswith("r2.") or r["phase"] == "bootstrap" else None
        k = out.setdefault(ph, 0.0)
        out[ph] = k + r["mm_flops"] + r["other_ops"]
    return out


rows = {}
for d in DIRS:
    for p in sorted(d.glob("*.count.json")):
        name = p.name.removesuffix(".count.json")
        scen, m = scen_method(name)
        if scen is None:
            continue
        c = json.loads(p.read_text())
        r = dict(method=m, status=c.get("status"), source=str(p.relative_to(ROOT.parent)))
        if c.get("status") == "done":
            ph = phases(c)
            r2 = c["rows"].get("r2.train", {})
            t = json.loads((d / f"{name}.time.json").read_text()) if (d / f"{name}.time.json").exists() else {}
            tr2 = (t.get("rows") or {}).get("r2.train", {})
            scale = tr2["executed"] / r2["executed"] if tr2.get("executed") and r2.get("executed") else 1.0
            r.update(bootstrap=ph.get("bootstrap", 0.0), train=ph.get("train", 0.0) * scale, other=ph.get("round_other", 0.0),
                     eval=ph.get("eval", 0.0), real_rows=c["rows"]["real_rows"], executed_rows=round((r2.get("executed") or 0) * scale),
                     useful_rows=int(r2.get("useful") or 0), train_scale=scale)
        rows.setdefault(scen, {})[m] = r

summary = {}
for scen, rs in rows.items():
    fed = rs.get("fedavg") if rs.get("fedavg", {}).get("status") == "done" else None
    fed_total = fed["bootstrap"] + 50 * (fed["train"] + fed["other"]) if fed else None
    for m, r in rs.items():
        if r["status"] != "done":
            continue
        if m == "canperfl":
            r["rounds"] = 10
            r["total"] = (fed_total + r["bootstrap"] + 10 * (r["train"] + r["other"])) if fed_total else None
        else:
            r["rounds"] = 50
            r["total"] = r["bootstrap"] + 50 * (r["train"] + r["other"])
        r["x_fedavg"] = r["total"] / fed_total if fed_total and r["total"] else None
    cols = ["method", "status", "bootstrap_tops", "train_tops_per_round", "other_tops_per_round", "eval_tops_per_round", "rounds",
            "total_tops", "x_fedavg", "real_rows", "executed_rows_r2", "useful_rows_r2", "source"]
    out = ROOT / "experiments" / scen / "compute.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in SPEC["methods"]:
            r = rs.get(m)
            if r is None:
                continue
            g = lambda k, s=T: "" if r.get(k) is None else f"{r[k] / s:.6g}"
            w.writerow([m, r["status"], g("bootstrap"), g("train"), g("other"), g("eval"), r.get("rounds", ""), g("total"),
                        "" if r.get("x_fedavg") is None else f"{r['x_fedavg']:.4f}", r.get("real_rows", ""),
                        r.get("executed_rows", ""), r.get("useful_rows", ""), r["source"]])
    summary[scen] = rs

# generated block of docs/research/compute.md: x FedAvg (single-model) / total Top (mixed)
doc = ROOT / "docs/research/compute.md"
scens = [s for s in SPEC["scenarios"] if s in summary]
L = ["<!-- compute:begin (scripts/compute/report.py) -->", "",
     "Tổng FL 50 round, đơn vị: **× FedAvg cùng kịch bản** (đơn model); mixed: **Tops = 1e12 op** (không có FedAvg). Panel 200 xe.", "",
     "| Phương pháp | " + " | ".join(scens) + " |", "|---|" + "---:|" * len(scens)]
for m in SPEC["methods"][1:]:
    cells = []
    for s in scens:
        r = summary[s].get(m)
        if r is None:
            cells.append("N/A" if s.startswith("mixed") and m not in SPEC["hetero"] else "")
        elif r["status"] != "done":
            cells.append(r["status"])
        elif s.startswith("mixed"):
            cells.append(f"{r['total'] / T:.1f}")
        else:
            cells.append(f"{r['x_fedavg']:.3f}" if r.get("x_fedavg") else "")
    L.append(f"| {SPEC['names'][m]} | " + " | ".join(cells) + " |")
fedrow = [f"{(summary[s]['fedavg']['bootstrap'] + 50 * (summary[s]['fedavg']['train'] + summary[s]['fedavg']['other'])) / T:.1f}"
          if summary[s].get("fedavg", {}).get("status") == "done" else "" for s in scens]
L += ["| *FedAvg tổng (Tops)* | " + " | ".join(fedrow) + " |", "", "<!-- compute:end -->"]
text = doc.read_text(encoding="utf-8")
a, b = text.index("<!-- compute:begin"), text.index("<!-- compute:end -->") + len("<!-- compute:end -->")
doc.write_text(text[:a] + "\n".join(L) + text[b:], encoding="utf-8")
print("\n".join(L))
