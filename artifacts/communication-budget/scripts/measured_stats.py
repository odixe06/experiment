"""Data-dependent and run-measured communication statistics (read-only on the repo and sibling caches).

 1. (client, class) pairs with non-zero train+val support: from ../mobihet/cache/fold_counts.npy (5343, 2 folds, 16)
    (sum = 37,361,273 = protocol train rows). Per mixed case/backbone group via core.model_assignment.
 2. MobiHet L4 measured bytes: rounds.jsonl bytes_up / bytes_down per round + bootstrap (config.json payload_bytes or the
    KM2_BYTES log line) for every complete 50-round local run.
 3. PerFed-SKD selected_frac per round where a rounds.jsonl exists locally.
Writes work/data_dependent_stats.json, work/l4_measured_bytes.csv, work/perfedskd_selected_frac.csv
"""
import csv, glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WORK = os.path.join(HERE, "..", "work")
sys.path.insert(0, ROOT)
from core.model_assignment import CASES, assignment  # noqa: E402

FOLD = "/home/odixe/project/mobihet/cache/fold_counts.npy"
res = {"source": FOLD, "basis": "measured (computed from per-client class counts)"}
fc = np.load(FOLD)
cnt = fc.sum(1)                                     # (5343, 16) train + val rows
assert cnt.shape == (5343, 16) and int(cnt.sum()) == 37361273
pres = cnt > 0
res.update(rows_total=int(cnt.sum()), n_clients=5343, n_pairs=int(pres.sum()),
           pairs_per_client_mean=float(pres.sum(1).mean()), pairs_per_client_min=int(pres.sum(1).min()),
           pairs_per_client_median=float(np.median(pres.sum(1))), pairs_per_client_max=int(pres.sum(1).max()),
           clients_with_zero_rows=int((cnt.sum(1) == 0).sum()), holders_per_class=pres.sum(0).tolist(),
           classes_with_no_holder=int((pres.sum(0) == 0).sum()), dense_pairs=5343 * 16)
res["cases"] = {}
for case in CASES:
    asg = assignment(case)
    g = {}
    for m in dict(CASES[case]):
        ids = [c for c, mm in asg.items() if mm == m]
        g[m] = {"clients": len(ids), "pairs": int(pres[ids].sum()), "pairs_per_client_mean": float(pres[ids].sum(1).mean())}
    res["cases"][case] = g
# MobiHet class-Q uplink depends on present classes only (per-vehicle packet = header + present rows)
json.dump(res, open(os.path.join(WORK, "data_dependent_stats.json"), "w"), indent=1)
with open(os.path.join(WORK, "data_dependent_stats.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["scope", "group", "clients", "client_class_pairs", "pairs_per_client_mean", "dense_pairs_16_per_client", "source"])
    w.writerow(["all", "all", 5343, res["n_pairs"], res["pairs_per_client_mean"], 5343 * 16, FOLD])
    for case, g in res["cases"].items():
        for m, v in g.items():
            w.writerow([case, m, v["clients"], v["pairs"], v["pairs_per_client_mean"], v["clients"] * 16, FOLD])

# ---- L4 measured -------------------------------------------------------------------------------------------------
def load_rounds(p):
    return [json.loads(l) for l in open(p) if l.strip()]

runs = []
for p in sorted(glob.glob(f"{ROOT}/runs/l4-20261120/pull/*/*/rounds.jsonl")):
    runs.append(("single", p))
for p in sorted(glob.glob(f"{ROOT}/runs/recovery-20261120/references/*/*-h2/*/rounds.jsonl")) + \
         sorted(glob.glob(f"{ROOT}/runs/recovery-20261120/references/bert-l4-artifacts/*-h2/*/rounds.jsonl")) + \
         sorted(glob.glob(f"{ROOT}/runs/mixed-l4/pilot/*/*/rounds.jsonl")):
    runs.append(("ref", p))
rows = []
for kind, p in runs:
    rr = load_rounds(p)
    d = os.path.dirname(p)
    cfgp = os.path.join(d, "config.json")
    boot = codec = None
    if os.path.exists(cfgp):
        c = json.load(open(cfgp)); pb = c.get("payload_bytes", {}) or {}
        boot = pb.get("bootstrap"); codec = pb.get("codec")
    if boot is None:
        for lg in glob.glob(os.path.join(os.path.dirname(d), "*.log")) + glob.glob(os.path.join(d, "*.log")):
            m = re.search(r'KM2_BYTES (\{.*?\})(?:\\n|\n)', open(lg, errors="ignore").read().replace('\\"', '"'))
            if m:
                try:
                    codec = json.loads(m.group(1)); boot = {"up": codec["up_mean"], "down": codec["down_mean"]}; break
                except Exception:
                    pass
    name = os.path.relpath(d, ROOT)
    ru, rd = [r["bytes_up"] for r in rr], [r["bytes_down"] for r in rr]
    rows.append(dict(run=name, kind=kind, rounds=len(rr), bootstrap_up=(boot or {}).get("up"), bootstrap_down=(boot or {}).get("down"),
                     bootstrap_broadcast_copy_down=(codec or {}).get("broadcast_copy"),
                     round_up_mean=float(np.mean(ru)), round_down_mean=float(np.mean(rd)), round_up_min=min(ru), round_up_max=max(ru),
                     round_down_min=min(rd), round_down_max=max(rd), round_up_sum=float(sum(ru)), round_down_sum=float(sum(rd))))
with open(os.path.join(WORK, "l4_measured_bytes.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---- PerFed-SKD selected_frac ------------------------------------------------------------------------------------
sel = []
for p in glob.glob(f"{ROOT}/runs/**/rounds.jsonl", recursive=True) + glob.glob(f"{ROOT}/bert/runs/**/rounds.jsonl", recursive=True):
    if "perfedskd" not in p or "worktree-consolidation" in p:
        continue
    rr = [r for r in load_rounds(p) if "selected_frac" in r]
    if len(rr) >= 5:
        fr = [r["selected_frac"] for r in rr]
        sel.append(dict(run=os.path.relpath(p, ROOT), rounds=len(rr), mean_selected_frac=float(np.mean(fr)),
                        first_round_frac=fr[0], min=min(fr), max=max(fr), sum_frac=float(sum(fr))))
with open(os.path.join(WORK, "perfedskd_selected_frac.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "rounds", "mean_selected_frac", "first_round_frac", "min", "max", "sum_frac"]); w.writeheader(); w.writerows(sel)
print(json.dumps({k: v for k, v in res.items() if k != "holders_per_class"}, indent=1))
for r in rows: print(r["run"], r["rounds"], r["bootstrap_up"], r["bootstrap_down"], round(r["round_up_mean"], 1), round(r["round_down_mean"], 1))
for r in sel: print(r)
