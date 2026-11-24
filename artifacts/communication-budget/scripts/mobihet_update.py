#!/usr/bin/env python3
"""Replace MobiHet L4 (LA on) by MobiHet (L4 without LA, owner 24/11/2026) in the communication budget tables.

The L4 rows are archived in ../mobihet/archive/l4-la-20261124/experiment/communication-budget/. MobiHet rows are measured from
the round logs of the audited MobiHet R50 runs cited in experiments/<scenario>/results.csv (method `mobihet`): per vehicle
the codec byte length that config.json payload_bytes (bootstrap) and rounds.jsonl bytes_up/bytes_down (every round) record,
times K = 5,343 (same formula as compute_budget.py: K*sum_r bytes_up_r | K*sum_r bytes_down_r | K*bootstrap_mean).
Then communication_summary.csv (mean, sample SD over seeds) and communication_ranking.csv (rank within scenario, x FedAvg)
are rebuilt for those rows; every other method's row is untouched.

    python3 artifacts/communication-budget/scripts/mobihet_update.py
"""
import csv
import json
import statistics as st
from pathlib import Path

REPO = Path("/home/odixe/project/experiment")
OUT = REPO / "artifacts/communication-budget"
K = 5343
BACKBONE = {"dagsnet": "dagsnet", "bert": "bert", "tabm": "tabm", "sae": "sae", "mlp": "mlp", "cnn-bigru": "cnn-bigru-attention",
            "mixed-case1": "bert+cnn-bigru-attention+dagsnet", "mixed-case2": "mlp+sae+tabm",
            "mixed-case3": "bert+cnn-bigru-attention+dagsnet+sae+tabm"}
BASIS = "measured (codec byte length per vehicle in simulation; all vehicles; not network traffic)"
PULLS = [REPO / "runs/ablation-l4/pulls", REPO / "runs/ablation-mh/pulls"]


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows, cols):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def run_dir(ref):
    owner, slug, _ = ref.split("/")
    for base in PULLS:
        pull = base / owner / slug
        if (pull / "ablation-audit.json").exists():
            audit = json.loads((pull / "ablation-audit.json").read_text())
            assert audit["ref"] == ref and audit["checks"]["pass"] and audit["last_round"] == 50, ref
            man = json.loads((pull / "experiment_manifest.json").read_text())
            return pull / man["run"]
    raise FileNotFoundError(ref)


def measure(scen, seed, ref):
    d = run_dir(ref)
    cfg = json.loads((d / "config.json").read_text())
    rows = [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines() if x.strip()]
    assert [r["round"] for r in rows] == list(range(1, 51)), ref
    boot = cfg["payload_bytes"]["bootstrap"]
    bu, bd = boot["up"] * K, boot["down"] * K
    ru, rd = sum(r["bytes_up"] for r in rows) * K, sum(r["bytes_down"] for r in rows) * K
    total = bu + bd + ru + rd
    return {"run_id": f"canonical:{ref}", "run_kind": "production", "scenario": scen, "backbone": BACKBONE[scen],
            "method": "mobihet", "seed": seed, "n_clients": K, "rounds_planned": 50, "rounds_executed": 50, "complete": True,
            "bootstrap_up_B": bu, "bootstrap_down_B": bd, "recurring_up_B": ru, "recurring_down_B": rd,
            "metadata_up_B": 0.0, "metadata_down_B": 0.0, "recurring_per_client_per_round_B": (ru + rd) / K / 50,
            "recurring_up_per_client_per_round_B": ru / K / 50, "recurring_down_per_client_per_round_B": rd / K / 50,
            "total_up_B": bu + ru, "total_down_B": bd + rd, "total_B": total, "total_per_client_B": total / K,
            "total_MB": total / 1e6, "total_GB": total / 1e9, "total_incl_attributed_B": total,
            "formula": "K*sum_r bytes_up_r | K*sum_r bytes_down_r | K*bootstrap_mean(up,down)", "basis": BASIS,
            "confidence": "high", "is_network_traffic": "no (simulated payload)", "rounds_source": str(d / "rounds.jsonl"),
            "config_source": "own", "notes": "MobiHet = L4 without LA (24/11/2026); audited R50 (scripts/audit_ablation.py)",
            "pilot_in_canonical": False}


def main():
    runs = []
    for scen in BACKBONE:
        for r in read(REPO / "experiments" / scen / "results.csv"):
            if r["method"] == "mobihet" and r["round"] == "50" and r["status"] == "audited":
                runs.append(measure(scen, int(r["seed"]), r["source"].split()[1]))
    per = read(OUT / "communication_per_run.csv")
    cols = list(per[0])
    per = [r for r in per if r["method"] != "mobihet_l4"] + [{c: r.get(c, "") for c in cols} for r in runs]
    write(OUT / "communication_per_run.csv", per, cols)
    summ = read(OUT / "communication_summary.csv")
    scols = list(summ[0])
    summ = [r for r in summ if r["method"] != "mobihet_l4"]
    for scen in BACKBONE:
        g = [r for r in runs if r["scenario"] == scen]
        if not g:
            continue
        rec = {"scenario": scen, "method": "mobihet", "backbone": BACKBONE[scen], "n_seeds": len(g),
               "seeds": ",".join(str(r["seed"]) for r in sorted(g, key=lambda r: r["seed"])), "n_clients": K, "rounds": 50}
        for col in ("bootstrap_up_B", "bootstrap_down_B", "recurring_up_B", "recurring_down_B", "metadata_up_B",
                    "metadata_down_B", "recurring_per_client_per_round_B", "total_per_client_B", "total_B",
                    "total_incl_attributed_B"):
            v = [float(r[col]) for r in g]
            rec[col + "_mean"] = st.mean(v)
            rec[col + "_std"] = st.stdev(v) if len(v) > 1 else float("nan")
        rec["total_GB_mean"] = rec["total_B_mean"] / 1e9
        rec["total_incl_attributed_GB_mean"] = rec["total_incl_attributed_B_mean"] / 1e9
        rec.update(basis=BASIS, confidence="high",
                   data_source="rounds.jsonl + config.json of canonical sessions (see communication_per_run.csv)")
        summ.append({c: rec.get(c, "") for c in scols})
    write(OUT / "communication_summary.csv", summ, scols)
    rank_cols = list(read(OUT / "communication_ranking.csv")[0])
    rank = []
    for scen in dict.fromkeys(r["scenario"] for r in summ):
        g = [r for r in summ if r["scenario"] == scen]
        fed = next((float(r["total_incl_attributed_B_mean"]) for r in g if r["method"] == "fedavg"), None)
        for r in g:
            t = float(r["total_incl_attributed_B_mean"])
            rank.append({**{c: r.get(c, "") for c in rank_cols}, "rank_in_scenario": 1 + sum(float(x["total_incl_attributed_B_mean"]) < t for x in g),
                         "x_fedavg": t / fed if fed else float("nan"), "total_incl_attributed_GB_mean": t / 1e9})
    rank.sort(key=lambda r: (r["scenario"], int(r["rank_in_scenario"])))
    write(OUT / "communication_ranking.csv", rank, rank_cols)
    for r in rank:
        if r["method"] == "mobihet":
            print(r["scenario"], "n", r["n_seeds"], "B/vehicle %.1f" % float(r["total_per_client_B_mean"]), "rank", r["rank_in_scenario"],
                  "of", sum(x["scenario"] == r["scenario"] for x in rank))


if __name__ == "__main__":
    main()
