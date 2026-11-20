"""Project full-horizon costs from schema-3 eager dispatch traces without mixing conventions."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "experiments/scenarios.json").read_text())
TOP = 10**12
# E = executed train rows, F = per-row model work, U = executed client steps, P = parameters.
COMPLEXITY = {
    "centralized": "O(50*(N*F+U*P))",
    "fedavg": "O(E*F+U*P+K*P)",
    "fedprox": "O(E*F+U*P+K*P)",
    "afpha": "O(E*(F+C)+U*P+K*P)",
    "fd_ids": "O(E*(F_train+F_teacher+C)+U*P+K*P)",
    "apfed": "O((E_private+E_shared)*F+(U_private+U_shared)*P+K*P)",
    "lwfednids": "O(E*F_pruned+U*P_pruned+K*P_pruned)",
    "v2x_htfl": "O(E*(F+D*H+H*C)+U*P+S*K*C*(H^2+H*C))",
    "canperfl": "O(50*(E*F+U*P+K*P)+10*(E*F+U*P))",
    "tinyproto": "O((E+N)*F+U*P+E*H+K*C*H)",
    "pfedes": "O((E_private+E_proxy)*(F+F_proxy)+U_private*P+U_proxy*P_proxy+K*P_proxy)",
    "nilm": "O(E*(F_private+F_proxy+C)+U*(P_private+P_proxy)+K*P_proxy)",
    "fedtgp": "O((E+N)*F+U*P+E*H+S*K*C^2*H)",
    "perfedskd": "O(E*(F_train+F_teacher+C)+U*P+K_sel*P)",
    "fedproto": "O(E*(F+H)+U*P+K*C*H)",
    "mobihet_l4": "O(E_mix*F+U*P+E_synth*(D^2+D*log B+C)+K*C^2)",
}
SIDES = ("client", "server", "other", "unattributed", "bootstrap", "preprocessing", "evaluation")
CONVENTIONS = ("formula", "arithmetic_estimate", "element_work")
PREFIX = re.compile(r"(dags-base|bert|tabm|sae|mlp|cnn-bigru|mixed-case[123])-(.+)\.count\.json$")

@contextmanager
def atomic_csv(path):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", newline="") as stream:
            yield stream
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def side(phase, scope):
    if phase == "bootstrap":
        return "bootstrap"
    if phase == "preprocessing":
        return "preprocessing"
    if phase.endswith(".eval"):
        return "evaluation"
    for label in reversed(scope.split("/")):
        owner = label.split(".")[0]
        if owner in {"client", "server", "other"}:
            return owner
    return "unattributed"


def trace(path):
    value = json.loads(path.read_text())
    if value.get("schema") != 3 or value.get("status") != "done":
        return
    values = Counter()
    reconciliation = Counter()
    for row in value["detail"]:
        phase, device, convention, n = (row[k] for k in ("phase", "device", "convention", "operations"))
        if convention not in CONVENTIONS or not isinstance(n, int) or n < 0:
            raise ValueError(f"{path}: invalid operation count")
        owner = side(phase, row["scope"])
        values[phase, owner, convention] += n
        reconciliation[phase, device, convention == "formula"] += n
    for row in value["ops"]:
        phase, device = row["phase"], row["device"]
        if (reconciliation.pop((phase, device, True), 0) != row["mm_flops"] or
                reconciliation.pop((phase, device, False), 0) != row["other_ops"]):
            raise ValueError(f"{path}: detailed operations do not reconcile")
    if any(reconciliation.values()):
        raise ValueError(f"{path}: op detail lacks coarse parent")
    for phase in ("r1.train", "r2.train"):
        if not any(r["phase"] == phase and "/backward" in r["scope"] and r["convention"] == "formula"
                   and r["operations"] for r in value["detail"]):
            raise ValueError(f"{path}: no formula FLOPs in training backward")
    if value["panel"] != 200 or value["rows"]["group_rows_requested"] != 4096:
        raise ValueError(f"{path}: wrong comparison panel or packing")
    return value, values


def projection(values, rounds):
    output = Counter()
    for (phase, owner, convention), n in values.items():
        if phase.startswith("r2."):
            weight = rounds - 1
        elif phase.startswith("r1.") or phase in {"bootstrap", "preprocessing"}:
            weight = 1
        else:
            continue
        output[owner, convention] += n * weight
    return output


def fmt(n):
    return f"{n / TOP:.8g}"


def gather(dirs):
    found = defaultdict(dict)
    for directory in dirs:
        for path in sorted(directory.rglob("*.count.json")):
            match = PREFIX.fullmatch(path.name)
            if not match:
                continue
            scenario, method = match.groups()
            if method not in SPEC["methods"]:
                continue
            data = trace(path)
            if data is None:
                continue
            scenario = "dagsnet" if scenario == "dags-base" else scenario
            found[scenario][method] = (path, *data)
    return found

def gather_central(dirs):
    found = {}
    for directory in dirs:
        for path in sorted(directory.rglob("central.count.json")):
            raw = json.loads(path.read_text())
            if raw.get("schema") != 2 or raw.get("status") != "done" or raw.get("panel") != 200 or raw.get("sample_steps") != 9:
                continue
            scenario = {"sae-ce": "sae", "cnn-bigru-attention": "cnn-bigru"}.get(raw["model"], raw["model"])
            training_scale = Fraction(raw["projected_rank_train_steps_per_epoch"] * raw["epochs"] *
                                      raw["projection_rank_multiplier"], raw["sample_steps"])
            eval_scale = Fraction(raw["test_rows"] * raw["epochs"], raw["sample_eval_rows_per_rank"])
            scopes = {"bootstrap": ("bootstrap", raw["projection_rank_multiplier"]),
                      "r1.train": ("server", training_scale), "r1.eval": ("evaluation", eval_scale)}
            estimated = Counter()
            for op in raw["detail"]:
                owner, scale = scopes[op["phase"]]
                estimated[owner, op["convention"]] += op["operations"] * scale
            if not any(op["phase"] == "r1.train" and "backward" in op["scope"] and
                       op["convention"] == "formula" and op["operations"] for op in raw["detail"]):
                raise ValueError(f"{path}: central training has no formula FLOPs in backward")
            found[scenario] = path, raw, estimated
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-out", type=Path, default=ROOT / "runs/compute-detailed-v3/activity.csv")
    parser.add_argument("dirs", type=Path, nargs="*", default=[ROOT / "runs/compute-detailed-v3"])
    args = parser.parse_args()
    found = gather(args.dirs)
    central = gather_central(args.dirs)
    activity = Counter()
    # Exclusive scope paths preserve all attribution boundaries, including backwards and optimizer work.
    for scenario, methods in found.items():
        for method, (_, raw, _) in methods.items():
            for op in raw["detail"]:
                activity[scenario, method, op["phase"], op["scope"], op["convention"]] += op["operations"]
    for scenario, (_, raw, _) in central.items():
        for op in raw["detail"]:
            activity[scenario, "centralized", op["phase"], op["scope"], op["convention"]] += op["operations"]
    args.activity_out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_csv(args.activity_out) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("scenario", "method", "sample_phase", "scope", "convention", "sample_operations"))
        writer.writerows((*key, n) for key, n in sorted(activity.items()))
    for scenario in SPEC["scenarios"]:
        mixed = scenario.startswith("mixed-")
        baseline = "fedproto" if mixed else "fedavg"
        methods = SPEC["hetero"] if mixed else SPEC["methods"]
        legacy_path = ROOT / "experiments" / scenario / "compute.csv"
        with legacy_path.open(newline="") as stream:
            legacy = {r["method"]: r for r in csv.DictReader(stream)}
        candidates, projected, totals = found[scenario], {}, {}
        for m, (path, raw, counts) in candidates.items():
            if m == "canperfl" and baseline not in projected:
                continue
            estimate = projection(counts, 10 if m == "canperfl" else 50)
            if m == "canperfl":
                # Baseline's first stage and fine-tuning each include their own initialization.
                estimate += projected[baseline]
            projected[m] = estimate
        # Paths can arrive in any lexical order, including CANPerFL before FedAvg.
        if "canperfl" in candidates and baseline in projected and "canperfl" not in projected:
            projected["canperfl"] = projection(candidates["canperfl"][2], 10) + projected[baseline]
        if scenario in central:
            path, raw, estimate = central[scenario]
            candidates["centralized"] = path, {"rows": {"panel_ids_sha256": raw["panel_ids_sha256"]}}, None
            projected["centralized"] = estimate
        for m, estimate in projected.items():
            totals[m] = sum(estimate[key, conv] for key in SIDES if key != "evaluation" for conv in CONVENTIONS[:2])
        if len({raw["rows"]["panel_ids_sha256"] for _, raw, _ in candidates.values()}) > 1:
            raise ValueError(f"{scenario}: input panels differ")
        rank = {m: i for i, m in enumerate(sorted(totals, key=totals.get), 1)}
        old_rank = {r["method"]: i for i, r in enumerate(sorted(
            (r for r in legacy.values() if r["total_tops"]), key=lambda r: float(r["total_tops"])), 1)}
        cols = ["method", "status", "baseline", "rank", "ratio", "total_tflop_estimate", "total_formula_tflops",
                "total_arithmetic_estimate_tops", "total_element_work_tops"] + [
                    f"{key}_{metric}" for key in SIDES for metric in ("formula_tflops", "arithmetic_estimate_tops")] + [
                "legacy_total_tops", "legacy_rank", "legacy_ratio", "legacy_minus_detailed_tops",
                "detailed_with_element_work_tops", "big_o", "source", "trace_sha256", "legacy_source"]
        out = legacy_path.with_name("compute-detail.csv")
        with atomic_csv(out) as stream:
            writer = csv.DictWriter(stream, fieldnames=cols, lineterminator="\n")
            writer.writeheader()
            for m in methods:
                old = legacy.get(m, {})
                base_old = legacy.get(baseline, {}).get("total_tops")
                old_ratio = float(old["total_tops"]) / float(base_old) if base_old and old.get("total_tops") else None
                row = dict(method=m, baseline=baseline, big_o=COMPLEXITY[m], legacy_total_tops=old.get("total_tops"),
                           legacy_rank=old_rank.get(m), legacy_ratio=f"{old_ratio:.6g}" if old_ratio is not None else "",
                           legacy_source=old.get("source"),
                           status="legacy-only" if old.get("total_tops") else "pending")
                if m in projected:
                    path, raw, _ = candidates[m]
                    counts = projected[m]
                    formula, arithmetic, element_work = (
                        sum(counts[k, conv] for k in SIDES if k != "evaluation") for conv in CONVENTIONS)
                    status = "centralized-probe-projected" if m == "centralized" else "eager-trace-projected"
                    row.update(status=status, rank=rank[m],
                               ratio=f"{float(totals[m]/totals[baseline]):.6g}" if baseline in totals else "",
                               total_tflop_estimate=fmt(totals[m]), total_formula_tflops=fmt(formula),
                               total_arithmetic_estimate_tops=fmt(arithmetic), total_element_work_tops=fmt(element_work),
                               detailed_with_element_work_tops=fmt(totals[m] + element_work),
                               legacy_minus_detailed_tops=f"{float(old['total_tops'])-float(totals[m]/TOP):.8g}"
                               if old.get("total_tops") else "",
                               source=path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path,
                               trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    for key in SIDES:
                        row[f"{key}_formula_tflops"] = fmt(counts[key, "formula"])
                        row[f"{key}_arithmetic_estimate_tops"] = fmt(counts[key, "arithmetic_estimate"])
                writer.writerow(row)
        print(f"{scenario}: {len(projected)}/{len(methods)} detailed; {len(old_rank)} legacy")


if __name__ == "__main__":
    main()
