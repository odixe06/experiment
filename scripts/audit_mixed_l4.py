"""Audit any committed mixed L4 session and emit a receipt for cross-session resume."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.metrics import TEN, ten
from scripts.audit_l4 import append_csv_row
from scripts.run_mixed_l4 import l4_hp


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pull", type=Path)
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path,
                        help="output inventory with checkpoint lengths and Torch zip headers")
    parser.add_argument("--ref", required=True, help="exact owner/slug/version")
    parser.add_argument("--csv", type=Path, help="append an audited r50 result")
    args = parser.parse_args()
    owner, slug, version = args.ref.split("/")
    if not owner or not slug or not version.isdigit() or args.notebook.stem != slug:
        parser.error("expected exact ref and matching executed notebook filename")

    p = args.pull
    manifest, status, source, hashes = (
        read(p / name) for name in ("experiment_manifest.json", "RUN_STATUS.json", "source_snapshot.json", "source_manifest.json"))
    assert hashes == {name: hashlib.sha256(body.encode()).hexdigest() for name, body in source.items()}
    frozen = "".join(read(args.notebook)["cells"][2]["source"]).splitlines()[0]
    assert source == ast.literal_eval(frozen.removeprefix("source = "))
    assert manifest["method"] == "mobihet_l4" and manifest["rounds"] == 50
    assert manifest["dataset"] == "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
    case, seed = manifest["case"], manifest["seed"]
    assert seed in (41, 42, 43)
    run = p / manifest["run"]
    cfg, checkpoint = read(run / "config.json"), read(run / "checkpoint.json")
    assert cfg["case"] == status["case"] == case and cfg["backend"] == "compiled" and cfg["world"] == 2
    assert cfg["hp"] == l4_hp(seed) and cfg["test_set"] == []
    assignment = source[f"experiments/mixed-model-v1/{case}.csv"].encode()
    assert cfg["assignment_sha256"] == hashlib.sha256(assignment).hexdigest()
    mapping = {int(row["client_id"]): row["model"]
               for row in csv.DictReader(io.StringIO(assignment.decode()))}
    assert set(mapping.values()) == set(cfg["groups"])
    last = status["last_round"]
    assert 1 <= last <= min(50, manifest["stop_after"]) and last == checkpoint["round"]
    assert status["complete"] == (last == 50) and status["seed"] == seed
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    assert status["config_sha256"] == checkpoint["config_sha256"] == digest
    log = (p / f"{slug}.log").read_text()
    assert "lockstep PASS" in log and "recompile_limit" not in log
    assert all(f"compile {m} True" in log and f"compile {m} False" not in log for m in cfg["groups"])
    if manifest["session"] > 1:
        assert "resume after round " in log
    rows = [json.loads(line) for line in (run / "rounds.jsonl").read_text().splitlines()]
    assert [row["round"] for row in rows] == list(range(1, last + 1))
    for row in rows:
        assert math.isfinite(row["train_loss"])
        for model in cfg["groups"]:
            prefix = f"health/{model}/"
            assert row[prefix + "finite"] and row[prefix + "applied_steps"] > 0
            assert 0 <= row[prefix + "skipped_frac"] < .01
    assert read(run / f"round_r{last:03}.json") == rows[-1]
    with np.load(run / f"cms_r{last:03}.npz", allow_pickle=False) as proof:
        ids, cm = proof["ids"], proof["client"]
        assert np.array_equal(ids, cfg["clients"]) and len(ids) == len(set(ids)) == len(mapping) == 5343
        assert set(ids) == set(mapping)
        assert cm.shape == (5343, 16, 16) and np.issubdtype(cm.dtype, np.integer)
        assert cm.min() >= 0 and (cm.sum(axis=(1, 2)) > 0).all() and int(cm.sum()) == 9347679
        support_path = ROOT / "runs/test_support.npy"
        if support_path.is_file():
            assert np.array_equal(cm.sum(axis=2), np.load(support_path, allow_pickle=False)[ids])
        scores = ten(cm.sum(axis=0))
        models = np.array([mapping[int(cid)] for cid in ids])
        per_model = {model: ten(cm[models == model].sum(axis=0)) for model in cfg["groups"]}
    for prefix, values in {"client/": scores, **{f"model/{m}/client/": s for m, s in per_model.items()}}.items():
        assert all(abs(values[k] - rows[-1][prefix + k]) < 1e-9 for k in TEN)

    inventory = read(args.inventory)["checkpoints"]
    expected = {f"{manifest['run']}/checkpoint-r{last:03}/state_rank{rank}.pt" for rank in range(2)}
    assert expected <= inventory.keys()
    checks = {name: inventory[name] for name in sorted(expected)}
    assert all(record["torch_zip_header"] and int(record["bytes"]) > 0 for record in checks.values())
    receipt = dict(ref=args.ref, last_round=last, config_sha256=digest,
                   checkpoint_remote=checks, scores={k: float(scores[k]) for k in TEN},
                   t_train_r2=rows[1]["t_train"] if last > 1 else None,
                   per_model_f1={m: float(s["f1_macro"]) for m, s in per_model.items()})
    receipt["pass"] = True
    (p / "recovery-audit.json").write_text(json.dumps(receipt, indent=2))
    print(args.ref, "PASS", last, receipt["scores"]["f1_macro"])

    if args.csv:
        if last != 50:
            parser.error("--csv requires the r50 endpoint")
        fields = ["mobihet_l4", str(seed), "client", "50", *(f"{scores[k]:.6f}" for k in TEN), "audited",
                  f"kaggle {args.ref} (audit_mixed_l4.py)"]
        print("appended" if append_csv_row(args.csv, fields) else "already recorded", args.csv)


if __name__ == "__main__":
    main()
