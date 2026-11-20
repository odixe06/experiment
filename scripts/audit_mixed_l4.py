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
from core.fl import cosine_lr
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
    parser.add_argument("--interrupted", action="store_true",
                        help="certify the last committed checkpoint of a terminal kernel with no final marker")
    args = parser.parse_args()
    owner, slug, version = args.ref.split("/")
    if not owner or not slug or not version.isdigit() or args.notebook.stem != slug:
        parser.error("expected exact ref and matching executed notebook filename")

    p = args.pull
    manifest, source, hashes = (
        read(p / name) for name in ("experiment_manifest.json", "source_snapshot.json", "source_manifest.json"))
    assert hashes == {name: hashlib.sha256(body.encode()).hexdigest() for name, body in source.items()}
    frozen = "".join(read(args.notebook)["cells"][2]["source"]).splitlines()[0]
    assert source == ast.literal_eval(frozen.removeprefix("source = "))
    assert manifest["method"] == "mobihet_l4" and manifest["rounds"] == 50
    assert manifest["dataset"] == "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
    case, seed = manifest["case"], manifest["seed"]
    assert seed in (41, 42, 43)
    expected_run = f"mixl4-{case}-s{seed}"
    session = manifest["session"]
    assert manifest["run"] == expected_run
    assert slug == (expected_run if session == 1 else f"{expected_run}-h{session}")
    run = p / manifest["run"]
    cfg, checkpoint = read(run / "config.json"), read(run / "checkpoint.json")
    marker = p / "RUN_STATUS.json"
    interrupted = args.interrupted and not marker.exists()
    assert cfg["case"] == case and cfg["backend"] == "compiled" and cfg["world"] == 2
    assert cfg["hp"] == l4_hp(seed) and cfg["test_set"] == []
    runtime_files = [name for name in source if name.startswith(("core/", "methods/")) and name.endswith(".py")]
    runtime_files += ["scripts/run_mixed_l4.py", "scripts/build_l4_fl.py", "scripts/run_fl.py", "scripts/run_p21_recovery.py"]
    source_digest = hashlib.sha256()
    for name in sorted(runtime_files):
        source_digest.update(name.encode())
        source_digest.update(source[name].encode())
    assert cfg["source_sha256"] == source_digest.hexdigest()
    assignment_path = f"experiments/mixed-model-v1/{case}.csv"
    assignment = source[assignment_path].encode()
    assert assignment == (ROOT / assignment_path).read_bytes()
    assert cfg["assignment_sha256"] == hashlib.sha256(assignment).hexdigest()
    mapping = {int(row["client_id"]): row["model"]
               for row in csv.DictReader(io.StringIO(assignment.decode()))}
    assert set(mapping.values()) == set(cfg["groups"])
    last = checkpoint["round"]
    assert 1 <= last <= min(50, manifest["stop_after"])
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    assert checkpoint["config_sha256"] == digest
    if not interrupted:
        status = read(marker)
        assert status["case"] == case and status["seed"] == seed
        assert status["last_round"] == last and status["complete"] == (last == 50)
        assert status["config_sha256"] == digest
    log = (p / f"{slug}.log").read_text()
    assert "lockstep PASS" in log and "recompile_limit" not in log
    assert all(f"compile {m} True" in log and f"compile {m} False" not in log for m in cfg["groups"])
    if manifest["session"] > 1:
        assert "resume after round " in log
    rows = [json.loads(line) for line in (run / "rounds.jsonl").read_text().splitlines()]
    with (run / "rounds.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if interrupted:
        rows, csv_rows = rows[:last], csv_rows[:last]
    assert len(csv_rows) == last
    assert [row["round"] for row in rows] == list(range(1, last + 1))
    ids = np.array(cfg["clients"])
    assert len(ids) == len(set(ids)) == len(mapping) == 5343 and set(ids) == set(mapping)
    models = np.array([mapping[int(cid)] for cid in ids])
    masks = {model: models == model for model in cfg["groups"]}
    support_path = ROOT / "runs/test_support.npy"
    support = np.load(support_path, allow_pickle=False)[ids] if support_path.is_file() else None
    for row, csv_row in zip(rows, csv_rows):
        rnd = row["round"]
        assert read(run / f"round_r{rnd:03}.json") == row and int(csv_row["round"]) == rnd
        assert math.isfinite(row["train_loss"])
        expected_lr = cosine_lr(rnd, 50, cfg["hp"]["lr_max"], cfg["hp"]["lr_min"])
        assert abs(row["lr"] - expected_lr) < 1e-14 and float(csv_row["lr"]) == row["lr"]
        for model in cfg["groups"]:
            prefix = f"health/{model}/"
            assert row[prefix + "finite"] and row[prefix + "applied_steps"] > 0
            assert 0 <= row[prefix + "skipped_frac"] < .01
        with np.load(run / f"cms_r{rnd:03}.npz", allow_pickle=False) as proof:
            assert np.array_equal(proof["ids"], ids)
            cm = proof["client"]
        assert cm.shape == (5343, 16, 16) and np.issubdtype(cm.dtype, np.integer)
        assert cm.min() >= 0 and (cm.sum(axis=(1, 2)) > 0).all() and int(cm.sum()) == 9347679
        if support is not None:
            assert np.array_equal(cm.sum(axis=2), support)
        scores = ten(cm.sum(axis=0))
        per_model = {model: ten(cm[mask].sum(axis=0)) for model, mask in masks.items()}
        for prefix, values in {"client/": scores, **{f"model/{m}/client/": s for m, s in per_model.items()}}.items():
            for key in TEN:
                metric = prefix + key
                assert abs(values[key] - row[metric]) < 1e-9, (rnd, metric)
                assert float(csv_row[metric]) == row[metric], (rnd, metric)

    inventory = read(args.inventory)["checkpoints"]
    expected = {f"{manifest['run']}/checkpoint-r{last:03}/state_rank{rank}.pt" for rank in range(2)}
    assert expected <= inventory.keys()
    checks = {name: inventory[name] for name in sorted(expected)}
    assert all(record["torch_zip_header"] and int(record["bytes"]) > 0 for record in checks.values())
    receipt = dict(ref=args.ref, last_round=last, config_sha256=digest,
                   checkpoint_remote=checks, scores={k: float(scores[k]) for k in TEN},
                   confusion_rounds=len(rows), client_class_support_checked=support is not None,
                   interrupted=interrupted,
                   t_train_r2=rows[1]["t_train"] if last > 1 else None,
                   per_model_f1={m: float(s["f1_macro"]) for m, s in per_model.items()})
    receipt["pass"] = True
    (p / "recovery-audit.json").write_text(json.dumps(receipt, indent=2))
    print(args.ref, "PASS", last, receipt["scores"]["f1_macro"])

    if args.csv:
        if last != 50:
            parser.error("--csv requires the committed r50 endpoint")
        fields = ["mobihet_l4", str(seed), "client", "50", *(f"{scores[k]:.6f}" for k in TEN), "audited",
                  f"kaggle {args.ref} (audit_mixed_l4.py)"]
        print("appended" if append_csv_row(args.csv, fields) else "already recorded", args.csv)


if __name__ == "__main__":
    main()
