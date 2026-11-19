"""Frozen model assignments for the three mixed-model cases (no data repartition)."""
import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

N_CLIENTS = 5343
ASSIGNMENT_SEED = 2026
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
CASES = {
    "case1": [("dagsnet", 30), ("bert", 30), ("cnn-bigru-attention", 40)],
    "case2": [("tabm", 30), ("sae", 30), ("mlp", 40)],
    "case3": [("dagsnet", 20), ("bert", 20), ("cnn-bigru-attention", 20),
              ("tabm", 20), ("sae", 20)],
}


def counts(case):
    """Largest remainder, ties resolved by the model order above; integer arithmetic."""
    ratios = CASES[case]
    values = [N_CLIENTS * pct // 100 for _, pct in ratios]
    order = sorted(range(len(ratios)), key=lambda i: (-(N_CLIENTS * ratios[i][1] % 100), i))
    for i in order[:N_CLIENTS - sum(values)]:
        values[i] += 1
    return dict(zip((model for model, _ in ratios), values))


def assignment(case):
    """A separate deterministic shuffle per case, independent of training RNG/seeds."""
    ids = list(range(N_CLIENTS))
    random.Random(f"{ASSIGNMENT_SEED}:{case}").shuffle(ids)
    result, start = {}, 0
    for model, count in counts(case).items():
        result.update((cid, model) for cid in ids[start:start + count])
        start += count
    return dict(sorted(result.items()))


def csv_bytes(case):
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["client_id", "model"])
    writer.writerows(assignment(case).items())
    return buf.getvalue().encode("utf-8")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def bundle():
    files = {f"{case}.csv": csv_bytes(case) for case in CASES}
    meta = {
        "schema_version": 1,
        "dataset": DATASET,
        "partition": "unchanged natural receiver-vehicle clients; train+val -> test",
        "client_count": N_CLIENTS,
        "client_id_range": [0, N_CLIENTS - 1],
        "assignment_seed": ASSIGNMENT_SEED,
        "assignment_algorithm": "Python Random(str(seed)+':'+case).shuffle(sorted client IDs)",
        "rounding": "largest remainder; ties by declared model order",
        "training_seeds": [41, 42, 43],
        "fixed_across": ["methods", "training_seeds", "rounds", "GPU_shards", "resume"],
        "cases": {case: {
            "ratios_percent": dict(CASES[case]), "counts": counts(case),
            "file": f"{case}.csv", "sha256": sha256(files[f"{case}.csv"]),
        } for case in CASES},
    }
    files["manifest.json"] = (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return files


def write_bundle(directory):
    """Idempotent creation; refuse to overwrite any differing frozen artifact."""
    directory = Path(directory)
    files = bundle()
    for name, raw in files.items():
        path = directory / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Frozen assignment differs: {path}; create a new version explicitly")
    directory.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        path = directory / name
        if not path.exists():
            path.write_bytes(raw)


def load_assignment(directory, case, client_ids=None):
    """Verify the lock and return a global-ID mapping, optionally in shard ID order.

    Future runners must record the CSV hash in checkpoint/config and compare on resume.
    Loading this module alone does not add mixed-model support to the existing runner.
    """
    directory = Path(directory)
    expected = bundle()
    if (directory / "manifest.json").read_bytes() != expected["manifest.json"]:
        raise ValueError("Assignment manifest differs from the locked specification")
    filename = f"{case}.csv"
    raw = (directory / filename).read_bytes()
    if raw != expected[filename]:
        raise ValueError(f"Assignment content/hash mismatch: {case}")
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    result = {int(row["client_id"]): row["model"] for row in rows}
    if Counter(result.values()) != counts(case) or set(result) != set(range(N_CLIENTS)):
        raise ValueError("Assignment coverage/count mismatch")
    if client_ids is None:
        return result
    ids = list(client_ids)
    if len(set(ids)) != len(ids) or not set(ids).issubset(result):
        raise ValueError("Duplicate or unknown client IDs in shard")
    return {cid: result[cid] for cid in ids}
