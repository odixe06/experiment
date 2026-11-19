"""Freeze v4 data identity and build the same row order as centralized DAGSNet."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import data
from core.central_windows import build_index


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--windows", action="store_true")
    a = ap.parse_args()
    root = Path(data.find_root())
    labels = json.loads((root/"label_mapping.json").read_text())
    expected_labels = json.loads((ROOT/"knowledge/label_mapping.json").read_text())["label_to_id"]
    if labels.get("label_to_id", labels) != expected_labels:
        raise ValueError("label mapping differs from the frozen v4 mapping")
    order = json.loads((root/"scaler.json").read_text())["order"]
    if order != data.ORDER[:71]:
        raise ValueError("scaler feature order differs from the frozen v4 feature order")
    if (a.cache/"identity.json").exists():
        raise FileExistsError("cache already sealed")
    data.build_npy(str(root), str(a.cache))
    identity = {k: digest(root/k) for k in ("label_mapping.json", "scaler.json")}
    for split, n in (("train",37361273), ("test",9347679)):
        x, y, ids, folds = [np.load(a.cache/f"{split}_{field}.npy", mmap_mode="r") for field in ("X","Y","ids","F")]
        if x.shape != (n,77) or not np.array_equal(ids,np.arange(5343)) or len(y) != n:
            raise ValueError(f"unexpected v4 {split} counts")
        expected = [32685166,4676107,0] if split == "train" else [0,0,9347679]
        if np.bincount(folds, minlength=3).tolist() != expected or y.min() != 0 or y.max() != 15:
            raise ValueError(f"unexpected v4 {split} folds/classes")
        if a.windows:
            build_index(root, split, a.cache)
        for field in ["X","Y","off","ids","F"] + (["prev"] if a.windows else []):
            name = f"{split}_{field}.npy"
            identity[name] = digest(a.cache/name)
    (a.cache/"identity.json").write_text(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
