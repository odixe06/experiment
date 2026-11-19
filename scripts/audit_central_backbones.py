"""Audit a committed centralized run using saved predictions, labels and counts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import metrics
from core.central_backbones import build_model


def audit(run, require_complete=True):
    cfg = json.loads((run/"config.json").read_text())
    dirs = sorted(run.glob("epoch_[0-9][0-9][0-9]"))
    if not dirs or (require_complete and len(dirs) != cfg["epochs"]):
        raise ValueError(f"found {len(dirs)} / {cfg['epochs']} committed epochs")
    model = build_model(cfg["model"])
    assert sum(p.numel() for p in model.parameters()) == cfg["params"]
    truth = [np.load(dirs[0]/f"truth_{r}.npy", mmap_mode="r") for r in range(cfg["world"])]
    for i, d in enumerate(dirs, 1):
        row = json.loads((d/"metrics.json").read_text())
        assert d.name == f"epoch_{i:03}" and row["epoch"] == i
        cm = np.zeros((16,16), np.int64)
        for r, y in enumerate(truth):
            p = np.load(d/f"pred_{r}.npy", mmap_mode="r")
            assert p.shape == y.shape and p.max() < 16 and y.min() >= 0 and y.max() < 16
            cm += np.bincount(y.astype(np.int64)*16+p, minlength=256).reshape(16,16)
        assert cm.sum() == row["eval_rows"]
        if cfg["device"] == "cuda":
            assert cm.sum() == 9347679 and row["train_rows"] == 37361273
        np.testing.assert_array_equal(cm, np.load(d/"cm.npy"))
        calculated = metrics.ten(cm)
        for key in [*metrics.TEN,"f1_per_class"]:
            np.testing.assert_allclose(calculated[key],row[key],rtol=0,atol=1e-12)
        model.load_state_dict(torch.load(d/"weights.pt",map_location="cpu",weights_only=True),strict=True)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert (d/"resume.pt").is_file()
    return len(dirs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    print("Verified committed epochs:",audit(args.run,not args.partial))
