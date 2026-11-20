#!/usr/bin/env python3
"""Audit and package missing non-checkpoint CNN-LSTM-feature centralized artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/central-cnn-lstm-feature-kaggle-output"
RUN = SOURCE / "central-cnn-lstm-feature"
ARCHIVE = ROOT / "artifacts/central-cnn-lstm-feature-supplement-2026-11-20.zip"
MANIFEST = ROOT / "artifacts/central-cnn-lstm-feature-supplement-2026-11-20.manifest.json"
TEST_ROWS = 9_347_679
TEN = (
    "accuracy", "precision_macro", "precision_micro", "precision_weighted",
    "recall_macro", "recall_micro", "recall_weighted",
    "f1_macro", "f1_micro", "f1_weighted",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_cm(cm: np.ndarray) -> dict[str, float]:
    values = cm.astype(np.float64)
    tp = np.diag(values)
    support = values.sum(axis=1)
    predicted = values.sum(axis=0)
    present = support > 0
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    total = values.sum()
    weights = support / total
    accuracy = tp.sum() / total
    return {
        "accuracy": accuracy,
        "precision_macro": precision[present].mean(),
        "precision_micro": accuracy,
        "precision_weighted": (precision * weights).sum(),
        "recall_macro": recall[present].mean(),
        "recall_micro": accuracy,
        "recall_weighted": (recall * weights).sum(),
        "f1_macro": f1[present].mean(),
        "f1_micro": accuracy,
        "f1_weighted": (f1 * weights).sum(),
    }


def audit() -> dict[str, object]:
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    assert config["model"] == "cnn-lstm-feature"
    assert int(config["epochs"]) == 50
    assert int(config["seed"]) == 42

    truth = [np.load(RUN / "epoch_001" / f"truth_{rank}.npy", mmap_mode="r") for rank in (0, 1)]
    assert sum(len(part) for part in truth) == TEST_ROWS
    audited = []
    for epoch in range(1, 51):
        folder = RUN / f"epoch_{epoch:03d}"
        metric = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        cm = np.load(folder / "cm.npy", allow_pickle=False)
        assert int(metric["epoch"]) == epoch
        assert cm.shape == (16, 16)
        assert np.issubdtype(cm.dtype, np.integer)
        assert np.all(cm >= 0)
        assert int(cm.sum()) == TEST_ROWS
        recomputed = metrics_from_cm(cm)
        for key in TEN:
            assert math.isfinite(float(metric[key]))
            assert math.isclose(float(metric[key]), float(recomputed[key]), rel_tol=0, abs_tol=1e-12)

        prediction_cm = np.zeros((16, 16), dtype=np.int64)
        for rank in (0, 1):
            prediction = np.load(folder / f"pred_{rank}.npy", mmap_mode="r")
            assert prediction.shape == truth[rank].shape
            assert prediction.min() >= 0 and prediction.max() < 16
            np.add.at(prediction_cm, (truth[rank], prediction), 1)
        assert np.array_equal(prediction_cm, cm)
        audited.append(epoch)
    return {
        "model": config["model"],
        "seed": config["seed"],
        "epochs": audited,
        "test_rows_per_epoch": TEST_ROWS,
        "checks": ["config", "metrics_finite", "cm_shape_count", "cm_to_10_metrics", "predictions_to_cm"],
        "status": "PASS",
    }


def package(audit_result: dict[str, object]) -> dict[str, object]:
    files = sorted(
        path for path in SOURCE.rglob("*")
        if path.is_file() and path.suffix.lower() not in {".pt", ".pth", ".ckpt"}
    )
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            relative = path.relative_to(SOURCE).as_posix()
            archive_path = "central-cnn-lstm-feature-supplement/" + relative
            bundle.write(path, archive_path)
            entries.append({
                "archive_path": archive_path,
                "source_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    with zipfile.ZipFile(ARCHIVE) as bundle:
        bad = bundle.testzip()
        assert bad is None, bad
        assert len(bundle.namelist()) == len(entries)
    manifest = {
        "source": "qwerasdf3004/veremi-central-cnn-lstm-feature-e50 v1",
        "scope": "all safe non-checkpoint output files",
        "audit": audit_result,
        "file_count": len(entries),
        "uncompressed_bytes": sum(item["bytes"] for item in entries),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = package(audit())
    print(json.dumps({key: result[key] for key in (
        "file_count", "uncompressed_bytes", "archive_bytes", "archive_sha256", "audit"
    )}, indent=2))
