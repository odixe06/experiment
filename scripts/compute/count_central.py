"""Count the real centralized runner's nine-step probe, then project 50 epochs on the same panel."""
import argparse
from functools import wraps
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

from op_counter import OpCounter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("dagsnet", "bert", "cnn-bigru-attention", "tabm", "sae-ce", "mlp"))
    parser.add_argument("--panel-cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    from core import central_backbones

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    identity = {}
    for split, fold in (("train", "train"), ("test", "val")):
        for field in ("X", "Y", "ids"):
            sources = list(args.panel_cache.glob(f"train_sub200_*_{fold}_{field}.npy"))
            if len(sources) != 1:
                raise ValueError(f"expected one frozen 200-vehicle file per fold and field: {sources}")
            src = sources[0].resolve()
            dst = cache / f"{split}_{field}.npy"
            if not dst.is_symlink():
                dst.symlink_to(src)
            elif dst.resolve() != src:
                raise ValueError(f"cache points to a different dataset: {dst}")
            with src.open("rb") as stream:
                identity[dst.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    train_ids = np.load(cache / "train_ids.npy")
    if not np.array_equal(train_ids, np.load(cache / "test_ids.npy")) or len(train_ids) != 200:
        raise ValueError("fold assignment differs")
    (cache / "identity.json").write_text(json.dumps(identity, sort_keys=True))
    train_rows, test_rows = (len(np.load(cache / f"{s}_Y.npy", mmap_mode="r")) for s in ("train", "test"))

    # Count a single eager rank using the production 4096 global batch split over two ranks.
    # Skip fused cuDNN RNN so the dispatcher sees its GEMMs. DDP collectives are outside the count.
    torch.backends.cudnn.enabled = False
    # Inference-mode dispatch can expose composite linear/GRU operators as opaque calls.
    # no_grad keeps the same evaluation while exposing their component operations.
    torch.inference_mode = torch.no_grad
    counter = OpCounter()
    counter.phase = "bootstrap"

    def tagged(function, label):
        @wraps(function)
        def wrapped(*a, **k):
            with counter.scope(label):
                return function(*a, **k)
        return wrapped

    original_model = central_backbones.build_model

    def build(model_name):
        model = original_model(model_name)
        original_train = model.train

        def train(mode=True):
            counter.phase = "r1.train" if mode else "r1.eval"
            return original_train(mode)

        model.train = train
        model.forward = tagged(model.forward, "model_forward")
        return model

    central_backbones.build_model = build
    backward = torch.autograd.backward

    def measured_backward(*a, **k):
        with counter.scope("backward"), counter.autograd():
            return backward(*a, **k)

    torch.autograd.backward = measured_backward
    amp_step = torch.amp.GradScaler.step
    torch.amp.GradScaler.step = tagged(amp_step, "optimizer")
    torch.amp.GradScaler.unscale_ = tagged(torch.amp.GradScaler.unscale_, "optimizer")
    torch.nn.utils.clip_grad_norm_ = tagged(torch.nn.utils.clip_grad_norm_, "optimizer")
    runner = ROOT / "scripts/run_central_backbones.py"
    batch = 2048
    sys.argv = [str(runner), "--model", args.model, "--cache", str(cache), "--out", str(out / "probe"),
                "--epochs", "50", "--batch", str(batch), "--probe", *(["--cpu"] if args.cpu else [])]
    with (counter, counter.scope("server.training"),
          torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)):
        runpy.run_path(str(runner), run_name="__main__")
    probe = json.loads((out / "probe/probe.json").read_text())
    sample_steps = min(9, probe["steps"])
    if probe["skipped"] or sample_steps != 9:
        raise ValueError("need nine full applied updates for the projected count")
    stats = {
        "schema": 2, "status": "done", "model": args.model, "panel": 200, "torch": torch.__version__,
        "device": "cpu" if args.cpu else torch.cuda.get_device_name(),
        "panel_ids_sha256": hashlib.sha256(train_ids.astype(np.int64).tobytes()).hexdigest(),
        "train_rows": train_rows, "test_rows": test_rows, "source": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "data": identity, "rank_batch": batch, "sample_steps": sample_steps,
        "sample_eval_rows_per_rank": min(32 * probe["micro"], test_rows),
        "projected_rank_train_steps_per_epoch": (train_rows + 4095) // 4096,
        "projection_rank_multiplier": 2, "epochs": 50, "ops": counter.table(), "detail": counter.detailed_table(),
    }
    if not any(row["phase"] == "r1.eval" and row["mm_flops"] > 0 for row in stats["ops"]):
        raise ValueError("evaluation omitted model FLOPs")
    path = out / "central.count.json"
    path.write_text(json.dumps(stats, indent=1))
    print("CENTRAL_OPERATION_PROFILE_DONE", path, flush=True)


if __name__ == "__main__":
    main()
