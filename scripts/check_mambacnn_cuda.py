"""Real-data, same-RNG CUDA scan equivalence gate before any production training."""
import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.mambacnn import MambaCNN, selective_scan_reference


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(42)
    reference = MambaCNN().cuda().train()
    candidate = copy.deepcopy(reference)
    models = (reference, candidate)
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4, fused=True) for m in models]
    scalers = [torch.amp.GradScaler("cuda") for _ in models]
    source_x = np.load(args.cache/"train_X.npy", mmap_mode="r")
    source_y = np.load(args.cache/"train_Y.npy", mmap_mode="r")
    rows = np.random.default_rng(42).choice(len(source_y), 704, replace=False)
    x = torch.tensor(source_x[rows], device="cuda")
    y = torch.tensor(source_y[rows], device="cuda", dtype=torch.long)
    records = []
    for step in range(20):
        values, grads, skips = [], [], []
        for index, (model, opt, scaler) in enumerate(zip(models, opts, scalers)):
            torch.manual_seed(1000+step)
            opt.zero_grad(set_to_none=True)
            backend = patch("core.mambacnn.selective_scan", selective_scan_reference) if index == 0 else nullcontext()
            with backend, torch.autocast("cuda", dtype=torch.float16):
                logits = model(x[step*32:(step+1)*32])
                loss = torch.nn.functional.cross_entropy(logits.float(), y[step*32:(step+1)*32])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad = torch.cat([p.grad.flatten() for p in model.parameters()])
            assert torch.isfinite(logits).all() and torch.isfinite(grad).all()
            values.append(logits.detach().float())
            grads.append(grad.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skips.append(int(scaler.get_scale() < scale))
        logit_error = float((values[1]-values[0]).abs().max())
        grad_relative = float((grads[1]-grads[0]).norm()/grads[0].norm().clamp_min(1e-8))
        assert logit_error < 0.005 and grad_relative < 0.01, (step, logit_error, grad_relative)
        assert skips == [0, 0], (step, skips)
        row = dict(step=step+1, max_logit_error=logit_error, gradient_relative_l2=grad_relative,
                   applied=[1-v for v in skips])
        records.append(row)
        print("MAMBA_EQ", row, flush=True)
    left = torch.cat([p.detach().flatten() for p in reference.parameters()])
    right = torch.cat([p.detach().flatten() for p in candidate.parameters()])
    weight_error = float((left-right).abs().max())
    differences = []
    identifiable_error = 0.
    for (name, p), (_, q) in zip(reference.named_parameters(), candidate.named_parameters()):
        diff = (p-q).detach().abs()
        index = int(diff.flatten().argmax())
        identifiable = diff.clone()
        if name in ("self_attention.in_proj_bias", "cross_attention.in_proj_bias"):
            # Key bias shifts every key score by the same q dot b; softmax cancels it.
            # Keep raw differences in evidence. Do not alter any trained parameter.
            identifiable[40:80] = 0
        identifiable_error = max(identifiable_error, float(identifiable.max()))
        differences.append(dict(name=name, max_error=float(diff.max()), flat_index=index,
                                identifiable_max_error=float(identifiable.max()),
                                reference_value=float(p.detach().flatten()[index]),
                                reference_gradient=float(p.grad.flatten()[index]),
                                candidate_gradient=float(q.grad.flatten()[index])))
    differences.sort(key=lambda v: v["max_error"], reverse=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        reference.eval()
        candidate.eval()
        with patch("core.mambacnn.selective_scan", selective_scan_reference):
            expected = reference(x[640:])
        actual = candidate(x[640:])
    eval_error = float((actual-expected).abs().max())
    disagreements = actual.argmax(-1) != expected.argmax(-1)
    top = expected.float().topk(2, dim=-1).values
    decisive_disagreements = int((disagreements & ((top[:, 0]-top[:, 1]) > 0.01)).sum())
    passed = identifiable_error < 0.002 and eval_error < 0.005 and decisive_disagreements == 0
    result = dict(status="PASS" if passed else "FAIL", acceptance="tolerance-equivalent", steps=records,
                  max_weight_error=weight_error, params=left.numel(), rows=rows.tolist(),
                  identifiable_max_weight_error=identifiable_error,
                  excluded_weight_coordinates="only key bias [40:80] in self/cross attention; exact softmax shift invariance",
                  parameter_differences=differences, eval_max_logit_error=float((actual-expected).abs().max()),
                  eval_prediction_disagreements=int(disagreements.sum()), eval_decisive_disagreements=decisive_disagreements,
                  scope="20 paired AMP optimizer steps, same microbatch32/RNG; not a full-training equivalence claim")
    args.out.write_text(json.dumps(result, indent=2))
    print("MAMBA_WEIGHT_DIAGNOSTIC", differences[:10], flush=True)
    print("MAMBA_EVAL_DIAGNOSTIC", result["eval_max_logit_error"], result["eval_prediction_disagreements"], flush=True)
    assert passed, (identifiable_error, eval_error, decisive_disagreements)
    print("MAMBA_CUDA_EQ_PASS", identifiable_error, flush=True)


if __name__ == "__main__":
    main()
