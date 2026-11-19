"""Centralized 50-epoch siblings of run_central.py. Launch with torchrun, one model per run."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics
from core.central_backbones import build_model, classification_loss, predict
from core.sae import batch_code_mean, sae_loss

MICRO = {"bert": 32, "cnn-bigru-attention": 256, "mlp": 2048, "tabm": 1024, "sae": 2048, "sae-ce": 2048, "dagsnet": 2048}


def commit_epoch(staging, destination):
    # Windows scanners can briefly hold newly written checkpoint files open.
    for attempt in range(21):
        try:
            staging.rename(destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 20:
                raise
            time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MICRO, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42, help="init, shuffle and dropout streams (default = the seed-42 runs)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--micro", type=int)
    ap.add_argument("--max-hours", type=float, default=11)
    ap.add_argument("--stop-after", type=int, help="session boundary; retains original LR horizon")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe", action="store_true", help="warm up 3 steps, time 6 full train steps and sampled eval")
    ap.add_argument("--wandb", default="", help="optional online project, uses process-scoped credentials")
    ap.add_argument("--cpu", action="store_true", help="CPU fixture only")
    a = ap.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    micro = a.micro or MICRO[a.model]
    if min(micro, a.batch, a.epochs, a.max_hours) <= 0 or a.batch % world:
        raise ValueError("positive sizes and a global batch divisible by world size required")
    dev = torch.device("cpu" if a.cpu else f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    if not a.cpu:
        torch.cuda.set_device(dev)
    if world > 1:
        dist.init_process_group("gloo" if a.cpu else "nccl")
    start = time.monotonic()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    def barrier():
        if world > 1:
            dist.barrier()
    def load_array(split, field):
        arr = np.load(a.cache / f"{split}_{field}.npy", mmap_mode="c")
        return torch.from_numpy(arr).to(dev)
    def stream(split):
        x, y = load_array(split, "X"), load_array(split, "Y")
        if x.shape != (len(y), 77):
            raise ValueError("invalid cache layout")
        return x, y
    torch.manual_seed(a.seed)
    model = build_model(a.model).to(dev)
    code = hashlib.sha256()
    for fn in [Path(__file__), ROOT/"core/central_backbones.py",
               ROOT/"core/data.py", ROOT/"core/metrics.py", ROOT/"knowledge/input_order.json"]:
        code.update(fn.read_bytes())
    if a.model in ("dagsnet", "sae", "sae-ce"):
        code.update((ROOT / "core" / f"{a.model.removesuffix('-ce')}.py").read_bytes())
    if a.model == "cnn-bigru-attention":
        code.update((ROOT/"core/cnn_bigru_attention.py").read_bytes())
    cfg = dict(model=a.model, epochs=a.epochs, batch=a.batch, micro=micro, seed=a.seed,
               world=world, device=dev.type, torch=str(torch.__version__), source=code.hexdigest(),
               data=json.loads((a.cache/"identity.json").read_text()),
               params=sum(p.numel() for p in model.parameters()), optimizer="AdamW", wd=1e-4,
               lr_start=1e-3, lr_end=1e-5, clip=1.)
    epochs = sorted(p for p in out.glob("epoch_[0-9][0-9][0-9]") if p.is_dir())
    first = len(epochs) + 1
    if [p.name for p in epochs] != [f"epoch_{i:03}" for i in range(1, first)]:
        raise RuntimeError("noncontiguous checkpoints")
    if a.resume != bool(epochs):
        raise RuntimeError("resume requires committed epochs; fresh run requires empty history")
    config = out/"config.json"
    # rank 0 checks then writes; the others check only after the barrier (20/11: rank 1 read a half-written file)
    if rank == 0 and config.exists() and json.loads(config.read_text()) != cfg:
        raise RuntimeError("config/source/data mismatch")
    if rank == 0:
        config.write_text(json.dumps(cfg, indent=2))
    barrier()
    if rank > 0 and json.loads(config.read_text()) != cfg:
        raise RuntimeError("config/source/data mismatch")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, fused=not a.cpu)
    scaler = torch.amp.GradScaler("cuda", enabled=not a.cpu)
    if a.model == "gru-attention" and not a.resume and rank == 0:
        model.fit_pca(np.load(a.cache/"train_X.npy", mmap_mode="r"))
        print("Fitted five PCA components on train_X", flush=True)
    # sae-ce trains the FL SAE with CE only: the decoder is stored but never in the loss (like the FL layout), so DDP must skip it
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=None if a.cpu else [dev.index],
                                                    find_unused_parameters=a.model == "sae-ce") if world > 1 else model
    if a.resume:
        model.load_state_dict(torch.load(epochs[-1]/"weights.pt", weights_only=True, map_location=dev))
        state = torch.load(epochs[-1]/"resume.pt", weights_only=True, map_location="cpu")
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
    run = None
    if rank == 0 and a.wandb:
        import wandb
        run = wandb.init(project=a.wandb, name=out.name, id=f"{out.name}-{cfg['source'][:10]}",
                         resume="allow", config=cfg)
    x, y = stream("train")
    vx, vy = stream("test")
    n = len(y)
    steps = math.ceil(n / a.batch)
    worst = 0.
    for ep in range(first, a.epochs + 1):
        tick = time.monotonic()
        # Epoch-local deterministic streams also recover dropout/shuffle after a process restart.
        torch.manual_seed(a.seed + ep * world + rank)
        gen = torch.Generator(device=dev).manual_seed(a.seed + ep)
        perm = torch.randperm(n, device=dev, generator=gen)
        lr = 1e-5 + .5 * (1e-3-1e-5) * (1 + math.cos(math.pi*(ep-1)/max(1, a.epochs-1)))
        opt.param_groups[0]["lr"] = lr
        model.train()
        total = torch.zeros((), dtype=torch.float64, device=dev)
        skipped = 0
        probe_warmup = 3
        probe_steps = min(steps, 9)
        for step in range(probe_steps if a.probe else steps):
            if a.probe and step == probe_warmup:
                if not a.cpu:
                    torch.cuda.synchronize()
                probe_start = time.monotonic()
            rows = perm[step*a.batch:(step+1)*a.batch]
            local = rows[rank::world]
            calls = math.ceil(math.ceil(len(rows)/world)/micro)
            batch_mean = None
            if a.model == "sae" and calls > 1:
                batch_mean = batch_code_mean(model, x, local, micro, len(rows), amp=not a.cpu)
            opt.zero_grad(set_to_none=True)
            for i in range(calls):
                ix = local[i*micro:(i+1)*micro]
                count = len(ix)
                if not count:
                    ix = rows[:1]  # equal DDP collective count, zero gradient weight
                sync = ddp.no_sync() if world > 1 and i+1 < calls else nullcontext()
                with sync:
                    bx = x[ix]
                    with torch.autocast(dev.type, dtype=torch.float16, enabled=not a.cpu):
                        inputs = bx.float() if a.cpu else bx
                        if a.model == "sae":
                            outputs = ddp(inputs, auxiliary=True)
                        else:
                            logits = ddp(inputs)
                    if a.model == "sae":
                        loss = sae_loss(outputs, y[ix].long(), count, len(rows), batch_mean)
                    else:
                        loss = classification_loss(logits, y[ix].long())
                    scaler.scale(loss*(world*count/len(rows))).backward()
                total += loss.detach()*count
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            if step == 0 or (step+1) % 250 == 0:
                print(f"{a.model} rank={rank} epoch={ep} step={step+1}/{steps} grad={float(norm):.4g} skipped={skipped}", flush=True)
        if not a.cpu:
            torch.cuda.synchronize()
        train_seconds = time.monotonic()-tick
        if a.model == "cnn-bigru-attention" and world > 1:
            # DDP updates running BatchNorm statistics locally on the final batch.
            # All test shards must use the same buffers saved in rank 0's checkpoint.
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)
        if a.probe:
            if probe_steps <= probe_warmup or skipped == probe_steps or not torch.isfinite(total):
                raise RuntimeError("probe has no finite applied training evidence")
            train_estimate = (time.monotonic()-probe_start)*steps/(probe_steps-probe_warmup)
            model.eval()
            begin = time.monotonic()
            sample = min(32*micro, len(vy)//world)
            with torch.inference_mode():
                for j in range(0, sample, micro*2):
                    ix = torch.arange(j,min(j+2*micro,sample),device=dev) + rank*(len(vy)//world)
                    bx = vx[ix]
                    with torch.autocast(dev.type, dtype=torch.float16, enabled=not a.cpu):
                        z = model(bx.float() if a.cpu else bx)
                    if not torch.isfinite(z).all():
                        raise RuntimeError("nonfinite probe eval")
                    predict(z).cpu()
            seconds = torch.tensor([train_estimate,(time.monotonic()-begin)*len(vy)/world/sample],device=dev)
            if world > 1:
                dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            if rank == 0:
                estimate = dict(model=a.model, micro=micro, batch=a.batch, steps=steps, skipped=skipped,
                                estimated_train_eval_seconds_per_epoch=seconds.tolist(),
                                estimated_hours_50_epochs=float(seconds.sum())*50/3600,
                                peak_allocated_gib=0 if a.cpu else torch.cuda.max_memory_allocated()/2**30)
                (out/"probe.json").write_text(json.dumps(estimate,indent=2))
                print("PROBE",estimate,flush=True)
            break
        if not torch.isfinite(total) or skipped > max(4, steps//100) or skipped == steps:
            raise RuntimeError(f"unstable training epoch {ep} skipped={skipped}")
        model.eval()
        tmp, final = out/f".epoch_{ep:03}", out/f"epoch_{ep:03}"
        if rank == 0:
            tmp.mkdir(exist_ok=True)
        barrier()
        lo, hi = rank*len(vy)//world, (rank+1)*len(vy)//world
        preds = np.empty(hi-lo, np.uint8)
        with torch.inference_mode():
            for j in range(lo, hi, micro*2):
                ix = torch.arange(j, min(j+micro*2, hi), device=dev)
                bx = vx[ix]
                with torch.autocast(dev.type, dtype=torch.float16, enabled=not a.cpu):
                    # Transformer encoder eval fast path can bypass autocast on older runtimes.
                    logits = model(bx.float() if a.cpu else bx)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite eval logits")
                preds[j-lo:j-lo+len(ix)] = predict(logits).cpu().numpy()
        truth = vy[lo:hi].cpu().numpy()
        np.save(tmp/f"pred_{rank}.npy", preds)
        if ep == 1:
            np.save(tmp/f"truth_{rank}.npy", truth)
        counts = np.bincount(truth.astype(np.int64)*16+preds, minlength=256)
        cm = torch.from_numpy(counts).to(dev)
        if world > 1:
            dist.all_reduce(cm)
            dist.all_reduce(total)
        if rank == 0:
            counts = cm.cpu().numpy().reshape(16, 16)
            assert counts.sum() == len(vy)
            result = metrics.ten(counts)
            row = dict(epoch=ep, lr=lr, loss=float(total)/n, applied=steps-skipped, skipped=skipped,
                       train_rows=n, eval_rows=len(vy), train_seconds=train_seconds,
                       eval_seconds=time.monotonic()-tick-train_seconds, **{k:result[k] for k in metrics.TEN},
                       f1_per_class=result["f1_per_class"])
            np.save(tmp/"cm.npy", counts)
            torch.save(model.state_dict(), tmp/"weights.pt")
            torch.save(dict(optimizer=opt.state_dict(), scaler=scaler.state_dict()), tmp/"resume.pt")
            (tmp/"metrics.json").write_text(json.dumps(row, allow_nan=False))
            with (tmp/"metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metrics.TEN)
                writer.writeheader()
                writer.writerow({key: row[key] for key in metrics.TEN})
            commit_epoch(tmp, final)  # atomic write boundary, shared filesystem
            if run:
                run.log({k:v for k,v in row.items() if not isinstance(v,list)}, step=ep)
            print(f"COMMITTED epoch={ep} f1_macro={result['f1_macro']:.6f}", flush=True)
        barrier()
        worst = max(worst, time.monotonic()-tick)
        stop = torch.tensor(int(time.monotonic()-start+1.15*worst > a.max_hours*3600), device=dev)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item() or ep == a.stop_after:
            break
    if run:
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.mha.set_fastpath_enabled(False)
    main()
