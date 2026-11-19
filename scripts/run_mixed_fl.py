"""One of 45 fixed mixed-model FL runs; shared server, per-model private banks."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from core import fl, local, grouped, metrics
from core.dagsnet import width
from core.model_assignment import CASES, load_assignment, sha256
from methods.mixed import Method, METHODS


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def source_hash():
    files = [p for folder in ("core", "methods") for p in (ROOT/folder).glob("*.py")]
    files.append(Path(__file__))
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=CASES, required=True)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--assignment-dir", type=Path, default=ROOT/"experiments/mixed-model-v1")
    ap.add_argument("--smoke-clients", type=int, default=0)
    ap.add_argument("--smoke-rows", type=int, default=0, help="Diagnostic only: cap each client's train/test rows")
    ap.add_argument("--stop-after", type=int, default=50)
    ap.add_argument("--max-hours", type=float, default=10.)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    mapping = load_assignment(args.assignment_dir, args.case)
    models = list(dict(CASES[args.case]))
    clients = None
    if args.smoke_clients:
        if args.smoke_clients < 2*len(models):
            raise ValueError("Smoke must include at least two clients per backbone")
        rng = np.random.default_rng(7)
        clients = []
        for i, model in enumerate(models):
            ids = [cid for cid, name in mapping.items() if name == model]
            count = args.smoke_clients//len(models) + (i < args.smoke_clients%len(models))
            clients.extend(rng.choice(ids, count, replace=False).tolist())
        clients.sort()
    hp = dict(rounds=50, seed=args.seed, batch=64, lr_max=1e-3, lr_min=1e-5,
              weight_decay=1e-4, clip=1., group_rows=8192, eval_rows=32768, amp=True)
    env = fl.Env(width(1), hp, clients=clients, cache=None)
    if args.smoke_rows:
        if not args.smoke_clients or args.smoke_rows < 1:
            raise ValueError("Row cap requires an explicit diagnostic client subset")
        from core.data import Split
        def capped(split):
            rows = torch.cat([torch.arange(int(split.off[i]), min(int(split.off[i+1]), int(split.off[i])+args.smoke_rows),
                                           device=env.dev) for i in range(env.K)])
            counts = (split.off[1:]-split.off[:-1]).clamp_max(args.smoke_rows)
            off = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            return Split(split.X[rows], split.Y[rows], off, split.ids)
        env.train, env.test = capped(env.train), capped(env.test)
        env.n = torch.as_tensor(env.train.sizes, dtype=torch.float32, device=env.dev)
    if not args.smoke_clients:
        assert set(env.all_ids.tolist()) == set(mapping)
        assert int(env.allreduce(env.n.double().sum())) == 37361273
        ntest = torch.tensor(len(env.test.Y), dtype=torch.int64, device=env.dev)
        assert int(env.allreduce(ntest)) == 9347679
    method = Method(env, args.method, mapping, models)
    backend = "eager"
    if args.compile:
        grouped.compile_all()
        # Preserve RNG; each model gate uses dropout-free cfg and only local data.
        rng, gen = torch.get_rng_state(), env.gen.get_state()
        cuda_rng = torch.cuda.get_rng_state(env.dev) if env.dev.type == "cuda" else None
        for group in method.groups:
            if not len(group.indices): continue
            gcfg = dict(group.cfg, dropout=0.)
            if "base" in gcfg: gcfg["base"] = dict(gcfg["base"], dropout=0.)
            gate_hp = dict(group.hp, group_rows=min(group.hp["group_rows"], 256))
            ok, report = local.compile_gate(group.lay, env.train, gcfg, gate_hp,
                                            torch.Generator(device=env.dev).manual_seed(0))
            print("compile", group.name, ok, report, flush=True)
            if not ok: raise RuntimeError(f"Mixed compile gate failed for {group.name}")
        torch.set_rng_state(rng); env.gen.set_state(gen)
        if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng, env.dev)
        backend = "compiled"
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(method=args.method, case=args.case, hp=hp, world=env.world,
                  clients=env.all_ids.tolist(), smoke_rows=args.smoke_rows,
                  assignment_sha256=sha256((args.assignment_dir/f"{args.case}.csv").read_bytes()),
                  source_sha256=source_hash(), backend=backend, payload_bytes=method.payload_bytes(),
                  groups={g.name: dict(cfg=g.cfg, stored_parameters=g.lay.P, buffers=g.lay.B,
                                      train_amp=g.hp.get("amp", True)) for g in method.groups})
    config_hash = sha256(json.dumps(config, sort_keys=True).encode())
    cp = args.out/"checkpoint.json"
    begin = 1
    if cp.exists():
        receipt = json.loads(cp.read_text())
        assert receipt["config_sha256"] == config_hash, "Resume config/source/assignment changed"
        round_no = int(receipt["round"])
        path = args.out/f"checkpoint-r{round_no:03d}"/f"state_rank{env.rank}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["round"] == round_no and state["config_sha256"] == config_hash
        method.load_state(state["method"])
        torch.set_rng_state(state["torch_rng"]); env.gen.set_state(state["gen"])
        if state["cuda_rng"] is not None: torch.cuda.set_rng_state(state["cuda_rng"], env.dev)
        begin = round_no+1
        if env.rank == 0:
            history = args.out/"rounds.jsonl"
            rows = [json.loads(line) for line in history.read_text().splitlines()]
            committed = [row for row in rows if row["round"] <= round_no]
            assert [row["round"] for row in committed] == list(range(1, round_no+1))
            history.write_text("".join(json.dumps(row)+"\n" for row in committed))
    elif (args.out/"rounds.jsonl").exists():
        raise RuntimeError("History exists without committed checkpoint; refusing fresh restart")
    if env.rank == 0:
        atomic_json(args.out/"config.json", config)
    if env.world > 1: torch.distributed.barrier()

    def save(r):
        folder = args.out/f"checkpoint-r{r:03d}"
        folder.mkdir(exist_ok=True)
        path = folder/f"state_rank{env.rank}.pt"
        body = dict(round=r, config_sha256=config_hash, method=method.state(), gen=env.gen.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(env.dev) if env.dev.type == "cuda" else None)
        torch.save(body, str(path)+".tmp"); os.replace(str(path)+".tmp", path)
        del body
        if env.world > 1: torch.distributed.barrier()
        if env.rank == 0:
            previous = json.loads(cp.read_text())["round"] if cp.exists() else None
            atomic_json(cp, dict(round=r, config_sha256=config_hash))
            if previous is not None and previous != r:
                old = args.out/f"checkpoint-r{int(previous):03d}"
                for rank in range(env.world): (old/f"state_rank{rank}.pt").unlink()
                old.rmdir()
        if env.world > 1: torch.distributed.barrier()

    saved, last, worst = begin-1, begin-1, 0.
    for r in range(begin, min(50, args.stop_after)+1):
        stop = torch.tensor(float(args.max_hours > 0 and time.monotonic()-started+1.2*worst > args.max_hours*3600), device=env.dev)
        if env.world > 1: torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
        if stop.item(): break
        t = time.monotonic()
        lr = fl.cosine_lr(r, 50, 1e-3, 1e-5)
        local.begin_round()
        info = method.round(r, lr)
        info.update(local.finish_round(env))
        if not math.isfinite(info["train_loss"]) or info["skipped_frac"] >= .01 or info["applied_steps"] <= 0:
            raise RuntimeError(f"Training health gate failed: {info}")
        train_seconds = time.monotonic()-t
        views = method.evaluate(args.out/"predictions"/f"r{r:03d}")
        row = dict(round=r, lr=lr, t_train=train_seconds, t_eval=time.monotonic()-t-train_seconds, **info)
        for view, cms in views.items():
            stat = metrics.summary(cms)
            for key, value in stat.items(): row[f"{view}/{key}"] = value
            for model in models:
                mask = np.array([mapping[int(cid)] == model for cid in env.all_ids])
                stat = metrics.summary(cms[mask])
                for key, value in stat.items(): row[f"model/{model}/{view}/{key}"] = value
        if env.rank == 0:
            np.savez_compressed(args.out/f"cms_r{r:03d}.npz", ids=env.all_ids, **views)
            atomic_json(args.out/f"round_r{r:03d}.json", row)
            with (args.out/"rounds.jsonl").open("a") as handle: handle.write(json.dumps(row)+"\n")
            history = [json.loads(line) for line in (args.out/"rounds.jsonl").read_text().splitlines()]
            with (args.out/"rounds.csv.tmp").open("w", newline="") as handle:
                columns = [k for k, v in row.items() if not isinstance(v, list)]
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader(); writer.writerows(history)
            os.replace(args.out/"rounds.csv.tmp", args.out/"rounds.csv")
            print(json.dumps({k: v for k, v in row.items() if k in ("round", "t_train", "t_eval", "skipped_frac", "client/f1_macro", "client_proto/f1_macro")}), flush=True)
        last = r
        if r % args.save_every == 0 or r == min(50, args.stop_after): save(r); saved = r
        worst = max(worst, time.monotonic()-t)
    if last > saved: save(last)
    if env.rank == 0:
        atomic_json(args.out/"RUN_STATUS.json", dict(method=args.method, case=args.case, seed=args.seed,
            last_round=last, target_round=50, complete=last == 50, config_sha256=config_hash))
    if env.world > 1: torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
