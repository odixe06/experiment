"""MobiHet L4 on one mixed-model case (methods/mobihet_mixed_l4.py): frozen assignment, protocol and L4 hp (seed only).

    torchrun --standalone --nproc_per_node=2 scripts/run_mixed_l4.py --case case1 --seed 42 --out runs/x --compile   # 2xT4
    python scripts/run_mixed_l4.py --case case2 --seed 42 --out runs/smoke --smoke-clients 30 --stop-after 2      # probe

Same checkpoint contract as scripts/run_mixed_fl.py: checkpoint-rNNN/state_rank<r>.pt written by every rank, then
checkpoint.json (round + config hash) by rank 0, previous checkpoint removed; resume truncates rounds.jsonl to the committed
round and refuses a changed config/source/assignment. Health gate per backbone every round: finite loss and weights,
skipped optimizer steps < 1 %, applied steps > 0.
"""
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
from methods.mobihet_mixed_l4 import Method
from scripts.build_l4_fl import MOBIHET_HP
from scripts.run_fl import PROTOCOL


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def source_hash():
    files = [p for folder in ("core", "methods") for p in (ROOT/folder).glob("*.py")]
    files += [Path(__file__), ROOT/"scripts/build_l4_fl.py", ROOT/"scripts/run_fl.py", ROOT/"scripts/run_p21_recovery.py"]
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


def l4_hp(seed, overrides=()):
    def val(v):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v                                                     # bare string, e.g. arm=none
    hp = dict(PROTOCOL, rounds=50)
    hp.update({k: val(v) for k, v in (s.split("=", 1) for s in MOBIHET_HP)})   # MobiHet = L4, la_mode=off
    hp["seed"] = seed
    hp.update({k: val(v) for k, v in (s.split("=", 1) for s in overrides)})
    return hp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=CASES, required=True)
    ap.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--assignment-dir", type=Path, default=ROOT/"experiments/mixed-model-v1")
    ap.add_argument("--cache", default=None, help="local probes: data cache directory")
    ap.add_argument("--smoke-clients", type=int, default=0, help="probe: this many clients, balanced over backbones")
    ap.add_argument("--clients", default="", help="probe: exact client ids (comma list; local caches hold fixed panels)")
    ap.add_argument("--test-set", nargs="*", default=[], help="unit tests only: hp overrides k=v (never production)")
    ap.add_argument("--stop-after", type=int, default=50)
    ap.add_argument("--max-hours", type=float, default=10.)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    os.environ.setdefault("MOBIHET_RECOVERY_DIR", str(args.out/"recovery"))
    mapping = load_assignment(args.assignment_dir, args.case)
    models = list(dict(CASES[args.case]))
    clients = None
    if args.clients:
        clients = sorted(int(c) for c in args.clients.split(","))
    elif args.smoke_clients:
        rng = np.random.default_rng(7)
        clients = []
        for i, model in enumerate(models):
            ids = [cid for cid in mapping if mapping[cid] == model]
            count = args.smoke_clients//len(models) + (i < args.smoke_clients % len(models))
            clients.extend(rng.choice(ids, count, replace=False).tolist())
        clients.sort()
    if clients is not None:
        per = {m: sum(mapping[c] == m for c in clients) for m in models}
        if min(per.values()) < 2:
            raise ValueError(f"probe needs >= 2 clients per backbone: {per}")
    hp = l4_hp(args.seed, args.test_set)
    env = fl.Env(width(1), hp, clients=clients, cache=args.cache)
    if clients is None:
        assert set(env.all_ids.tolist()) == set(mapping)
        assert int(env.allreduce(env.n.double().sum())) == 37361273
        ntest = torch.tensor(len(env.test.Y), dtype=torch.int64, device=env.dev)
        assert int(env.allreduce(ntest)) == 9347679
    method = Method(env, mapping, models)
    backend = "eager"
    if args.compile:
        grouped.compile_all()
        rng, gen = torch.get_rng_state(), env.gen.get_state()
        cuda_rng = torch.cuda.get_rng_state(env.dev) if env.dev.type == "cuda" else None
        for genv in method.envs:
            gcfg = dict(genv.cfg, dropout=0.)
            gcfg["base"] = dict(gcfg["base"], dropout=0.)
            gate_hp = dict(genv.hp, group_rows=min(genv.hp["group_rows"], 64 * min(4, genv.K)))   # <= 4 real clients, no empty pad
            ok, report = local.compile_gate(genv.lay, genv.train, gcfg, gate_hp, torch.Generator(device=env.dev).manual_seed(0))
            print("compile", genv.model, ok, report, flush=True)
            if not ok:
                raise RuntimeError(f"compile gate failed for {genv.model}")
        torch.set_rng_state(rng); env.gen.set_state(gen)
        if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng, env.dev)
        backend = "compiled"
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(method="mobihet_l4", case=args.case, hp=hp, world=env.world, clients=env.all_ids.tolist(),
                  test_set=args.test_set, assignment_sha256=sha256((args.assignment_dir/f"{args.case}.csv").read_bytes()),
                  source_sha256=source_hash(), backend=backend, payload_bytes=method.payload_bytes(),
                  groups={e.model: dict(cfg=e.cfg, hp=e.hp, stored_parameters=e.lay.P, buffers=e.lay.B,
                                        clients=len(e.all_ids)) for e in method.envs})
    config_hash = sha256(json.dumps(config, sort_keys=True, default=str).encode())
    cp = args.out/"checkpoint.json"
    begin = 1
    if cp.exists():
        receipt = json.loads(cp.read_text())
        assert receipt["config_sha256"] == config_hash, "Resume config/source/assignment changed"
        round_no = int(receipt["round"])
        state = torch.load(args.out/f"checkpoint-r{round_no:03d}"/f"state_rank{env.rank}.pt", map_location="cpu",
                           weights_only=True)
        assert state["round"] == round_no and state["config_sha256"] == config_hash
        method.load_state(state["method"])
        torch.set_rng_state(state["torch_rng"]); env.gen.set_state(state["gen"])
        if state["cuda_rng"] is not None: torch.cuda.set_rng_state(state["cuda_rng"], env.dev)
        del state
        begin = round_no+1
        if env.rank == 0:
            history = args.out/"rounds.jsonl"
            rows = [json.loads(line) for line in history.read_text().splitlines()]
            committed = [row for row in rows if row["round"] <= round_no]
            assert [row["round"] for row in committed] == list(range(1, round_no+1))
            history.write_text("".join(json.dumps(row)+"\n" for row in committed))
        print(f"resume after round {round_no}", flush=True)
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
        lr = fl.cosine_lr(r, 50, hp["lr_max"], hp["lr_min"])
        local.begin_round()
        info = method.round(r, lr)
        info.update(local.finish_round(env))
        bad = [m for m in models if not (info[f"health/{m}/finite"] and info[f"health/{m}/skipped_frac"] < .01
                                         and info[f"health/{m}/applied_steps"] > 0)]
        if not math.isfinite(info["train_loss"]) or bad:
            raise RuntimeError(f"Training health gate failed at round {r}: {bad} {info}")
        train_seconds = time.monotonic()-t
        views = method.evaluate()
        row = dict(round=r, lr=lr, t_train=train_seconds, t_eval=time.monotonic()-t-train_seconds, **info)
        for view, cms in views.items():
            for key, value in metrics.summary(cms).items(): row[f"{view}/{key}"] = value
            for model in models:
                mask = np.array([mapping[int(cid)] == model for cid in env.all_ids])
                for key, value in metrics.summary(cms[mask]).items(): row[f"model/{model}/{view}/{key}"] = value
        if env.rank == 0:
            np.savez_compressed(args.out/f"cms_r{r:03d}.npz", ids=env.all_ids, **views)
            atomic_json(args.out/f"round_r{r:03d}.json", row)
            with (args.out/"rounds.jsonl").open("a") as handle: handle.write(json.dumps(row, default=float)+"\n")
            history = [json.loads(line) for line in (args.out/"rounds.jsonl").read_text().splitlines()]
            with (args.out/"rounds.csv.tmp").open("w", newline="") as handle:
                columns = [k for k, v in row.items() if not isinstance(v, (list, dict))]
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader(); writer.writerows(history)
            os.replace(args.out/"rounds.csv.tmp", args.out/"rounds.csv")
            print(json.dumps({k: v for k, v in row.items() if k in ("round", "t_train", "t_eval", "train_loss", "skipped_frac",
                              "real_frac", "client/f1_macro", "client/client_f1_macro_mean")
                              or k.endswith("/skipped_frac") or (k.startswith("model/") and k.endswith("/client/f1_macro"))},
                             default=float), flush=True)
        last = r
        if r % args.save_every == 0 or r == min(50, args.stop_after): save(r); saved = r
        worst = max(worst, time.monotonic()-t)
    if last > saved: save(last)
    if env.rank == 0:
        atomic_json(args.out/"RUN_STATUS.json", dict(method="mobihet_l4", case=args.case, seed=args.seed, last_round=last,
                    target_round=50, complete=last == 50, config_sha256=config_hash))
    if env.world > 1: torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
