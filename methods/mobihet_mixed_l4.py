"""MobiHet L4 on the mixed-model cases (owner 20/11/2026): one private model per vehicle, backbone from the frozen case
assignment (core.model_assignment), each the MixedAdapter of that backbone (core.mixed_backbones.config, the model the four
prototype baselines of the mixed cases train). No parameter is aggregated.

Every backbone group is an unchanged methods.mobihet_l4 instance over its own vehicles (GroupEnv: the rank's rows of those
vehicles, the group's cfg/layout/schedule). What L4 shares through the server is a sum over vehicles (env.allreduce):
bootstrap copula/KM statistics, class-Q packets and their counts, loss/step/byte telemetry. Across groups these sums must
cover every vehicle of the case, as in one network. Lockstep runs the groups as virtual ranks: each group runs in its own
thread, one thread at a time in group order, and switches only at an allreduce; the last group adds the contributions in
group order, all-reduces the total across GPU ranks and writes it back into every group's tensor. Bootstrap tables and Q
therefore come out the same in every group (checked), each group keeps its backbone's loss, synthetic rows, real-row
selection and compact execution unchanged.

Module globals that L4 swaps during a round (core.local.local_epoch wrappers of budget/sim/refine, core.local._round_steps
step telemetry) are saved and restored per thread at every switch, so each group sees only its own wrappers and its own
optimizer-step counts (per-backbone health gate). TorchDispatchModes of the calling thread (the FLOP counter of
scripts/compute) are entered in every group thread, so its ops are counted; its dynamo/inductor config (thread-local:
recompile_limit) is copied in. The one rank-only collective of L4 is its round-1 compile guard (a count
over GPU ranks); GroupL4 marks it.
"""
import contextlib
from contextvars import copy_context
import sys
import threading
import warnings

import numpy as np
import torch

from core import data as D, fl, local
from core.data import Split
from core.flat import Layout
from core.mixed_backbones import config
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from methods.mobihet_l4 import Method as L4

SWAPPED = ("local_epoch", "_round_steps")
SHARED = ("cop_cdf", "cop_mass", "cop_mu", "cop_L", "km_p", "km_mu", "km_L", "mu", "L")


class Abort(Exception):
    pass


def compiler_configs():
    import torch._inductor.config as inductor
    return [(m, m.get_config_copy()) for m in (torch._dynamo.config, inductor)]


def load_configs(configs):
    """Thread-local compiler settings of the calling thread (grouped.compile_all: recompile_limit 64; a new thread reads the
    default 8, and past it dynamo silently runs that frame eager)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for module, saved in configs:
            current = module.get_config_copy()
            for key, value in saved.items():
                if current.get(key) != value:
                    setattr(module, key, value)


class Lockstep:
    """Run one callable per group as virtual ranks of a sum-allreduce (see module docstring)."""

    def __init__(self, n, reduce):
        self.n, self.reduce, self.cv, self.active = n, reduce, threading.Condition(), False

    def run(self, fns, dev):
        n = self.n
        assert len(fns) == n
        main = {k: getattr(local, k) for k in SWAPPED}
        self.slots = [{"local_epoch": main["local_epoch"], "_round_steps": []} for _ in range(n)]
        self.turn, self.posts, self.error, self.ended, self.calls = 0, [None] * n, None, False, 0
        results = [None] * n
        modes = _get_current_dispatch_mode_stack()                       # thread-local: an op counter of the caller
        configs = compiler_configs()                                     # thread-local too (torch 2.13): recompile_limit ...

        def body(i):
            try:
                if dev.type == "cuda":
                    torch.cuda.set_device(dev)
                load_configs(configs)
                with contextlib.ExitStack() as stack:
                    for mode in modes:
                        stack.enter_context(mode)
                    with self.cv:
                        self._wait(i)
                        self._restore(i)
                    results[i] = fns[i]()
                    self.sync(i, None, "END")
            except Abort:
                pass
            except BaseException as exc:                                 # noqa: BLE001  re-raised by the main thread
                with self.cv:
                    if self.error is None:
                        self.error = exc
                    self.cv.notify_all()

        threads = [threading.Thread(target=copy_context().run, args=(body, i), daemon=True) for i in range(n)]
        self.active = True
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            self.active = False
            for k, v in main.items():
                setattr(local, k, v)
        if self.error is not None:
            raise self.error
        return results

    def _wait(self, i):
        while self.turn != i and self.error is None and not self.ended:
            self.cv.wait()
        if self.error is not None:
            raise Abort()

    def _save(self, i):
        self.slots[i] = {k: getattr(local, k) for k in SWAPPED}

    def _restore(self, i):
        for k, v in self.slots[i].items():
            setattr(local, k, v)

    def sync(self, i, t, tag):
        with self.cv:
            self._save(i)
            self.posts[i] = (tag, t)
            if i < self.n - 1:
                self.turn = i + 1
                self.cv.notify_all()
            else:
                tags = [p[0] for p in self.posts]
                if len(set(tags)) != 1:
                    self.error = RuntimeError(f"lockstep collective mismatch at call {self.calls}: {tags}")
                    self.cv.notify_all()
                    raise Abort()
                self.calls += 1
                if tag == "END":
                    self.ended = True
                    self.cv.notify_all()
                    return None
                ts = [p[1] for p in self.posts]
                assert all(x.shape == t.shape and x.dtype == t.dtype and x.device == t.device for x in ts), \
                    (tag, [(tuple(x.shape), x.dtype) for x in ts])
                total = ts[0].clone()
                for x in ts[1:]:
                    total += x
                total = self.reduce(total)
                for x in ts:
                    x.copy_(total)
                self.turn = 0
                self.cv.notify_all()
            if tag == "END":
                while not self.ended and self.error is None:
                    self.cv.wait()
                return None
            self._wait(i)
            self._restore(i)
        return t


def subset(split, idx):
    """Rows of local clients idx (ascending) of a rank's Split, same order; new offsets."""
    off = split.off
    rows = torch.cat([torch.arange(int(off[i]), int(off[i + 1]), device=off.device) for i in idx])
    counts = off[1:][idx] - off[:-1][idx]
    new_off = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    return Split(split.X[rows], split.Y[rows], new_off, split.ids[idx])


class GroupEnv(fl.Env):
    """The rank's vehicles of one backbone, as an fl.Env (no data loading)."""

    def __init__(self, env, model, local_idx, all_ids, hp, step, k):
        self.world, self.rank, self.dev, self.gen = env.world, env.rank, env.dev, env.gen
        self.model, self.cfg, self.hp = model, config(model), hp
        self.lay = Layout(self.cfg)
        idx = torch.as_tensor(local_idx, dtype=torch.long, device=env.train.off.device)
        self.train, self.test = subset(env.train, idx), subset(env.test, idx)
        assert (self.train.ids == self.test.ids).all()
        self.ids = self.train.ids.cpu().numpy()
        assert (self.ids == env.ids[local_idx]).all()
        self.K = len(self.ids)
        self.all_ids = np.sort(np.asarray(all_ids))
        self.n = torch.as_tensor(self.train.sizes, device=self.dev, dtype=torch.float32)
        self.groups = local.group_schedule(self.train.sizes, list(range(self.K)), hp["batch"], hp["group_rows"])
        self._reduce, self.step, self.k, self.rank_only_next = env.allreduce, step, k, 0

    def allreduce(self, t):
        caller = sys._getframe(1)
        tag = f"{caller.f_code.co_filename.rsplit('/', 1)[-1]}:{caller.f_lineno}"
        if self.rank_only_next:
            assert tag.startswith("mobihet_next.py") and t.numel() == 1, tag     # L4's round-1 compile guard
            self.rank_only_next -= 1
            return self._reduce(t)
        if not self.step.active:                                         # evaluation: confusion matrices, ranks only
            return self._reduce(t)
        return self.step.sync(self.k, t, tag)


class GroupL4(L4):
    def round(self, r, lr):
        if r == 1:
            self.env.rank_only_next = 1
        return super().round(r, lr)


def group_hp(hp, model):
    """Engine packing only (as methods.mixed): BERT attention activations limit grouped rows; batch/epochs unchanged."""
    return dict(hp, group_rows=min(hp["group_rows"], 256), eval_rows=256) if model == "bert" else dict(hp)


class Method:
    name = "mobihet_l4"
    per_client = True

    def __init__(self, env, assignment, models):
        self.env, self.models = env, list(models)
        self.step = Lockstep(len(models), env.allreduce)
        self.envs = []
        for k, model in enumerate(self.models):
            local_idx = [i for i, cid in enumerate(env.ids) if assignment[int(cid)] == model]
            all_ids = [int(c) for c in env.all_ids if assignment[int(c)] == model]
            assert local_idx, f"rank {env.rank} has no {model} vehicle"
            self.envs.append(GroupEnv(env, model, local_idx, all_ids, group_hp(env.hp, model), self.step, k))
        assert sum(e.K for e in self.envs) == env.K
        env.train = env.test = None                                      # group copies hold every row now
        if env.dev.type == "cuda":
            torch.cuda.empty_cache()
        self.groups = self.step.run([lambda e=e: GroupL4(e) for e in self.envs], env.dev)
        self._check_shared()

    def _check_shared(self):
        g0 = self.groups[0]
        for g in self.groups[1:]:
            for key in SHARED:
                if hasattr(g0, key):
                    assert torch.equal(getattr(g0, key), getattr(g, key)), f"bootstrap {key} differs across groups"
            assert g.bootstrap_bytes == g0.bootstrap_bytes
            assert (g.Q is None) == (g0.Q is None) and (g.Q is None or torch.equal(g.Q, g0.Q)), "class Q differs across groups"

    def round(self, r, lr):
        outs = self.step.run([lambda g=g: g.round(r, lr) for g in self.groups], self.env.dev)
        self._check_shared()
        info, health = {}, {}
        for model, env, g, slot in zip(self.models, self.envs, self.groups, self.step.slots):
            steps = env._reduce(torch.stack(slot["_round_steps"]).sum(0).double())
            if local._round_steps is not None:
                local._round_steps.extend(slot["_round_steps"])
            finite = env._reduce(torch.tensor(float(torch.isfinite(g.priv_p).all() and torch.isfinite(g.priv_b).all()),
                                              device=env.dev))
            health.update({f"health/{model}/applied_steps": float(steps[0]), f"health/{model}/skipped_steps": float(steps[1]),
                           f"health/{model}/skipped_frac": float(steps[1] / steps.sum().clamp_min(1)),
                           f"health/{model}/finite": float(finite) == env.world})
        for key in sorted(set().union(*outs)):
            vals = [o.get(key) for o in outs]
            if all(v == vals[0] for v in vals):
                info[key] = vals[0]
            else:
                info.update({f"group/{m}/{key}": v for m, v in zip(self.models, vals)})
        return {**info, **health}

    def evaluate(self):
        env = self.env
        full = np.zeros((len(env.all_ids), D.N_CLS, D.N_CLS), dtype=np.int64)
        for e, g in zip(self.envs, self.groups):
            views = g.evaluate()
            assert list(views) == ["private"], list(views)
            full[np.searchsorted(env.all_ids, e.all_ids)] = views["private"]
        return {"client": full}

    def payload_bytes(self):
        return {"bootstrap": self.groups[0].bootstrap_bytes, "codec": getattr(self.groups[0], "codec_bytes", None),
                "per_round": "rounds.jsonl bytes_up/bytes_down (class-Q packets, per vehicle)"}

    def state(self):
        return {"groups": {m: {"ids": e.ids.tolist(), "state": g.state()}
                           for m, e, g in zip(self.models, self.envs, self.groups)}}

    def load_state(self, state):
        assert list(state["groups"]) == self.models
        for m, e, g in zip(self.models, self.envs, self.groups):
            saved = state["groups"][m]
            assert saved["ids"] == e.ids.tolist(), f"client/model bank mismatch for {m}"
            g.load_state(saved["state"])
