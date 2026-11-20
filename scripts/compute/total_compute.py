"""Total compute of one method run, split by phase, one harness for every method and repo (owner 19/11/2026; copied from
../mobihet/scripts/total_compute.py on 20/11/2026, docs/research/compute.md). Changes: op_counter from this folder, fold counts
from $V4_CACHE/fold_counts.npy, and the mixed-model runner (scripts/run_mixed_fl.py imports methods.mixed directly) is wrapped.

    python scripts/total_compute.py count|time OUT.json --repo REPO [--panel 200] -- SCRIPT ARGS...

SCRIPT runs inside REPO exactly as it trains (scripts/sim_fold_val.py of MobiHet / fl_veremi_bert, or scripts/run_fl.py of a
baseline repo with --panel N: the same fold-train panel and fold-val split as sim_fold_val, applied here).
Phases: bootstrap = the method's constructor (copula / KM statistics, teacher, proxy setup ...); train = every local_epoch of a
round (client forward/backward/optimizer, synthetic rows, per-step loss terms); round_other = the rest of the round (server
aggregation, codecs, packets, prototypes ...); eval = the method's evaluate (reported apart, owner 19/11). Data loading and
the compile gate are outside every phase.
  count: eager (compile replaced by an eager TRAIN wrapper so method guards still see a non-reference TRAIN), every op
         counted by scripts/op_counter.py; also rows executed / trained by the grouped engine per round.
  time:  production path (--compile), wall time per phase with a CUDA synchronize at every phase boundary.
"""
import hashlib
import importlib
import json
import os
import runpy
import sys
import time
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

HERE = Path(__file__).resolve().parent
mode, out = sys.argv[1], Path(sys.argv[2]).resolve()
argv = sys.argv[3:]
cut = argv.index("--")
opts, script_argv = argv[:cut], argv[cut + 1:]
repo = Path(opts[opts.index("--repo") + 1]).resolve()
panel = int(opts[opts.index("--panel") + 1]) if "--panel" in opts else 0
method = "mobihet_l4" if Path(script_argv[0]).name == "run_mixed_l4.py" else script_argv[script_argv.index("--method") + 1]
assert mode in ("count", "time")
os.chdir(repo)
sys.path.insert(0, str(repo))
sys.path.insert(1, str(HERE))                                        # op_counter only (no repo package shadowing)

import torch  # noqa: E402
from op_counter import OpCounter  # noqa: E402

torch.backends.mha.set_fastpath_enabled(False)                       # fused MHA has no FLOP formula; decompose it
counter = OpCounter() if mode == "count" else None
phases = ContextVar("compute_phases", default=())
acc, t0 = {}, [None]
rows = {}
runner = Path(script_argv[0]).name
source_files = [repo / script_argv[0], *repo.glob("core/*.py"), *repo.glob("methods/*.py"), *HERE.glob("*.py")]
source_hash = hashlib.sha256()
for source in sorted(source_files, key=lambda p: str(p.relative_to(repo) if p.is_relative_to(repo) else p.name)):
    source_hash.update(source.name.encode() + b"\0" + source.read_bytes() + b"\0")


def scope(label):
    return counter.scope(label) if counter is not None else nullcontext()


def tag(owner, name, label):
    original = getattr(owner, name, None)
    if not callable(original):
        return

    @wraps(original)
    def measured(*args, **kwargs):
        with scope(label):
            return original(*args, **kwargs)

    setattr(owner, name, measured)



def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def push(ph):
    stack = phases.get()
    if mode == "time":
        sync()
        now = time.perf_counter()
        if stack:
            acc[stack[-1]] = acc.get(stack[-1], 0.0) + now - t0[0]
        t0[0] = now
    phases.set((*stack, ph))
    if counter is not None:
        counter.phase = ph


def pop():
    stack = phases.get()
    if mode == "time":
        sync()
        now = time.perf_counter()
        acc[stack[-1]] = acc.get(stack[-1], 0.0) + now - t0[0]
        t0[0] = now
    stack = stack[:-1]
    phases.set(stack)
    if counter is not None:
        counter.phase = stack[-1] if stack else None


def save(status):
    for v in rows.values():
        if isinstance(v, dict) and v.get("useful") is not None and not isinstance(v["useful"], int):
            v["useful"] = int(v["useful"])
    res = {"mode": mode, "repo": str(repo), "method": method, "argv": script_argv, "panel": panel, "status": status,
           "torch": torch.__version__, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
           "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, "rows": rows}
    if counter is not None:
        res.update(schema=3, ops=counter.table(), detail=counter.detailed_table(),
                   source_sha256=source_hash.hexdigest())
    else:
        res["seconds"] = acc
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(res, indent=1, default=str))
    temporary.replace(out)


cur = {"round": 0}


def wrap_instance(M):
    """Per-instance wrappers: run_fl calls M.round / M.evaluate; super() calls inside stay unwrapped."""
    rnd, ev = M.round, M.evaluate

    def round_(r, lr):
        cur["round"] = r
        push(f"r{r}.round_other")
        try:
            return rnd(r, lr)
        finally:
            pop()
            save("running")

    def evaluate(*a, **k):
        push(f"r{cur['round']}.eval")
        try:
            return ev(*a, **k)
        finally:
            pop()
    M.round, M.evaluate = round_, evaluate


def patch_local():
    """The engine module the methods actually use (fl_veremi_bert swaps core.local for core.local_t3)."""
    loc = sys.modules["core.local"]
    if getattr(loc, "_tc_patched", False):
        return
    epoch = loc.local_epoch
    if runner == "run_t3.py" and counter is not None:
        tag(loc.GroupAdamW, "step", "optimizer")

    def counted(*a, **k):
        stack = phases.get()
        top = stack[-1] if stack else ""
        push(top.replace("round_other", "train") if top.endswith("round_other") else top or "unphased.train")
        try:
            loss = k.get("loss_fn")
            role = "proxy" if getattr(loss, "__name__", "") == "step2" else "private_proxy_joint" if k.get("models") == 2 else "local"
            with scope(f"client.{role}"):
                return epoch(*a, **k)
        finally:
            pop()
    loc.local_epoch, loc._tc_patched = counted, True
    grouped = sys.modules["core.grouped"]

    def wrap_train(f):
        def train(P, B, x, cfg, mask=None, train=False, **k):
            stack = phases.get()
            if stack and stack[-1].endswith(".train"):
                r = rows.setdefault(stack[-1], {"executed": 0, "useful": None})
                key = cfg.get("backbone", cfg.get("model", "bert" if repo.name == "bert" or runner == "run_t3.py" else "dagsnet"))
                grouped_rows = r.setdefault("by_backbone", {}).setdefault(key, {"executed": 0, "aux_executed": 0})
                executed = x.shape[0] * x.shape[1]
                if mask is None:                                      # teacher / auxiliary forward (fd_ids, pfedes): ops
                    grouped_rows["aux_executed"] += executed
                    r["aux_executed"] = r.get("aux_executed", 0) + executed   # counted, rows kept apart
                else:
                    grouped_rows["executed"] += executed
                    r["executed"] += executed
                    phase = counter.phase if counter is not None else None
                    if counter is not None:
                        counter.phase = None
                    try:
                        useful = mask.sum()
                        r["useful"] = useful if r["useful"] is None else r["useful"] + useful
                    finally:
                        if counter is not None:
                            counter.phase = phase
            label = "model_forward" if train else "aux_forward"
            with scope(label):
                return f(P, B, x, cfg, mask=mask, train=train, **k)
        return train
    if mode == "count":                                               # eager, and TRAIN is not grouped.forward
        ev_rows = [int(h.split("=", 1)[1]) for h in script_argv if h.startswith("eval_rows=")]
        evm = sys.modules.get("core.evaluate")
        if ev_rows and evm is not None and hasattr(evm, "class_sums"):
            # eager prototype pass (fedtgp ...) OOMs 4 GB at its default 32768-row chunk (19/11): chunk at eval_rows, as the
            # count run already does for evaluate; same ops per row, only the chunk padding differs (rows kept in JSON)
            cs = evm.class_sums

            def class_sums(*a, max_rows=1 << 15, **k):
                rows["class_sums_max_rows"] = min(max_rows, ev_rows[-1])
                return cs(*a, max_rows=min(max_rows, ev_rows[-1]), **k)
            evm.class_sums = class_sums
        if evm is not None:
            tag(evm, "class_sums", "client.prototype_forward")
        fwd = grouped.forward
        grouped.compile_all = lambda: setattr(grouped, "TRAIN", wrap_train(fwd))
        loc.compile_gate = lambda *a, **k: (True, {"total_compute": "eager count, gate skipped"})
        grouped.TRAIN = wrap_train(fwd)
    else:
        comp = grouped.compile_all

        def compile_and_wrap():
            comp()
            grouped.TRAIN = wrap_train(grouped.TRAIN)
        grouped.compile_all = compile_and_wrap


real_import = importlib.import_module

if counter is not None and (repo / "core/cost_scope.py").is_file():
    real_import("core.cost_scope").scope = counter.scope

# Helpers are shared by the single-model and mixed runners.
if counter is not None:
    def autograd_entry(function):
        @wraps(function)
        def apply(*args, **kwargs):
            phase = counter.phase
            if runner == "run_t3.py" and phase and phase.endswith(".train"):
                with scope("backward"), counter.autograd():
                    return function(*args, **kwargs)
            with counter.autograd():
                return function(*args, **kwargs)
        return apply

    torch.autograd.backward = autograd_entry(torch.autograd.backward)
    torch.autograd.grad = autograd_entry(torch.autograd.grad)

    for name in ("fedtgp", "v2x_htfl"):
        adapter = real_import(f"methods.{name}").Method
        tag(adapter, "_server", "server.training")
        tag(adapter, "_gather", "server.prototype_aggregation")
    if runner == "run_t3.py":
        core = real_import("core")
        core.local = sys.modules["core.local"] = real_import("core.local_t3")
    l4_name = "mobihet_budget" if runner == "run_t3.py" else "mobihet_l4"
    if (repo / f"methods/{l4_name}.py").is_file():
        l4 = real_import(f"methods.{l4_name}").Method
        for name, label in {
            "_synth": "other.synthetic_data", "_real_rows": "client.real_row_selection",
            "_bootstrap": "global_statistics", "_copula_bootstrap": "global_statistics",
            "_kmeans": "global_statistics", "_codec_acct": "other.serialization",
        }.items():
            tag(l4, name, label)
        tag(real_import("methods.mobihet_lossless"), "encode", "other.serialization")
        tag(real_import("methods.mobihet_lossless"), "decode", "other.serialization")



def import_module(name, *a, **k):
    mod = real_import(name, *a, **k)
    if name in (f"methods.{method}", "methods.mixed", "methods.mobihet_mixed_l4") and not getattr(mod.Method, "_tc_wrapped", False):
        init = mod.Method.__init__

        def __init__(self, env, *a2, **k2):
            patch_local()
            rows["real_rows"] = int(sum(int(v) for v in env.train.sizes))
            rows["group_rows_requested"] = env.hp.get("group_rows")
            push("bootstrap")
            try:
                init(self, env, *a2, **k2)
            finally:
                pop()
            groups = getattr(self, "envs", None) or getattr(self, "groups", None)
            if groups:
                rows["group_rows_effective"] = {
                    getattr(g, "model", getattr(g, "name", "?")): g.hp.get("group_rows") for g in groups}
            wrap_instance(self)
        mod.Method.__init__, mod.Method._tc_wrapped = __init__, True
    return mod


importlib.import_module = import_module


def apply_panel(n):
    """scripts/sim_fold_val.py of MobiHet: fold-train panel of n vehicles, fold val as the test split."""
    import numpy as np
    from core import data as D
    from core import fl
    cnt = np.load(Path(os.environ.get("V4_CACHE", "cache")) / "fold_counts.npy")
    tr, va = cnt[:, 0].sum(1), cnt[:, 1].sum(1)                       # = sim_fold_val.pick(n, seed=7)
    ok = np.flatnonzero((tr > 0) & (va >= 32))
    edges = np.quantile(tr[ok], np.linspace(0, 1, 11))
    strata = np.clip(np.searchsorted(edges, tr[ok], side="right") - 1, 0, 9)
    rng = np.random.default_rng(7)
    ids = np.sort(np.concatenate([rng.choice(ok[strata == s], n // 10, replace=False) for s in range(10)]))
    rows["panel_ids_sha256"] = hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()
    orig = D.load

    def load(root, split, clients=None, cache=None, folds=None, with_fold=False):
        return orig(root, "train", clients, cache, folds=["train" if split == "train" else "val"])
    D.load = load
    fl.Env._sizes = staticmethod(lambda root, cache: cnt[:, 0].sum(1))
    init = fl.Env.__init__

    def env_init(self, cfg, hp, clients=None, cache=None, root=None):
        push("preprocessing")
        try:
            init(self, cfg, hp, sorted(int(c) for c in ids), cache or os.environ.get("V4_CACHE"), root)
        finally:
            pop()
    fl.Env.__init__ = env_init


if panel:
    apply_panel(panel)
if Path(script_argv[0]).name == "run_mixed_fl.py":                    # `from methods.mixed import Method`: wrap it up front
    import_module("methods.mixed")
if Path(script_argv[0]).name == "run_mixed_l4.py":                    # same for the mixed MobiHet L4 runner (20/11); its
    import_module("methods.mobihet_mixed_l4")                          # group threads enter this counter (Lockstep)
sys.argv = script_argv
status = "error"
try:
    if counter is not None:
        with counter:
            runpy.run_path(str(repo / script_argv[0]), run_name="__main__")
    else:
        runpy.run_path(str(repo / script_argv[0]), run_name="__main__")
    status = "done"
finally:
    save(status)
