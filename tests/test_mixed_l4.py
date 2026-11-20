"""Mixed-model MobiHet L4 (methods/mobihet_mixed_l4.py), CPU.

  lockstep      toy collectives: pooled sums in group order, per-thread core.local globals, mismatch detection
  subset        a group's rows are exactly its vehicles' rows
  pooled boot   case2 groups on the 20-vehicle panel: bootstrap tables = single-model L4 on the union of vehicles
  identity      one group holding every vehicle = plain L4 on the same env (states, Q, confusion bit-exact, 1 round)
  assignment    group ids = frozen assignment; evaluation rows land on their client ids; every test row counted once
  resume        case2: r1 -> state -> r2 equals restore -> r2 (states and confusion bit-exact)

    V4_ROOT=../mobihet/knowledge python tests/test_mixed_l4.py
Real rows of the cached 20-vehicle panel (../mobihet/cache), production L4 hp except km2_frac=1 (5 sampled vehicles of 20
cannot hold the class rows the KM codec asserts; plumbing test only). Eager forward behind a wrapper: L4 refuses to run
uncompiled (its compile guard checks grouped.TRAIN is not grouped.forward).
"""
import copy
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("V4_ROOT", "/home/odixe/project/mobihet/knowledge")
import numpy as np
import torch

from core import fl, grouped, local
from core.mixed_backbones import config
from core.model_assignment import load_assignment
from methods import mobihet_mixed_l4 as mx
from methods.mobihet_l4 import Method as L4
from scripts.run_mixed_l4 import l4_hp

CACHE = "/home/odixe/project/mobihet/cache"
PANEL = np.sort(np.random.default_rng(7).choice(5343, 20, replace=False)).tolist()
CASE2 = ["tabm", "sae", "mlp"]


def exact(a, b, path=""):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b) or (a.is_floating_point() and torch.equal(a.isnan(), b.isnan())
                                     and torch.equal(a.nan_to_num(), b.nan_to_num())), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (path, set(a) ^ set(b))
        for k in a: exact(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): exact(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    else:
        assert a == b, (path, a, b)


def test_lockstep():
    calls = []
    step = mx.Lockstep(3, lambda t: t * 1)                              # one "GPU rank": reduce = identity
    orig = local.local_epoch

    def job(k):
        def run():
            local.local_epoch = f"wrapper{k}"                           # each group installs its own wrapper
            local._round_steps.append(torch.tensor([k + 1.0, 0.0]))
            a = step.sync(k, torch.tensor([float(k + 1)]), "a")         # 1 + 2 + 3
            calls.append((k, "after_a", local.local_epoch))
            b = step.sync(k, a * (k + 1), "b")                          # 6 * (1 + 2 + 3)
            return float(a), float(b), local.local_epoch
        return run

    local.begin_round()
    out = step.run([job(k) for k in range(3)], torch.device("cpu"))
    assert out == [(6.0, 36.0, "wrapper0"), (6.0, 36.0, "wrapper1"), (6.0, 36.0, "wrapper2")], out
    assert [c[0] for c in calls] == [0, 1, 2] and all(c[2] == f"wrapper{c[0]}" for c in calls), calls
    assert local.local_epoch is orig and local._round_steps == []      # main thread globals restored
    assert [float(s["_round_steps"][0][0]) for s in step.slots] == [1.0, 2.0, 3.0]

    def bad(k):
        return lambda: step.sync(k, torch.zeros(1), "x" if k else "y")
    try:
        step.run([bad(k) for k in range(3)], torch.device("cpu"))
    except RuntimeError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("mismatched collectives must fail")
    assert threading.active_count() == 1
    limit, seen = torch._dynamo.config.recompile_limit, []               # thread-local compiler config is carried over
    torch._dynamo.config.recompile_limit = 64
    try:
        step.run([lambda: seen.append(torch._dynamo.config.recompile_limit) for _ in range(3)], torch.device("cpu"))
    finally:
        torch._dynamo.config.recompile_limit = limit
    assert seen == [64, 64, 64], seen
    print("lockstep PASS")


class Eager:                                                            # L4 runs only "compiled": wrap eager forward
    def __enter__(self):
        self.train = grouped.TRAIN
        grouped.TRAIN = lambda *a, **k: grouped.forward(*a, **k)

    def __exit__(self, *exc):
        grouped.TRAIN = self.train


def hp():
    return l4_hp(42, ["km2_frac=1.0"])


def make_env(cfg=None):
    torch.manual_seed(0)
    return fl.Env(cfg or config("mlp"), hp(), clients=PANEL, cache=CACHE)


def test_subset():
    env = make_env()
    idx = [1, 4, 5, 17]
    s = mx.subset(env.train, torch.tensor(idx))
    for j, i in enumerate(idx):
        a, b = int(env.train.off[i]), int(env.train.off[i + 1])
        c, d = int(s.off[j]), int(s.off[j + 1])
        assert torch.equal(env.train.X[a:b], s.X[c:d]) and torch.equal(env.train.Y[a:b], s.Y[c:d])
    assert s.ids.tolist() == [int(env.ids[i]) for i in idx]
    print("subset PASS")


def build(models, mapping):
    env = make_env()
    return env, mx.Method(env, mapping, models)


def test_pooled_bootstrap_and_assignment():
    mapping = load_assignment(ROOT/"experiments/mixed-model-v1", "case2")
    env, mixed = build(CASE2, mapping)
    plain = L4(make_env())
    for key in ("cop_cdf", "cop_mass"):                                  # sums of 1.0 in float64: exact in any order
        exact(getattr(plain, key), getattr(mixed.groups[0], key), key)
    for key in ("cop_mu", "cop_L", "mu", "L"):
        torch.testing.assert_close(getattr(mixed.groups[0], key), getattr(plain, key), rtol=1e-5, atol=1e-6, msg=key)
    # KM: float64 sums in another order can flip a class eigenvector's sign, so k-means starts from mu -/+ d instead of
    # mu +/- d: the same mixture with its clusters relabelled. Compare per class up to that permutation.
    from scipy.optimize import linear_sum_assignment
    g = mixed.groups[0]
    for c in range(g.km_p.shape[0]):
        a = torch.cat([g.km_mu[c], g.km_p[c][:, None]], 1).double()
        b = torch.cat([plain.km_mu[c], plain.km_p[c][:, None]], 1).double()
        perm = linear_sum_assignment(torch.cdist(a, b).numpy())[1].tolist()
        for key in ("km_p", "km_mu", "km_L"):
            torch.testing.assert_close(getattr(g, key)[c], getattr(plain, key)[c][perm], rtol=1e-5, atol=1e-5,
                                       msg=f"{key} class {c} perm {perm}")
    for e, g in zip(mixed.envs, mixed.groups):
        assert all(mapping[int(c)] == e.model for c in e.ids) and all(mapping[int(c)] == e.model for c in e.all_ids)
        pos = np.searchsorted(plain.env.ids, e.ids)
        exact(plain.cnt[torch.as_tensor(pos)], g.cnt, f"{e.model}/cnt")
        exact(plain.la_shift[torch.as_tensor(pos)], g.la_shift, f"{e.model}/la_shift")
    assert sorted(int(c) for e in mixed.envs for c in e.ids) == PANEL
    with Eager():
        local.begin_round()
        info = mixed.round(1, 1e-3)
        local.finish_round(env)
    for m in CASE2:
        assert info[f"health/{m}/finite"] and info[f"health/{m}/applied_steps"] > 0 and info[f"health/{m}/skipped_steps"] == 0
    assert np.isfinite(info["train_loss"]) and mixed.groups[0].Q is not None
    cms = mixed.evaluate()["client"]
    test_rows = make_env().test
    assert cms.shape == (20, 16, 16) and cms.sum() == len(test_rows.Y)
    for i, cid in enumerate(env.all_ids):
        k = int(np.searchsorted(test_rows.ids.numpy(), cid))
        truth = np.bincount(test_rows.Y[test_rows.off[k]:test_rows.off[k + 1]].numpy(), minlength=16)
        assert np.array_equal(cms[i].sum(1), truth), cid
    print("pooled bootstrap + assignment + round health PASS", {k: v for k, v in info.items() if k.startswith("health")})


def test_identity():
    """One group holding every vehicle: lockstep, GroupEnv and the state wrapper change nothing."""
    cfg = config("tabm")
    mapping = {c: "tabm" for c in range(5343)}
    env_a = make_env(cfg)
    mixed = mx.Method(env_a, mapping, ["tabm"])
    env_b = make_env(cfg)
    plain = L4(env_b)
    with Eager():
        for M, env in ((mixed, env_a), (plain, env_b)):
            torch.manual_seed(1)
            local.begin_round()
            M.round(1, 1e-3)
            local.finish_round(env)
    exact(plain.state(), mixed.groups[0].state(), "state")
    exact(plain.evaluate()["private"], mixed.evaluate()["client"], "cms")
    print("identity PASS")


def test_resume():
    mapping = load_assignment(ROOT/"experiments/mixed-model-v1", "case2")
    with Eager():
        env, mixed = build(CASE2, mapping)
        torch.manual_seed(3)
        local.begin_round(); mixed.round(1, 1e-3); local.finish_round(env)
        state = copy.deepcopy(mixed.state())
        rng, gen = torch.get_rng_state(), env.gen.get_state()
        local.begin_round(); mixed.round(2, 9e-4); local.finish_round(env)
        want, want_cm = mixed.state(), mixed.evaluate()["client"]
        env2, again = build(CASE2, mapping)
        again.load_state(state)
        torch.set_rng_state(rng); env2.gen.set_state(gen)
        local.begin_round(); again.round(2, 9e-4); local.finish_round(env2)
        exact(want, again.state(), "state")
        exact(want_cm, again.evaluate()["client"], "cms")
    print("resume PASS")


if __name__ == "__main__":
    torch.set_num_threads(8)
    import tempfile
    os.environ.setdefault("MOBIHET_RECOVERY_DIR", tempfile.mkdtemp(prefix="mixed-l4-recovery-"))
    test_lockstep()
    if "--toy" in sys.argv:                                              # Kaggle notebook: no local data panel there
        sys.exit(0)
    test_subset()
    test_pooled_bootstrap_and_assignment()
    test_identity()
    test_resume()
    print("ALL PASS")
