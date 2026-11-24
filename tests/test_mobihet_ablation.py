"""MobiHet ablations (methods/mobihet_ablation.py, methods/mobihet_ablation_mixed.py), CPU, 20-vehicle panel, 2 rounds.
MobiHet = mobihet_l4 with MOBIHET_HP (L4 hp + la_mode=off, owner 24/11/2026).

  mobihet   la_mode=off == L4 (la_mode=full) with la_shift zeroed by hand = the former ablation a3 (whose audited R50 runs
            are MobiHet's rows); bootstrap tables and bytes identical to L4
  control   ablation=none == MobiHet (states incl. Q and row CE, confusion), every single-model backbone, 2 rounds
  a1        nothing sent (bootstrap 0 B, no generator tables), Q never built, 0 B per round, one optimizer step per
            ceil(n/64) real batch (no floor), loss and gradient = plain CE(z) on real rows
  a2        no stored row CE; same trained-row fraction as MobiHet
  a4        round 1 == MobiHet (Q unused until round 2); Q fp32 within one uint8 level of MobiHet's, not fp16-rounded;
            1024 B each way
  resume    r1 -> state -> r2 == continuous r2, every arm
  mixed     case2: none == mobihet_mixed_l4; every arm one healthy round

    python tests/test_mobihet_ablation.py [--quick]          # --quick: control on mlp only
Data and hp as tests/test_mixed_l4.py (km2_frac=1: 20 vehicles cannot hold the KM sample the codec asserts); eval_rows=4096
(evaluation packing only: CNN-BiGRU eval of two banks does not fit a 4 GB GPU otherwise).
"""
import copy
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("V4_ROOT", "/home/odixe/project/mobihet/knowledge")
os.environ.setdefault("MOBIHET_RECOVERY_DIR", tempfile.mkdtemp(prefix="ablation-recovery-"))
import numpy as np
import torch
import torch.nn.functional as F

from core import fl, grouped, local
from core.cnn_bigru_attention import config as bigru_config
from core.dagsnet import width
from core.mlp import config as mlp_config
from core.model_assignment import load_assignment
from core.sae import fl_config as sae_config
from core.tabm import config as tabm_config
from methods import mobihet_ablation as A
from methods import mobihet_ablation_mixed as AM
from methods import mobihet_mixed_l4 as MX
from methods.mobihet_l4 import Method as L4
from scripts.run_mixed_l4 import l4_hp
from tests.test_mixed_l4 import CACHE, PANEL, Eager, exact

CFG = {"dagsnet": lambda: width(1), "tabm": tabm_config, "sae": sae_config, "mlp": mlp_config,
       "cnn-bigru-attention": bigru_config}


def env_for(backbone, arm=None):
    torch.manual_seed(0)
    extra = A.HP[arm] if arm else []
    return fl.Env(CFG[backbone](), l4_hp(42, ["km2_frac=1.0", "eval_rows=4096", *extra]), clients=PANEL, cache=CACHE)


def run(M, env, rounds, seed=1):
    infos = []
    with Eager():
        torch.manual_seed(seed)
        for r in rounds:
            local.begin_round()
            info = M.round(r, fl.cosine_lr(r, 50, 1e-3, 1e-5))
            info.update(local.finish_round(env))
            infos.append(info)
    return infos


def pair(backbone, arm, rounds=(1, 2), patch=None):
    """(ablation method, its infos), (MobiHet method, its infos) on identical envs and RNG."""
    ea, eb = env_for(backbone, arm), env_for(backbone)
    a, b = A.Method(ea), L4(eb)
    if patch:
        patch(b)
    return (a, run(a, ea, rounds)), (b, run(b, eb, rounds))


def test_control(backbone):
    (a, ia), (b, ib) = pair(backbone, "none")
    exact(b.state(), a.state(), f"{backbone}/state")
    exact(b.evaluate()["private"], a.evaluate()["private"], f"{backbone}/cms")
    assert [x["bytes_up"] for x in ia] == [x["bytes_up"] for x in ib] and a.bootstrap_bytes == b.bootstrap_bytes
    print(f"control {backbone} PASS (2 rounds bit-exact, F1 inputs identical)")


def test_a1(backbone):
    env = env_for(backbone, "a1")
    m = A.Method(env)
    assert m.bootstrap_bytes == {"up": 0, "down": 0}
    assert not any(hasattr(m, k) for k in ("cop_cdf", "cop_mu", "km_p", "mu", "_score")), "a1 built generator/selection"
    ref = L4(env_for(backbone))
    exact(ref.cnt, m.cnt, "cnt")
    exact(ref.la_shift, m.la_shift, "la_shift")
    assert not bool(m.la_shift.any()), "a1 has a logit adjustment"
    infos = run(m, env, (1, 2))
    n = env.train.off[1:] - env.train.off[:-1]
    want = int(torch.ceil(n.float() / 64).sum())                         # one step per real batch of 64, no floor
    for i in infos:
        assert i["applied_steps"] == want and i["skipped_steps"] == 0, (i["applied_steps"], want)
        assert i["bytes_up"] == 0 and i["bytes_down"] == 0 and i["real_frac"] == 1.0 and np.isfinite(i["train_loss"])
    assert m.Q is None and "row_score" not in m.state()
    # loss and gradient of one step: CE(z + shift) mean over each vehicle's real rows
    G, b, dev = 3, 64, m.env.dev
    z = torch.randn(G, b, 16, dtype=torch.float64, device=dev, requires_grad=True)
    y = torch.randint(16, (G, b), device=dev)
    mask = torch.rand(G, b, device=dev) < .7
    shift = m.la_shift[:G].double()
    keep = grouped.TRAIN
    grouped.TRAIN = lambda P, B, x, cfg, mask, train: (z if m.env.cfg.get("model") != "tabm" else z[:, :, None], {})
    try:
        m._shift = shift
        if m.env.cfg.get("model") == "sae":
            reg = grouped.sae_regularizer
            grouped.sae_regularizer = lambda *a, **k: 0.0
        loss, _ = m._loss_real(SimpleNamespace(P={}, B={}, x=None, y=y, mask=mask))
        if m.env.cfg.get("model") == "sae":
            grouped.sae_regularizer = reg
    finally:
        grouped.TRAIN = keep
    ce = F.cross_entropy((z + shift[:, None]).reshape(-1, 16), y.reshape(-1), reduction="none").view(G, b)
    torch.testing.assert_close(loss, (ce * mask).sum(1) / mask.sum(1))
    loss.sum().backward()
    grad = (torch.softmax(z + shift[:, None], -1) - F.one_hot(y, 16)) * (mask / mask.sum(1, keepdim=True))[..., None]
    torch.testing.assert_close(z.grad, grad.detach())
    print(f"a1 {backbone} PASS (steps {want}/round, 0 B/round, bootstrap 0 B, loss/grad = CE(z))")


def test_a2(backbone):
    (a, ia), (b, ib) = pair(backbone, "a2")
    assert a.select == "random" and "row_score" not in a.state() and "row_score" in b.state()
    for x, y in zip(ia, ib):
        assert abs(x["real_frac"] - y["real_frac"]) < 1e-12 and x["synth_rows"] == y["synth_rows"]
        assert x["applied_steps"] == y["applied_steps"]
    assert a.Q is not None and a.bootstrap_bytes == b.bootstrap_bytes
    assert not torch.equal(a.priv_p, b.priv_p), "random selection trained the same rows as top"
    print(f"a2 {backbone} PASS (same budget/steps/bytes, no row CE)")


def test_mobihet_is_l4_without_la(backbone):
    """MobiHet (la_mode=off) == L4 with la_shift zeroed by hand, i.e. the former ablation a3, over two rounds."""
    torch.manual_seed(0)
    el4 = fl.Env(CFG[backbone](), l4_hp(42, ["km2_frac=1.0", "eval_rows=4096", "la_mode=full"]), clients=PANEL, cache=CACHE)
    emh = env_for(backbone)
    assert emh.hp["la_mode"] == "off" and el4.hp["la_mode"] == "full"
    l4, mh = L4(el4), L4(emh)
    assert bool(l4.la_shift.abs().sum() > 0) and not bool(mh.la_shift.any())
    for k in ("cnt", "cop_cdf", "cop_mass", "cop_mu", "cop_L", "km_p", "km_mu", "km_L"):
        exact(getattr(l4, k), getattr(mh, k), k)
    assert mh.bootstrap_bytes == l4.bootstrap_bytes
    l4.la_shift = torch.zeros_like(l4.la_shift)
    ia, ib = run(mh, emh, (1, 2)), run(l4, el4, (1, 2))
    exact(l4.state(), mh.state(), f"{backbone}/state")
    exact(l4.evaluate()["private"], mh.evaluate()["private"], "cms")
    assert [x["bytes_up"] for x in ia] == [x["bytes_up"] for x in ib]
    print(f"mobihet {backbone} PASS (== L4 with zero shift = former a3; bootstrap unchanged)")


def test_a4(backbone):
    (a, ia), (b, ib) = pair(backbone, "a4", rounds=(1,))
    exact(b.priv_p, a.priv_p, "r1 private weights")
    exact(b.priv_b, a.priv_b, "r1 private buffers")
    assert ia[0]["bytes_up"] == 1024 and ia[0]["bytes_down"] == 1024 and ib[0]["bytes_up"] < 300
    assert a.Q.dtype == torch.float32 and float((a.Q - b.Q).abs().max()) <= 2 / 255   # uint8 up + down
    assert not torch.equal(a.Q, a.Q.half().float()), "Q went through an fp16 cast"
    torch.testing.assert_close(a.Q.sum(1), torch.ones(16, device=a.Q.device), rtol=0, atol=1e-5)
    assert a.q_bits == 8 and a.q_codec == "mean_fp16"                   # restored after the round
    print(f"a4 {backbone} PASS (r1 == L4, Q fp32 |dQ|max {float((a.Q - b.Q).abs().max()):.2e}, 1024+1024 B)")


def test_resume(backbone, arm):
    env = env_for(backbone, arm)
    m = A.Method(env)
    run(m, env, (1,), seed=3)
    state = copy.deepcopy(m.state())
    rng, gen = torch.get_rng_state(), env.gen.get_state()
    run(m, env, (2,), seed=4)
    env2 = env_for(backbone, arm)
    again = A.Method(env2)
    again.load_state(state)
    torch.set_rng_state(rng); env2.gen.set_state(gen)
    run(again, env2, (2,), seed=4)
    exact(m.state(), again.state(), "state")
    exact(m.evaluate()["private"], again.evaluate()["private"], "cms")
    print(f"resume {backbone} {arm} PASS")


def test_mixed():
    mapping = load_assignment(ROOT / "experiments/mixed-model-v1", "case2")
    models = ["tabm", "sae", "mlp"]

    def build(cls, arm=None):
        torch.manual_seed(0)
        env = fl.Env(mlp_config(), l4_hp(42, ["km2_frac=1.0", *(A.HP[arm] if arm else [])]), clients=PANEL, cache=CACHE)
        return env, cls(env, mapping, models)

    n_test = len(env_for("mlp").test.Y)
    ea, a = build(AM.Method, "none")
    eb, b = build(MX.Method)
    run(a, ea, (1,)); run(b, eb, (1,))
    exact(b.state(), a.state(), "mixed state")
    exact(b.evaluate()["client"], a.evaluate()["client"], "mixed cms")
    assert MX.GroupL4 is not AM.GroupAblation
    for arm, up in (("a1", 0.0), ("a2", None), ("a4", 1024.0)):
        env, m = build(AM.Method, arm)
        assert all(isinstance(g, AM.GroupAblation) and g.ablation == arm for g in m.groups)
        info = run(m, env, (1, 2))[-1]
        for model in models:
            assert info[f"health/{model}/finite"] and info[f"health/{model}/skipped_steps"] == 0, (arm, model)
            assert info[f"health/{model}/applied_steps"] > 0
        assert np.isfinite(info["train_loss"]) and (up is None or info["bytes_up"] == up), (arm, info["bytes_up"])
        assert m.evaluate()["client"].sum() == n_test
        if arm == "a1":
            assert m.payload_bytes()["bootstrap"] == {"up": 0, "down": 0} and all(g.Q is None for g in m.groups)
        print(f"mixed case2 {arm} PASS (2 rounds, health per backbone, bytes_up {info['bytes_up']})")
    print("mixed case2 control PASS (== MobiHet mixed, 1 round bit-exact)")


if __name__ == "__main__":
    torch.set_num_threads(8)
    quick = "--quick" in sys.argv
    for bb in (["mlp"] if quick else ["mlp", "tabm", "sae", "dagsnet", "cnn-bigru-attention"]):
        test_control(bb)
    for bb in ("mlp", "tabm", "sae"):
        test_a1(bb)
    test_a2("mlp")
    test_mobihet_is_l4_without_la("mlp")
    test_mobihet_is_l4_without_la("tabm")
    for bb in ("mlp", "tabm"):
        test_a4(bb)
    for arm in ("a1", "a2", "a4"):
        test_resume("mlp", arm)
    test_mixed()
    print("ALL PASS")
