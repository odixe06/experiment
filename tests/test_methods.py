"""Every method runs 2 rounds on a toy CPU environment, produces finite models and per-client/global confusion matrices,
and its state round-trips through state()/load_state()."""
import importlib

import numpy as np
import pytest
import torch

from core import evaluate, local
from core.dagsnet import width
from core.data import Split
from core.flat import Layout

METHODS = ["fedavg", "fedprox", "scaffold", "mobihet", "mobihet_rel", "mobihet_pair", "mobihet_stats", "pfedes", "fedproto", "fedtgp", "tinyproto", "nilm", "fd_ids", "afpha", "perfedskd", "lwfednids", "apfed", "v2x_htfl"]


class ToyEnv:
    def __init__(self, K=6, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = width(0.25)
        self.hp = {"batch": 8, "group_rows": 32, "weight_decay": 1e-4, "clip": 1.0, "amp": False, "seed": 42,
                   "lr_max": 1e-3, "lr_min": 1e-5, "rounds": 2}
        self.lay = Layout(self.cfg)
        self.dev = torch.device("cpu")
        self.world, self.rank = 1, 0

        def split(lo, hi):
            n = rng.integers(lo, hi, K)
            off = np.concatenate([[0], np.cumsum(n)])
            y = torch.from_numpy(rng.integers(0, 16, off[-1]).astype(np.int8))
            x = torch.from_numpy(rng.standard_normal((int(off[-1]), 77))).float() + 0.3 * y[:, None].float()
            return Split(x, y, torch.from_numpy(off), torch.arange(K))
        self.train, self.test = split(10, 50), split(4, 12)
        self.K = K
        self.ids = np.arange(K)
        self.all_ids = np.arange(K)
        self.n = torch.as_tensor(self.train.sizes, dtype=torch.float32)
        self.groups = local.group_schedule(self.train.sizes, list(range(K)), self.hp["batch"], self.hp["group_rows"])
        self.gen = torch.Generator().manual_seed(0)

    def allreduce(self, t):
        return t

    def gather_cms(self, cm):
        return np.asarray(cm)

    def groups_idx(self):
        for g in self.groups:
            yield torch.as_tensor(g)

    def eval_global(self, p, b):
        return evaluate.global_cm(self.lay, p, b, self.test, self.cfg)

    def eval_clients(self, bp, bb, **kwargs):
        return evaluate.own_cm(self.lay, bp, bb, self.test, self.cfg, torch.arange(self.K), **kwargs)


@pytest.mark.parametrize("name", METHODS)
def test_method_runs(name):
    env = ToyEnv()
    M = importlib.import_module(f"methods.{name}").Method(env)
    for r in (1, 2):
        info = M.round(r, 1e-3)
        assert np.isfinite(info["train_loss"])
    views = M.evaluate()
    for v, cms in views.items():
        assert cms.shape == (env.K, 16, 16)
        assert cms.sum() == len(env.test.Y), v
    s = M.state()
    M2 = importlib.import_module(f"methods.{name}").Method(ToyEnv())
    M2.load_state(s)
    for v in views:
        assert (M2.evaluate()[v] == views[v]).all()


def test_canperfl_finetunes_fedavg_state(tmp_path):
    env = ToyEnv()
    p, b = env.lay.init(42, 1)
    torch.save({"method": "fedavg", "cfg": env.cfg, "round": 50, "method_state": {"p": p[0], "b": b[0]}},
               tmp_path / "state.pt")
    env.hp["init"] = str(tmp_path / "*.pt")
    M = importlib.import_module("methods.canperfl").Method(env)
    before = M.evaluate()["client"]
    assert (before == M.evaluate()["global"]).all()                  # untouched clients = the FedAvg model
    assert np.isfinite(M.round(1, 1e-3)["train_loss"])
    assert not torch.equal(M.bank_p[0], p[0])


@pytest.mark.parametrize("name", ["mobihet_stats", "mobihet_stats_realbn", "mobihet_stats_adaptive", "mobihet_stats_massnorm", "mobihet_stats_nogauss", "mobihet_stats_prior", "mobihet_stats_hist"])
def test_mobihet_stats_has_no_proxy_or_round_payload(name):
    env = ToyEnv()
    method = importlib.import_module("methods." + name).Method(env)
    assert not hasattr(method, "g_p")
    info = method.round(1, 1e-3)
    assert info["bytes_up"] == info["bytes_down"] == 0
    assert set(method.evaluate()) == {"private", "private_local"}


def test_mobihet_stats_adaptive_missing_classes_keep_full_support():
    from methods.mobihet_stats_adaptive import support_weights
    w = support_weights(torch.tensor([0, 64, 640]), 64)
    assert w[0] == 1 and w[1] == 0.5 and 0 < w[2] < w[1]


def test_mobihet_stats_massnorm_conserves_per_client_weight():
    from methods.mobihet_stats_massnorm import Method
    m = Method(ToyEnv())
    assert torch.isfinite(m.gaussian_weights).all()
    assert (m.gaussian_weights > 0).all()
    assert torch.allclose(m.gaussian_weights.mean(1), torch.ones(m.env.K))


def test_mobihet_stats_nogauss_does_not_sample_gaussians():
    from methods.mobihet_stats_nogauss import Method
    m = Method(ToyEnv())
    before = m.gen_g.get_state().clone()
    m.mu.fill_(float("nan"))  # Must never enter a forward pass.
    info = m.round(1, 1e-3)
    assert np.isfinite(info["train_loss"])
    assert torch.equal(before, m.gen_g.get_state())


def test_mobihet_stats_prior_recovers_uniform_sampling_posterior():
    from methods.mobihet_stats_prior import Method
    m = Method(ToyEnv())
    likelihood = torch.arange(1, 17, dtype=torch.float32)
    global_logits = likelihood.log() + m.log_pg
    adjusted = m._synthetic_logits(global_logits)
    assert torch.allclose(adjusted.softmax(-1), likelihood / likelihood.sum())
    # Unequal prior must not be corrected in the opposite direction.
    assert not torch.allclose(global_logits.softmax(-1), adjusted.softmax(-1))


def test_mobihet_stats_hist_preserves_binary_mass_and_constant_support():
    from methods.mobihet_stats_hist import hist_stats, transport
    x = torch.tensor([[0., -1.]] * 75 + [[1., -1.]] * 25)
    y = torch.zeros(100, dtype=torch.long)
    count, total = hist_stats(x, y, classes=1)
    assert torch.equal(count.sum(-1), torch.full((1, 2), 100., dtype=torch.float64))
    cdf = (count.cumsum(-1) / 100).float()
    values = (total / count.clamp_min(1)).float()
    gen = torch.Generator().manual_seed(123)
    z = torch.randn(20000, 1, 2, generator=gen)
    sampled = transport(z, torch.eye(2)[None], torch.tensor([0]), cdf, values)
    assert torch.isin(sampled[..., 0], torch.tensor([0., 1.])).all()
    assert (sampled[..., 1] == -1).all()
    assert abs(float(sampled[..., 0].mean()) - 0.25) < 0.015
    # Histogram stats are additive across clients/ranks.
    c1, s1 = hist_stats(x[:40], y[:40], classes=1)
    c2, s2 = hist_stats(x[40:], y[40:], classes=1)
    assert torch.equal(c1+c2, count) and torch.equal(s1+s2, total)


def test_mobihet_stats_hist_accounts_for_extra_bootstrap():
    from methods.mobihet_stats_hist import Method, BINS
    m = Method(ToyEnv())
    base = 4 * 16 * (1 + 77 + 77*78//2)
    assert m.bootstrap_bytes['up'] == base + 16*16*77*BINS
    assert m.bootstrap_bytes['down'] == base + 8*16*77*BINS


def test_mobihet_stats_sampler_hook_preserves_control():
    from methods.mobihet_stats import Method
    m = Method(ToyEnv())
    z = torch.randn(3, m.aux, 77, generator=torch.Generator().manual_seed(123))
    expected = (m.mu[m.labels] + (m.L[m.labels] @ z[..., None]).squeeze(-1)).clamp(-4, 4)
    assert torch.equal(m._gaussian_rows(z), expected)


def test_mobihet_stats_histmask_repairs_only_missing_values():
    from methods.mobihet_stats_histmask import enforce_missing, PAIRS
    x = torch.arange(3*77, dtype=torch.float32).reshape(3,77)
    for flag, value in PAIRS:
        x[:,flag] = torch.tensor([0.,1.,0.])
    before = x.clone()
    out = enforce_missing(x)
    expected = x.clone()
    for flag,value in PAIRS:
        expected[1,value] = 0
    assert torch.equal(out,expected) and torch.equal(x,before)
    assert torch.equal(enforce_missing(out),out)


def test_mobihet_stats_histmask_no_extra_payload_or_rng():
    from methods.mobihet_stats_hist import Method as Hist
    from methods.mobihet_stats_histmask import Method, PAIRS
    base, fixed = Hist(ToyEnv()), Method(ToyEnv())
    z = torch.randn(2, base.aux, 77)
    a, b = base._gaussian_rows(z), fixed._gaussian_rows(z)
    assert base.payload_bytes() == fixed.payload_bytes()
    assert torch.equal(base.gen_g.get_state(), fixed.gen_g.get_state())
    for flag,value in PAIRS:
        assert torch.equal(a[...,flag],b[...,flag])
    assert fixed.round(1,1e-3)['bytes_up'] == 0
    assert not hasattr(fixed,'g_p')
def test_mobihet_core_floor_and_teacher():
    from methods.mobihet_core import Method
    env = ToyEnv()
    m = Method(env)
    r1 = m.round(1, .001)
    assert r1['applied_steps'] == 16*env.K and r1['bytes_up']==0
    r2 = m.round(2,.001)
    assert r2['q_forward_rows']==len(env.train.Y)
    assert (r2['bytes_up'],r2['bytes_down'])==(576,512)
    assert torch.allclose(m.q.sum(1),torch.ones(16),atol=1e-6)
    assert torch.isfinite(m.L).all() and torch.isfinite(m.la_shift).all()


def test_mobihet_core_blend_targets(monkeypatch):
    from methods.mobihet_core import Method
    import torch.nn.functional as F
    env = ToyEnv()
    env.hp.update(q_interval=2, q_hard_weight=0.5)
    method = Method(env)
    method.q.fill_(1/16)
    original = F.kl_div
    checked = []

    def check_target(logp, target, **kwargs):
        hard = F.one_hot(method.labels, num_classes=16).to(target)
        expected = (0.5*hard + 0.5/16).expand_as(target)
        assert torch.allclose(target, expected)
        checked.append(True)
        return original(logp, target, **kwargs)

    monkeypatch.setattr(F, "kl_div", check_target)
    info = method.round(3, .001)
    assert checked and info["q_forward_rows"] == 0 and info["bytes_up"] == 0
    restored = Method(env)
    restored.load_state(method.state())
    assert torch.equal(restored.q, method.q)


def test_v2x_cosine_normalize_bounds_near_zero_gradient():
    from methods.v2x_htfl import COS_EPS, cosine_normalize
    x = torch.zeros(4, 32, requires_grad=True)
    (64 * cosine_normalize(x).sum()).backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().max() <= 64 / COS_EPS
