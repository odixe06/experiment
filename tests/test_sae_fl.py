"""SAE grouped equivalence and one-round smoke for every selected baseline."""
import importlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from core import evaluate, grouped, local
from core.data import Split
from core.flat import Layout
from core.sae import SupervisedSAE, config

METHODS = ["fedavg", "fedprox", "afpha", "fd_ids", "apfed", "lwfednids", "tinyproto",
           "pfedes", "nilm", "fedtgp", "perfedskd", "fedproto", "v2x_htfl"]


class ToyEnv:
    def __init__(self, K=3, seed=7):
        rng = np.random.default_rng(seed)
        self.cfg = config()
        self.hp = {"batch": 4, "group_rows": 16, "weight_decay": 1e-4, "clip": 1.0, "amp": False,
                   "seed": 41, "lr_max": 1e-3, "lr_min": 1e-5, "rounds": 2, "server_epochs": 2}
        self.lay, self.dev, self.world, self.rank = Layout(self.cfg), torch.device("cpu"), 1, 0

        def split(lo, hi):
            n = rng.integers(lo, hi, K)
            off = np.concatenate([[0], np.cumsum(n)])
            y = torch.from_numpy(rng.integers(0, 16, off[-1]).astype(np.int8))
            x = torch.from_numpy(rng.standard_normal((int(off[-1]), 77))).float()
            return Split(x, y, torch.from_numpy(off), torch.arange(K))

        self.train, self.test = split(5, 9), split(3, 6)
        self.K, self.ids, self.all_ids = K, np.arange(K), np.arange(K)
        self.n = torch.as_tensor(self.train.sizes, dtype=torch.float32)
        self.groups = local.group_schedule(self.train.sizes, list(range(K)), self.hp["batch"], self.hp["group_rows"])
        self.gen = torch.Generator().manual_seed(11)

    def allreduce(self, t):
        return t

    def gather_cms(self, cm):
        return np.asarray(cm)

    def groups_idx(self):
        for g in self.groups:
            yield torch.as_tensor(g)

    def eval_global(self, p, b):
        return evaluate.global_cm(self.lay, p, b, self.test, self.cfg)

    def eval_clients(self, p, b, **kwargs):
        return evaluate.own_cm(self.lay, p, b, self.test, self.cfg, torch.arange(self.K), **kwargs)


def test_sae_grouped_matches_reference():
    cfg = config()
    lay = Layout(cfg)
    p, b = lay.init(41, 3)
    x = torch.randn(3, 7, 77)
    mask = torch.tensor([[1] * 7, [1] * 5 + [0] * 2, [1] * 3 + [0] * 4], dtype=torch.bool)
    got, hidden, _ = grouped.forward(lay.params(p), lay.bufs(b), x, cfg, hidden=True)
    refs, codes, regs = [], [], []
    for i in range(3):
        model = SupervisedSAE(cfg)
        model.load_state_dict(lay.state_dict(p[i], b[i]))
        unit = model.unit_input(x[i])
        code = model.encoder(unit)
        refs.append(model.head(torch.cat((unit, code), dim=1)))
        codes.append(code)
        valid = mask[i]
        reconstruction = model.decoder(code[valid])
        rec = (reconstruction - unit[valid]).square().sum(1).mean()
        mean = code[valid].mean(0).clamp(1e-6, 1 - 1e-6)
        rho = cfg["sparse_rho"]
        kl = rho * (np.log(rho) - mean.log()) + (1-rho) * (np.log1p(-rho) - torch.log1p(-mean))
        regs.append(cfg["reconstruction_weight"] * rec + cfg["sparsity_weight"] * kl.sum())
    assert torch.allclose(got, torch.stack(refs), atol=1e-6, rtol=1e-6)
    assert torch.allclose(hidden, torch.stack(codes), atol=1e-6, rtol=1e-6)
    reg = grouped.sae_regularizer(lay.params(p), x, hidden, mask, cfg)
    assert torch.allclose(reg, torch.stack(regs), atol=1e-5, rtol=1e-5)
    assert lay.P == 7613 and lay.B == 0


def test_selected_methods_run_one_sae_round():
    import methods.afpha as afpha
    from core import data as data_module
    old = afpha.pd.read_parquet
    old_root = data_module.find_root
    data_module.find_root = lambda: "."
    afpha.pd.read_parquet = lambda _: pd.DataFrame({"client_id": range(3), "scenario": [0, 0, 1], "run": [0, 1, 0]})
    try:
        for name in METHODS:
            env = ToyEnv()
            method = importlib.import_module(f"methods.{name}").Method(env)
            info = method.round(1, 1e-3)
            assert np.isfinite(info["train_loss"]), name
            for view, cm in method.evaluate().items():
                assert cm.shape == (env.K, 16, 16), (name, view)
                assert cm.sum() == len(env.test.Y), (name, view)
    finally:
        afpha.pd.read_parquet = old
        data_module.find_root = old_root


def test_canperfl_sae_parent():
    with tempfile.TemporaryDirectory() as tmp:
        env = ToyEnv()
        p, b = env.lay.init(41, 1)
        path = Path(tmp) / "state.pt"
        torch.save({"method": "fedavg", "cfg": env.cfg, "round": 50,
                    "method_state": {"p": p[0], "b": b[0]}}, path)
        env.hp["init"] = str(path)
        method = importlib.import_module("methods.canperfl").Method(env)
        assert np.isfinite(method.round(1, 1e-3)["train_loss"])


if __name__ == "__main__":
    test_sae_grouped_matches_reference()
    test_selected_methods_run_one_sae_round()
    test_canperfl_sae_parent()
    print("SAE FL SMOKE PASS")
