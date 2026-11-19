"""TabM grouped equivalence and one-round smoke for every selected baseline."""
import importlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core import evaluate, grouped, local
from core.data import Split
from core.flat import Layout
from core.tabm import TabM, config

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


def test_tabm_grouped_matches_reference():
    cfg = dict(config(), dropout=0.0)
    lay = Layout(cfg)
    p, b = lay.init(41, 3)
    x = torch.randn(3, 7, 77)
    got, h, _ = grouped.forward(lay.params(p), lay.bufs(b), x, cfg, hidden=True)
    refs, hidden = [], []
    for i in range(3):
        m = TabM(cfg)
        m.load_state_dict(lay.state_dict(p[i], b[i]))
        z = m.blocks(x[i].unsqueeze(1))
        refs.append(torch.einsum("bki,kio->bko", z, m.weight) + m.bias)
        hidden.append(z.mean(1))
    assert torch.equal(got, torch.stack(refs))
    assert torch.equal(h, torch.stack(hidden))
    assert lay.P == 6548 and lay.B == 0


def test_selected_methods_run_one_tabm_round(monkeypatch=None):
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
            if name == "v2x_htfl":
                assert info["applied_steps"] > 0 and info["skipped_steps"] == 0
                assert info["skipped_frac"] == 0
            for view, cm in method.evaluate().items():
                assert cm.shape == (env.K, 16, 16), (name, view)
                assert cm.sum() == len(env.test.Y), (name, view)
    finally:
        afpha.pd.read_parquet = old
        data_module.find_root = old_root


def test_canperfl_tabm_parent():
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
    test_tabm_grouped_matches_reference()
    test_selected_methods_run_one_tabm_round()
    test_canperfl_tabm_parent()
    print("TABM FL SMOKE PASS")
