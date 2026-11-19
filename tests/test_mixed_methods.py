"""Mixed-round aggregation, all-five-method health and exact CPU resume gates."""
import numpy as np
import torch
import tempfile
from pathlib import Path
import importlib
from unittest.mock import patch
from core.flat import Layout
from core.mixed_backbones import config, MixedAdapter
from methods import v2x_htfl

from core import data, local, evaluate
from methods.mixed import Method, METHODS


class ToyEnv:
    def __init__(self):
        self.dev = torch.device("cpu")
        self.rank, self.world, self.K = 0, 1, 6
        self.ids = self.all_ids = np.arange(6)
        self.hp = dict(seed=42, batch=4, group_rows=4, amp=False, weight_decay=1e-4,
                       clip=1., server_epochs=1, eval_rows=16)
        self.gen = torch.Generator().manual_seed(42000)
        rng = torch.Generator().manual_seed(18)
        x = torch.randn(30, 77, generator=rng)
        y = torch.arange(30) % 16
        off = torch.arange(0, 31, 5)
        self.train = data.Split(x, y, off, torch.arange(6))
        self.test = self.train
        self.n = torch.full((6,), 5.)

    def allreduce(self, t):
        return t

    def gather_cms(self, cm):
        return np.asarray(cm)


NAMES = ("dagsnet", "bert", "tabm", "sae", "mlp", "cnn-bigru-attention")


def exact(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: exact(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): exact(x, y)
    else:
        assert a == b


def run():
    torch.set_num_threads(2)
    mapping = dict(enumerate(NAMES))
    for name in METHODS:
        env = ToyEnv()
        model = Method(env, name, mapping, NAMES)
        local.begin_round()
        info = model.round(1, 1e-3)
        telemetry = local.finish_round(env)
        assert np.isfinite(info["train_loss"])
        assert telemetry["skipped_steps"] == 0 and telemetry["applied_steps"] > 0
        state = model.state()
        rng, gen = torch.get_rng_state(), env.gen.get_state()
        model.round(2, 9e-4)
        expected_state, expected_cm = model.state(), model.evaluate()
        for cm in expected_cm.values(): assert cm.sum() == 30
        restored_env = ToyEnv()
        restored = Method(restored_env, name, mapping, NAMES)
        restored.load_state(state)
        torch.set_rng_state(rng)
        restored_env.gen.set_state(gen)
        restored.round(2, 9e-4)
        exact(expected_state, restored.state())
        with tempfile.TemporaryDirectory() as folder:
            actual_cm = restored.evaluate(folder)
            for view, cm in actual_cm.items():
                np.testing.assert_array_equal(cm, expected_cm[view])
                rebuilt = np.zeros_like(cm)
                for path in Path(folder).glob(f"rank0-*-{view}.npz"):
                    with np.load(path) as arr:
                        for i, cid in enumerate(arr['ids']):
                            start, stop = arr['offsets'][i:i+2]
                            code = arr['y_true'][start:stop].astype('int64')*16 + arr['y_pred'][start:stop]
                            rebuilt[cid] += np.bincount(code, minlength=256).reshape(16,16)
                np.testing.assert_array_equal(cm, rebuilt)
        print(name, 'six-backbone round / exact state+CM resume PASS', flush=True)

    sums, counts = torch.zeros(16, 128), torch.zeros(16)
    env = ToyEnv()
    subset = ("bert", "tabm", "sae", "mlp")
    mapping = {i: subset[i % 4] for i in range(6)}
    model = Method(env, "fedproto", mapping, subset)
    for group in model.groups:
        group.cfg["dropout"] = 0.
        S, C = evaluate.class_sums(group.lay, group.p, group.b, env.train, group.cfg, group.indices, max_rows=16)
        sums += S.sum(0); counts += C.sum(0)
    model.round(1, 0.)
    torch.testing.assert_close(model.proto, sums/counts.clamp_min(1)[:, None], atol=2e-6, rtol=2e-5)
    assert torch.equal(model.has, counts > 0)
    print('joint cross-backbone FedProto aggregation PASS', flush=True)

    # Homogeneous limiting case must retain the established method, not just run.
    for name in METHODS:
        env = ToyEnv()
        env.cfg = config('mlp', projected=name != 'pfedes')
        env.lay = Layout(env.cfg)
        env.groups_idx = lambda: (torch.tensor(ix) for ix in local.group_schedule(env.train.sizes, list(range(env.K)), 4, 4))
        env.eval_clients = lambda p,b: evaluate.own_cm(env.lay,p,b,env.test,env.cfg,torch.arange(env.K),max_rows=16)
        with patch('methods.v2x_htfl.V2XNet', v2x_htfl.projected_model(MixedAdapter)):
            original = importlib.import_module('methods.'+name).Method(env)
        for r in (1,2): original.round(r,1e-3)
        old = original.evaluate()
        mixed_env = ToyEnv()
        mixed = Method(mixed_env,name,{i:'mlp' for i in range(6)},('mlp',))
        for r in (1,2): mixed.round(r,1e-3)
        torch.testing.assert_close(original.bank_p,mixed.groups[0].p,atol=3e-6,rtol=3e-5)
        for view,cm in mixed.evaluate().items(): np.testing.assert_array_equal(cm,old[view])
        print(name,'homogeneous original-method weight/CM parity PASS',flush=True)


if __name__ == '__main__':
    run()
