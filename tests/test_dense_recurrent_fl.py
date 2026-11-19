"""Reference equivalence gates for new FL backbones."""
import torch
import copy
import importlib
import io
from unittest.mock import patch
from core import grouped
from core.flat import Layout
from core.mlp import config as mlp_config
from core.cnn_bigru_attention import config as bigru_config


def test_resume(fixture):
    table = fixture.pd.DataFrame({"client_id": range(3), "scenario": [0, 0, 1], "run": [0, 1, 0]})
    with patch("core.data.find_root", return_value="."), patch("methods.afpha.pd.read_parquet", return_value=table):
        for config in (mlp_config, bigru_config):
            fixture.config = config
            for name in fixture.METHODS:
                env = fixture.ToyEnv()
                cls = importlib.import_module("methods."+name).Method
                method = cls(env)
                method.round(1, 1e-3)
                stream = io.BytesIO()
                torch.save(copy.deepcopy(method.state()), stream)
                rng, gen = torch.get_rng_state(), env.gen.get_state()
                method.round(2, 1e-4)
                expected = method.evaluate()
                restored_env = fixture.ToyEnv()
                restored = cls(restored_env)
                stream.seek(0)
                restored.load_state(torch.load(stream, weights_only=True))
                torch.set_rng_state(rng)
                restored_env.gen.set_state(gen)
                restored.round(2, 1e-4)
                actual = restored.evaluate()
                for view in expected:
                    assert (expected[view] == actual[view]).all(), (config()["model"], name, view)


def test_training_reference():
    for config in (mlp_config, bigru_config):
        cfg = dict(config(), dropout=0.0)
        lay = Layout(cfg)
        p, b = lay.init(41, 2)
        p.requires_grad_(True)
        x = torch.randn(2, 3, 77)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        actual, nb = grouped.forward(lay.params(p), lay.bufs(b), x, cfg, mask=mask, train=True)
        actual[mask].square().sum().backward()
        for i in range(2):
            model = lay.make(cfg).train()
            state = lay.state_dict(p[i], b[i])
            for key in model.state_dict():
                if key.endswith("num_batches_tracked"):
                    state[key] = torch.tensor(0)
            model.load_state_dict(state, strict=True)
            expected = model(x[i, mask[i]])
            expected.square().sum().backward()
            torch.testing.assert_close(actual[i, mask[i]], expected, atol=3e-5, rtol=3e-5)
            grad = torch.cat([dict(model.named_parameters())[k].grad.flatten() for k, _ in lay.p])
            torch.testing.assert_close(p.grad[i], grad, atol=2e-4, rtol=2e-3)
            for key in nb:
                torch.testing.assert_close(nb[key][i], model.state_dict()[key], atol=2e-5, rtol=2e-5)


def test_reference():
    for config in (mlp_config, bigru_config):
        cfg = config()
        lay = Layout(cfg)
        p, b = lay.init(41, 2)
        x = torch.randn(2, 3, 77)
        actual, _ = grouped.forward(lay.params(p), lay.bufs(b), x, cfg)
        for i in range(2):
            model = lay.make(cfg).eval()
            state = lay.state_dict(p[i], b[i])
            for key in model.state_dict():
                if key.endswith("num_batches_tracked"):
                    state[key] = torch.tensor(0)
            model.load_state_dict(state, strict=True)
            torch.testing.assert_close(actual[i], model(x[i]), atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    import test_sae_fl as fixture
    torch.set_num_threads(2)
    test_reference()
    test_training_reference()
    for config in (mlp_config, bigru_config):
        fixture.config = config
        fixture.test_selected_methods_run_one_sae_round()
        fixture.test_canperfl_sae_parent()
        expected = 7600 if config()["model"] == "mlp" else 118288
        assert Layout(config()).P == expected
    print("DENSE RECURRENT REFERENCE AND 28 METHOD SMOKES PASS")
    test_resume(fixture)
    print("26 METHOD ROUND-BOUNDARY RESUME CM CHECKS PASS")
