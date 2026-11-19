"""Regression checks for the DAGSNet-aligned FL revision (CPU, no dataset)."""
import importlib.util
from pathlib import Path
import torch
from core import grouped, local
from core.flat import Layout
from core.sae import SupervisedSAE, fl_config
from methods.tinyproto import Method as TinyProto


def main():
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('sae_tests', Path(__file__).with_name('test_sae_fl.py'))
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    suite.config = fl_config
    cfg = fl_config()
    lay = Layout(cfg)
    p, b = lay.init(42, 2)
    P = {k: v.detach().requires_grad_(True) for k, v in lay.params(p).items()}
    x = torch.randn(2, 7, 77)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    y = torch.randint(0, 16, (2, 7))
    z, h, _ = grouped.forward(P, lay.bufs(b), x, cfg, hidden=True)
    assert h.shape == (2, 7, 93)
    refs = []
    for i in range(2):
        model = SupervisedSAE(cfg)
        model.load_state_dict(lay.state_dict(p[i], b[i]))
        refs.append(model(x[i]))
    assert torch.allclose(z, torch.stack(refs), atol=1e-6)
    loss = local.supervised_loss(P, x, z, h, y, mask, cfg)
    assert torch.equal(loss, local.ce_loss(z, y, mask))
    loss.sum().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in P.values())
    assert all(not v.grad.any() for k, v in P.items() if k.startswith('decoder.'))
    env = suite.ToyEnv()
    method = TinyProto(env)
    assert method.mu == 0.00015
    local.begin_round()
    method.round(1, 1e-3)
    health = local.finish_round(env)
    assert health['applied_steps'] > 0 and health['skipped_steps'] == 0
    suite.test_selected_methods_run_one_sae_round()
    suite.test_canperfl_sae_parent()
    print('ALIGNED FL: CE/reference/93D/decoder-zero-gradient/TinyProto/telemetry/14-method smoke PASS')


if __name__ == '__main__':
    main()
