"""Regression for scaled fp16 projection-bias reduction overflow."""
import torch
from unittest.mock import patch
from core import local
from core.mlp import config as mlp_config
from methods.v2x_htfl import Method
from methods.v2x_htfl import project


def test_projection_bias_gradient_under_autocast():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h = torch.ones(1, 64, 32, device=device, requires_grad=True)
    p = {'proj_v': torch.eye(32, device=device)[None].requires_grad_(),
         'proj_g': torch.ones(1, 32, device=device, requires_grad=True),
         'proj_b': torch.zeros(1, 32, device=device, requires_grad=True)}
    with torch.autocast(device.type, dtype=torch.float16):
        out = project(p, h)
    assert out.dtype == torch.float32
    (out.sum() * 4096).backward()
    assert torch.equal(p['proj_b'].grad, torch.full_like(p['proj_b'], 64 * 4096))
    assert all(torch.isfinite(v.grad).all() for v in p.values())
    assert torch.isfinite(h.grad).all()


def test_mlp_v2x_loss_scaled_gradients():
    """Exercise the actual round-1 loss, including the MLP backbone, at scale4096."""
    import test_sae_fl as fixture

    class ProbeDone(Exception):
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in (41, 42, 43):
        with patch.object(fixture, 'config', lambda: dict(mlp_config(), dropout=0.)):
            env = fixture.ToyEnv(K=1)
        env.hp['seed'] = seed
        env.dev = device
        env.n = env.n.to(device)
        method = Method(env)

        def probe(lay, flat, bufs, data, clients, cfg, hp, lr, gen, loss_fn):
            ctx = local.Ctx()
            ctx.P = {k: v.detach().clone().requires_grad_(True) for k, v in lay.params(flat).items()}
            ctx.B = lay.bufs(bufs)
            ctx.x = torch.randn(1, 64, 77, generator=torch.Generator().manual_seed(7)).to(device)
            ctx.y = (torch.arange(64, device=device) % 16)[None]
            ctx.mask = torch.ones(1, 64, device=device, dtype=torch.bool)
            with torch.autocast(device.type, dtype=torch.float16):
                loss, _ = loss_fn(ctx)
            assert torch.isfinite(loss).all(), seed
            (loss.sum() * 4096).backward()
            bad = [k for k, v in ctx.P.items() if v.grad is None or not torch.isfinite(v.grad).all()]
            assert not bad, (seed, bad)
            assert ctx.P['proj_b'].grad.abs().sum() > 0
            raise ProbeDone

        with patch.object(local, 'local_epoch', probe):
            try:
                method.round(1, 1e-3)
            except ProbeDone:
                continue
            raise AssertionError('V2X loss probe did not run')


if __name__ == '__main__':
    torch.set_num_threads(2)
    test_projection_bias_gradient_under_autocast()
    test_mlp_v2x_loss_scaled_gradients()
    print('V2X PROJECTION PRECISION PASS', flush=True)
