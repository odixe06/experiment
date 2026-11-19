"""Reference forward, gradient and masked BatchNorm gates for mixed adapters."""
import torch
from core import grouped
from core.flat import Layout
from core.mixed_backbones import config


def run():
    torch.set_num_threads(2)
    for name in ("dagsnet", "bert", "tabm", "sae", "mlp", "cnn-bigru-attention"):
        for projected in (False, True):
            cfg = config(name, projected)
            cfg["dropout"] = 0.0
            if projected:
                cfg["base"]["dropout"] = 0.0
            lay = Layout(cfg)
            p, b = lay.init(41, 2)
            p.requires_grad_(True)
            x = torch.randn(2, 3, 77)
            mask = torch.tensor([[True, True, True], [True, True, False]])
            z, h, nb = grouped.forward(lay.params(p), lay.bufs(b), x, cfg, mask, True, True)
            z[mask].square().sum().backward()
            for i in range(2):
                model = lay.make(cfg).train()
                sd = lay.state_dict(p[i], b[i])
                for k in model.state_dict():
                    if k.endswith("num_batches_tracked"):
                        sd[k] = torch.tensor(0)
                model.load_state_dict(sd, strict=True)
                zz = model(x[i, mask[i]])
                torch.testing.assert_close(z[i, mask[i]], zz, atol=3e-5, rtol=3e-4)
                zz.square().sum().backward()
                grads = lay.params(p.grad)
                named = dict(model.named_parameters())
                for k, _ in lay.p:
                    expected = named[k].grad
                    if expected is None:
                        expected = torch.zeros_like(named[k])
                    torch.testing.assert_close(grads[k][i], expected, atol=2e-3, rtol=2e-3, msg=f"{name}/{projected}/{k}")
                for k, value in nb.items():
                    torch.testing.assert_close(value[i], model.state_dict()[k], atol=3e-5, rtol=3e-4)
            if projected:
                assert h.shape == (2, 3, 128)
            print(name, 'projected' if projected else 'native', lay.P, 'PASS', flush=True)


if __name__ == '__main__':
    run()
