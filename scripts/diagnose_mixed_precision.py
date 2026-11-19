"""Synthetic localization only; real-data GPU confirmation is separate."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, '.')
import torch
from core import grouped, local
from core.flat import Layout
from core.mixed_backbones import MixedAdapter, config
from methods.mixed import Method
from methods import v2x_htfl as v2x

torch.set_num_threads(2)
linear = grouped._linear
def fp32_projection(c, name, x):
    if name == 'projection':
        with torch.autocast(x.device.type, enabled=False):
            return linear(c, name, x.float())
    return linear(c, name, x)

results = []
for name in ('sae', 'tabm', 'mlp'):
    cfg = config(name); cfg['dropout'] = 0.
    lay = Layout(cfg, make=v2x.projected_model(MixedAdapter))
    p,b = lay.init(42,1)
    method = Method.__new__(Method)
    method.name, method.s, method.gamma, method.m, method.proto = 'v2x_htfl',64.,.99,.5,None
    method.V = v2x.etf(128,53,'cpu')
    for mode in ('old_amp', 'projection_fp32', 'fp32'):
        ctx = local.Ctx(); ctx.G = ctx.Gfix = 1
        ctx.P = {k:v.detach().clone().requires_grad_() for k,v in lay.params(p).items()}
        ctx.B = lay.bufs(b)
        ctx.x = torch.randn(1,64,77,generator=torch.Generator().manual_seed(7))
        ctx.y = (torch.arange(64)%16)[None]; ctx.mask = torch.ones(1,64,dtype=torch.bool)
        with patch.object(grouped, '_linear', fp32_projection if mode == 'projection_fp32' else linear):
            with torch.autocast('cpu',dtype=torch.float16,enabled=mode != 'fp32'):
                loss,_ = method._loss(ctx,SimpleNamespace(cfg=cfg),18,torch.ones(1))
            (loss.sum()*4096).backward()
        bad = [k for k,v in ctx.P.items() if not torch.isfinite(v.grad).all()]
        row = dict(model=name,mode=mode,loss=float(loss.detach()),bad=bad)
        results.append(row); print(json.dumps(row),flush=True)
from pathlib import Path
Path('runs/mixed128-precision-synthetic.json').write_text(json.dumps(results,indent=2))
