"""Small synthetic gradient localization; does not certify real-data GPU training."""
import math
import sys
sys.path.insert(0, '.')
import torch
from core.flat import Layout
from core.tabm import config
from core import grouped
from methods.v2x_htfl import TabMV2X, project, cosine_normalize, etf
torch.set_num_threads(2)
for seed in (41,42,43):
    cfg = dict(config(), dropout=0.)
    lay = Layout(cfg, make=TabMV2X)
    p,b = lay.init(seed,1)
    x = torch.randn(1,64,77, generator=torch.Generator().manual_seed(7))
    y = torch.arange(64).reshape(1,64)%16
    for amp,scale in ((False,4096),(True,4096),(True,1)):
        P={k:v.detach().clone().requires_grad_(True) for k,v in lay.params(p).items()}
        with torch.autocast('cpu', dtype=torch.float16, enabled=amp):
            z,h,_=grouped.forward(P,lay.bufs(b),x,cfg,hidden=True)
            g=project(P,h).float()
            cos=cosine_normalize(g)@etf(32,seed+11,'cpu')
            cy=cos.gather(-1,y[...,None]).squeeze(-1)
            sy=(1-cy*cy).clamp_min(0).sqrt()
            phi=torch.where(cy>math.cos(math.pi-.5),cy*math.cos(.5)-sy*math.sin(.5),cy-math.sin(math.pi-.5)*.5)
            logits=(64*.99*cos).scatter(-1,y[...,None],64*phi[...,None])
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,16),y.reshape(-1))+0*z.sum()
        (loss*scale).backward()
        bad=[k for k,v in P.items() if not torch.isfinite(v.grad).all()]
        print(seed,amp,scale,'bad',bad,flush=True)
