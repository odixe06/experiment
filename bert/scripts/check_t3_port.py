"""Real cached rows: T3 tier loss/gradient equals each client's unpadded BERT loss."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.run_t3
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from core import grouped, local
from core.bert import cfg_full
from core.flat import Layout
from methods.mobihet_compute import Method

torch.set_num_threads(2)
cfg=cfg_full(); cfg['dropout']=0.0
lay=Layout(cfg); assert lay.P==77904 and lay.B==0
x=torch.from_numpy(np.load('cache/train_sub100_83466334_X.npy',mmap_mode='r')[:80].copy()).float()
y=torch.from_numpy(np.load('cache/train_sub100_83466334_Y.npy',mmap_mode='r')[:8].copy()).long()
m=Method.__new__(Method); m.env=SimpleNamespace(cfg=cfg); m.aux=64
m._rows_k=torch.tensor([64,32,16]); m.Q=torch.eye(16)*.9+.1/16
m._shift=torch.zeros(3,16); m._csum=torch.zeros(3,16,16)
p,b=lay.init(41,3); p.requires_grad_()
lab=torch.arange(64).remainder(16)[None].expand(3,-1); xg=x[8:72][None].expand(3,-1,-1)
m._synth=lambda ctx:(xg,lab)
ctx=SimpleNamespace(Gfix=3,x=x[:8][None].expand(3,-1,-1),y=y[None].expand(3,-1),clients=torch.arange(3),mask=torch.ones(3,8,dtype=torch.bool),active=torch.ones(3,dtype=torch.bool),P=lay.params(p),B={})
loss,_=m._loss_tier(ctx); grad=torch.autograd.grad(loss.sum(),p)[0]
for i,a in enumerate([64,32,16]):
    q=p[i:i+1].detach().clone().requires_grad_()
    z,_=grouped.forward(lay.params(q),{},torch.cat([ctx.x[i:i+1],xg[i:i+1,:a]],1),cfg,train=True)
    lt=m.Q.clamp_min(1e-6).log()[lab[i:i+1,:a]]
    ref=F.cross_entropy(z[:,:8].reshape(-1,16),y)+(lt.exp()*(lt-z[:,8:].log_softmax(-1))).sum(-1).mean()
    torch.testing.assert_close(loss[i],ref,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(grad[i],torch.autograd.grad(ref,q)[0][0],atol=3e-6,rtol=3e-4)
assert torch.isfinite(grad).all()
print('T3_BERT_REAL_LOSS_GRAD_PASS')
