"""Regression for scaled-half SAE gradients and the mixed V2X precision policy."""
import torch
from core import grouped
from core.flat import Layout
from core.mixed_backbones import config
from methods.mixed import Method
from test_mixed_methods import ToyEnv, NAMES


def run():
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for amp in (True,False):
        cfg=config('sae')
        lay=Layout(cfg)
        p,b=lay.init(42,1,device);p.requires_grad_()
        x=torch.randn(1,64,77,generator=torch.Generator().manual_seed(7)).to(device)
        with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
            z,h,_=grouped.forward(lay.params(p),lay.bufs(b),x,cfg,hidden=True)
        (h.sum()*4096+z.sum()*0).backward()
        bias=lay.params(p.grad)['projection.bias']
        if not amp:
            assert torch.equal(bias,torch.full_like(bias,64*4096))
            assert torch.isfinite(p.grad).all()
        else:
            assert not torch.isfinite(bias).all(), 'Expected original fp16 overflow not reproduced'
    for name in ('v2x_htfl','fedproto','pfedes'):
        env=ToyEnv();env.hp['amp']=True
        method=Method(env,name,dict(enumerate(NAMES)),NAMES)
        for group in method.groups:
            assert group.hp['amp']==(not(name=='v2x_htfl' and group.name in ('sae','bert')))
    print('SAE FP32 GRADIENT AND MIXED PRECISION POLICY PASS',flush=True)


if __name__=='__main__':run()
