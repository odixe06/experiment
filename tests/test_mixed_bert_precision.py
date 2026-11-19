"""Compare V2X BERT's actual loss gradients with scaled fp16 and fp32."""
import json
from pathlib import Path
from unittest.mock import patch
import torch
from core import grouped, local
from methods.mixed import Method
from test_mixed_methods import ToyEnv, NAMES

def run():
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]
    for seed in (41,42,43):
        env=ToyEnv();env.hp['seed']=seed
        method=Method(env,'v2x_htfl',dict(enumerate(NAMES)),NAMES)
        group=next(g for g in method.groups if g.name=='bert')
        method.V=method.V.to(device)
        for amp in (True,False):
            q=local.Ctx();q.G=q.Gfix=1
            p=group.p.detach().to(device).clone()
            q.P={k:v.detach().requires_grad_() for k,v in group.lay.params(p).items()}
            q.B={k:v.to(device) for k,v in group.lay.bufs(group.b).items()}
            q.x=torch.randn(1,64,77,generator=torch.Generator().manual_seed(7)).to(device)
            q.y=torch.zeros(1,64,dtype=torch.long,device=device)
            q.mask=torch.ones_like(q.y,dtype=torch.bool)
            torch.manual_seed(123)
            with patch.object(grouped,'TRAIN',grouped.forward):
                with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
                    loss,_=method._loss(q,group,1,torch.ones(1,device=device))
                (loss.sum()*4096).backward()
            bad=[k for k,v in q.P.items() if v.grad is None or not torch.isfinite(v.grad).all()]
            row=dict(seed=seed,amp=amp,loss=float(loss.detach()),bad=bad)
            rows.append(row)
            assert torch.isfinite(loss).all()
            if not amp:assert not bad,row
    assert any(r['bad'] for r in rows if r['amp']), 'Original overflow not reproduced'
    print('BERT_PRECISION_EVIDENCE',json.dumps(rows),flush=True)
    return rows

if __name__=='__main__':run()
