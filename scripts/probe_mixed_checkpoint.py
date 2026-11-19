"""Replay original checkpoint; capture first real SAE failure without changing updates."""
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import torch
from core import grouped, local
from core.optim import GroupAdamW

original_step = GroupAdamW.step
captured = False
def probe_step(self, flat, grad, lr, active):
    global captured
    frame = inspect.currentframe().f_back
    ctx = frame.f_locals['ctx']
    cfg = frame.f_locals['cfg']
    bad = active & ~torch.isfinite(grad).all(1)
    if not captured and cfg.get('backbone') == 'sae' and bool(bad.any()):
        captured = True
        caller = frame.f_back.f_locals
        method, group, r = caller['self'], caller['group'], caller['r']
        i = int(torch.where(bad)[0][0])
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        fixture = dict(P={k:v[i:i+1].detach().cpu().clone() for k,v in ctx.P.items()},
            B={k:v[i:i+1].detach().cpu().clone() for k,v in ctx.B.items()},
            x=ctx.x[i:i+1].cpu(), y=ctx.y[i:i+1].cpu(), mask=ctx.mask[i:i+1].cpu(),
            cfg=cfg, r=r, proto=method.proto.cpu(), has=method.has.cpu(), V=method.V.cpu(),
            lam=ctx.pad(caller['lam'])[i:i+1].cpu())
        out = Path('/kaggle/working')
        torch.save(fixture,out/f'failure_rank{os.environ["RANK"]}.pt')
        linear = grouped._linear
        def projection32(c,name,x):
            if name == 'projection':
                with torch.autocast('cuda',enabled=False): return linear(c,name,x.float())
            return linear(c,name,x)
        rows = []
        for mode in ('old_amp','amp_scale1','projection_fp32','loss_fp32','both_fp32','fp32'):
            q = local.Ctx(); q.G = q.Gfix = 1
            q.P = {k:v.to(flat.device).requires_grad_() for k,v in fixture['P'].items()}
            q.B = {k:v.to(flat.device) for k,v in fixture['B'].items()}
            q.x,q.y,q.mask = [fixture[k].to(flat.device) for k in ('x','y','mask')]
            def forward(*args,**kwargs):
                with torch.autocast('cuda',dtype=torch.float16,enabled=mode != 'fp32'):
                    return grouped.forward(*args,**kwargs)
            with torch.enable_grad(), patch.object(grouped,'TRAIN',forward), patch.object(grouped,'_linear',
                    projection32 if mode in ('projection_fp32','both_fp32') else linear):
                with torch.autocast('cuda',dtype=torch.float16,enabled=mode not in ('loss_fp32','both_fp32','fp32')):
                    loss,_ = method._loss(q,group,r,fixture['lam'].to(flat.device))
                (loss.sum()*(1 if mode == 'amp_scale1' else 4096)).backward()
            bad_names = [k for k,v in q.P.items() if not torch.isfinite(v.grad).all()]
            rows.append(dict(mode=mode,loss=float(loss.detach()),bad=bad_names))
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
        evidence = dict(rank=int(os.environ['RANK']),round=r,client=int(method.env.ids[int(ctx.clients[i])]),rows=rows)
        (out/f'precision_rank{os.environ["RANK"]}.json').write_text(json.dumps(evidence,indent=2))
        print('PRECISION_EVIDENCE',json.dumps(evidence),flush=True)
    return original_step(self,flat,grad,lr,active)

GroupAdamW.step = probe_step
sys.argv[0] = 'scripts/run_mixed_fl.py'
runpy.run_path(str(Path('scripts/run_mixed_fl.py').resolve()),run_name='__main__')
