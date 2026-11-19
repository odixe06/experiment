"""MobiHet Core. Simulated per-client wire payload, excluding transport/security framing."""
import struct
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from core import evaluate, grouped, local
from methods.mobihet_stats import Method as Stats
from methods.mobihet_stats_hist import hist_stats, transport, BINS


class Method(Stats):
    name = "mobihet_core"
    use_q = True

    def _bootstrap(self):
        env = self.env
        env.hp.setdefault("synthetic_step_floor", 16)
        self.aux = 64
        x, y = env.train.X, env.train.Y.long()
        c, d = 16, x.shape[1]
        self.labels = torch.arange(self.aux, device=x.device) % c
        clients = torch.repeat_interleave(torch.arange(env.K, device=x.device), env.train.off.diff())
        counts = torch.bincount(clients*c+y, minlength=env.K*c).reshape(-1,c)
        self.la_shift = ((counts+1)/(counts.sum(1, keepdim=True)+c)).log()
        hist, _ = hist_stats(x, y)
        hist = env.allreduce(hist)
        mass = hist.sum(-1, keepdim=True)
        if (mass == 0).any():
            raise ValueError("bootstrap requires train support for every class")
        encoded = (65535 * hist.cumsum(-1) / mass).round().cpu().numpy().astype("<u2")
        raw = encoded.tobytes()
        packet = struct.pack("<4sIII16x", b"MCDF", c,d,zlib.crc32(raw)) + zlib.compress(raw,6)
        decoded = zlib.decompress(packet[32:])
        assert decoded == raw and zlib.crc32(decoded) == struct.unpack("<4sIII16x",packet[:32])[3]
        self.cdf = torch.from_numpy(np.frombuffer(decoded,dtype="<u2").astype(np.float32).reshape(c,d,BINS)).to(x.device)/65535
        prev = F.pad(self.cdf[:,:,:-1], (1,0))
        scores = torch.special.ndtri(((self.cdf+prev)/2).clamp(1e-6,1-1e-6))
        cols = torch.arange(d,device=x.device)
        sums = x.new_zeros((c,d),dtype=torch.float64)
        cross = x.new_zeros((c,d,d),dtype=torch.float64)
        for label in range(c):
            for start in range(0,len(y),131072):
                sub = x[start:start+131072][y[start:start+131072]==label]
                bins = ((sub.float()+4)*32).long().clamp(0,BINS-1)
                z = scores[label,cols,bins].double()
                sums[label] += z.sum(0)
                cross[label] += z.T @ z
        mean = env.allreduce(sums)/mass[:,0]
        cov = env.allreduce(cross)/mass[:,0,None] - mean[:,:,None]*mean[:,None,:]
        sd = cov.diagonal(dim1=1,dim2=2).clamp_min(1e-10).sqrt()
        corr = cov/(sd[:,:,None]*sd[:,None,:])
        corr.diagonal(dim1=1,dim2=2).fill_(1)
        self.L = torch.linalg.cholesky(corr+1e-4*torch.eye(d,device=x.device)).half().float()
        centers = (torch.arange(BINS,device=x.device)+.5)/32-4
        self.values = centers.expand(c,d,BINS).clone()
        self.values[:,-6:] = self.values[:,-6:].round().clamp(0,1)
        self.q = torch.eye(c,device=x.device)
        tri = d*(d+1)//2
        self.bootstrap_bytes = {"up":4*c*d*BINS + 8*c*(d+tri), "down":len(packet)+2*c*tri}
        batch = env.hp["batch"]
        steps = (env.n.long()+batch-1)//batch
        floor = env.hp["synthetic_step_floor"]
        self.floor_extra = int(env.allreduce((floor-steps).clamp_min(0).sum()))
        self.syn_rows = int(env.allreduce(steps.clamp_min(floor).sum()))*64

    def payload_bytes(self):
        return {"bootstrap":self.bootstrap_bytes,"per_round":{"up":576 if self.use_q else 0,
                "down":512 if self.use_q else 0}}

    def _loss(self, ctx):
        z = torch.randn(ctx.Gfix,64,77,device=ctx.x.device,generator=self.gen_g)
        with torch.autocast(ctx.x.device.type, enabled=False):
            fake = transport(z,self.L,self.labels,self.cdf,self.values).to(ctx.x.dtype)
        mask = torch.cat((ctx.mask,ctx.active[:,None].expand(-1,64)),1)
        logits, buffers = grouped.TRAIN(ctx.P,ctx.B,torch.cat((ctx.x,fake),1),self.env.cfg,mask=mask,train=True)
        b = ctx.x.shape[1]
        real = local.ce_loss(logits[:,:b]+self._shift[:,None],ctx.y,ctx.mask)
        logp = logits[:,b:].float().log_softmax(-1)
        target = self.q[self.labels]
        hard_weight = self.env.hp.get("q_hard_weight",0.0)
        if hard_weight:
            hard = F.one_hot(self.labels,num_classes=16).to(target.dtype)
            target = (1-hard_weight)*target + hard_weight*hard
        target = target.expand_as(logp)
        return real + F.kl_div(logp,target,reduction="none").sum(-1).mean(1), buffers

    def round(self,r,lr):
        e = self.env
        # Pre-training snapshot. First round uses hard labels in both arms.
        refresh = self.use_q and r >= 2 and (r-2) % e.hp.get("q_interval",1) == 0
        if refresh:
            sums, counts = evaluate.class_sums(e.lay,self.priv_p,self.priv_b,e.train,e.cfg,
                torch.arange(e.K,device=e.dev),d=16,probabilities=True,max_rows=8192)
            # Upload fp16 class means + uint32 counts; restore weighted sums before pooling.
            means = (sums/counts.clamp_min(1)[...,None]).half().float()
            mass = e.allreduce(counts.sum(0))
            q = e.allreduce((means*counts[...,None]).sum(0))/mass.clamp_min(1)[:,None]
            q = q.half().float()
            self.q = q/q.sum(1,keepdim=True).clamp_min(1e-12)
        info = super().round(r,lr)
        payload = self.payload_bytes()["per_round"]
        info.update(bytes_up=payload["up"] if refresh else 0,bytes_down=payload["down"] if refresh else 0,
                    synthetic_rows=self.syn_rows, floor_extra_steps=self.floor_extra,
                    q13=self.q[13].tolist(),q_forward_rows=int(e.allreduce(e.n.double().sum())) if refresh else 0)
        return info

    def state(self):
        state = super().state()
        if self.env.hp.get("q_interval",1) > 1:
            state["q"] = self.q.cpu()
        return state

    def load_state(self,state):
        super().load_state(state)
        if self.env.hp.get("q_interval",1) > 1:
            self.q = state["q"].to(self.env.dev)
