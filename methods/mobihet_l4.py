"""MobiHet FB-L, configuration L4 (owner 19/11/2026), on every single-model backbone of this repo.

methods/mobihet*.py are byte copies of ../mobihet (method a4147a1b..., L4 r50 audited there for DAGSNet and FeatureBERT).
This subclass only adapts the per-step loss to backbones whose forward differs from DAGSNet's (G, n, 16) logits:

  tabm  logits (G, n, K members, 16), trained like the TabM baselines (core.local.ce_loss): every loss term is computed per
        member and averaged over members. Row score for real_select = member-mean CE; class-Q packet = member-mean
        probabilities (the ensemble prediction core.evaluate scores).
  sae   FL objective "ce" leaves the decoder unused; core.grouped.sae_regularizer adds its explicit zero gradient, as the
        SAE baselines do.

dagsnet / mlp / cnn-bigru-attention: unchanged mobihet_budget (DAGSNet path bit-exact with ../mobihet).
"""
import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet_budget import Method as Budget
from methods.mobihet_wstudy import active


class Method(Budget):
    def _loss_tier(self, ctx):
        model = self.env.cfg.get("model")
        if model == "sae":
            loss, new_b = super()._loss_tier(ctx)
            return loss + grouped.sae_regularizer(ctx.P, ctx.x, None, ctx.mask, self.env.cfg), new_b
        if model != "tabm":
            return super()._loss_tier(ctx)
        assert self.budget, "TabM path is defined for the L4 row budget"
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        xg, lab = self._synth(ctx)
        rows = self._rows_k[ctx.clients][:, None]
        pos = torch.arange(a, device=lab.device)[None]
        first = (rows == 8) * (8 * (ctx.step % 2)) if bool((self._rows_k == 8).any()) else 0
        keep = active(ctx)[:, None] & (pos >= first) & (pos < first + rows) & (not self.synth_off)
        real, qrow = self._real_rows(ctx)
        x, y, idx = ctx.x, ctx.y, (ctx.rows if self.select != "random" else None)
        if self.compact:
            wr = 0 if ctx.step >= self._real_steps else self._wr
            if self.select != "random":
                order = (~real).to(torch.uint8).argsort(dim=1, stable=True)
                x, y, idx = x.gather(1, order[..., None].expand_as(x)), y.gather(1, order), idx.gather(1, order)
                real, qrow = real.gather(1, order), qrow.gather(1, order)
                idx = idx[:, :wr]
            x, y, real, qrow = x[:, :wr], y[:, :wr], real[:, :wr], qrow[:, :wr]
            xg, lab, keep, a = xg[:, :self._ws], lab[:, :self._ws], keep[:, :self._ws], self._ws
            b = wr
        z, new_b = grouped.TRAIN(ctx.P, ctx.B, torch.cat([x, xg], 1), self.env.cfg,
                                 mask=torch.cat([real, keep], 1), train=True)
        K = z.shape[2]                                                   # (G, b + a, K, 16)
        zr = z[:, :b] + self._shift[:, None, None, :]
        row_ce = F.cross_entropy(zr.reshape(-1, 16), y[:, :, None].expand(G, b, K).reshape(-1),
                                 reduction="none").view(G, b, K).mean(2)
        ce = local.masked_mean(row_ce, real)
        logp = F.log_softmax(z, -1)
        with torch.no_grad():
            at = (torch.arange(G, device=lab.device)[:, None].expand_as(y), y)
            self._csum.index_put_(at, torch.where(qrow[..., None], logp[:, :b].detach().exp().mean(2), 0.0),
                                  accumulate=True)
            self._seen.index_put_(at, qrow.float(), accumulate=True)
            if hasattr(self, "_trained"):
                self._trained += real.sum()
            if self.select != "random":
                sink = self._score.numel() - 1
                self._score.index_put_((torch.where(real, idx, sink).reshape(-1),),
                                       row_ce.detach().float().reshape(-1))
        if self.Q is None:
            synth = F.cross_entropy(z[:, b:].reshape(-1, 16), lab[:, :, None].expand(G, a, K).reshape(-1),
                                    reduction="none").view(G, a, K).mean(2)
            return ce + local.masked_mean(synth, keep), new_b
        lq = self.Q.clamp_min(1e-6).log()[lab][:, :, None]               # (G, a, 1, 16)
        synth = (lq.exp() * (lq - logp[:, b:])).sum(-1).mean(2)
        loss = ce + local.masked_mean(synth, keep)
        if self.q_real:
            lr_ = self.Q.clamp_min(1e-6).log()[y][:, :, None]
            loss = loss + local.masked_mean((lr_.exp() * (lr_ - logp[:, :b])).sum(-1).mean(2), real)
        return loss, new_b
