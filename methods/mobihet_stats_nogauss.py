"""Diagnostic removal of Gaussian rows and CE; retain real logit-adjusted CE."""
from core import grouped, local
from methods.mobihet_stats import Method as Stats


class Method(Stats):
    name = "mobihet_stats_nogauss"

    def _loss(self, ctx):
        logits, buffers = grouped.TRAIN(ctx.P, ctx.B, ctx.x, self.env.cfg, mask=ctx.mask, train=True)
        return local.ce_loss(logits + self._shift[:, None, :], ctx.y, ctx.mask), buffers
