"""Align uniform Gaussian CE with the global-prior target of real adjusted CE."""
from methods.mobihet_stats import Method as Stats


class Method(Stats):
    name = "mobihet_stats_prior"

    def _synthetic_logits(self, logits):
        # log(uniform prior) is a class-constant and cancels in softmax.
        return logits - self.log_pg
