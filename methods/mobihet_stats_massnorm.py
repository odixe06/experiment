"""Isolate class allocation from total Gaussian loss mass."""
from methods.mobihet_stats_adaptive import Method as Adaptive


class Method(Adaptive):
    name = "mobihet_stats_massnorm"

    def __init__(self, env):
        super().__init__(env)
        self.gaussian_weights = self.gaussian_weights / self.gaussian_weights.mean(1, keepdim=True)
