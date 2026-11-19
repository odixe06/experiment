"""Train-count weighted Gaussian CE, batch/(batch+local class count)."""
import torch

from methods.mobihet_stats import Method as Stats


def support_weights(counts, batch):
    return batch / (counts.float() + batch)


class Method(Stats):
    name = "mobihet_stats_adaptive"

    def __init__(self, env):
        super().__init__(env)
        counts = torch.stack([torch.bincount(env.train.Y[env.train.off[k]:env.train.off[k+1]].long(),
                                             minlength=16) for k in range(env.K)])
        self.gaussian_weights = support_weights(counts, env.hp["batch"])

    def _synthetic_weights(self, ctx):
        return self.gaussian_weights[ctx.clients][:, self.labels]
