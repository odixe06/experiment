"""Stats ablation: synthetic CE retained, BatchNorm moments from real rows only."""
import torch

from methods.mobihet_stats import Method as Stats


class Method(Stats):
    name = "mobihet_stats_realbn"

    def _normalization_mask(self, real_mask, active, groups, auxiliary):
        return torch.cat([real_mask, torch.zeros(groups, auxiliary, dtype=torch.bool,
                                                device=real_mask.device)], 1)
