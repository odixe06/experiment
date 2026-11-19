"""GRANDE, Marton et al. (ICLR 2024), Eqs. 2–7. VeReMi: 32 trees, depth 4."""
import torch
from torch import nn
from torch.nn import functional as F


class Entmax15(torch.autograd.Function):
    """Sparse feature selector, with the entmax-1.5 Jacobian on its support."""
    @staticmethod
    def forward(ctx, scores):
        z = (scores - scores.amax(-1, keepdim=True)) / 2
        ordered = z.sort(descending=True).values
        k = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
        mean = ordered.cumsum(-1) / k
        variance = ordered.square().cumsum(-1) / k - mean.square()
        tau = mean - (1 / k - variance).clamp_min(0).sqrt()
        support = (ordered >= tau).sum(-1, keepdim=True)
        root = (z - tau.gather(-1, support - 1)).clamp_min(0)
        ctx.save_for_backward(root)
        return root.square()

    @staticmethod
    def backward(ctx, grad):
        root, = ctx.saved_tensors
        center = (grad * root).sum(-1, keepdim=True) / root.sum(-1, keepdim=True)
        return root * (grad - center)


class GRANDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_scores = nn.Parameter(torch.empty(32, 15, 77))
        self.thresholds = nn.Parameter(torch.empty_like(self.feature_scores))
        self.leaf_weights = nn.Parameter(torch.empty(32, 16))
        self.leaf_logits = nn.Parameter(torch.empty(32, 16, 16))
        for p in self.parameters():
            nn.init.normal_(p, std=0.05)

    def forward(self, x):
        # Preserve near-threshold decisions and the entmax support under outer AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(self.thresholds.dtype)
            soft = Entmax15.apply(self.feature_scores)
            hard = F.one_hot(self.feature_scores.argmax(-1), 77).to(soft.dtype)
            selector = hard + (soft - soft.detach())
            selected = F.linear(x, selector.flatten(0, 1)).reshape(-1, 32, 15)
            margin = (selector * self.thresholds).sum(-1) - selected
            soft_left = (F.softsign(margin) + 1) / 2
            left = soft_left.round() + (soft_left - soft_left.detach())
            path = torch.ones_like(left[..., :1])
            for start, count in ((0, 1), (1, 2), (3, 4), (7, 8)):
                branch = left[..., start:start + count]
                path = torch.stack((path * branch, path * (1 - branch)), -1).flatten(-2)
            weights = (path * self.leaf_weights).sum(-1).softmax(-1)
            values = torch.einsum("btl,tlc->btc", path, self.leaf_logits)
            return (weights.unsqueeze(-1) * values).sum(1)
