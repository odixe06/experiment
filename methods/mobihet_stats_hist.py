"""Train-only histogram marginal transport of the existing correlated Gaussian latent.

256 fixed bins on [-4,4]; occupied-bin means preserve isolated atoms (including binary flags).
This is an approximate marginal model, not a fitted Gaussian copula or exact joint-constraint repair.
"""
import torch

from methods.mobihet_stats import Method as Stats


BINS = 256


def hist_stats(x, y, classes=16):
    d = x.shape[1]
    count = torch.zeros(classes * d * BINS, dtype=torch.float64, device=x.device)
    total = torch.zeros_like(count)
    features = torch.arange(d, device=x.device)[None, :]
    for start in range(0, len(y), 8192):
        values = x[start:start+8192].double()
        bins = ((values + 4) * (BINS / 8)).floor().long().clamp(0, BINS-1)
        indices = ((y[start:start+8192].long()[:, None] * d + features) * BINS + bins).reshape(-1)
        count += torch.bincount(indices, minlength=count.numel()).double()
        total += torch.bincount(indices, weights=values.reshape(-1), minlength=total.numel())
    return count.reshape(classes, d, BINS), total.reshape(classes, d, BINS)


def transport(z, factors, labels, cdf, values):
    centered = (factors[labels] @ z[..., None]).squeeze(-1)
    sigma = factors.square().sum(-1).sqrt()[labels]
    u = (0.5 * (1 + torch.erf(centered / sigma / (2 ** 0.5)))).clamp(1e-7, 1-1e-7)
    # Search per (class slot, feature), across groups; never materialize a group x histogram bank.
    query = u.permute(1, 2, 0).contiguous()
    bins = torch.searchsorted(cdf[labels].contiguous(), query, right=False).clamp_max(BINS-1)
    return values[labels].gather(-1, bins).permute(2, 0, 1).contiguous()


class Method(Stats):
    name = 'mobihet_stats_hist'

    def _bootstrap(self):
        super()._bootstrap()
        count, total = hist_stats(self.env.train.X, self.env.train.Y)
        count, total = self.env.allreduce(count), self.env.allreduce(total)
        assert (count.sum(-1) > 0).all()
        self.hist_cdf = (count.cumsum(-1) / count.sum(-1, keepdim=True)).float()
        self.hist_values = (total / count.clamp_min(1)).float()
        self.hist_cdf[..., -1] = 1
        # Each client uploads float64 bin counts/sums; server broadcasts fp32 CDF/bin means.
        self.bootstrap_bytes['up'] += 16 * count.numel()
        self.bootstrap_bytes['down'] += 8 * count.numel()

    def _gaussian_rows(self, z):
        return transport(z, self.L, self.labels, self.hist_cdf, self.hist_values)
