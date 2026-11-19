"""Supervised sparse autoencoder from TAS-VANET (Eqs. 9–12), adapted to VeReMi v4."""
import math

import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F


def config():
    return {"model": "sae", "n_features": 77, "n_classes": 16, "encoder_hidden": 32,
            "head_hidden": 16, "decoder_hidden": 32, "dropout": 0.0, "patch_len": 7,
            "sparse_rho": 0.05, "reconstruction_weight": 1.0,
            "sparsity_weight": 1.2, "classification_weight": 1.5}


def fl_config():
    """Owner 15/11: CE-only FL variant; full pre-classifier representation."""
    return dict(config(), objective="ce", code_dim=16, head_hidden=93,
                classification_weight=1.0, reconstruction_weight=0.0, sparsity_weight=0.0)


class SupervisedSAE(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = config() if cfg is None else cfg
        eh, code, dh = cfg["encoder_hidden"], cfg.get("code_dim", cfg["head_hidden"]), cfg["decoder_hidden"]
        self.encoder = nn.Sequential(nn.Linear(77, eh), nn.ReLU(), nn.Linear(eh, code), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(code, dh), nn.ReLU(), nn.Linear(dh, 77), nn.Sigmoid())
        self.head = nn.Linear(77 + code, 16)

    @staticmethod
    def unit_input(x):
        # Fixed bounds from the v4 normal-score scaler. Last six columns are binary flags.
        return torch.cat(((x[:, :71].float() + 3.72) / 7.44, x[:, 71:].float()), dim=1)

    def forward(self, x, *, auxiliary=False):
        x = self.unit_input(x)
        code = self.encoder(x)
        logits = self.head(torch.cat((x, code), dim=1))
        if auxiliary:
            return logits, code, self.decoder(code), x
        return logits


@torch.no_grad()
def batch_code_mean(model, x, local, micro, global_size, *, amp):
    """Prepass only when a logical batch needs multiple forwards on one GPU."""
    total = torch.zeros(16, device=x.device)
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
        for rows in local.split(micro):
            code = model.encoder(model.unit_input(x[rows]))
            total += code.float().sum(0)
    if dist.is_initialized():
        dist.all_reduce(total)
    return total / global_size


def sae_loss(outputs, labels, count, global_size, batch_mean=None):
    logits, code, reconstruction, inputs = outputs
    local_mean = code.float().mean(dim=0)
    if batch_mean is None:
        total = local_mean.detach() * count
        if dist.is_initialized():
            dist.all_reduce(total)
        batch_mean = total / global_size
    # KL is evaluated once per logical batch. The runner weights each local gradient by
    # its row count, so the local mean supplies the exact derivative after DDP averaging.
    mean = (batch_mean + (local_mean - local_mean.detach())).clamp(1e-6, 1 - 1e-6)
    rho = 0.05
    kl = rho * (math.log(rho) - mean.log()) + (1-rho) * (math.log1p(-rho) - torch.log1p(-mean))
    reconstruction_loss = (reconstruction.float() - inputs).square().sum(dim=1).mean()
    return reconstruction_loss + 1.2 * kl.sum() + 1.5 * F.cross_entropy(logits.float(), labels)
