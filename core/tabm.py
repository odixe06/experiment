"""TabM k=4 backbone used by centralized and federated experiments."""
import torch
from torch import nn


def config():
    return {"model": "tabm", "n_features": 77, "n_classes": 16,
            "members": 4, "head_hidden": 32, "dropout": 0.1, "patch_len": 7}


class EnsembleBlock(nn.Module):
    def __init__(self, n_in, first, hidden=32, members=4, dropout=0.1):
        super().__init__()
        self.shared = nn.Linear(n_in, hidden, bias=False)
        self.r = nn.Parameter(torch.ones(members, n_in))
        self.s = nn.Parameter(torch.ones(members, hidden))
        self.bias = nn.Parameter(torch.empty(members, hidden))
        nn.init.uniform_(self.bias, -n_in**-0.5, n_in**-0.5)
        if first:
            with torch.no_grad():
                self.r.bernoulli_(0.5).mul_(2).sub_(1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop((self.shared(x * self.r) * self.s + self.bias).relu())


class TabM(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = config() if cfg is None else cfg
        h, k, p = cfg["head_hidden"], cfg.get("members", 4), cfg.get("dropout", 0.1)
        self.blocks = nn.Sequential(EnsembleBlock(cfg["n_features"], True, h, k, p),
                                    EnsembleBlock(h, False, h, k, p))
        self.weight = nn.Parameter(torch.empty(k, h, cfg.get("n_classes", 16)))
        self.bias = nn.Parameter(torch.empty(k, cfg.get("n_classes", 16)))
        for q in (self.weight, self.bias):
            nn.init.uniform_(q, -h**-0.5, h**-0.5)

    def forward(self, x):
        z = self.blocks(x.unsqueeze(1))
        return torch.einsum("bki,kio->bko", z, self.weight) + self.bias


def log_probs(logits, ensemble=False):
    """Predictive log-probability after averaging member probabilities."""
    if not ensemble and logits.ndim != 4:
        return logits.float().log_softmax(-1)
    return torch.logsumexp(logits.float().log_softmax(-1), dim=-2) - \
        torch.log(torch.as_tensor(logits.shape[-2], device=logits.device, dtype=torch.float32))


def predict(logits):
    return log_probs(logits, ensemble=True).argmax(-1)
