"""Centralized MLP architecture with named FL feature and classifier modules."""
from torch import nn


def config():
    return dict(model="mlp", n_features=77, n_classes=16, first_hidden=64,
                head_hidden=32, dropout=0.1, patch_len=7)


class MLP(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = config() if cfg is None else cfg
        self.features = nn.Sequential(nn.Linear(77, cfg["first_hidden"]), nn.ReLU(),
                                      nn.Dropout(cfg["dropout"]),
                                      nn.Linear(cfg["first_hidden"], cfg["head_hidden"]), nn.ReLU())
        self.drop = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(cfg["head_hidden"], 16)

    def forward(self, x):
        return self.head(self.drop(self.features(x)))
