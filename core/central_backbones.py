"""Centralized VeReMi backbones of the experiment scenarios. See docs/research/backbones/central-backbones.md."""
import torch
from torch import nn
from torch.nn import functional as F

from core.tabm import TabM


class FeatureBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(77, 64))
        self.identity = nn.Parameter(torch.empty(77, 64))
        self.cls = nn.Parameter(torch.empty(1, 1, 64))
        for param in (self.scale, self.identity, self.cls):
            nn.init.normal_(param, std=0.02)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(64, 4, 128, 0.1, "gelu", batch_first=True)
            for _ in range(2)])
        self.head = nn.Linear(64, 16)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.scale + self.identity
        tokens = torch.cat((self.cls.expand(len(x), -1, -1), tokens), dim=1)
        return self.head(self.blocks(tokens)[:, 0])


def build_model(name):
    if name == "cnn-bigru-attention":
        from core.cnn_bigru_attention import CNNBiGRUAttention
        return CNNBiGRUAttention()
    if name == "sae":
        from core.sae import SupervisedSAE
        return SupervisedSAE()
    if name == "sae-ce":  # same network as the FL SAE (core.sae.fl_config), trained with plain CE like the FL runs (owner 20/11)
        from core.sae import SupervisedSAE, fl_config
        return SupervisedSAE(fl_config())
    if name == "dagsnet":
        from core.dagsnet import DAGSNet, width
        return DAGSNet(width(1))
    if name == "mlp":
        return nn.Sequential(nn.Linear(77, 64), nn.ReLU(), nn.Dropout(0.1),
                             nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 16))
    return {"bert": FeatureBERT, "tabm": TabM}[name]()


def classification_loss(logits, target):
    if logits.ndim == 3:  # individual CE per TabM member, shared minibatch
        return F.cross_entropy(logits.float().transpose(1, 2), target[:, None].expand(-1, logits.size(1)))
    return F.cross_entropy(logits.float(), target)


def predict(logits):
    if logits.ndim == 3:
        return logits.float().softmax(-1).mean(1).argmax(-1)
    return logits.argmax(-1)
