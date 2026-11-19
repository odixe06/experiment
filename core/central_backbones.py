"""Five centralized VeReMi backbones. See docs/research/central-backbones.md."""
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


class TCNNBlock(nn.Module):
    """Khan Table 3. Keras MHA projects to 32*64 even for a one-channel input."""
    def __init__(self, channels, pool):
        super().__init__()
        self.qkv = nn.Linear(channels, 3 * 32 * 64)
        self.out = nn.Linear(32 * 64, channels)
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(channels, 256, 5, padding=2)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()
        self.output_norm = nn.LayerNorm(256)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, 32, 64).permute(2, 0, 3, 1, 4).unbind()
        h = F.scaled_dot_product_attention(q, k, v)
        h = self.out(h.transpose(1, 2).reshape(b, n, 2048))
        z = self.conv((x + self.norm(h)).transpose(1, 2)).relu() + h.transpose(1, 2)
        return self.output_norm(self.pool(z).transpose(1, 2))


class TCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*(TCNNBlock(1 if i == 0 else 256, i < 4) for i in range(5)))
        widths = (4 * 256, 128, 512, 128, 256, 128, 256, 64, 32, 64, 128, 512, 256, 128)
        layers = []
        for n_in, n_out in zip(widths, widths[1:]):
            layers.extend([nn.Linear(n_in, n_out), nn.ReLU()])
        self.head = nn.Sequential(*layers, nn.Linear(128, 16))

    def forward(self, x):
        return self.head(self.blocks(x.unsqueeze(-1)).flatten(1))


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(77, 512, 1), nn.Tanh(), nn.Conv1d(512, 256, 1), nn.Tanh())
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True)
        self.head = nn.Linear(128, 16)

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.lstm(z)
        return self.head(z[:, -1])


class CNNLSTMFeature(nn.Module):
    """One message: 77 scalar feature positions, not a temporal message window.

    Budget adaptation: Conv256/128, two LSTM96 layers, 196,240 parameters.
    The historical CNNLSTM above remains available for its existing checkpoints.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 256, 1), nn.Tanh(),
                                  nn.Conv1d(256, 128, 1), nn.Tanh())
        self.lstm = nn.LSTM(128, 96, num_layers=2, batch_first=True)
        self.head = nn.Linear(96, 16)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 77:
            raise ValueError("cnn-lstm-feature expects (batch, 77), one message per row")
        z = self.conv(x.unsqueeze(1)).transpose(1, 2)
        z, _ = self.lstm(z)
        return self.head(z[:, -1])


def build_model(name):
    if name == "cnn-lstm-feature":
        return CNNLSTMFeature()
    if name == "mambacnn":
        from core.mambacnn import MambaCNN
        return MambaCNN()
    if name == "cnn-bigru-attention":
        from core.cnn_bigru_attention import CNNBiGRUAttention
        return CNNBiGRUAttention()
    if name == "gru-attention":
        from core.gru_attention import GRUAttention
        return GRUAttention()
    if name == "sae":
        from core.sae import SupervisedSAE
        return SupervisedSAE()
    if name == "grande":
        from core.grande import GRANDE
        return GRANDE()
    if name == "dagsnet":
        from core.dagsnet import DAGSNet, width
        return DAGSNet(width(1))
    if name == "mlp":
        return nn.Sequential(nn.Linear(77, 64), nn.ReLU(), nn.Dropout(0.1),
                             nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 16))
    return {"bert": FeatureBERT, "tcnn": TCNN, "cnn-lstm": CNNLSTM, "tabm": TabM}[name]()


def classification_loss(logits, target):
    if logits.ndim == 3:  # individual CE per TabM member, shared minibatch
        return F.cross_entropy(logits.float().transpose(1, 2), target[:, None].expand(-1, logits.size(1)))
    return F.cross_entropy(logits.float(), target)


def predict(logits):
    if logits.ndim == 3:
        return logits.float().softmax(-1).mean(1).argmax(-1)
    return logits.argmax(-1)
