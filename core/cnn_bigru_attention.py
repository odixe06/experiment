"""Table-2 CNN-BiGRU-Attention adapted to 77 features and 16 classes.

Uses native PyTorch GRU as approved by the owner. See
papers/mnkash-2026-cnn-bigru/rebuild.md for the paper discrepancies.
"""
import torch
from torch import nn


def config():
    return dict(model="cnn-bigru-attention", n_features=77, n_classes=16,
                conv1=64, conv2=128, gru_hidden=64, attention_hidden=128,
                head_hidden=128, dropout=0.3, patch_len=7)


class CNNBiGRUAttention(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = config() if cfg is None else cfg
        a, b, h, d = cfg["conv1"], cfg["conv2"], cfg["gru_hidden"], cfg["attention_hidden"]
        self.conv = nn.Sequential(
            nn.Conv1d(1, a, 3),
            nn.BatchNorm1d(a),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(a, b, 3),
            nn.ReLU(),
        )
        self.gru = nn.GRU(b, h, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2*h, d)
        self.score = nn.Linear(d, 1, bias=False)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(2*h, 16)

    def forward(self, x):
        sequence = self.conv(x.unsqueeze(1)).transpose(1, 2)
        states, _ = self.gru(sequence)
        scores = self.score(torch.tanh(self.attention(states)))
        weights = scores.float().softmax(dim=1)
        context = (weights * states.float()).sum(dim=1)
        return self.head(self.dropout(context))
