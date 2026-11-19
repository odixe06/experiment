"""Saran & Kesswani (2025), Fig. 3, adapted to the 16 VeReMi classes."""
import numpy as np
import torch
from torch import nn


class GRUAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(77))
        self.register_buffer("components", torch.zeros(77, 5))
        self.gru = nn.GRU(1, 96, batch_first=True)
        self.key = nn.Linear(96, 96)
        self.query = nn.Linear(96, 96, bias=False)
        self.score = nn.Linear(96, 1, bias=False)
        self.head = nn.Linear(96, 16)

    def fit_pca(self, train_x):
        # Float64 sufficient statistics from the already-scaled training split.
        total = np.zeros(77, dtype=np.float64)
        cross = np.zeros((77, 77), dtype=np.float64)
        for start in range(0, len(train_x), 65536):
            rows = np.asarray(train_x[start:start + 65536], dtype=np.float64)
            total += rows.sum(axis=0)
            cross += rows.T @ rows
        center = total / len(train_x)
        _, vectors = np.linalg.eigh(cross - np.outer(total, center))
        components = vectors[:, -5:][:, ::-1].copy()
        pivots = np.abs(components).argmax(axis=0)
        components *= np.sign(components[pivots, np.arange(5)])
        self.center.copy_(torch.from_numpy(center))
        self.components.copy_(torch.from_numpy(components))

    def forward(self, x):
        projected = (x - self.center) @ self.components
        states, _ = self.gru(projected.unsqueeze(-1))
        scores = self.score(torch.tanh(self.key(states) + self.query(states[:, -1:])))
        context = (scores.softmax(dim=1) * states).sum(dim=1)
        return self.head(context)
