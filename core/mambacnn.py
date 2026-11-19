"""Supervised VeReMi adaptation of Feng et al. (2026), not the paper's KD pipeline.

Paper leaves student dimensions unspecified; chosen dimensions are documented in
papers/feng-2026-mambacnn/rebuild.md. Selective SSM follows Mamba-1's real-valued
recurrence (state-spaces/mamba). CUDA uses the pinned official selective-scan
extension; CPU/double precision retains the PyTorch reference.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def selective_scan_reference(u, delta, a, b, c, skip):
    """B,L,H inputs; H,N negative a; B,L,N input-dependent b/c.

    Parallel prefix composition of h[t] = exp(delta[t]*a)*h[t-1]
    + delta[t]*b[t]*u[t]. No division by cumulative products (can underflow).
    Caller disables autocast and supplies fp32, or fp64 for numerical tests.
    """
    decay = torch.exp(delta.unsqueeze(-1) * a)
    state = (delta * u).unsqueeze(-1) * b.unsqueeze(2)
    offset = 1
    while offset < u.size(1):
        state = torch.cat((state[:, :offset], state[:, offset:]
                           + decay[:, offset:] * state[:, :-offset]), dim=1)
        decay = torch.cat((decay[:, :offset], decay[:, offset:]
                           * decay[:, :-offset]), dim=1)
        offset *= 2
    return (state * c.unsqueeze(2)).sum(-1) + u * skip


def selective_scan(u, delta, a, b, c, skip):
    if not u.is_cuda or u.dtype == torch.float64:
        return selective_scan_reference(u, delta, a, b, c, skip)
    # Fail explicitly if unavailable: silently using the reference takes ~22h/epoch.
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    return selective_scan_fn(u.transpose(1, 2), delta.transpose(1, 2), a,
                             b.transpose(1, 2), c.transpose(1, 2), skip,
                             delta_softplus=False).transpose(1, 2)


class MambaBlock(nn.Module):
    def __init__(self, width=40, state=16):
        super().__init__()
        inner = 2 * width
        self.rank = math.ceil(width / 16)
        self.state = state
        self.norm = nn.RMSNorm(width, eps=1e-5)
        self.in_proj = nn.Linear(width, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, 4, padding=3, groups=inner)
        self.x_proj = nn.Linear(inner, self.rank + 2 * state, bias=False)
        self.dt_proj = nn.Linear(self.rank, inner)
        self.a_log = nn.Parameter(torch.arange(1, state + 1).float().log().repeat(inner, 1))
        self.skip = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, width, bias=False)
        nn.init.uniform_(self.dt_proj.weight, -self.rank**-0.5, self.rank**-0.5)
        with torch.no_grad():
            dt = torch.exp(torch.rand(inner) * math.log(100) + math.log(0.001))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        normalized = self.norm(x.to(self.norm.weight.dtype)).to(x.dtype)
        u, gate = self.in_proj(normalized).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2))
        dt, b, c = self.x_proj(u).split((self.rank, self.state, self.state), dim=-1)
        # Mamba's exponential recurrence and gate accumulate in fp32 under AMP.
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            delta = F.softplus(F.linear(dt.to(dtype), self.dt_proj.weight.to(dtype),
                                       self.dt_proj.bias.to(dtype)))
            y = selective_scan(u.to(dtype), delta, -self.a_log.to(dtype).exp(),
                               b.to(dtype), c.to(dtype), self.skip.to(dtype))
            y = y * F.silu(gate.to(dtype))
        return x + self.out_proj(y.to(u.dtype))


class MambaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(1, 40)
        self.branches = nn.ModuleList([
            nn.Sequential(MambaBlock(), MambaBlock()) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(40, 4, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(40, 4, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.SiLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.SiLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(144, 64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 16))

    def forward(self, x):
        tokens = self.embedding(x.unsqueeze(-1))
        left, right, own = [branch(tokens) for branch in self.branches]
        cross, _ = self.cross_attention(left, right, right, need_weights=False)
        own, _ = self.self_attention(own, own, own, need_weights=False)
        local = self.conv(x.unsqueeze(1)).mean(-1)
        features = torch.cat((cross.mean(1), own.mean(1), local), dim=-1)
        return self.head(features)
