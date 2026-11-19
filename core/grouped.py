"""G independent DAGSNets in one forward: the function of core/dagsnet.DAGSNet, vectorised over clients.

Parameters/buffers are dicts keyed by the reference state_dict names, each tensor with a leading G axis. Activations are
channels-last (G, b, L, C); a Conv1d is k shifted windows concatenated on C and one bmm, so a grouped step is a few large
fp16 matmuls instead of G small launches. BatchNorm statistics are masked: padded rows (clients shorter than the step's
batch) never enter mean/var, loss or running stats. Equivalence to the reference module: tests/test_grouped.py.
"""
import math

import torch
import torch.nn.functional as F

from core.dagsnet import DAGSNet

EPS, MOM = 1e-5, 0.1


def _f32(t):
    """fp16 activations -> fp32 for statistics/outputs; fp64 (tests) stays fp64."""
    return t if t.dtype == torch.float64 else t.float()


def init_state(cfg, seed, g=1):
    """Reference-module initialisation under `seed`, broadcast to G copies (FL: every client starts from θ^0)."""
    torch.manual_seed(seed)
    sd = DAGSNet(cfg).state_dict()
    params = {k: v.unsqueeze(0).repeat(g, *[1] * v.dim()).float() for k, v in sd.items()
              if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    bufs = {k: v.unsqueeze(0).repeat(g, *[1] * v.dim()).float() for k, v in sd.items()
            if k.endswith(("running_mean", "running_var"))}
    return params, bufs


def to_state_dict(params, bufs, i):
    """Client i as a reference DAGSNet state_dict (num_batches_tracked is not used by the model)."""
    sd = {k: v[i].detach().clone() for k, v in {**params, **bufs}.items()}
    return sd


class _Ctx:
    def __init__(self, P, B, mask, train):
        self.P, self.B, self.train, self.newB = P, B, train, {}
        if mask is not None:
            self.m = mask.to(torch.float32)[:, :, None, None]          # (G, b, 1, 1)
            self.n = mask.to(torch.float32).sum(1)                     # (G,) real rows


def _conv(c, name, x, k):
    w = c.P[name + ".weight"]                                         # (G, O, C, k)
    G, b, L, C = x.shape
    if k > 1:
        p = k // 2
        xp = F.pad(x, (0, 0, p, p))
        x = torch.cat([xp[:, :, j:j + L] for j in range(k)], dim=-1)  # (G, b, L, k*C), index j*C + c
    wm = w.permute(0, 3, 2, 1).reshape(G, k * C, -1)                   # (G, k*C, O)
    return torch.bmm(x.reshape(G, b * L, k * C), wm.to(x.dtype)).view(G, b, L, -1)


def _bn(c, name, x):
    w, bias = c.P[name + ".weight"], c.P[name + ".bias"]
    rm, rv = c.B[name + ".running_mean"], c.B[name + ".running_var"]
    if c.train:
        xf = _f32(x)
        L = x.shape[2]
        n = (c.n * L).clamp_min(1)[:, None]                              # (G, 1)
        mean = (xf * c.m).sum((1, 2)) / n                                # (G, C)
        var = (((xf - mean[:, None, None]) ** 2) * c.m).sum((1, 2)) / n
        nu = n / (n - 1).clamp_min(1)
        c.newB[name + ".running_mean"] = (1 - MOM) * rm + MOM * mean.detach()
        c.newB[name + ".running_var"] = (1 - MOM) * rv + MOM * (var * nu).detach()
    else:
        mean, var = rm, rv
    scale = w * torch.rsqrt(var + EPS)
    shift = bias - mean * scale
    return (x * scale[:, None, None].to(x.dtype) + shift[:, None, None].to(x.dtype))


def _cbr(c, name, x, k):
    return torch.relu(_bn(c, name + ".1", _conv(c, name + ".0", x, k)))


# Max-pool padding: every pooled tensor is a ReLU output (>= 0), so padding with 0 equals PyTorch's -inf padding exactly,
# and avoids the -inf the compiled backward turned into non-finite gradients (29/10, LESSONS).
def _pool3(x):                                                        # MaxPool1d(3, 1, 1) on L
    xp = F.pad(x, (0, 0, 1, 1))
    L = x.shape[2]
    return torch.maximum(torch.maximum(xp[:, :, :L], xp[:, :, 1:L + 1]), xp[:, :, 2:L + 2])


def _pool2(x):                                                        # MaxPool1d(2, ceil_mode=True) on L
    G, b, L, C = x.shape
    if L % 2:
        x = F.pad(x, (0, 0, 0, 1))
    return torch.maximum(x[:, :, 0::2], x[:, :, 1::2])               # not .amax: its fp16 compiled backward gave NaN


def _combined(c, cfg, x):
    G, b, _ = x.shape
    pl = cfg["patch_len"]
    fm = x.view(G, b, cfg["n_features"] // pl, pl)                     # channels-last of view(B,11,7).transpose(1,2)
    outs = []
    # DenseNet
    h = _cbr(c, "stems.0", fm, 1)
    for i in range(cfg["dense_layers"]):
        h = torch.cat([h, _cbr(c, f"dense.blocks.{i}", h, 3)], dim=-1)
    outs.append(h)
    # GoogleNet
    h = _cbr(c, "stems.1", fm, 1)
    for i in range(cfg["incep_modules"]):
        p = f"google.net.{i}"
        h = torch.cat([_cbr(c, p + ".b1", h, 1),
                       _cbr(c, p + ".b3.1", _cbr(c, p + ".b3.0", h, 1), 3),
                       _cbr(c, p + ".b5.1", _cbr(c, p + ".b5.0", h, 1), 5),
                       _cbr(c, p + ".bp.1", _pool3(h), 1)], dim=-1)
    outs.append(h)
    # AlexNet
    h = _cbr(c, "stems.2", fm, 1)
    h = _pool2(_cbr(c, "alex.net.0", h, 3))
    h = _pool2(_cbr(c, "alex.net.2", h, 3))
    outs.append(_cbr(c, "alex.net.4", h, 3))
    # SqueezeNet
    h = _cbr(c, "stems.3", fm, 1)
    for i in range(cfg["fire_modules"]):
        p = f"squeeze.net.{i}"
        s = _cbr(c, p + ".squeeze", h, 1)
        h = torch.cat([_cbr(c, p + ".e1", s, 1), _cbr(c, p + ".e3", s, 3)], dim=-1)
    outs.append(h)
    return torch.cat([_f32(o).mean(2) for o in outs], dim=-1)        # (G, b, comb), Eq. (47)


def _linear(c, name, x):
    w, b = c.P[name + ".weight"], c.P[name + ".bias"]
    return torch.baddbmm(b[:, None, :].to(x.dtype), x, w.transpose(1, 2).to(x.dtype))


def sae_input(x):
    return torch.cat(((x[..., :71].float() + 3.72) / 7.44, x[..., 71:].float()), dim=-1)


def sae_decode(P, code):
    c = _Ctx(P, {}, None, False)
    h = torch.relu(_linear(c, "decoder.0", code))
    return torch.sigmoid(_linear(c, "decoder.2", h))


def sae_regularizer(P, x, code, mask, cfg):
    """Per-client reconstruction + KL sparsity terms; classification is supplied by each FL method."""
    if cfg.get("objective") == "ce":
        # Decoder remains in the legacy parameter layout; explicitly zero its gradient.
        zero = sum(v.float().sum(tuple(range(1, v.ndim))) * 0
                   for k, v in P.items() if k.startswith("decoder."))
        return zero
    target = sae_input(x)
    reconstruction = sae_decode(P, code)
    rec = ((reconstruction.float() - target) ** 2).sum(-1)
    rec = torch.where(mask, rec, 0.0).sum(1) / mask.sum(1).clamp_min(1)
    mean = torch.where(mask[..., None], code.float(), 0.0).sum(1) / mask.sum(1).clamp_min(1)[:, None]
    mean = mean.clamp(1e-6, 1 - 1e-6)
    rho = cfg.get("sparse_rho", 0.05)
    kl = rho * (math.log(rho) - mean.log()) + (1 - rho) * (math.log1p(-rho) - torch.log1p(-mean))
    return cfg.get("reconstruction_weight", 1.0) * rec + cfg.get("sparsity_weight", 1.2) * kl.sum(1)


def forward(P, B, x, cfg, mask=None, train=False, hidden=False):
    """x (G, b, 77) -> logits (G, b, 16) [, hidden (G, b, head_hidden)], new BN buffers (train only, else {}).

    mask (G, b) bool marks real rows; required when train=True. Dropout is active only when train=True.
    """
    if cfg.get("model") in ("bert", "mixed-adapter"):
        from core.mixed_backbones import bert_forward, forward as mixed_forward
        fn = bert_forward if cfg["model"] == "bert" else mixed_forward
        return fn(P, B, x, cfg, mask, train, hidden)
    if cfg.get("model") == "cnn-bigru-attention":
        from core.grouped_bigru import forward as bigru_forward
        return bigru_forward(P, B, x, cfg, mask, train, hidden)
    if cfg.get("model") == "mlp":
        c = _Ctx(P, B, mask, train)
        h = torch.relu(_linear(c, "features.0", x))
        h = F.dropout(h, cfg["dropout"], train)
        h = torch.relu(_linear(c, "features.3", h))
        logits = _f32(_linear(c, "head", F.dropout(h, cfg["dropout"], train)))
        return (logits, _f32(h), {}) if hidden else (logits, {})
    if cfg.get("model") == "tabm":
        h = x[:, :, None, :] * P["blocks.0.r"][:, None]
        h = torch.einsum("gbki,ghi->gbkh", h, P["blocks.0.shared.weight"].to(h.dtype))
        h = torch.relu(h * P["blocks.0.s"][:, None].to(h.dtype) + P["blocks.0.bias"][:, None].to(h.dtype))
        h = F.dropout(h, cfg["dropout"], train)
        h = h * P["blocks.1.r"][:, None].to(h.dtype)
        h = torch.einsum("gbki,ghi->gbkh", h, P["blocks.1.shared.weight"].to(h.dtype))
        h = torch.relu(h * P["blocks.1.s"][:, None].to(h.dtype) + P["blocks.1.bias"][:, None].to(h.dtype))
        h = F.dropout(h, cfg["dropout"], train)
        logits = torch.einsum("gbki,gkio->gbko", h, P["weight"].to(h.dtype)) + P["bias"][:, None].to(h.dtype)
        representation = _f32(h) if cfg.get("member_hidden") else _f32(h).mean(2)
        return (_f32(logits), representation, {}) if hidden else (_f32(logits), {})
    if cfg.get("model") == "sae":
        c = _Ctx(P, B, mask, train)
        u = sae_input(x)
        h = torch.relu(_linear(c, "encoder.0", u))
        code = torch.sigmoid(_linear(c, "encoder.2", h))
        logits = _f32(_linear(c, "head", torch.cat((u, code.float()), dim=-1)))
        representation = torch.cat((u, code.float()), dim=-1) if cfg.get("objective") == "ce" else _f32(code)
        return (logits, representation, {}) if hidden else (logits, {})
    c = _Ctx(P, B, mask, train)
    z = _combined(c, cfg, x)
    z = F.layer_norm(z, (z.shape[-1],)) * P["head.0.weight"][:, None] + P["head.0.bias"][:, None]
    z = F.dropout(z, cfg["dropout"], train)
    h = torch.relu(_linear(c, "head.2", z.to(x.dtype)))
    logits = _f32(_linear(c, "head.5", F.dropout(h, cfg["dropout"], train)))
    return (logits, _f32(h), c.newB) if hidden else (logits, c.newB)


# Entry points the train loop / evaluator call; scripts swap in torch.compile'd versions (static shapes for train,
# dynamic for eval). Plain `forward` stays the reference for tests.
TRAIN = forward
EVAL = forward


def compile_all():
    global TRAIN, EVAL
    from core import optim
    torch._dynamo.config.recompile_limit = 64                          # methods legitimately use several shapes
    TRAIN = torch.compile(forward, dynamic=False)
    EVAL = torch.compile(forward, dynamic=True)
    optim.compile_all()
