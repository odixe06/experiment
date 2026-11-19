"""Mixed-model adapters: native backbone -> learned 128D projection -> classifier.

Native output layers are retained with explicit zero gradient for compatibility with
the audited grouped extractors. Count them separately from prediction-active params.
TabM keeps four member outputs/CEs; prototypes average member representations.
"""
import torch
from torch import nn
from torch.nn import functional as F

from core.central_backbones import FeatureBERT
from core.dagsnet import DAGSNet, width
from core.tabm import TabM, config as tabm_config
from core.sae import SupervisedSAE, fl_config as sae_config
from core.mlp import MLP, config as mlp_config
from core.cnn_bigru_attention import CNNBiGRUAttention, config as bigru_config


def native_config(name):
    if name == "dagsnet":
        return width(1)
    if name == "bert":
        return dict(model="bert", n_features=77, n_classes=16, head_hidden=64,
                    dropout=0.1, patch_len=7)
    return {"tabm": tabm_config, "sae": sae_config, "mlp": mlp_config,
            "cnn-bigru-attention": bigru_config}[name]()


def config(name, projected=True):
    base = native_config(name)
    if not projected:
        return base
    return dict(model="mixed-adapter", backbone=name, base=base, n_features=77,
                n_classes=16, head_hidden=128, dropout=base["dropout"], patch_len=7)


def make_native(cfg):
    if cfg.get("model") == "bert":
        model = FeatureBERT()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = cfg["dropout"]
            if isinstance(module, nn.MultiheadAttention):
                module.dropout = cfg["dropout"]
        return model
    return {"tabm": TabM, "sae": SupervisedSAE, "mlp": MLP,
            "cnn-bigru-attention": CNNBiGRUAttention}.get(cfg.get("model"), DAGSNet)(cfg)


class MixedAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.base = make_native(cfg["base"])
        d = cfg["base"]["head_hidden"]
        self.projection = nn.Linear(d, 128)
        if cfg["backbone"] == "tabm":
            self.member_weight = nn.Parameter(torch.empty(4, 128, 16))
            self.member_bias = nn.Parameter(torch.empty(4, 16))
            nn.init.uniform_(self.member_weight, -128**-0.5, 128**-0.5)
            nn.init.uniform_(self.member_bias, -128**-0.5, 128**-0.5)
        else:
            self.classifier = nn.Linear(128, 16)

    def forward(self, x, hidden=False):
        b, name = self.base, self.cfg["backbone"]
        if name == "dagsnet":
            h = b.hidden(x)
        elif name == "bert":
            t = x.unsqueeze(-1) * b.scale + b.identity
            h = b.blocks(torch.cat((b.cls.expand(len(x), -1, -1), t), 1))[:, 0]
        elif name == "tabm":
            h = b.blocks(x.unsqueeze(1))
        elif name == "sae":
            u = b.unit_input(x)
            h = torch.cat((u, b.encoder(u)), -1)
        elif name == "mlp":
            h = b.features(x)
        else:
            states, _ = b.gru(b.conv(x.unsqueeze(1)).transpose(1, 2))
            weights = b.score(torch.tanh(b.attention(states))).float().softmax(1)
            h = (weights * states.float()).sum(1)
        h = self.projection(h)
        q = h if name in ("tabm", "sae", "bert") else F.dropout(h, self.cfg["dropout"], self.training)
        logits = (torch.einsum("bki,kio->bko", q, self.member_weight) + self.member_bias
                  if name == "tabm" else self.classifier(q))
        # All stored native parameters have defined gradients, including unused heads/SAE decoder.
        logits = logits + sum(p.sum() * 0 for p in self.base.parameters())
        return (logits, h.mean(1) if name == "tabm" else h) if hidden else logits


def bert_forward(P, B, x, cfg, mask=None, train=False, hidden=False):
    from core.grouped import _Ctx, _linear, _f32
    c = _Ctx(P, B, mask, train)
    G, batch, _ = x.shape
    h = x[..., None] * P["scale"][:, None] + P["identity"][:, None]
    h = torch.cat((P["cls"].expand(G, batch, 1, 64), h), 2)
    p = cfg["dropout"]
    for i in range(2):
        prefix = f"blocks.{i}"
        t = h.reshape(G, batch * 78, 64)
        w = P[prefix + ".self_attn.in_proj_weight"]
        bias = P[prefix + ".self_attn.in_proj_bias"]
        qkv = torch.baddbmm(bias[:, None].to(t.dtype), t, w.transpose(1, 2).to(t.dtype))
        q, k, v = qkv.reshape(G, batch, 78, 3, 4, 16).permute(3, 0, 1, 4, 2, 5).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=p if train else 0.0)
        a = a.transpose(-3, -2).reshape(G, batch * 78, 64)
        a = _linear(c, prefix + ".self_attn.out_proj", a).reshape(G, batch, 78, 64)
        def norm(tensor, suffix):
            key = prefix + suffix
            return F.layer_norm(_f32(tensor), (64,), eps=1e-5) * P[key + ".weight"][:, None, None] + P[key + ".bias"][:, None, None]
        h = norm(h + F.dropout(a, p, train), ".norm1")
        z = F.gelu(_linear(c, prefix + ".linear1", h.reshape(G, batch * 78, 64)))
        z = _linear(c, prefix + ".linear2", F.dropout(z, p, train)).reshape(G, batch, 78, 64)
        h = norm(h + F.dropout(z, p, train), ".norm2")
    h = h[:, :, 0]
    logits = _f32(_linear(c, "head", h))
    return (logits, _f32(h), {}) if hidden else (logits, {})


def forward(P, B, x, cfg, mask=None, train=False, hidden=False):
    from core import grouped
    base_p = {k[5:]: v for k, v in P.items() if k.startswith("base.")}
    base_b = {k[5:]: v for k, v in B.items() if k.startswith("base.")}
    base_cfg = dict(cfg["base"], dropout=cfg["dropout"], member_hidden=True)
    old, h, nb = grouped.forward(base_p, base_b, x, base_cfg, mask, train, True)
    c = grouped._Ctx(P, B, mask, train)
    if cfg["backbone"] == "tabm":
        G, batch, members, d = h.shape
        h = grouped._linear(c, "projection", h.reshape(G, batch * members, d)).reshape(G, batch, members, 128)
        logits = torch.einsum("gbki,gkio->gbko", h, P["member_weight"].to(h.dtype)) + P["member_bias"][:, None].to(h.dtype)
        representation = h.float().mean(2)
    else:
        h = grouped._linear(c, "projection", h)
        q = h if cfg["backbone"] in ("sae", "bert") else F.dropout(h, cfg["dropout"], train)
        logits = grouped._linear(c, "classifier", q)
        representation = grouped._f32(h)
    zero = sum(v.sum(tuple(range(1, v.ndim))) * 0 for v in base_p.values())
    logits = grouped._f32(logits) + zero.reshape(-1, *([1] * (logits.ndim - 1)))
    nb = {"base." + k: v for k, v in nb.items()}
    return (logits, representation, nb) if hidden else (logits, nb)
