"""Dependency-group L1 pruning; gate rows and both GRU directions stay aligned."""
import torch
from core.flat import Layout


def prune(cfg, sd, ratio):
    specs = {}
    def add(key, axis, group, copies=1):
        specs.setdefault(key, []).append((axis, group, copies))
    if cfg["model"] == "mlp":
        sizes = {"first_hidden": cfg["first_hidden"], "head_hidden": cfg["head_hidden"]}
        for layer, group, consumer in (("features.0", "first_hidden", "features.3"),
                                       ("features.3", "head_hidden", "head")):
            add(layer+".weight", 0, group); add(layer+".bias", 0, group)
            add(consumer+".weight", 1, group)
    else:
        sizes = {k: cfg[k] for k in ("conv1", "conv2", "gru_hidden", "attention_hidden")}
        for layer, group in (("conv.0", "conv1"), ("conv.4", "conv2")):
            add(layer+".weight", 0, group); add(layer+".bias", 0, group)
        for key in ("weight", "bias", "running_mean", "running_var"):
            add("conv.1."+key, 0, "conv1")
        add("conv.4.weight", 1, "conv1")
        for suffix in ("", "_reverse"):
            for kind in ("ih", "hh"):
                add(f"gru.weight_{kind}_l0{suffix}", 0, "gru_hidden", 3)
                add(f"gru.bias_{kind}_l0{suffix}", 0, "gru_hidden", 3)
                add(f"gru.weight_{kind}_l0{suffix}", 1, "conv2" if kind == "ih" else "gru_hidden")
        add("attention.weight", 1, "gru_hidden", 2)
        add("head.weight", 1, "gru_hidden", 2)
        add("attention.weight", 0, "attention_hidden")
        add("attention.bias", 0, "attention_hidden")
        add("score.weight", 1, "attention_hidden")
    scores = {k: torch.zeros(n) for k, n in sizes.items()}
    for key, axes in specs.items():
        if key.endswith(("running_mean", "running_var")):
            continue
        for axis, group, copies in axes:
            value = sd[key].abs().movedim(axis, 0)
            value = value.reshape(value.shape[0], -1).sum(1)
            scores[group] += value.reshape(copies, sizes[group]).sum(0)
    keep = {k: scores[k].topk(n-int(ratio*n)).indices.sort().values for k, n in sizes.items()}
    out = {}
    for key, value in sd.items():
        for axis, group, copies in specs.get(key, []):
            indices = torch.cat([keep[group]+i*sizes[group] for i in range(copies)])
            value = value.index_select(axis, indices)
        out[key] = value.clone()
    new = dict(cfg, **{k: len(v) for k, v in keep.items()})
    if cfg["model"] == "cnn-bigru-attention":
        new["head_hidden"] = 2*new["gru_hidden"]
    Layout(new).make(new).load_state_dict(out, strict=True)
    return new, out
