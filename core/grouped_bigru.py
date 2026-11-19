"""Grouped native-PyTorch GRU equations, including reset-after recurrent candidate."""
import torch
import torch.nn.functional as F


def forward(P, B, x, cfg, mask, train, hidden):
    from core.grouped import _Ctx, _conv, _bn, _linear, _f32
    c = _Ctx(P, B, mask, train)
    z = _conv(c, "conv.0", x[..., None], 3)[:, :, 1:-1]
    z = torch.relu(_bn(c, "conv.1", z + P["conv.0.bias"][:, None, None]))
    length = z.shape[2] // 2 * 2
    z = torch.maximum(z[:, :, :length:2], z[:, :, 1:length:2])
    z = torch.relu(_conv(c, "conv.4", z, 3)[:, :, 1:-1] + P["conv.4.bias"][:, None, None])
    G, batch, length, channels = z.shape
    outputs = []
    for reverse in (False, True):
        suffix = "_reverse" if reverse else ""
        def affine(value, kind):
            w = P[f"gru.weight_{kind}_l0{suffix}"]
            bias = P[f"gru.bias_{kind}_l0{suffix}"]
            return torch.baddbmm(bias[:, None].to(value.dtype), value, w.transpose(1, 2).to(value.dtype))
        inputs = affine(z.reshape(G, batch*length, channels), "ih").reshape(G, batch, length, -1)
        state = z.new_zeros(G, batch, cfg["gru_hidden"])
        states = []
        order = range(length-1, -1, -1) if reverse else range(length)
        for t in order:
            ir, iz, inn = inputs[:, :, t].chunk(3, -1)
            hr, hz, hn = affine(state, "hh").chunk(3, -1)
            reset, update = torch.sigmoid(ir+hr), torch.sigmoid(iz+hz)
            candidate = torch.tanh(inn + reset*hn)
            state = (1-update)*candidate + update*state
            states.append(state)
        if reverse:
            states.reverse()
        outputs.append(torch.stack(states, 2))
    states = torch.cat(outputs, -1)
    a = torch.tanh(_linear(c, "attention", states.reshape(G, batch*length, -1)))
    scores = torch.bmm(a, P["score.weight"].transpose(1, 2).to(a.dtype)).reshape(G, batch, length, 1)
    context = (scores.float().softmax(2)*states.float()).sum(2)
    logits = _f32(_linear(c, "head", F.dropout(context, cfg["dropout"], train)))
    return (logits, context, c.newB) if hidden else (logits, c.newB)
