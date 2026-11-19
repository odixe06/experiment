"""Private-only MobiHet: one-time pooled class statistics, no trainable proxy or per-round payload."""
import torch
import torch.nn.functional as F

from core import grouped, local
from methods.mobihet import Method as MobiHet


class Method(MobiHet):
    name = "mobihet_stats"

    def __init__(self, env):
        self.env = env
        p0, b0 = env.lay.init(env.hp["seed"], 1, env.dev)
        self.priv_p = p0.repeat(env.K, 1)
        self.priv_b = b0.repeat(env.K, 1)
        self.version = 0
        self.aux = env.hp["batch"] // 2
        self.gen_g = torch.Generator(device=env.dev).manual_seed(env.hp["seed"] * 11 + env.rank)
        self._bootstrap()

    def payload_bytes(self):
        return {"bootstrap": self.bootstrap_bytes, "per_round": {"up": 0, "down": 0}}

    def _loss(self, ctx):
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        active = ctx.mask.any(1)
        z = torch.randn(G, a, self.mu.shape[1], device=ctx.x.device, generator=self.gen_g)
        xg = self._gaussian_rows(z).to(ctx.x.dtype)
        x = torch.cat([ctx.x, xg], 1)
        mask = self._normalization_mask(ctx.mask, active, G, a)
        logits, new_b = grouped.TRAIN(ctx.P, ctx.B, x, self.env.cfg, mask=mask, train=True)
        shift = self._shift[:, None, :]
        real = local.masked_mean(
            F.cross_entropy((logits[:, :b] + shift).reshape(-1, 16), ctx.y.reshape(-1), reduction="none").view(G, b),
            ctx.mask,
        )
        synthetic = F.cross_entropy(
            self._synthetic_logits(logits[:, b:]).reshape(-1, 16), self.labels.repeat(G), reduction="none"
        ).view(G, a)
        synthetic = (synthetic * self._synthetic_weights(ctx)).mean(1)
        return real + synthetic, new_b

    def _normalization_mask(self, real_mask, active, groups, auxiliary):
        return torch.cat([real_mask, active[:, None].expand(groups, auxiliary)], 1)

    def _gaussian_rows(self, z):
        return (self.mu[self.labels] + (self.L[self.labels] @ z[..., None]).squeeze(-1)).clamp(-4, 4)

    def _synthetic_weights(self, ctx):
        return 1.0

    def _synthetic_logits(self, logits):
        return logits

    def round(self, r, lr):
        env = self.env
        loss = torch.zeros(env.K, device=env.dev)
        counts = torch.zeros(3, device=env.dev)
        for gi, idx in enumerate(env.groups_idx()):
            flat = self.priv_p[idx].clone()
            bufs = self.priv_b[idx].clone()
            self._shift = local.pad_rows(self.la_shift[idx], local.group_size(env.train.off, idx, env.hp))
            state = local.local_epoch(env.lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen,
                                      loss_fn=self._loss)
            counts += torch.stack([state["applied"].sum(), state["skipped"].sum(),
                                   (state["applied"] == 0).sum()])
            if not torch.isfinite(flat).all() or not torch.isfinite(bufs).all():
                raise RuntimeError(f"non-finite MobiHet-stats update, rank={env.rank}, group={gi}")
            self.priv_p[idx], self.priv_b[idx] = flat, bufs
            loss[idx] = state["loss"]
        self.version = r
        counts = env.allreduce(counts)
        if counts[2] > 0 or counts[1] > counts[0]:
            raise RuntimeError(f"MobiHet-stats invalid training: applied/skipped/idle={counts.tolist()}")
        total = env.allreduce(env.n.sum().clone())
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / total),
                "applied_steps": float(counts[0]), "skipped_steps": float(counts[1]),
                "skipped_frac": float(counts[1] / counts[:2].sum().clamp_min(1)),
                "bytes_up": 0, "bytes_down": 0}

    def evaluate(self):
        return {"private": self.env.eval_clients(self.priv_p, self.priv_b),
                "private_local": self.env.eval_clients(self.priv_p, self.priv_b, logit_bias=self.la_shift)}

    def state(self):
        return {"priv_p": self.priv_p.cpu(), "priv_b": self.priv_b.cpu(), "version": self.version,
                "gen_g": self.gen_g.get_state()}

    def load_state(self, state):
        self.priv_p = state["priv_p"].to(self.env.dev)
        self.priv_b = state["priv_b"].to(self.env.dev)
        self.version = state["version"]
        self.gen_g.set_state(state["gen_g"])
