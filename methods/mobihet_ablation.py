"""MobiHet ablations (owner 24/11/2026, docs/research/mobihet-ablation.md). Each arm starts from full MobiHet = L4 without
logit adjustment (hp la_mode=off, scripts/build_l4_fl.py MOBIHET_HP); hp `ablation`:

  none  MobiHet unchanged (control: must reproduce the MobiHet run of the same seed).
  a1    w/o synthetic data (C4 + C6). Each vehicle trains plain CE(z, y) on every real row, batch 64, no step floor
        (hp synth_step_floor=0). With no synthetic row nothing reads the copula/KM generator (C1), the class-Q teacher (C2)
        or the stored row CE (C5), and without LA nothing reads the class counts, so nothing is sent: bootstrap 0 bytes and
        every round 0 bytes (local training only).
  a2    w/o loss-aware selection (C5): hp real_select=random only (same real/synthetic budget, losses and schedule; no code).
  a4    w/o Q compression: class-Q packets raw fp32, dense 16 x 16 both ways (4 B per value, 1024 B up and 1024 B down per
        vehicle per round). The vehicle's per-class mean is not quantised (no uint8, no fp16 cast, no bitmask, no zlib), the
        server's Q is not cast either; rows are renormalised as both L4 decoders do. Count-weighted aggregation, KD
        schedule and bootstrap codec unchanged.

The former a3 (w/o logit adjustment) is MobiHet itself since 24/11. The L4-based ablations (LA on) are archived in
../mobihet/archive/l4-la-20261124/.

Works in this repo (methods.mobihet_l4: DAGSNet, TabM, SAE, MLP, CNN-BiGRU, mixed adapters) and in the frozen FeatureBERT
harness (no mobihet_l4 there: its L4 ran methods.mobihet_budget, which mobihet_l4 equals off TabM/SAE; the notebook
overlays ../mobihet's mobihet_budget.py with la_mode).
"""
import torch

from core import grouped, local

try:
    from methods.mobihet_l4 import Method as L4
except ModuleNotFoundError:                                              # FeatureBERT harness (fl_veremi_bert 49b2d88)
    from methods.mobihet_budget import Method as L4

ARMS = ("none", "a1", "a2", "a4")
HP = {"none": ["ablation=none"], "a1": ["ablation=a1", "synth_step_floor=0"],          # added after MOBIHET_HP + seed
      "a2": ["ablation=a2", "real_select=random"], "a4": ["ablation=a4"]}
RAW_Q_BYTES = 16 * 16 * 4


class Ablation:
    """Mixin placed before an L4 class (also the mixed GroupL4)."""

    def __init__(self, env):
        hp = env.hp
        self.ablation = hp.get("ablation", "none")
        assert self.ablation in ARMS, self.ablation
        assert (hp.get("real_select") == "random") == (self.ablation == "a2"), "a2 is exactly real_select=random"
        assert (hp.get("synth_step_floor") == 0) == (self.ablation == "a1"), "a1 is exactly synth_step_floor=0"
        assert hp.get("la_mode") == "off", "ablations start from MobiHet (la_mode=off)"
        super().__init__(env)
        if self.ablation == "a1":
            self.select = "random"                                       # no stored row CE: nothing ranks real rows
            for k in ("_score", "_sgen"):
                self.__dict__.pop(k, None)
        elif self.ablation == "a4":
            assert self.q_bits == 8 and self.q_codec == "mean_fp16" and self.q_agg == "count" and self.kd_bins == 1

    def _bootstrap(self):
        if self.ablation != "a1":
            return super()._bootstrap()
        env, C = self.env, 16
        cnt = torch.zeros(env.K, C, device=env.dev)
        client = torch.repeat_interleave(torch.arange(env.K, device=env.dev), env.train.off[1:] - env.train.off[:-1])
        cnt.index_put_((client, env.train.Y.long()), torch.ones(len(client), device=env.dev), accumulate=True)
        self.cnt = cnt                                                   # local only: no LA, so no global count is needed
        self.la_shift = torch.zeros(env.K, C, device=env.dev)
        self.labels = torch.arange(self.aux, device=env.dev) % C
        self.bootstrap_bytes = {"up": 0, "down": 0}

    def _loss_real(self, ctx):
        """a1: CE(z + la_shift) over the step's real rows (TabM: per member, averaged), as MobiHet's real-row term
        (la_shift = 0 under la_mode=off, so plain CE)."""
        cfg = self.env.cfg
        z, new_b = grouped.TRAIN(ctx.P, ctx.B, ctx.x, cfg, mask=ctx.mask, train=True)
        shift = self._shift[:, None, None, :] if z.dim() == 4 else self._shift[:, None, :]
        loss = local.ce_loss(z + shift, ctx.y, ctx.mask)
        if cfg.get("model") == "sae":                                    # as mobihet_l4: explicit zero gradient, decoder
            loss = loss + grouped.sae_regularizer(ctx.P, ctx.x, None, ctx.mask, cfg)
        return loss, new_b

    def _round_private(self, r, lr):
        if self.ablation == "a1":
            return self._round_real(r, lr)
        if self.ablation != "a4":
            return super()._round_private(r, lr)
        # Raw fp32: mobihet_sim's 8-bit codec delegates at q_bits=16; mobihet.Method's q_codec "legacy" branch sums the
        # vehicles' fp32 class sums (= n_kc x fp32 mean after mobihet_budget's rescale) without the fp16 packet or Q cast.
        self.q_bits, self.q_codec = 16, "legacy"
        try:
            info = super()._round_private(r, lr)
        finally:
            self.q_bits, self.q_codec = 8, "mean_fp16"
        if self.Q is not None:      # row renormalisation as both L4 decoders (a class untrained that round adds 0 mass)
            self.Q = self.Q / self.Q.sum(-1, keepdim=True).clamp_min(1e-12)
        info.update(bytes_up=float(RAW_Q_BYTES), bytes_down=RAW_Q_BYTES)
        return info

    def _round_real(self, r, lr):
        env, lay, dev = self.env, self.env.lay, self.env.dev
        loss = torch.zeros(env.K, device=dev)
        steps = torch.zeros(2, device=dev)
        missing = torch.zeros((), device=dev)
        for idx in env.groups_idx():
            self._shift = local.pad_rows(self.la_shift[idx], local.group_size(env.train.off, idx, env.hp))
            flat, bufs = self.priv_p[idx], self.priv_b[idx]
            st = local.local_epoch(lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen, loss_fn=self._loss_real)
            self.priv_p[idx], self.priv_b[idx] = flat, bufs
            loss[idx] = st["loss"]
            steps += torch.stack([st["applied"].sum(), st["skipped"].sum()])
            missing += (st["applied"] <= 0).sum()
        N = env.allreduce(env.n.sum().clone())
        steps = env.allreduce(steps)
        missing = int(env.allreduce(missing))
        assert missing == 0, "Client received no optimizer update"
        self.version = r
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / N),
                "skipped_frac": float(steps[1] / steps.sum().clamp_min(1)), "clients_without_applied": missing,
                "real_frac": 1.0, "synth_rows": 0, "bytes_up": 0.0, "bytes_down": 0}


class Method(Ablation, L4):
    pass
