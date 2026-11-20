"""Five existing baseline mechanisms with heterogeneous private model banks.

All backbone groups train against the SAME previous-round server state; aggregate
once after all groups, across both GPU ranks. Native single-model code is untouched.
"""
import math
import time
from pathlib import Path
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from core import cost_scope
from core import evaluate, grouped, local
from core.flat import Layout
from core.mixed_backbones import MixedAdapter, config
from methods import fedtgp, pfedes, v2x_htfl as v2x

METHODS = ("fedproto", "fedtgp", "tinyproto", "v2x_htfl", "pfedes")


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


class Method:
    def __init__(self, env, name, assignment, model_order):
        assert name in METHODS
        self.env, self.name, self.d = env, name, 128
        hp, dev = env.hp, env.dev
        self.groups = []
        for model in model_order:
            ids = [i for i, cid in enumerate(env.ids) if assignment[int(cid)] == model]
            cfg = config(model, projected=name != "pfedes")
            make = v2x.projected_model(MixedAdapter) if name == "v2x_htfl" else None
            lay = Layout(cfg, make=make)
            p, b = lay.init(hp["seed"], len(ids), dev)
            # BERT attention activation budget; batch64 and client epochs stay unchanged.
            ghp = dict(hp, group_rows=min(hp["group_rows"], 256) if model == "bert" else hp["group_rows"])
            # Scaled fp16 gradients overflow in SAE's encoder and BERT attention
            # biases. Keep the same objective in fp32 for these V2X groups.
            if name == "v2x_htfl" and model in ("sae", "bert"):
                ghp["amp"] = False
            indices = torch.tensor(ids, dtype=torch.long, device=dev)
            sizes = env.train.sizes[ids]
            schedule = local.group_schedule(sizes, list(range(len(ids))), hp["batch"], ghp["group_rows"])
            self.groups.append(SimpleNamespace(name=model, cfg=cfg, lay=lay, p=p, b=b, indices=indices,
                                               n=env.n[indices], hp=ghp, schedule=schedule))
        self.proto = None
        self.has = torch.zeros(16, dtype=torch.bool, device=dev)
        self.n_total = env.allreduce(env.n.sum().clone())
        self.lam = 1.0 if name in ("fedproto", "tinyproto") else 0.1
        self.mu, self.delta = 0.1, 0.0
        self.gen_s = torch.Generator(device=dev).manual_seed(hp["seed"] + 9)
        if name == "tinyproto":
            self.mu, self.s = 0.00015, math.ceil(self.d / 10)
            perm = torch.randperm(self.d, generator=torch.Generator().manual_seed(hp["seed"] + 13))
            self.mask = torch.zeros(16, self.d, dtype=torch.bool, device=dev)
            for j in range(16):
                self.mask[j, perm[(j*self.s + torch.arange(self.s)) % self.d].to(dev)] = True
        elif name in ("fedtgp", "v2x_htfl"):
            self.S = hp.get("server_epochs", 100)
            torch.manual_seed(hp["seed"] + 5)
            if name == "fedtgp":
                self.tau = 100.0
                self.tgp = fedtgp.TGP(16, self.d).to(dev)
                self.opt = torch.optim.SGD(self.tgp.parameters(), lr=0.01)
            else:
                self.s, self.m, self.gamma, self.tau = 64.0, 0.5, 0.99, 0.5
                self.V = v2x.etf(self.d, hp["seed"] + 11, dev)
                self.gp = v2x.GlobalProto(self.d).to(dev)
                self.opt = torch.optim.Adam(self.gp.parameters(), lr=0.01)
        elif name == "pfedes":
            self.proxy_cfg = dict(n_features=77, patch_len=7)
            self.glay = Layout(self.proxy_cfg, make=pfedes.ProxyExtractor)
            self.proxy, _ = self.glay.init(hp["seed"] + 1, 1, dev)
            self.proxy = self.proxy[0]

    def payload_bytes(self):
        if self.name == "pfedes":
            up = down = 4 * self.glay.P
        elif self.name == "tinyproto":
            up = down = 4 * 16 * self.s
        else:
            up = 4 * 16 * self.d + (64 if self.name in ("fedproto", "v2x_htfl") else 0)
            down = 4 * 16 * self.d
        return dict(up=up, down=down, masks_once=256 if self.name == "tinyproto" else 0,
                    accounting="per-client worst-case 16 present classes; fp32")

    def _loss(self, ctx, group, r, lam):
        z, h, nb = grouped.TRAIN(ctx.P, ctx.B, ctx.x, group.cfg, mask=ctx.mask, train=True, hidden=True)
        if self.name == "v2x_htfl":
            gn = v2x.cosine_normalize(v2x.project(ctx.P, h))
            cos = gn @ self.V
            cy = cos.gather(-1, ctx.y[..., None]).squeeze(-1)
            sy = (1-cy*cy).clamp_min(0).sqrt()
            th, mm = math.cos(math.pi-self.m), math.sin(math.pi-self.m)*self.m
            phi = torch.where(cy > th, cy*math.cos(self.m)-sy*math.sin(self.m), cy-mm)
            logits = (self.s*self.gamma**r*cos).scatter(-1, ctx.y[..., None], self.s*phi[..., None])
            per = F.cross_entropy(logits.flatten(0, 1), ctx.y.flatten(), reduction="none").view_as(ctx.y)
            if self.proto is not None:
                pn = F.normalize(self.proto, dim=-1)[ctx.y]
                ls = torch.where(self.has[ctx.y], 1-(gn*pn).sum(-1), 0.)
                per = per + self.mu*ctx.pad(lam)[:, None]*ls
            return local.masked_mean(per, ctx.mask) + z.sum(tuple(range(1, z.ndim)))*0, nb
        loss = local.ce_loss(z, ctx.y, ctx.mask)
        if self.proto is not None:
            diff = ((h-self.proto[ctx.y])**2).mean(-1)
            valid = ctx.mask & self.has[ctx.y] if self.name == "fedproto" else ctx.mask
            loss = loss + self.lam * torch.where(valid, diff, 0.).sum(1)/ctx.mask.sum(1).clamp_min(1)
        return loss, nb

    def round(self, r, lr):
        env, dev = self.env, self.env.dev
        S = torch.zeros(env.K, 16, self.d, device=dev)
        Cn = torch.zeros(env.K, 16, device=dev)
        loss_sum = torch.zeros((), device=dev)
        group_health = {}
        if self.name == "pfedes":
            gacc = torch.zeros_like(self.proxy)
        for group in self.groups:
            t_group = time.monotonic()
            step_counts = torch.zeros(2, device=dev, dtype=torch.float64)
            for ix in group.schedule:
                idx = torch.tensor(ix, device=dev)
                data_ids = group.indices[idx]
                Gf = local.group_size(env.train.off, data_ids, group.hp)
                if self.name == "fedproto":
                    acc = torch.zeros(Gf, 16, self.d, device=dev)
                    cnt = torch.zeros(Gf, 16, device=dev)
                if self.name == "pfedes":
                    Pg = self.glay.params(self.proxy[None].repeat(Gf, 1))
                    def loss_fn(ctx):
                        with torch.no_grad():
                            xh = pfedes.g_forward(Pg, ctx.x, self.proxy_cfg)
                        xx, mm = torch.cat((xh, ctx.x), 1), torch.cat((ctx.mask, ctx.mask), 1)
                        z, h, nb = grouped.TRAIN(ctx.P, ctx.B, xx, group.cfg, mask=mm, train=True, hidden=True)
                        b = ctx.x.shape[1]
                        l1 = local.supervised_loss(ctx.P, xh, z[:, :b], h[:, :b], ctx.y, ctx.mask, group.cfg)
                        l2 = local.supervised_loss(ctx.P, ctx.x, z[:, b:], h[:, b:], ctx.y, ctx.mask, group.cfg)
                        return 0.5*(l1+l2), nb
                elif self.name == "fedproto":
                    def loss_fn(ctx):
                        z, h, nb = grouped.TRAIN(ctx.P, ctx.B, ctx.x, group.cfg, mask=ctx.mask, train=True, hidden=True)
                        loss = local.ce_loss(z, ctx.y, ctx.mask)
                        if self.proto is not None:
                            diff = ((h-self.proto[ctx.y])**2).mean(-1)
                            loss += torch.where(ctx.mask & self.has[ctx.y], diff, 0.).sum(1)/ctx.mask.sum(1).clamp_min(1)
                        with torch.no_grad():
                            gi = torch.arange(Gf, device=dev)[:, None].expand_as(ctx.y)
                            acc.index_put_((gi, ctx.y), torch.where(ctx.mask[..., None], h.detach(), 0.), accumulate=True)
                            cnt.index_put_((gi, ctx.y), ctx.mask.float(), accumulate=True)
                        return loss, nb
                else:
                    lam = self.n_total/(self.n_total+group.n[idx])
                    loss_fn = lambda ctx: self._loss(ctx, group, r, lam)
                flat, bufs = group.p[idx], group.b[idx]
                st = local.local_epoch(group.lay, flat, bufs, env.train, data_ids, group.cfg,
                                       group.hp, lr, env.gen, loss_fn=loss_fn)
                group.p[idx], group.b[idx] = flat, bufs
                loss_sum += (st["loss"]*group.n[idx]).sum()
                step_counts += torch.stack((st["applied"].double().sum(), st["skipped"].double().sum()))
                if self.name == "fedproto":
                    S[data_ids], Cn[data_ids] = acc[:len(idx)], cnt[:len(idx)]
                elif self.name == "pfedes":
                    Fp, Fb = group.lay.params(local.pad_rows(flat, Gf)), group.lay.bufs(local.pad_rows(bufs, Gf))
                    def step2(ctx):
                        z, _ = grouped.TRAIN(Fp, Fb, pfedes.g_forward(ctx.P, ctx.x, self.proxy_cfg), group.cfg, train=False)
                        return local.ce_loss(z, ctx.y, ctx.mask), {}
                    gflat = self.proxy.repeat(len(idx), 1)
                    gb = torch.zeros(len(idx), 0, device=dev)
                    proxy_st = local.local_epoch(self.glay, gflat, gb, env.train, data_ids, group.cfg,
                                                 group.hp, lr, env.gen, loss_fn=step2)
                    step_counts += torch.stack((proxy_st["applied"].double().sum(), proxy_st["skipped"].double().sum()))
                    with cost_scope.scope("server.model_aggregation"):
                        gacc += (group.n[idx][:, None]*gflat).sum(0)
            if self.name not in ("pfedes", "fedproto") and len(group.indices):
                sums, counts = evaluate.class_sums(group.lay, group.p, group.b, env.train, group.cfg,
                    group.indices, max_rows=256 if group.name == "bert" else env.hp.get("eval_rows", 32768),
                    feat=v2x.project if self.name == "v2x_htfl" else None, d=self.d)
                S[group.indices], Cn[group.indices] = sums, counts
            step_counts = env.allreduce(step_counts)
            finite = torch.tensor(float(torch.isfinite(group.p).all() and torch.isfinite(group.b).all()), device=dev)
            finite = env.allreduce(finite)
            skip = float(step_counts[1]/step_counts.sum().clamp_min(1))
            if float(finite) != env.world or skip >= .01 or float(step_counts[0]) <= 0:
                raise RuntimeError(f"Backbone health failure: {group.name}, skip={skip}")
            group_health[f"health/{group.name}/skipped_frac"] = skip
            group_health[f"health/{group.name}/applied_steps"] = float(step_counts[0])
            group_health[f"timing/{group.name}/rank_seconds"] = time.monotonic()-t_group
        info = {"train_loss": float(env.allreduce(loss_sum)/self.n_total), **group_health}
        if self.name == "pfedes":
            with cost_scope.scope("server.model_aggregation"):
                self.proxy = env.allreduce(gacc)/self.n_total
        elif self.name == "fedproto":
            # Preserve existing accumulation precision/order within each rank.
            with cost_scope.scope("server.prototype_aggregation"):
                totals, counts = env.allreduce(S.double().sum(0)), env.allreduce(Cn.double().sum(0))
                self.has = counts > 0
                self.proto = (totals/counts.clamp_min(1)[:, None]).float()
        elif self.name == "tinyproto":
            self.local_proto = (S/Cn.clamp_min(1)[..., None]).half()
            self.local_has = Cn > 0
            with cost_scope.scope("server.prototype_aggregation"):
                scaled = env.allreduce((S*self.mask[None]).sum(0))
                holders = env.allreduce((Cn > 0).sum(0).float())
                totals = env.allreduce(Cn.sum(0))
                self.proto = self.mu*scaled/holders.clamp_min(1)[:, None]
                info.update(aps_mu=self.mu, aps_gain_max=float((self.mu*totals/holders.clamp_min(1)).max()))
        else:
            P, y = fedtgp.Method._gather(self, S, Cn)
            if self.name == "fedtgp":
                fedtgp.Method._server(self, P, y)
            else:
                v2x.Method._server(self, P, y)
            info.update(delta=self.delta, n_protos=len(y))
        return info

    def evaluate(self, prediction_dir=None):
        env = self.env
        views = ["client"] if self.name == "pfedes" else ["client", "client_proto"]
        result = {v: np.zeros((env.K, 16, 16), dtype=np.int64) for v in views}
        def score(group, args, kw, view):
            if prediction_dir is None:
                return evaluate.own_cm(*args, **kw)
            cms, pred, offsets = evaluate.own_cm(*args, **kw, predictions=True)
            folder = Path(prediction_dir)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/f"rank{env.rank}-{group.name}-{view}.npz"
            truth = torch.cat([env.test.Y[env.test.off[i]:env.test.off[i+1]] for i in group.indices]).byte().cpu().numpy()
            with open(str(path)+".tmp", "wb") as handle:
                np.savez_compressed(handle, ids=env.ids[group.indices.cpu().numpy()], offsets=offsets,
                                    y_true=truth, y_pred=pred)
            os.replace(str(path)+".tmp", path)
            return cms
        for group in self.groups:
            if not len(group.indices):
                continue
            args = (group.lay, group.p, group.b, env.test, group.cfg, group.indices)
            kw = dict(max_rows=256 if group.name == "bert" else env.hp.get("eval_rows", 32768))
            if self.name == "v2x_htfl":
                kw["head"] = lambda P, h: v2x.cosine_normalize(v2x.project(P, h)) @ self.V
            ids = group.indices.cpu().numpy()
            result["client"][ids] = score(group, args, kw, "client")
            if self.name == "pfedes":
                continue
            if self.name == "v2x_htfl":
                pn = v2x.cosine_normalize(self.proto, dim=1)
                kw["head"] = lambda P, h: (v2x.cosine_normalize(v2x.project(P, h)) @ pn.T).masked_fill(~self.has, -2.)
            elif self.name == "tinyproto":
                kw["protos"] = (self.local_proto[group.indices], self.local_has[group.indices])
            else:
                has = self.has if self.name == "fedproto" else torch.ones(16, dtype=torch.bool, device=env.dev)
                kw["protos"] = (self.proto, has)
            result["client_proto"][ids] = score(group, args, kw, "client_proto")
        return {v: env.gather_cms(cm) for v, cm in result.items()}

    def state(self):
        out = {"groups": {g.name: {"p": g.p, "b": g.b, "ids": self.env.ids[g.indices.cpu().numpy()].tolist()}
                          for g in self.groups}, "proto": self.proto, "has": self.has,
               "delta": self.delta, "gen_s": self.gen_s.get_state()}
        for key in ("local_proto", "local_has", "proxy"):
            if hasattr(self, key): out[key] = getattr(self, key)
        for key in ("tgp", "gp", "opt"):
            if hasattr(self, key): out[key] = getattr(self, key).state_dict()
        return cpu_tree(out)

    def load_state(self, state):
        for g in self.groups:
            saved = state["groups"][g.name]
            assert saved["ids"] == self.env.ids[g.indices.cpu().numpy()].tolist(), "Client/model bank mismatch"
            assert saved["p"].shape == g.p.shape and saved["b"].shape == g.b.shape
            g.p, g.b = saved["p"].to(self.env.dev), saved["b"].to(self.env.dev)
        for key in ("proto", "has", "local_proto", "local_has", "proxy"):
            if key in state:
                setattr(self, key, None if state[key] is None else state[key].to(self.env.dev))
        self.delta = state["delta"]
        self.gen_s.set_state(state["gen_s"].cpu())
        for key in ("tgp", "gp", "opt"):
            if key in state: getattr(self, key).load_state_dict(state[key])
