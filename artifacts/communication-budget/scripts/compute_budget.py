"""Communication budget of every executed FL run (simulated client<->server payload, NOT measured network traffic).

    ~/Tools/miniconda3/envs/nckh/bin/python artifacts/communication-budget/scripts/compute_budget.py

Inputs (produced by scripts/inventory.py and scripts/payload_specs.py, see method_formulas.md):
  work/run_inventory_raw.csv   every run dir/session found (repo, ORICO, ORICO zip archives), canonical flags
  work/backbone_params.csv     P, B, d per backbone (instantiated on CPU)
  work/data_dependent_stats.csv  (client, class) pairs with >= 1 train+val row, total and per mixed backbone group
Per-round logs (rounds.jsonl) and config.json are re-read here from the canonical session paths (not from worker sums),
so L4 measured bytes and PerFed-SKD selection come straight from the run logs.

Outputs (artifacts/communication-budget/): run_inventory.csv, communication_per_run.csv, communication_summary.csv,
communication_ranking.csv, work/crosscheck.json. Units: bytes; MB = 1e6 B, GB = 1e9 B.
"""
import io
import json
import math
import os
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
WORK = os.path.join(OUT, "work")
ARCH = "/run/media/odixe/ORICO/experiment-data/experiment/archives"
C = 16

DENSE = {"fedavg", "fedprox", "afpha", "fd_ids", "apfed", "nilm"}
PROTO = {"fedproto", "fedtgp", "v2x_htfl", "tinyproto"}
SCEN_BACKBONE = {"dagsnet": "dagsnet", "bert": "bert (FeatureBERT)", "tabm": "tabm", "sae": "sae", "mlp": "mlp",
                 "cnn-bigru": "cnn-bigru-attention"}
MIXED_GROUP_NAME = {"cnn-bigru-attention": "cnn-bigru-attention"}


# ---------------------------------------------------------------------------------------------------------------- io
_zips = {}


def read_text(path):
    """Plain path, or '<archive>.zip!/<member>' (archives on ORICO)."""
    if not isinstance(path, str) or not path:
        return None
    if ".zip!" in path:
        z, m = path.split("!", 1)
        if z not in _zips:
            _zips[z] = zipfile.ZipFile(os.path.join(ARCH, os.path.basename(z)))
        try:
            return _zips[z].read(m.lstrip("/")).decode("utf-8")
        except KeyError:
            return None
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_rounds(path):
    txt = read_text(path)
    if txt is None:
        return None
    rows = {}
    for line in txt.splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            if "round" in d:
                rows[int(d["round"])] = d          # cumulative logs: last occurrence of a round wins (identical on overlap)
    return [rows[r] for r in sorted(rows)]


def read_config(path):
    txt = read_text(path)
    return json.loads(txt) if txt else None


# ------------------------------------------------------------------------------------------------------- constants
bp = pd.read_csv(os.path.join(WORK, "backbone_params.csv"))


def bp_row(backbone, variant_prefix):
    x = bp[(bp.backbone == backbone) & bp.variant.str.startswith(variant_prefix)]
    assert len(x) == 1, (backbone, variant_prefix, len(x))
    return x.iloc[0]


SINGLE = {}
for scen, bb in SCEN_BACKBONE.items():
    native = bp_row(bb, "FeatureBERT FL native" if scen == "bert" else "single-FL native")
    lw = bp_row(bb, "LwFedNIDS")
    d = int(native.proto_dim_d)
    SINGLE[scen] = {"dense": int(native.bytes_fp32), "P": int(native.n_params), "B": int(native.n_buffers),
                    "lw": int(lw.bytes_fp32), "d": d, "s": math.ceil(d / 10)}
PFEDES = int(bp[bp.variant.str.startswith("pFedES")].iloc[0].bytes_fp32)
MIXED_D = 128

ds = pd.read_csv(os.path.join(WORK, "data_dependent_stats.csv"))
NP_ALL = int(ds[ds.scope == "all"].client_class_pairs.iloc[0])
K_ALL = int(ds[ds.scope == "all"].clients.iloc[0])
GROUPS = {case: {r.group: (int(r.clients), int(r.client_class_pairs)) for r in g.itertuples()}
          for case, g in ds[ds.scope != "all"].groupby("scope")}
for case, g in GROUPS.items():                 # per-group pairs must partition the population
    assert sum(v[0] for v in g.values()) == K_ALL and sum(v[1] for v in g.values()) == NP_ALL, case


def per_class_bytes(method, d, s):
    """Uplink bytes per (client, present class): fp32 prototype (+ fp32 count where the server weights by it)."""
    return {"fedproto": 4 * d + 4, "fedtgp": 4 * d, "v2x_htfl": 4 * d, "tinyproto": 4 * s}[method]


def proto_down(method, d, s):
    return 4 * C * (s if method == "tinyproto" else d)


# ---------------------------------------------------------------------------------------------------------- model
def blank():
    return dict(boot_up=0.0, boot_down=0.0, rec_up=0.0, rec_down=0.0, meta_up=0.0, meta_down=0.0,
                down_unique=0.0, dense_upper_up=None, basis="", confidence="", formula="", notes=[])


def run_params(method, scen, cfg, own):
    """d, s, model bytes for this run: from the run's own config payload_bytes when present (older generations differ,
    e.g. SAE v1 head_hidden=16), else the current backbone constants (instantiated, backbone_params.csv)."""
    mixed = scen.startswith("mixed")
    prm = {"d": MIXED_D if mixed else SINGLE[scen]["d"], "dense": None if mixed else SINGLE[scen]["dense"],
           "lw": None if mixed else SINGLE[scen]["lw"], "pfedes": PFEDES, "src": "backbone constants"}
    prm["s"] = math.ceil(prm["d"] / 10)
    pb = (cfg or {}).get("payload_bytes") if own else None
    if isinstance(pb, dict) and "down" in pb and isinstance(pb["down"], (int, float)):
        if method in DENSE or method == "perfedskd":
            prm["dense"] = int(pb["down"])
        elif method == "lwfednids":
            prm["lw"] = int(pb["down"])
        elif method == "pfedes":
            prm["pfedes"] = int(pb["down"])
        elif method == "tinyproto":
            prm["s"] = int(pb["down"]) // (4 * C)
            prm["d"] = int(pb["masks_once"]) * 8 // C
        elif method in PROTO:
            prm["d"] = int(pb["down"]) // (4 * C)
        prm["src"] = "own config payload_bytes"
    return prm


def compute(method, scen, K, R, rounds, cfg, prm):
    """Totals over all clients and rounds 1..R for one run."""
    o = blank()
    mixed = scen.startswith("mixed")
    if method == "mobihet_l4" or method.startswith("mobihet"):
        has = rounds and all("bytes_up" in x and "bytes_down" in x for x in rounds)
        boot = (cfg or {}).get("payload_bytes", {}).get("bootstrap") if cfg else None
        if not has:
            o["basis"], o["confidence"] = "not computable", "none"
            o["notes"].append("rounds.jsonl has no bytes_up/bytes_down")
            return o
        up = [float(x["bytes_up"]) for x in rounds]
        dn = [float(x["bytes_down"]) for x in rounds]
        o["rec_up"], o["rec_down"] = K * sum(up), K * sum(dn)
        o["down_unique"] = sum(dn)
        if isinstance(boot, dict) and "up" in boot:
            o["boot_up"], o["boot_down"] = K * float(boot["up"]), K * float(boot["down"])
            codec = cfg["payload_bytes"].get("codec") or {}
            if "broadcast_copy" in codec:
                o["down_unique"] += float(codec["broadcast_copy"])
                o["notes"].append("bootstrap downlink unique = codec broadcast_copy (personalised parts would be one copy)")
            else:
                o["down_unique"] = float("nan")
                o["notes"].append("bootstrap broadcast copy not logged in this config; down_unique left blank")
        else:
            o["notes"].append("no bootstrap bytes in config")
        if method == "mobihet_l4":
            o["basis"] = "measured (codec byte length per vehicle in simulation; all vehicles; not network traffic)"
            o["confidence"] = "high"
        else:
            o["basis"] = "measured-in-log (probe variant; per-round byte semantics not re-audited)"
            o["confidence"] = "low"
        o["formula"] = "K*sum_r bytes_up_r | K*sum_r bytes_down_r | K*bootstrap_mean(up,down)"
        return o

    d, s, dense = prm["d"], prm["s"], prm["dense"]

    if method in DENSE or method == "lwfednids":
        m = prm["lw"] if method == "lwfednids" else dense
        o["rec_up"] = o["rec_down"] = K * R * m
        o["down_unique"] = R * m
        o["formula"] = "K*R*4(P'+B')" if method == "lwfednids" else "K*R*4(P+B)"
        o["basis"], o["confidence"] = "calculated (tensor size fp32; P,B by instantiation)", "high"
        if method == "afpha":
            o["meta_down"] = 4 * K * (R - 1)       # personal mu_i (fp32) computed on the server after rounds 1..R-1
            o["formula"] += " + 4*K*(R-1) mu_i"
    elif method == "perfedskd":
        nsel = [round(float(x["selected_frac"]) * K) for x in rounds]
        o["rec_up"] = o["rec_down"] = dense * sum(nsel)
        o["down_unique"] = R * dense
        o["meta_up"] = 4 * K * (R - 1)              # a_m from every device, used for selection in rounds 2..R
        o["formula"] = "4(P+B)*sum_r n_sel_r + 4*K*(R-1) a_m"
        o["basis"], o["confidence"] = "calculated from logged selected_frac (selected devices only)", "high"
        o["notes"].append(f"participants/round mean {sum(nsel) / len(nsel):.1f} of {K}")
    elif method == "canperfl":
        o["boot_down"] = K * dense                  # one download of theta^50 to start fine-tuning; phase 2 sends nothing
        o["down_unique"] = dense
        o["formula"] = "K*4(P+B) once (theta^50); phase-1 FedAvg attributed separately"
        o["basis"], o["confidence"] = "calculated", "high"
    elif method == "pfedes":
        o["rec_up"] = o["rec_down"] = K * R * prm["pfedes"]
        o["down_unique"] = R * prm["pfedes"]
        o["formula"] = "K*R*4*P_G (P_G=351)"
        o["basis"], o["confidence"] = "calculated", "high"
    elif method in PROTO:
        if K != K_ALL:
            o["basis"], o["confidence"] = "not computable (client-class pairs only known for K=5343)", "none"
            return o
        pc = per_class_bytes(method, d, s)
        o["rec_up"] = R * NP_ALL * pc
        o["dense_upper_up"] = R * K * C * pc
        o["rec_down"] = K * R * proto_down(method, d, s)
        o["down_unique"] = R * proto_down(method, d, s)
        o["formula"] = f"R*N_p*{pc} up | K*R*{proto_down(method, d, s)} down (N_p={NP_ALL})"
        if method == "tinyproto":
            # CPS masks: a seeded permutation (methods/tinyproto.py:34-38) rebuilt on every client -> 0 B (owner 21/11)
            o["notes"].append(f"masks 0 B (seed-derived); payload_bytes masks_once={C * d // 8} B not counted")
        o["basis"] = "calculated (present classes only; N_p from fold_counts.npy)"
        o["confidence"] = "high"
    else:
        o["basis"], o["confidence"] = f"unknown method {method}", "none"
    return o


def mixed_breakdown(method, scen, R):
    """Per-backbone-group totals for calculated mixed baselines (K = 5343, all clients every round)."""
    out = {}
    d, s = MIXED_D, math.ceil(MIXED_D / 10)
    for g, (kg, npg) in GROUPS[scen.replace("mixed-", "")].items():
        if method == "pfedes":
            up = dn = kg * R * PFEDES
            boot = 0
        else:
            up = R * npg * per_class_bytes(method, d, s)
            dn = kg * R * proto_down(method, d, s)
            boot = 0                                # TinyProto masks are seed-derived: 0 B
        out[g] = {"clients": kg, "pairs": npg, "up": up, "down": dn + boot,
                  "per_client_total": (up + dn + boot) / kg}
    return out


# ----------------------------------------------------------------------------------------------------------- runs
inv = pd.read_csv(os.path.join(WORK, "run_inventory_raw.csv"))


def leaf(sid):
    return str(sid).rstrip("/").split("/")[-1]


def pick_runs():
    """Canonical production + one session per executed pilot / probe / superseded logical run (largest round)."""
    sel = [r for _, r in inv[(inv.canonical == "yes") & (inv.run_kind == "production")].iterrows()]
    extra = inv[inv.run_kind.isin(["pilot", "probe", "superseded-rerun"]) & inv.rounds_path.notna()].copy()
    extra["key"] = extra.apply(lambda r: (r.run_kind, r.scenario, r.method, r.seed,
                                          leaf(r.session_id).split("-r")[0] if r.run_kind == "probe" else leaf(r.session_id)
                                          ), axis=1)
    extra = extra.sort_values(["round_last", "location"], ascending=[False, True])
    for _, g in extra.groupby("key", sort=False):
        sel.append(g.iloc[0])
    return sel


def find_config(r):
    cfg = read_config(r.config_path) if isinstance(r.config_path, str) else None
    src = "own"
    if cfg is None:                             # same logical run, any session with a config (payload is not seed-specific)
        ok = ["production", "production-duplicate", "continuation-session", "config-only"]
        sib = inv[(inv.scenario == r.scenario) & (inv.method == r.method) & (inv.seed == r.seed) & inv.config_path.notna()
                  & inv.run_kind.isin(ok if r.run_kind == "production" else [r.run_kind])]
        for _, s in sib.iterrows():
            cfg = read_config(s.config_path)
            if cfg is not None:
                src = "inherited:" + str(s.session_id)
                break
    return cfg, src


rows, cross = [], {"config_payload_mismatch": [], "round_gaps": [], "pilot_overlap": []}
canon_rounds = {}
for r in pick_runs():
    rounds = read_rounds(r.rounds_path)
    cfg, cfg_src = find_config(r)
    if not rounds:
        continue
    R = len(rounds)
    if [x["round"] for x in rounds] != list(range(1, R + 1)):
        cross["round_gaps"].append(r.session_id)
    k_note = ""
    if pd.notna(r.n_clients):
        K = int(r.n_clients)
    elif r.run_kind == "production":
        K, k_note = K_ALL, "K=5343 from protocol + per-backbone manifest (no own config.json)"
    elif cfg_src == "own" and cfg and "clients" in cfg:
        K = len(cfg["clients"]) if isinstance(cfg["clients"], list) else int(cfg["clients"])
    else:
        K = None
    if K is None:
        print("skip (K unknown):", r.session_id)
        continue
    if r.run_kind == "production":
        canon_rounds[(r.scenario, r.method, int(r.seed))] = rounds
    prm = run_params(r.method, r.scenario, cfg, cfg_src == "own")
    o = compute(r.method, r.scenario, K, R, rounds, cfg, prm)
    if k_note:
        o["notes"].append(k_note)
    if cfg_src != "own":
        o["notes"].append("config inherited from " + cfg_src if cfg else "no config found")
    # cross-check own formula vs the run's own payload_bytes (per client per round, dense convention)
    pb = (cfg or {}).get("payload_bytes", {})
    cst = run_params(r.method, r.scenario, None, False)
    if r.method in DENSE | {"lwfednids", "perfedskd", "pfedes"} | PROTO and isinstance(pb, dict) and "down" in pb:
        mine = (cst["lw"] if r.method == "lwfednids" else cst["pfedes"] if r.method == "pfedes"
                else proto_down(r.method, cst["d"], cst["s"]) if r.method in PROTO else cst["dense"])
        if int(pb["down"]) != mine:
            cross["config_payload_mismatch"].append([r.run_kind, r.session_id, pb, mine, cfg_src])
    total_up = o["boot_up"] + o["rec_up"] + o["meta_up"]
    total_down = o["boot_down"] + o["rec_down"] + o["meta_down"]
    rec_pc_round = (o["rec_up"] + o["rec_down"]) / (K * R) if K and R else float("nan")
    row = {
        "run_id": r.logical_run_id if r.run_kind == "production" else f"{r.run_kind}:{leaf(r.session_id)}",
        "run_kind": r.run_kind, "scenario": r.scenario, "backbone": r.backbone, "method": r.method,
        "seed": int(r.seed) if pd.notna(r.seed) else None, "n_clients": K,
        "clients_per_backbone": r.clients_per_backbone if pd.notna(r.clients_per_backbone) else "",
        "rounds_planned": r.rounds_planned, "rounds_executed": R,
        "complete": bool(pd.notna(r.rounds_planned) and R >= int(r.rounds_planned)),
        "bootstrap_up_B": o["boot_up"], "bootstrap_down_B": o["boot_down"],
        "recurring_up_B": o["rec_up"], "recurring_down_B": o["rec_down"],
        "metadata_up_B": o["meta_up"], "metadata_down_B": o["meta_down"],
        "recurring_per_client_per_round_B": rec_pc_round,
        "recurring_up_per_client_per_round_B": o["rec_up"] / (K * R) if K and R else float("nan"),
        "recurring_down_per_client_per_round_B": o["rec_down"] / (K * R) if K and R else float("nan"),
        "total_up_B": total_up, "total_down_B": total_down, "total_B": total_up + total_down,
        "total_per_client_B": (total_up + total_down) / K if K else float("nan"),
        "total_MB": (total_up + total_down) / 1e6, "total_GB": (total_up + total_down) / 1e9,
        "dense_upper_bound_up_B": o["dense_upper_up"],
        "server_down_unique_B": o["down_unique"],
        "canperfl_phase1_attributed_B": None, "total_incl_attributed_B": total_up + total_down,
        "payload_params": json.dumps({k: prm[k] for k in ("d", "s", "dense", "lw", "pfedes", "src")}),
        "formula": o["formula"], "basis": o["basis"], "confidence": o["confidence"],
        "is_network_traffic": "no (simulated payload)",
        "rounds_source": r.rounds_path, "config_source": cfg_src if cfg else "none",
        "notes": "; ".join(o["notes"]),
    }
    if r.scenario.startswith("mixed") and r.method in PROTO | {"pfedes"} and K == K_ALL:
        row["per_backbone_breakdown"] = json.dumps(mixed_breakdown(r.method, r.scenario, R))
    elif r.scenario.startswith("mixed") and r.method == "mobihet_l4":
        row["per_backbone_breakdown"] = "pooled: bytes logged as one mean over all vehicles of the case; not separable"
    else:
        row["per_backbone_breakdown"] = ""
    rows.append(row)

df = pd.DataFrame(rows)

# CANPerFL phase 1 = the FedAvg run of the same scenario and seed (state reused, not re-run)
fa = df[(df.run_kind == "production") & (df.method == "fedavg")].set_index(["scenario", "seed"]).total_B
for i in df.index[(df.method == "canperfl")]:
    key = (df.at[i, "scenario"], df.at[i, "seed"])
    if key in fa.index:
        df.at[i, "canperfl_phase1_attributed_B"] = float(fa[key])
        df.at[i, "total_incl_attributed_B"] = df.at[i, "total_B"] + float(fa[key])
        df.at[i, "notes"] = (df.at[i, "notes"] + "; " if df.at[i, "notes"] else "") + \
            "phase 1 = FedAvg same scenario/seed (assumed parent; hp.init glob)"

# pilots that are the first session of a canonical chain would be double counted
for i in df.index[df.run_kind == "pilot"]:
    key = (df.at[i, "scenario"], df.at[i, "method"], df.at[i, "seed"])
    pr = read_rounds(df.at[i, "rounds_source"])
    cr = canon_rounds.get(key)
    same = bool(cr) and df.at[i, "n_clients"] == K_ALL and all(
        abs(float(a.get("train_loss", 0)) - float(b.get("train_loss", 1))) < 1e-9 and a.get("bytes_up") == b.get("bytes_up")
        for a, b in zip(pr, cr))
    df.at[i, "notes"] = (df.at[i, "notes"] + "; " if df.at[i, "notes"] else "") + \
        ("rounds identical to canonical prefix -> already inside canonical total, do not add" if same
         else "independent of canonical chain")
    df.loc[i, "pilot_in_canonical"] = same
    cross["pilot_overlap"].append([df.at[i, "run_id"], same])

df.sort_values(["run_kind", "scenario", "method", "seed"]).to_csv(os.path.join(OUT, "communication_per_run.csv"), index=False)

# ---------------------------------------------------------------------------------------------------- summary
prod = df[df.run_kind == "production"]
agg = []
for (sc, m), g in prod.groupby(["scenario", "method"]):
    rec = {"scenario": sc, "method": m, "backbone": g.backbone.iloc[0], "n_seeds": len(g),
           "seeds": ",".join(str(s) for s in sorted(g.seed)), "n_clients": int(g.n_clients.iloc[0]),
           "rounds": int(g.rounds_executed.iloc[0])}
    for col in ["bootstrap_up_B", "bootstrap_down_B", "recurring_up_B", "recurring_down_B", "metadata_up_B",
                "metadata_down_B", "recurring_per_client_per_round_B", "total_per_client_B", "total_B",
                "total_incl_attributed_B"]:
        v = g[col].astype(float)
        rec[col + "_mean"] = v.mean()
        rec[col + "_std"] = v.std(ddof=1) if len(v) > 1 else float("nan")
    rec["total_GB_mean"] = rec["total_B_mean"] / 1e9
    rec["total_incl_attributed_GB_mean"] = rec["total_incl_attributed_B_mean"] / 1e9
    rec["basis"] = g.basis.iloc[0]
    rec["confidence"] = g.confidence.min()
    rec["data_source"] = "rounds.jsonl + config.json of canonical sessions (see communication_per_run.csv)"
    agg.append(rec)
summ = pd.DataFrame(agg)
summ.to_csv(os.path.join(OUT, "communication_summary.csv"), index=False)

rank = summ[["scenario", "method", "backbone", "n_seeds", "total_incl_attributed_B_mean", "total_incl_attributed_B_std",
             "total_B_mean", "total_per_client_B_mean", "recurring_per_client_per_round_B_mean", "basis", "confidence"]].copy()
rank["rank_in_scenario"] = rank.groupby("scenario").total_incl_attributed_B_mean.rank(method="min").astype(int)
fa_s = rank[rank.method == "fedavg"].set_index("scenario").total_incl_attributed_B_mean
rank["x_fedavg"] = rank.apply(lambda r: r.total_incl_attributed_B_mean / fa_s[r.scenario] if r.scenario in fa_s.index
                              else float("nan"), axis=1)
rank["total_incl_attributed_GB_mean"] = rank.total_incl_attributed_B_mean / 1e9
rank.sort_values(["scenario", "rank_in_scenario"]).to_csv(os.path.join(OUT, "communication_ranking.csv"), index=False)

# --------------------------------------------------------------------------------------------- run inventory
iv = inv.copy()
iv["in_budget"] = "no"
iv.loc[(iv.canonical == "yes") & (iv.run_kind == "production"), "in_budget"] = "yes (canonical)"
kept = set(df.rounds_source)
iv.loc[iv.run_kind.isin(["pilot", "probe", "superseded-rerun"]) & iv.rounds_path.isin(kept), "in_budget"] = "per-run only (not in summary)"
reason = {"continuation-session": "rounds subsumed by canonical session (cumulative log)",
          "production-duplicate": "duplicate copy of canonical session", "smoke": "toy smoke test",
          "compute-trace": "FLOP-count panel run, not an experiment", "centralized": "no client-server exchange",
          "config-only": "no rounds.jsonl", "dropped-method": "SCAFFOLD dropped by owner",
          "probe": "older session of a probe chain", "pilot": "", "superseded-rerun": "older session of a superseded chain"}
iv["budget_exclusion_reason"] = iv.apply(lambda r: "" if r.in_budget != "no" else reason.get(r.run_kind, ""), axis=1)
iv["data_source"] = iv.location
iv["confidence"] = iv.apply(lambda r: "high" if r.in_budget == "yes (canonical)" and r.results_source_matches_session in (True, "True", "yes")
                            else "medium" if r.in_budget != "no" else "", axis=1)
iv.to_csv(os.path.join(OUT, "run_inventory.csv"), index=False)

with open(os.path.join(WORK, "crosscheck.json"), "w") as f:
    json.dump(cross, f, indent=1, default=str)
print("runs:", len(df), df.run_kind.value_counts().to_dict())
print("config payload mismatches:", len(cross["config_payload_mismatch"]), " round gaps:", len(cross["round_gaps"]))
print("pilot overlap:", cross["pilot_overlap"])
