"""Instantiate each FL method on a toy CPU env (as scripts/cost.py) and measure the tensors that cross client/server.

    cd <repo>; python artifacts/communication-budget/scripts/probe_methods.py --root <repo|repo/bert> --out probe.json

Per (method, backbone): payload_bytes() of the method, layout P/B, prototype dim, and (after round 1 for exchange-by-tensor
methods) the real server-state tensors: shape, dtype, nbytes, nonzeros per class for TinyProto, n uploaded (client,class)
prototypes for FedTGP/V2X (info['n_protos']). Read-only on the repo; no real data.
"""
import argparse, importlib, io, json, os, sys, traceback
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--methods", nargs="*")
a = ap.parse_args()
ROOT = os.path.abspath(a.root)
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
os.chdir(ROOT)
import numpy as np, torch
torch.set_num_threads(4)
IS_BERT = os.path.basename(ROOT) == "bert"
from core import local  # noqa
from core.flat import Layout  # noqa
from test_methods import ToyEnv  # noqa
if IS_BERT:
    from core.bert import cfg_full
    BACKBONES = {"bert": cfg_full}
else:
    from core.dagsnet import width
    from core.tabm import config as tabm_cfg
    from core.sae import fl_config as sae_cfg
    from core.mlp import config as mlp_cfg
    from core.cnn_bigru_attention import config as gru_cfg
    BACKBONES = {"dagsnet": lambda: width(1), "tabm": tabm_cfg, "sae": sae_cfg, "mlp": mlp_cfg,
                 "cnn-bigru-attention": gru_cfg}
METHODS = a.methods or ["fedavg", "fedprox", "afpha", "fd_ids", "apfed", "lwfednids", "perfedskd", "nilm", "pfedes",
                        "fedproto", "fedtgp", "tinyproto", "v2x_htfl"]


def env_for(cfg, K=6):
    env = ToyEnv(K=K)
    env.cfg = cfg
    env.lay = Layout(cfg)
    return env


def ser(t):
    b = io.BytesIO(); torch.save(t.detach().cpu().contiguous(), b); return b.getbuffer().nbytes


def tinfo(t):
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "nbytes": int(t.numel() * t.element_size()),
            "torch_save_bytes": ser(t)}


def probe(name, bb):
    env = env_for(BACKBONES[bb]())
    note = ""
    try:
        M = importlib.import_module(f"methods.{name}").Method(env)
    except Exception as ex:
        if name != "afpha":
            raise
        # AFPHA __init__ reads clients.parquet (full dataset absent locally); payload_bytes/p/b are FedAvg's (inherited)
        M = importlib.import_module("methods.fedavg").Method(env)
        note = "afpha probed via inherited FedAvg layout (clients.parquet absent): " + str(ex)[:60]
    rec = {"method": name, "backbone": bb, "note": note, "payload_bytes": {k: v for k, v in M.payload_bytes().items()}}
    e = M.env
    rec["P"], rec["B"], rec["d_cfg_head_hidden"] = e.lay.P, e.lay.B, e.cfg.get("head_hidden")
    if name == "v2x_htfl":
        rec["P_with_projection"] = e.lay.P
    T = {}
    if name == "perfedskd":
        T["theta_params"], T["theta_bn_buffers"] = tinfo(M.g_p), tinfo(M.g_b)
    elif name in ("fedavg", "fedprox", "afpha", "fd_ids", "apfed", "lwfednids"):
        T["theta_params"], T["theta_bn_buffers"] = tinfo(M.p), tinfo(M.b)
        if name == "apfed":
            T["classifier_only_P"] = int(e.lay.mask(["head.5"]).sum()) if bb == "dagsnet" else None
        if name == "perfedskd":
            pass
    elif name == "nilm":
        T["proxy_params"], T["proxy_bn_buffers"] = tinfo(M.g_p), tinfo(M.g_b)
    elif name == "pfedes":
        T["proxy_extractor_G"] = tinfo(M.g); rec["P_G"] = M.glay.P
    else:  # prototype methods need one round
        info = M.round(1, 1e-3)
        ytr = env.train.Y.long()
        cnt = torch.zeros(env.K, 16)
        for k in range(env.K):
            cnt[k] = torch.bincount(ytr[env.train.off[k]:env.train.off[k + 1]], minlength=16)
        rec["toy_present_pairs"] = int((cnt > 0).sum()); rec["toy_K"] = env.K
        if name == "fedproto":
            T["global_proto_down"] = tinfo(M.proto)
        elif name == "fedtgp":
            T["global_proto_down"] = tinfo(M.proto); rec["n_protos_uploaded"] = info.get("n_protos")
        elif name == "v2x_htfl":
            T["global_proto_down"] = tinfo(M.proto); rec["n_protos_uploaded"] = info.get("n_protos")
            rec["P"] = M.env.lay.P
        elif name == "tinyproto":
            nz = (M.target != 0).sum(1)
            T["target_dense_down"] = tinfo(M.target)
            rec["s"] = M.s; rec["mask_true_per_class"] = M.mask.sum(1).tolist()
            rec["target_nonzero_per_class"] = nz.tolist()
            T["target_sparse_wire_bytes_calc"] = {"nbytes": int(16 * M.s * 4)}
            T["mask_bits_once"] = {"nbytes": int(16 * e.cfg["head_hidden"] // 8)}
    rec["tensors"] = T
    return rec


out = []
for bb in BACKBONES:
    for m in METHODS:
        try:
            out.append(probe(m, bb))
            print("ok", bb, m, flush=True)
        except Exception as ex:
            out.append({"method": m, "backbone": bb, "error": f"{type(ex).__name__}: {ex}",
                        "trace": traceback.format_exc()[-600:]})
            print("ERR", bb, m, ex, flush=True)
json.dump(out, open(a.out, "w"), indent=1)
