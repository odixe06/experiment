"""Mixed (heterogeneous) cases: build methods.mixed.Method on a toy CPU env (2 clients per backbone), run round 1, and read
the tensors that cross the boundary + stored parameter counts per backbone group. Read-only; no real data.

    python artifacts/communication-budget/scripts/probe_mixed.py --out artifacts/communication-budget/work/probe_mixed.json
"""
import argparse, json, os, sys, traceback
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
os.chdir(ROOT)
import torch
torch.set_num_threads(4)
from core.model_assignment import CASES, counts
from methods import mixed
from test_methods import ToyEnv


def run(case, name):
    models = [m for m, _ in CASES[case]]
    env = ToyEnv(K=2 * len(models))
    assignment = {i: models[i // 2] for i in range(env.K)}
    M = mixed.Method(env, name, assignment, models)
    rec = {"case": case, "method": name, "payload_bytes": M.payload_bytes(), "d": M.d,
           "groups": {g.name: {"P": g.lay.P, "B": g.lay.B, "cfg_head_hidden": g.cfg["head_hidden"], "n_clients_toy": len(g.indices)}
                      for g in M.groups}}
    info = M.round(1, 1e-3)
    t = {}
    if name in ("fedproto", "tinyproto", "v2x_htfl", "fedtgp"):
        t["global_proto_down"] = {"shape": list(M.proto.shape), "dtype": str(M.proto.dtype), "nbytes": M.proto.numel() * M.proto.element_size()}
    if name == "pfedes":
        t["proxy"] = {"shape": list(M.proxy.shape), "nbytes": M.proxy.numel() * 4, "P_G": M.glay.P}
    if name == "tinyproto":
        rec["s"] = M.s; rec["target_nonzero_per_class"] = (M.proto != 0).sum(1).tolist()
        t["masks_once_bits_nbytes"] = 16 * M.d // 8
    rec["tensors"] = t
    rec["n_protos_uploaded"] = info.get("n_protos")
    return rec


out = []
for case in CASES:
    for name in mixed.METHODS:
        try:
            out.append(run(case, name)); print("ok", case, name, flush=True)
        except Exception as ex:
            out.append({"case": case, "method": name, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-500:]})
            print("ERR", case, name, ex, flush=True)
out.append({"assignment_counts": {c: counts(c) for c in CASES}})
json.dump(out, open(a.out, "w"), indent=1)
