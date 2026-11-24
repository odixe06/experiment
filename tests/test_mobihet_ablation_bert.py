"""MobiHet ablations in the frozen FeatureBERT harness (the source the BERT L4 chain ran: fl_veremi_bert 49b2d88 + overlay,
identical manifest in all six flb-fbl-l4-top sessions) with ../mobihet's methods/mobihet_budget.py (frozen + la_mode, as
scripts/build_bert_l4_ablation.py embeds it) plus methods/mobihet_ablation.py from this repo. hp = MOBIHET_HP (la_mode=off).

  mobihet   budget la_mode=off == budget la_mode=full with la_shift zeroed by hand (the former a3), 2 rounds
  control   ablation=none == mobihet_budget (state, Q, row CE, confusion), 2 rounds
  a1/a2/a4  one/two rounds: a1 nothing sent, Q None, 0 B; a2 no row CE, same real_frac; a4 round 1 == budget, Q fp32, 1024 B

    python tests/test_mobihet_ablation_bert.py [snapshot dir]
Snapshot default: runs/recovery-20261120/references/bert-l4-artifacts/flb-fbl-l4-top-s41-r50/provenance (source/ + manifest).
20-vehicle panel of tests/test_mixed_l4.py; production L4 hp except km2_frac=1 (panel), group_rows=256 and
eval_rows=2048 (4 GB local GPU; packing only). Eager forward behind a wrapper as tests/test_mixed_l4.py.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "runs/recovery-20261120/references/bert-l4-artifacts/flb-fbl-l4-top-s41-r50/provenance"


def materialise(snap, dest):
    manifest = json.loads((snap / "source_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        src = snap / "source" / name
        assert hashlib.sha256(src.read_bytes()).hexdigest() == digest, name
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest / name)
    shutil.copyfile(ROOT / "methods/mobihet_ablation.py", dest / "methods/mobihet_ablation.py")
    shutil.copyfile(ROOT.parent / "mobihet/methods/mobihet_budget.py", dest / "methods/mobihet_budget.py")
    shutil.copyfile(Path(__file__), dest / "ablation_child.py")


def child():
    import numpy as np
    import torch
    import core
    from core import local_t3
    core.local = local_t3
    sys.modules["core.local"] = local_t3
    from core import fl, grouped, local
    from core.bert import cfg_full
    from methods import mobihet_ablation as A
    from methods.mobihet_budget import Method as Budget
    import importlib.util                                                # the snapshot's own scripts/ shadows this repo's
    spec = importlib.util.spec_from_file_location("build_l4_fl", Path(os.environ["ABLATION_REPO"]) / "scripts/build_l4_fl.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    MOBIHET_HP = builder.MOBIHET_HP

    assert A.L4 is Budget, "FeatureBERT harness must resolve to mobihet_budget"
    panel = np.sort(np.random.default_rng(7).choice(5343, 20, replace=False)).tolist()
    cache = os.environ.get("V4_CACHE", "/home/odixe/project/mobihet/cache")
    protocol = {"rounds": 50, "batch": 64, "lr_max": 1e-3, "lr_min": 1e-5, "weight_decay": 1e-4, "clip": 1.0, "seed": 42,
                "group_rows": 8192, "amp": True}

    def val(v):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v

    def env_for(arm=None, extra=()):
        torch.manual_seed(0)
        hp = dict(protocol)
        hp.update({k: val(v) for k, v in (s.split("=", 1) for s in
                   [*MOBIHET_HP, "seed=42", "km2_frac=1.0", "group_rows=256", "eval_rows=2048", *(A.HP[arm] if arm else []),
                    *extra])})
        return fl.Env(cfg_full(), hp, clients=panel, cache=cache)

    def run(M, rounds, seed=1):
        train = grouped.TRAIN
        grouped.TRAIN = lambda *a, **k: grouped.forward(*a, **k)
        try:
            torch.manual_seed(seed)
            return [M.round(r, fl.cosine_lr(r, 50, 1e-3, 1e-5)) for r in rounds]
        finally:
            grouped.TRAIN = train

    def exact(a, b, path=""):
        if torch.is_tensor(a):
            if path.endswith("/row_score"):
                a, b = a[:-1], b[:-1]                                    # sink slot of masked writes
            assert a.dtype == b.dtype and a.shape == b.shape and (
                torch.equal(a, b) or torch.equal(a.nan_to_num(), b.nan_to_num())), path
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a: exact(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, np.ndarray):
            assert np.array_equal(a, b), path
        else:
            assert a == b, (path, a, b)

    a, b = Budget(env_for()), Budget(env_for(extra=["la_mode=full"]))
    assert not bool(a.la_shift.any()) and bool(b.la_shift.abs().sum() > 0) and a.bootstrap_bytes == b.bootstrap_bytes
    b.la_shift = torch.zeros_like(b.la_shift)
    run(a, (1, 2)); run(b, (1, 2))
    exact(b.state(), a.state(), "mobihet state"); exact(b.evaluate()["private"], a.evaluate()["private"], "mobihet cms")
    print("bert mobihet PASS (la_mode=off == L4 with zero shift = former a3)", flush=True)

    a, b = A.Method(env_for("none")), Budget(env_for())
    run(a, (1, 2)); run(b, (1, 2))
    exact(b.state(), a.state(), "state"); exact(b.evaluate()["private"], a.evaluate()["private"], "cms")
    print("bert control PASS (2 rounds bit-exact with mobihet_budget)", flush=True)

    m = A.Method(env_for("a1"))
    assert m.bootstrap_bytes == {"up": 0, "down": 0} and not hasattr(m, "cop_cdf") and not bool(m.la_shift.any())
    i = run(m, (1, 2))[-1]
    assert m.Q is None and i["bytes_up"] == 0 and i["real_frac"] == 1.0 and np.isfinite(i["train_loss"])
    print("bert a1 PASS", flush=True)

    a, b = A.Method(env_for("a2")), Budget(env_for())
    ia, ib = run(a, (1,)), run(b, (1,))
    assert "row_score" not in a.state() and abs(ia[0]["real_frac"] - ib[0]["real_frac"]) < 1e-12
    print("bert a2 PASS", flush=True)

    a, b = A.Method(env_for("a4")), Budget(env_for())
    ia, ib = run(a, (1,)), run(b, (1,))
    exact(b.priv_p, a.priv_p, "a4 r1")
    dq = float((a.Q - b.Q).abs().max())
    print("bert a4 bytes_up", ia[0]["bytes_up"], "bytes_down", ia[0]["bytes_down"], "|dQ|max", dq, flush=True)
    # |dQ| = the 8-bit codec error of the control (uint8 up + uint8 down, then row renormalisation in both decoders, which
    # rescales a row whose rounded entries do not sum to 1): 2/255 from the roundings plus that rescale. 24/11: .00827 on the
    # MobiHet (no LA) BERT panel, just above 2/255 = .00784; 3/255 bounds it.
    assert ia[0]["bytes_up"] == 1024 and dq <= 3 / 255
    assert not torch.equal(a.Q, a.Q.half().float())
    print("bert a4 PASS", flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        sys.path.insert(0, os.getcwd())
        child()
        sys.exit(0)
    snap = Path(sys.argv[1]) if len(sys.argv) > 1 else SNAP
    with tempfile.TemporaryDirectory(prefix="bert-ablation-") as tmp:
        materialise(snap, Path(tmp))
        env = {**os.environ, "PYTHONPATH": tmp, "V4_ROOT": os.environ.get("V4_ROOT", "/home/odixe/project/mobihet/knowledge"),
               "MOBIHET_RECOVERY_DIR": str(Path(tmp) / "recovery"), "ABLATION_REPO": str(ROOT)}
        sys.exit(subprocess.run([sys.executable, "ablation_child.py", "--child"], cwd=tmp, env=env).returncode)
