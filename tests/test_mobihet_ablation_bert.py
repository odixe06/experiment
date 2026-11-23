"""MobiHet L4 ablations in the frozen FeatureBERT harness (the source the BERT L4 chain ran: fl_veremi_bert 49b2d88 + overlay,
identical manifest in all six flb-fbl-l4-top sessions) plus methods/mobihet_ablation.py from this repo.

  control   ablation=none == mobihet_budget (state, Q, row CE, confusion), 2 rounds
  a1..a4    one/two rounds: a1 counts-only bootstrap, Q None, 0 B; a2 no row CE, same real_frac; a3 == budget with zero shift;
            a4 round 1 == budget, Q fp32, 1024 B

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
    L4_HP = builder.L4_HP

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

    def env_for(arm=None):
        torch.manual_seed(0)
        hp = dict(protocol)
        hp.update({k: val(v) for k, v in (s.split("=", 1) for s in
                   [*L4_HP, "seed=42", "km2_frac=1.0", "group_rows=256", "eval_rows=2048", *(A.HP[arm] if arm else [])])})
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

    a, b = A.Method(env_for("none")), Budget(env_for())
    run(a, (1, 2)); run(b, (1, 2))
    exact(b.state(), a.state(), "state"); exact(b.evaluate()["private"], a.evaluate()["private"], "cms")
    print("bert control PASS (2 rounds bit-exact with mobihet_budget)", flush=True)

    m = A.Method(env_for("a1"))
    assert m.bootstrap_bytes == {"up": 64, "down": 64} and not hasattr(m, "cop_cdf")
    i = run(m, (1, 2))[-1]
    assert m.Q is None and i["bytes_up"] == 0 and i["real_frac"] == 1.0 and np.isfinite(i["train_loss"])
    print("bert a1 PASS", flush=True)

    a, b = A.Method(env_for("a2")), Budget(env_for())
    ia, ib = run(a, (1,)), run(b, (1,))
    assert "row_score" not in a.state() and abs(ia[0]["real_frac"] - ib[0]["real_frac"]) < 1e-12
    print("bert a2 PASS", flush=True)

    a, b = A.Method(env_for("a3")), Budget(env_for())
    b.la_shift = torch.zeros_like(b.la_shift)
    run(a, (1, 2)); run(b, (1, 2))
    exact(b.state(), a.state(), "a3 state")
    print("bert a3 PASS (== budget with zero shift)", flush=True)

    a, b = A.Method(env_for("a4")), Budget(env_for())
    ia, ib = run(a, (1,)), run(b, (1,))
    exact(b.priv_p, a.priv_p, "a4 r1")
    assert ia[0]["bytes_up"] == 1024 and float((a.Q - b.Q).abs().max()) <= 2 / 255
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
