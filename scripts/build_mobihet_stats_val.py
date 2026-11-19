"""Build private W&B-free train->val control vs proxy-free MobiHet-stats notebook."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OWNER = "hoangdang1711"
SLUG = "mobihet-v4-stats-val-r10"


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", OWNER,
                    "--slug", SLUG, "--method", "mobihet_stats", "--no-wandb", "--max-hours", "11"], check=True)
    path = ROOT / "notebooks" / SLUG / f"{SLUG}.ipynb"
    nb = nbformat.read(path, as_version=4)
    paths = sorted([*ROOT.glob("core/*.py"), *ROOT.glob("methods/*.py"), *ROOT.glob("tests/*.py"),
                    *ROOT.glob("knowledge/*.json"), ROOT / "scripts/run_fl.py"])
    source = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths}
    hashes = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in source.items()}
    nb.cells[0].source = f'''# {SLUG}

Train-only → full-val cost/utility screening; official test is not decoded or read. Matched arms use the 50-round cosine
horizon and stop after r10: baseline `mobihet`, then proxy-free `mobihet_stats`. Candidate exchanges pooled per-class
count/mean/covariance once at bootstrap; each private model trains on local real rows plus global Gaussian CE every round.
No trainable proxy, teacher KD, Q3 update, or per-round payload. W&B disabled.

Gate locked before execution: private macro/client-mean/p10 deltas >= -0.005, class13 delta >= 0,
mean train time r3-r10 reduced >=30%, and bytes_up=bytes_down=0 every candidate round.'''
    nb.cells[2].source = f'''from pathlib import Path
import hashlib, json
source = {source!r}
hashes = {hashes!r}
for name, content in source.items():
    p = Path("/kaggle/temp/src") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[name]
Path("/kaggle/working/source_manifest.json").write_text(json.dumps(hashes, indent=2))
Path("/kaggle/working/source_snapshot.json").write_text(json.dumps(source))
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_methods.py", "tests/test_mobihet_resume.py",
                "-k", "mobihet_stats"], cwd="/kaggle/temp/src", check=True,
               env={{**os.environ, "OMP_NUM_THREADS": "2", "PYTHONPATH": "/kaggle/temp/src"}})
'''
    nb.cells[4].source = '''t = time.monotonic()
sys.path.insert(0, "/kaggle/temp/src")
from core import data as D
D.build_npy(D.find_root(), "/kaggle/temp/v4npy", splits=("train",))
assert not Path("/kaggle/temp/v4npy/test_X.npy").exists()
print(f"train-only npy cache built in {time.monotonic() - t:.0f}s", flush=True)
'''
    smoke = '''os.environ.update(V4_NPY="/kaggle/temp/v4npy", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                  TORCHINDUCTOR_COMPILE_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  WANDB_MODE="disabled")
import numpy as np, torch
F = np.load("/kaggle/temp/v4npy/train_F.npy")
assert set(np.unique(F)) == {0, 1} and not Path("/kaggle/temp/v4npy/test_X.npy").exists()
base = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_fl.py", "--method", "mobihet_stats",
        "--out", "/kaggle/working/smoke-stats", "--width", "1", "--rounds", "50", "--subset", "32",
        "--compile", "--cache", "", "--train-fold", "train", "--eval-fold", "val"]
for stop in (1, 2):
    subprocess.run(base + ["--stop-after", str(stop)], cwd="/kaggle/temp/src", check=True)
rows = [json.loads(s) for s in Path("/kaggle/working/smoke-stats/rounds.jsonl").read_text().splitlines()]
assert [r["round"] for r in rows] == [1, 2]
assert all(r["applied_steps"] > 0 and r["skipped_frac"] < 0.01 for r in rows)
assert all(r["bytes_up"] == 0 and r["bytes_down"] == 0 for r in rows)
for rank in (0, 1):
    p = Path("/kaggle/working/smoke-stats") / ("state.pt" if rank == 0 else "state_rank1.pt")
    state = torch.load(p, map_location="cpu", weights_only=True)
    assert state["round"] == 2
    assert set(state["method_state"]) == {"priv_p", "priv_b", "version", "gen_g"}
    if rank == 0:
        assert state["method"] == "mobihet_stats"
Path("/kaggle/working/SMOKE_PASS.json").write_text(json.dumps({"rounds": 2, "clients": 32,
                                                               "resume": True, "train": "train", "eval": "val",
                                                               "proxy": False, "round_bytes": 0}))
print("STATS SMOKE PASS", flush=True)
'''
    nb.cells.insert(5, nbformat.v4.new_code_cell(smoke))
    production = '''common = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_fl.py", "--width", "1",
          "--rounds", "50", "--stop-after", "10", "--compile", "--cache", "",
          "--train-fold", "train", "--eval-fold", "val"]
arms = [("mobihet", "mobihet-v4-control-stats-val-r10"), ("mobihet_stats", "mobihet-v4-stats-val-r10")]
for method, run in arms:
    cmd = common + ["--method", method, "--out", f"/kaggle/working/{run}"]
    print(cmd, flush=True)
    subprocess.run(cmd, cwd="/kaggle/temp/src", check=True)
    rows = [json.loads(s) for s in Path(f"/kaggle/working/{run}/rounds.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rows] == list(range(1, 11))
    assert all(r["applied_steps"] > 0 and r["skipped_frac"] < 0.01 for r in rows)
    if method == "mobihet":
        for p in Path(f"/kaggle/working/{run}").glob("state*.pt"):
            p.unlink()
'''
    nb.cells[6].source = production
    gate = '''control = [json.loads(s) for s in Path("/kaggle/working/mobihet-v4-control-stats-val-r10/rounds.jsonl").read_text().splitlines()]
candidate = [json.loads(s) for s in Path("/kaggle/working/mobihet-v4-stats-val-r10/rounds.jsonl").read_text().splitlines()]
c, s = control[-1], candidate[-1]
deltas = {"class13_f1": s["private/f1_per_class"][13] - c["private/f1_per_class"][13],
          "client_f1_mean": s["private/client_f1_macro_mean"] - c["private/client_f1_macro_mean"],
          "macro_f1": s["private/f1_macro"] - c["private/f1_macro"],
          "client_p10": s["private/client_f1_macro_p10"] - c["private/client_f1_macro_p10"]}
control_time = sum(r["t_train"] for r in control[2:]) / len(control[2:])
candidate_time = sum(r["t_train"] for r in candidate[2:]) / len(candidate[2:])
time_reduction = 1 - candidate_time / control_time
zero_round_bytes = all(r["bytes_up"] == 0 and r["bytes_down"] == 0 for r in candidate)
gate = {"round": 10, "split": "train_to_val", "deltas": deltas,
        "mean_train_seconds_r3_r10": {"control": control_time, "candidate": candidate_time},
        "time_reduction": time_reduction, "zero_round_bytes": zero_round_bytes,
        "pass": deltas["macro_f1"] >= -0.005 and deltas["client_f1_mean"] >= -0.005
                and deltas["client_p10"] >= -0.005 and deltas["class13_f1"] >= 0
                and time_reduction >= 0.30 and zero_round_bytes,
        "rule": "macro/mean/p10 >=-0.005; class13 >=0; train time -30%; per-round bytes=0"}
Path("/kaggle/working/STATS_GATE.json").write_text(json.dumps(gate, indent=2))
print("STATS GATE", json.dumps(gate), flush=True)
'''
    nb.cells.insert(7, nbformat.v4.new_code_cell(gate))
    nb.cells[-1].source = '''for run in ("mobihet-v4-control-stats-val-r10", "mobihet-v4-stats-val-r10"):
    p = Path(f"/kaggle/working/{run}/rounds.jsonl")
    print(run, p.read_text()[-3000:], flush=True)
print(Path("/kaggle/working/STATS_GATE.json").read_text(), flush=True)
'''
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source)
            if index != 2:
                assert "import wandb" not in cell.source and "--wandb " not in cell.source
    nbformat.validate(nb)
    nbformat.write(nb, path)
    meta = json.loads((path.parent / "kernel-metadata.json").read_text())
    assert meta["is_private"] and meta["enable_gpu"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == ["odixe0502/veremi-nextgen-v4-scaled-per-vehicle"]
    assert not meta["kernel_sources"]
    print(path)
    print("source files:", len(source), "sha256:", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
