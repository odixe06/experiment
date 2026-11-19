"""Build private W&B-free final MobiHet-stats train+val -> official-test notebook."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat


ROOT = Path(__file__).resolve().parents[1]
OWNER = "hoangdang1711"
SLUG = "mobihet-v4-stats-final-r50"


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", OWNER,
                    "--slug", SLUG, "--method", "mobihet_stats", "--no-wandb", "--max-hours", "11"], check=True)
    path = ROOT / "notebooks" / SLUG / f"{SLUG}.ipynb"
    nb = nbformat.read(path, as_version=4)
    paths = sorted([*ROOT.glob("core/*.py"), *ROOT.glob("methods/*.py"), *ROOT.glob("tests/*.py"),
                    *ROOT.glob("knowledge/*.json"), ROOT / "scripts/run_fl.py"])
    source = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths}
    hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()}
    nb.cells[0].source = f'''# {SLUG}

Final locked run after audited train→val gate PASS in `hoangdang1711/mobihet-v4-stats-val-r10/1`.
Train on train+val, evaluate official full test every round, 50-round shared cosine schedule. Method is private-only
`mobihet_stats`: one-time pooled class count/mean/covariance bootstrap, local real CE plus pooled Gaussian CE.
No trainable proxy, teacher KD, Q3 update, per-round payload, W&B, or validation-selected hyperparameter change.'''
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
D.build_npy(D.find_root(), "/kaggle/temp/v4npy", splits=("train", "test"))
import numpy as np
F = np.load("/kaggle/temp/v4npy/train_F.npy")
assert set(np.unique(F)) == {0, 1}
assert Path("/kaggle/temp/v4npy/test_X.npy").exists()
print(f"train+test npy cache built in {time.monotonic() - t:.0f}s", flush=True)
'''
    smoke = '''os.environ.update(V4_NPY="/kaggle/temp/v4npy", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                  TORCHINDUCTOR_COMPILE_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  WANDB_MODE="disabled")
import json, torch
base = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_fl.py", "--method", "mobihet_stats",
        "--out", "/kaggle/working/smoke-stats-final", "--width", "1", "--rounds", "50", "--subset", "32",
        "--compile", "--cache", ""]
for stop in (1, 2):
    subprocess.run(base + ["--stop-after", str(stop)], cwd="/kaggle/temp/src", check=True)
rows = [json.loads(s) for s in Path("/kaggle/working/smoke-stats-final/rounds.jsonl").read_text().splitlines()]
assert [r["round"] for r in rows] == [1, 2]
assert all(r["applied_steps"] > 0 and r["skipped_frac"] < 0.01 for r in rows)
assert all(r["bytes_up"] == 0 and r["bytes_down"] == 0 for r in rows)
for rank in (0, 1):
    p = Path("/kaggle/working/smoke-stats-final") / ("state.pt" if rank == 0 else "state_rank1.pt")
    state = torch.load(p, map_location="cpu", weights_only=True)
    assert state["round"] == 2
    assert set(state["method_state"]) == {"priv_p", "priv_b", "version", "gen_g"}
Path("/kaggle/working/SMOKE_PASS.json").write_text(json.dumps(
    {"rounds": 2, "clients": 32, "resume": True, "train": "train+val", "eval": "test",
     "proxy": False, "round_bytes": 0}))
print("STATS FINAL SMOKE PASS", flush=True)
'''
    nb.cells.insert(5, nbformat.v4.new_code_cell(smoke))
    nb.cells[6].source = '''run = "/kaggle/working/mobihet-v4-stats-final-r50"
cmd = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_fl.py", "--width", "1",
       "--rounds", "50", "--method", "mobihet_stats", "--out", run, "--compile", "--cache", ""]
print(cmd, flush=True)
subprocess.run(cmd, cwd="/kaggle/temp/src", check=True)
rows = [json.loads(s) for s in Path(run, "rounds.jsonl").read_text().splitlines()]
assert [r["round"] for r in rows] == list(range(1, 51))
assert all(r["applied_steps"] > 0 and r["skipped_frac"] < 0.01 for r in rows)
assert all(r["bytes_up"] == 0 and r["bytes_down"] == 0 for r in rows)
summary = {"round": 50, "train": "train+val", "eval": "official_test", "proxy": False,
           "per_round_bytes": 0, "val_gate_kernel": "hoangdang1711/mobihet-v4-stats-val-r10/1",
           "final": rows[-1]}
Path("/kaggle/working/FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2))
'''
    nb.cells[-1].source = '''p = Path("/kaggle/working/mobihet-v4-stats-final-r50/rounds.jsonl")
print(p.read_text()[-5000:], flush=True)
print(Path("/kaggle/working/FINAL_SUMMARY.json").read_text(), flush=True)
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
