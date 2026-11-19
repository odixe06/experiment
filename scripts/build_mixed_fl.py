"""Build private mixed-model pilot or production notebook, never launch implicitly."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.model_assignment import CASES, load_assignment

METHODS = ("fedproto", "fedtgp", "tinyproto", "v2x_htfl", "pfedes")
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def build(case, method, seed, owner, pilot=False, session_hours=10.5, revision="a1"):
    mapping = load_assignment(ROOT/"experiments/mixed-model-v1", case)
    assert revision in ("a1", "a2")
    slug = f"mix128-{case}-{method.replace('_', '-')}-s{seed}" + ("-probe" if pilot else "-"+revision)
    paths = sorted([*ROOT.glob("core/*.py"), *ROOT.glob("methods/*.py"), *ROOT.glob("knowledge/*.json"),
                    *ROOT.glob("experiments/mixed-model-v1/*"), ROOT/"scripts/run_mixed_fl.py",
                    ROOT/"tests/test_mixed_backbones.py", ROOT/"tests/test_mixed_methods.py",
                    ROOT/"tests/test_mixed_precision.py", ROOT/"tests/test_mixed_bert_precision.py"])
    source = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths if p.is_file()}
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in source.items()}
    manifest = dict(case=case, method=method, seed=seed, assignment_seed=2026,
                    assignment_sha256=hashes[f"experiments/mixed-model-v1/{case}.csv"],
                    representation_dim=None if method == "pfedes" else 128,
                    rounds=50, pilot=pilot, clients=len(mapping), dataset=DATASET,
                    proposal=False, protocol="mixed128-v1", revision=revision, source_hashes=hashes)
    manifest["training_precision"] = {name: "fp32" if method == "v2x_htfl" and name in ("sae", "bert") else "amp-fp16"
                                      for name in set(mapping.values())}
    setup = '''import os, sys, json, time, hashlib, subprocess, shutil
from pathlib import Path
T0 = time.monotonic()
os.environ.update(OMP_NUM_THREADS="2", TORCHINDUCTOR_COMPILE_THREADS="1",
                  PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", WANDB_MODE="disabled")
import torch
assert torch.cuda.device_count() == 2
assert all(torch.cuda.get_device_capability(i) == (7, 5) and "T4" in torch.cuda.get_device_name(i) for i in range(2))
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(2)], flush=True)
'''
    embed = f'''source = {source!r}
hashes = {hashes!r}
for name, body in source.items():
    assert hashlib.sha256(body.encode()).hexdigest() == hashes[name]
    path = Path("/kaggle/temp/src")/name
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(body)
Path("/kaggle/working/source_snapshot.json").write_text(json.dumps(source))
Path("/kaggle/working/source_manifest.json").write_text(json.dumps(hashes, indent=2))
Path("/kaggle/working/experiment_manifest.json").write_text(json.dumps({manifest!r}, indent=2))
os.environ["PYTHONPATH"] = "/kaggle/temp/src"
for test in ("tests/test_mixed_backbones.py", "tests/test_mixed_methods.py"):
    subprocess.run([sys.executable, test], cwd="/kaggle/temp/src", check=True)
if {revision == 'a2'!r}:
    subprocess.run([sys.executable, "tests/test_mixed_precision.py"], cwd="/kaggle/temp/src", check=True)
    if {method == 'v2x_htfl' and 'bert' in set(mapping.values())!r}:
        subprocess.run([sys.executable, "tests/test_mixed_bert_precision.py"], cwd="/kaggle/temp/src", check=True)
sys.path.insert(0, "/kaggle/temp/src")
from core import data as D
D.build_npy(D.find_root(), "/kaggle/temp/v4npy", splits=("train", "test"))
os.environ["V4_NPY"] = "/kaggle/temp/v4npy"
'''
    base = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_mixed_fl.py",
            "--case", case, "--method", method, "--seed", str(seed), "--compile"]
    smoke = f'''base = {base!r}
smoke = Path("/kaggle/working/smoke")
for stop in (1, 2):
    subprocess.run(base + ["--out", str(smoke), "--smoke-clients", "30", "--smoke-rows", "128",
                          "--stop-after", str(stop), "--max-hours", "2"], cwd="/kaggle/temp/src", check=True)
rows = [json.loads(line) for line in (smoke/"rounds.jsonl").read_text().splitlines()]
assert [row["round"] for row in rows] == [1, 2]
assert all(row["applied_steps"] > 0 and row["skipped_frac"] < .01 for row in rows)
Path("/kaggle/working/SMOKE_PASS.json").write_text(json.dumps({{"case": {case!r}, "method": {method!r},
    "seed": {seed}, "rounds": 2, "resume": True, "clients": 30, "max_rows_per_client": 128,
    "assignment_sha256": {manifest['assignment_sha256']!r}, "last": rows[-1]}}, indent=2))
print("MIXED GPU SMOKE PASS", flush=True)
'''
    cells = [nbformat.v4.new_markdown_cell(f"# {slug}\n\nFive-baseline mixed-model protocol; unchanged vehicle partition. "
             "Prototype methods use learned128D adapters; native output layers retained with zero gradient. "
             "pFedES keeps native classifiers and shared77→77proxy. TabM keeps four members. "
             "Diagnostic row caps apply only to smoke; production uses every training row."), setup, embed, smoke]
    if not pilot:
        cells.append(f'''remaining = {session_hours!r} - (time.monotonic()-T0)/3600 - .5
assert remaining > .5, "Insufficient session budget after smoke"
out = Path("/kaggle/working/{slug}")
subprocess.run(base + ["--out", str(out), "--max-hours", str(remaining)], cwd="/kaggle/temp/src", check=True)
status = json.loads((out/"RUN_STATUS.json").read_text())
Path("/kaggle/working/RUN_STATUS.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
''')
    code_cells = []
    for cell in cells:
        if isinstance(cell, str):
            ast.parse(cell); code_cells.append(nbformat.v4.new_code_cell(cell))
        else: code_cells.append(cell)
    nb = nbformat.v4.new_notebook(cells=code_cells)
    nb.metadata.update(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                       language_info=dict(name="python", version="3.12"),
                       kaggle=dict(accelerator="nvidiaTeslaT4", isGpuEnabled=True, isInternetEnabled=False,
                                   language="python", sourceType="notebook"))
    directory = ROOT/"notebooks"/slug
    directory.mkdir(parents=True, exist_ok=True)
    nbformat.validate(nb)
    nbformat.write(nb, directory/f"{slug}.ipynb")
    metadata = dict(id=f"{owner}/{slug}", title=slug, code_file=f"{slug}.ipynb", language="python",
                    kernel_type="notebook", is_private=True, enable_gpu=True, enable_internet=False,
                    machine_shape="NvidiaTeslaT4", docker_image=IMAGE, dataset_sources=[DATASET],
                    kernel_sources=[], competition_sources=[])
    (directory/"kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return dict(case=case, method=method, seed=seed, pilot=pilot, owner=owner, slug=slug,
                directory=str(directory), notebook_sha256=hashlib.sha256((directory/f"{slug}.ipynb").read_bytes()).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--session-hours", type=float, default=10.5)
    p.add_argument("--revision", choices=("a1", "a2"), default="a1")
    a = p.parse_args()
    print(json.dumps(build(a.case, a.method, a.seed, a.owner, a.pilot, a.session_hours, a.revision), indent=2))
