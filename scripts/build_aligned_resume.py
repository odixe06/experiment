"""Build one private checkpoint-resume notebook for an incomplete aligned-v2 direct run."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--backbone", required=True, choices=("mlp", "cnn-bigru-attention"))
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", required=True, type=int, choices=(41, 42, 43))
    ap.add_argument("--from-round", required=True, type=int)
    ap.add_argument("--max-hours", required=True, type=float)
    a = ap.parse_args()
    assert a.parent.split("/", 1)[0] == a.owner, "private parent must use same owner"
    run_name = a.parent.split("/", 1)[1]
    manifest = {"experiment": f"single-backbone-{a.backbone}-baselines", "method": a.method,
                "seed": a.seed, "backbone": a.backbone, "rounds": 50, "parent": a.parent,
                "resume_from_round": a.from_round, "dataset": DATASET,
                "protocol_revision": "dagsnet-aligned-v2"}
    setup = '''import glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
T0 = time.monotonic()
import torch
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
assert all(torch.cuda.get_device_name(i) == "Tesla T4" for i in range(2))
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(2)], flush=True)
'''
    restore_source = f'''snapshots = list(Path("/kaggle/input").glob("**/source_snapshot.json"))
manifests = list(Path("/kaggle/input").glob("**/source_manifest.json"))
assert len(snapshots) == len(manifests) == 1, (snapshots, manifests)
source, hashes = json.loads(snapshots[0].read_text()), json.loads(manifests[0].read_text())
assert source.keys() == hashes.keys()
for name, content in source.items():
    assert hashlib.sha256(content.encode()).hexdigest() == hashes[name], name
    path = Path("/kaggle/temp/src") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
Path("/kaggle/working/source_snapshot.json").write_text(json.dumps(source))
Path("/kaggle/working/source_manifest.json").write_text(json.dumps(hashes, indent=2))
Path("/kaggle/working/experiment_manifest.json").write_text(json.dumps({manifest!r}, indent=2))
'''
    restore_state = f'''run_name = {run_name!r}
candidates = [p for p in Path("/kaggle/input").glob(f"**/{{run_name}}/state.pt")
              if p.parent.name == run_name]
assert len(candidates) == 1, candidates
src, dst = candidates[0].parent, Path("/kaggle/working") / run_name
dst.mkdir(parents=True, exist_ok=True)
for name in ("state.pt", "state_rank1.pt", "rounds.jsonl"):
    inp, out = src / name, dst / name
    assert inp.is_file(), inp
    part = out.with_name(out.name + ".part")
    shutil.copyfile(inp, part)
    assert part.stat().st_size == inp.stat().st_size
    os.replace(part, out)
state = torch.load(dst / "state.pt", map_location="cpu", weights_only=True)
rank1 = torch.load(dst / "state_rank1.pt", map_location="cpu", weights_only=True)
assert state["round"] == rank1["round"] == {a.from_round}
assert state["method"] == {a.method!r} and state["hp"]["seed"] == {a.seed}
rows = [json.loads(line) for line in (dst / "rounds.jsonl").read_text().splitlines() if line.strip()]
assert [r["round"] for r in rows] == list(range(1, {a.from_round + 1}))
Path("/kaggle/working/SMOKE_PASS.json").write_text(json.dumps({{"resume": True,
    "parent": {a.parent!r}, "from_round": {a.from_round}}}))
del state, rank1
print("RESUME PREFLIGHT PASS", run_name, {a.from_round}, flush=True)
'''
    cache = '''sys.path.insert(0, "/kaggle/temp/src")
from core import data as D
D.build_npy(D.find_root(), "/kaggle/temp/v4npy", splits=("train", "test"))
os.environ.update(V4_NPY="/kaggle/temp/v4npy", WANDB_MODE="disabled", PYTHONUNBUFFERED="1",
                  OMP_NUM_THREADS="2", TORCHINDUCTOR_COMPILE_THREADS="2",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
'''
    train = f'''budget = {a.max_hours} - (time.monotonic() - T0) / 3600 - 0.35
assert budget > 1.0, budget
cmd = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_fl.py",
       "--method", {a.method!r}, "--backbone", {a.backbone!r},
       "--out", "/kaggle/working/{run_name}", "--compile", "--cache", "",
       "--rounds", "50", "--set", "seed={a.seed}", "--save-every", "10",
       "--max-hours", f"{{budget:.3f}}"]
subprocess.run(cmd, cwd="/kaggle/temp/src", check=True)
run = Path("/kaggle/working/{run_name}")
rows = [json.loads(line) for line in (run / "rounds.jsonl").read_text().splitlines() if line.strip()]
status = {{"method": {a.method!r}, "seed": {a.seed}, "last_round": rows[-1]["round"],
          "target_round": 50, "complete": rows[-1]["round"] == 50,
          "all_round_metrics": len(rows) == rows[-1]["round"], "final": rows[-1],
          "resumed_from": {a.parent!r}}}
Path("/kaggle/working/RUN_STATUS.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
'''
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        f"# {a.slug}\n\nResume `{a.parent}` from round {a.from_round} to round 50."),
        *[nbformat.v4.new_code_cell(cell) for cell in
          (setup, restore_source, restore_state, cache, train)]])
    nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                        "language_info": {"name": "python", "version": "3.12"},
                        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                                   "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}})
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    folder = ROOT / "notebooks" / a.slug
    folder.mkdir(parents=True, exist_ok=True)
    notebook = folder / f"{a.slug}.ipynb"
    nbformat.validate(nb)
    nbformat.write(nb, notebook)
    metadata = {"id": f"{a.owner}/{a.slug}", "title": a.slug, "code_file": notebook.name,
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
                "docker_image": IMAGE, "dataset_sources": [DATASET], "kernel_sources": [a.parent],
                "competition_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(notebook)
    print("sha256", hashlib.sha256(notebook.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
