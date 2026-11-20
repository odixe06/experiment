"""Build one private 2xT4 notebook: MobiHet L4 on one mixed-model case (scripts/run_mixed_l4.py), one seed.

Embedded source as scripts/build_l4_fl.py; data from a verified or freshly built v4 NPY cache; frozen assignment
experiments/mixed-model-v1. --stop-after 2 is the pilot: same run directory and cosine horizon (50), so a later session
continues it from checkpoint r2 with --parent. A session > 1 copies the parent's run directory from /kaggle/input
(checkpoint.json + checkpoint-rNNN + history) before run_mixed_l4.py resumes it; the config hash refuses any source change.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat

from kaggle_npy_cache import cache_input

ROOT = Path(__file__).resolve().parents[1]
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def code(text):
    ast.parse(text)
    return nbformat.v4.new_code_cell(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--case", required=True, choices=("case1", "case2", "case3"))
    ap.add_argument("--seed", required=True, type=int, choices=(41, 42, 43))
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--parent", default="", help="<owner>/<slug> of the previous session (session > 1)")
    sources = ap.add_mutually_exclusive_group()
    sources.add_argument("--parent-artifacts", type=Path, help="audited parent pull with frozen source and checkpoint receipt")
    sources.add_argument("--source-artifacts", type=Path, help="reuse a pilot source snapshot for a new seed")
    ap.add_argument("--npy-cache", type=Path, help="preflight receipt for a verified full v4 NPY cache")
    ap.add_argument("--stop-after", type=int, default=50)
    ap.add_argument("--max-hours", type=float, default=11.5)
    args = ap.parse_args()
    if (args.session > 1) != bool(args.parent):
        ap.error("--parent is required exactly when --session > 1")
    if args.npy_cache and not args.parent_artifacts:
        ap.error("--npy-cache requires audited parent artifacts")
    run = f"mixl4-{args.case}-s{args.seed}"                         # run directory: the same in every session
    slug = run if args.session == 1 else f"{run}-h{args.session}"
    paths = sorted([*ROOT.glob("core/*.py"), *ROOT.glob("methods/*.py"), *ROOT.glob("knowledge/*.json"),
                    *ROOT.glob("experiments/mixed-model-v1/*"), ROOT/"tests/test_mixed_l4.py",
                    ROOT/"scripts/run_mixed_l4.py", ROOT/"scripts/build_l4_fl.py", ROOT/"scripts/run_fl.py",
                    ROOT/"scripts/run_p21_recovery.py"])
    source = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths}
    parent_receipt = None
    parent_sizes = {}
    if args.parent_artifacts:
        if not args.parent:
            ap.error("--parent-artifacts requires --parent")
        pull = args.parent_artifacts
        audited = json.loads((pull / "recovery-audit.json").read_text())
        parent_config = json.loads((pull / run / "config.json").read_text())
        parent_receipt = json.loads((pull / run / "checkpoint.json").read_text())
        if (audited["pass"] is not True or parent_config["case"] != args.case or
                parent_config["hp"]["seed"] != args.seed or parent_receipt["round"] != audited["last_round"] or
                parent_receipt["config_sha256"] != audited["config_sha256"] or
                not 1 <= parent_receipt["round"] < args.stop_after):
            ap.error("parent audit, checkpoint, or requested continuation disagree")
        parent_sizes = {Path(name).name: int(rec["bytes"]) for name, rec in audited["checkpoint_remote"].items()}
        if set(parent_sizes) != {"state_rank0.pt", "state_rank1.pt"} or min(parent_sizes.values()) <= 0:
            ap.error("parent audit must cover two checkpoint ranks")
        source = json.loads((pull / "source_snapshot.json").read_text())
    if args.source_artifacts:
        source = json.loads((args.source_artifacts / "source_snapshot.json").read_text())
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in source.items()}
    frozen = args.parent_artifacts or args.source_artifacts
    if frozen and hashes != json.loads((frozen / "source_manifest.json").read_text()):
        ap.error("frozen parent source and manifest disagree")
    manifest = {"experiment": f"mixed-{args.case}-mobihet-l4", "method": "mobihet_l4", "case": args.case, "seed": args.seed,
                "rounds": 50, "stop_after": args.stop_after, "session": args.session, "parent": args.parent or None,
                "run": run, "dataset": DATASET, "client_partition": "fixed-all-5343",
                "assignment": "experiments/mixed-model-v1", "model": "core.mixed_backbones.config(backbone) (MixedAdapter)"}

    setup = '''import os, subprocess, sys, time, json, hashlib, glob, shutil
from pathlib import Path
T0 = time.monotonic()
import torch
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
assert all(torch.cuda.get_device_name(i) == "Tesla T4" for i in range(2))
assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(2))
print(torch.__version__, sys.version, [torch.cuda.get_device_name(i) for i in range(2)], flush=True)
'''
    embed = f'''source = {source!r}
hashes = {hashes!r}
for name, content in source.items():
    path = Path("/kaggle/temp/src") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]
Path("/kaggle/working/source_snapshot.json").write_text(json.dumps(source))
Path("/kaggle/working/source_manifest.json").write_text(json.dumps(hashes, indent=2))
Path("/kaggle/working/experiment_manifest.json").write_text(json.dumps({manifest!r}, indent=2))
subprocess.run([sys.executable, "tests/test_mixed_l4.py", "--toy"], cwd="/kaggle/temp/src", check=True,
               env={{**os.environ, "PYTHONPATH": "/kaggle/temp/src", "OMP_NUM_THREADS": "2"}})
'''
    cache = '''sys.path.insert(0, "/kaggle/temp/src")
from core import data as D
t = time.monotonic()
mounted_npy = Path("/kaggle/temp/v4npy")
D.build_npy(D.find_root(), str(mounted_npy), splits=("train", "test"))
print(f"NPY cache {time.monotonic()-t:.1f}s", flush=True)
'''
    kernel_sources = [args.parent] if args.parent else []
    if args.npy_cache:
        cache_ref, cache = cache_input(args.npy_cache)
        kernel_sources.append(cache_ref)
    cache += '''os.environ.update(V4_NPY=str(mounted_npy), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                  TORCHINDUCTOR_COMPILE_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  WANDB_MODE="disabled", MOBIHET_RECOVERY_DIR="/kaggle/working/recovery")
'''
    run_dir = f"/kaggle/working/{run}"
    resume = f'''run = Path({run_dir!r})
parent = {args.parent!r}
if parent:
    found = [Path(p).parent for p in glob.glob("/kaggle/input/**/{run}/checkpoint.json", recursive=True)]
    assert len(found) == 1, ("parent run directory", found)
    upstream = found[0]
    cp = json.loads((upstream / "checkpoint.json").read_text())
    expected = {parent_receipt!r}
    if expected:
        assert cp == expected, ("wrong parent checkpoint", cp)
        assert json.loads((upstream.parent / "source_manifest.json").read_text()) == hashes
        for name, size in {parent_sizes!r}.items():
            saved = upstream / f"checkpoint-r{{cp['round']:03d}}" / name
            assert saved.stat().st_size == size, name
            with saved.open("rb") as handle:
                assert handle.read(4) == bytes.fromhex("504b0304"), name
    shutil.copytree(upstream, run, copy_function=shutil.copyfile)
    rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines()]
    assert [x["round"] for x in rows][:cp["round"]] == list(range(1, cp["round"] + 1)), "parent rows"
    for rank in (0, 1):
        assert (run / f"checkpoint-r{{cp['round']:03d}}" / f"state_rank{{rank}}.pt").exists(), rank
    print(f"resume from {{found[0]}} at round {{cp['round']}}", flush=True)
else:
    assert not run.exists()
'''
    base = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_mixed_l4.py", "--case", args.case,
            "--seed", str(args.seed), "--out", run_dir, "--compile", "--stop-after", str(args.stop_after)]
    train = f'''cmd = {base!r} + ["--max-hours", f"{{{args.max_hours}-(time.monotonic()-T0)/3600-0.4:.3f}}"]
print(cmd, flush=True)
subprocess.run(cmd, cwd="/kaggle/temp/src", check=True)
status = json.loads((run / "RUN_STATUS.json").read_text())
rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines()]
assert [x["round"] for x in rows] == list(range(1, rows[-1]["round"] + 1)) and rows[-1]["round"] == status["last_round"]
status.update(session={args.session}, stop_after={args.stop_after})
Path("/kaggle/working/RUN_STATUS.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
'''
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(
        f"# {slug}\n\nMixed {args.case} · MobiHet L4 · seed {args.seed} · session {args.session} · stop after round "
        f"{args.stop_after} · all 5,343 clients."), code(setup), code(embed), code(resume), code(cache), code(train)]
    nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                        "language_info": {"name": "python", "version": "3.12"},
                        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                                   "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}})
    out = ROOT / "notebooks" / slug
    out.mkdir(parents=True, exist_ok=True)
    notebook = out / f"{slug}.ipynb"
    nbformat.validate(nb)
    nbformat.write(nb, notebook)
    meta = {"id": f"{args.owner}/{slug}", "title": slug, "code_file": notebook.name, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4", "docker_image": IMAGE, "dataset_sources": [DATASET],
            "kernel_sources": kernel_sources, "competition_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(notebook)
    print("source", len(source), "sha256", hashlib.sha256(notebook.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
