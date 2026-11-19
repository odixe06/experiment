"""Build one private 2xT4 TabM/SAE FL notebook with embedded dirty-worktree source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
METHODS = {"fedavg", "fedprox", "afpha", "fd_ids", "apfed", "lwfednids", "canperfl", "tinyproto",
           "pfedes", "nilm", "fedtgp", "perfedskd", "fedproto", "v2x_htfl"}


def code(text):
    ast.parse(text)
    return nbformat.v4.new_code_cell(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--backbone", choices=("tabm", "sae", "mlp", "cnn-bigru-attention"), default="tabm")
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--seed", required=True, type=int, choices=(41, 42, 43))
    ap.add_argument("--slug", default="")
    ap.add_argument("--parent", default="", help="FedAvg kernel source for CANPerFL")
    ap.add_argument("--max-hours", type=float, default=11.0)
    args = ap.parse_args()
    if (args.method == "canperfl") != bool(args.parent):
        ap.error("--parent is required only for canperfl")
    model = {"tabm": {"display": "TabM", "parameters": 6548, "test": "tests/test_tabm_fl.py"},
             "sae": {"display": "SAE", "parameters": 7613, "test": "tests/test_sae_fl.py"},
             "mlp": {"display": "MLP", "parameters": 7600, "test": "tests/test_dense_recurrent_fl.py"},
             "cnn-bigru-attention": {"display": "CNN-BiGRU-Attention", "parameters": 118288,
                                      "test": "tests/test_dense_recurrent_fl.py"}}[args.backbone]
    slug = args.slug or f"{args.backbone}-fl-{args.method.replace('_', '-')}-s{args.seed}"
    run = slug
    paths = sorted([*ROOT.glob("core/*.py"), *ROOT.glob("methods/*.py"),
                    ROOT / model["test"], ROOT / "tests/test_sae_fl.py", ROOT / "tests/test_aligned_fl.py", ROOT / "scripts/run_fl.py",
                    *ROOT.glob("knowledge/*.json")])
    source = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths}
    if args.method == 'v2x_htfl':
        source['tests/test_v2x_precision.py'] = (ROOT / 'tests/test_v2x_precision.py').read_text(encoding='utf-8')
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in source.items()}
    manifest = {"experiment": f"single-backbone-{args.backbone}-baselines", "method": args.method, "seed": args.seed,
                "backbone": model["display"], "parameters": model["parameters"], "rounds": 10 if args.method == "canperfl" else 50,
                "parent": args.parent or None, "dataset": DATASET, "client_partition": "fixed-all-5343",
                "seed_scope": "initialization-dropout-shuffle", "centralized_warm_start": False,
                "protocol_revision": "dagsnet-aligned-v2", "tinyproto_mu": 0.00015,
                "sae_variant": "ce-preclassifier93" if args.backbone == "sae" else None}

    setup = '''import os, subprocess, sys, time, json, hashlib, glob, shutil
from pathlib import Path
T0 = time.monotonic()
import torch
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
assert all(torch.cuda.get_device_name(i) == "Tesla T4" for i in range(2))
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
subprocess.run([sys.executable, {model['test']!r}], cwd="/kaggle/temp/src", check=True,
               env={{**os.environ, "PYTHONPATH": "/kaggle/temp/src", "OMP_NUM_THREADS": "2"}})
subprocess.run([sys.executable, "tests/test_aligned_fl.py"], cwd="/kaggle/temp/src", check=True,
               env={{**os.environ, "PYTHONPATH": "/kaggle/temp/src", "OMP_NUM_THREADS": "2"}})
'''
    cache = '''sys.path.insert(0, "/kaggle/temp/src")
from core import data as D
t = time.monotonic()
D.build_npy(D.find_root(), "/kaggle/temp/v4npy", splits=("train", "test"))
print(f"NPY cache {time.monotonic()-t:.1f}s", flush=True)
os.environ.update(V4_NPY="/kaggle/temp/v4npy", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                  TORCHINDUCTOR_COMPILE_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  WANDB_MODE="disabled")
'''
    if args.method == 'v2x_htfl':
        embed += '''subprocess.run([sys.executable, "tests/test_v2x_precision.py"], cwd="/kaggle/temp/src", check=True,
               env={**os.environ, "PYTHONPATH": "/kaggle/temp/src"})
'''
    base = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_fl.py", "--method", args.method,
            "--backbone", args.backbone, "--out", f"/kaggle/working/{run}", "--compile", "--cache", ""]
    if args.method == "canperfl":
        parent_slug = args.parent.rstrip("/").split("/")[-1]
        base += ["--rounds", "10", "--set", f"seed={args.seed}",
                 'init=' + json.dumps(f"/kaggle/input/**/{parent_slug}/state.pt")]
    else:
        base += ["--rounds", "50", "--set", f"seed={args.seed}"]
    if args.method == "tinyproto":
        base += ["mu=0.00015"]
    smoke_base = list(base)
    smoke_base[smoke_base.index(f"/kaggle/working/{run}")] = "/kaggle/working/smoke"
    smoke_base += ["--subset", "32"]
    smoke = f'''base = {smoke_base!r}
for stop in (1, 2):
    subprocess.run(base + ["--stop-after", str(stop)], cwd="/kaggle/temp/src", check=True)
rows = [json.loads(x) for x in Path("/kaggle/working/smoke/rounds.jsonl").read_text().splitlines()]
assert [x["round"] for x in rows] == [1, 2]
assert all(x["skipped_frac"] < 0.01 and x["applied_steps"] > 0 for x in rows)
Path("/kaggle/working/SMOKE_PASS.json").write_text(json.dumps({{"method": {args.method!r}, "seed": {args.seed},
    "rounds": 2, "resume": True, "backbone": {args.backbone!r}, "parameters": {model['parameters']}}}))
print({(model['display'] + ' PRODUCTION SMOKE PASS')!r}, flush=True)
'''
    train = f'''run = Path("/kaggle/working/{run}")
cmd = {base!r} + ["--save-every", "10", "--max-hours", f"{{{args.max_hours}-(time.monotonic()-T0)/3600-0.4:.3f}}"]
print(cmd, flush=True)
subprocess.run(cmd, cwd="/kaggle/temp/src", check=True)
rows = [json.loads(x) for x in (run/"rounds.jsonl").read_text().splitlines()]
expected = {10 if args.method == 'canperfl' else 50}
status = {{"method": {args.method!r}, "seed": {args.seed}, "last_round": rows[-1]["round"],
          "target_round": expected, "complete": rows[-1]["round"] == expected,
          "all_round_metrics": len(rows) == rows[-1]["round"], "final": rows[-1]}}
Path("/kaggle/working/RUN_STATUS.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
'''
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(
        f"# {slug}\n\n{model['display']} single-backbone FL · {args.method} · seed {args.seed} · all 5,343 clients."),
        code(setup), code(embed), code(cache), code(smoke), code(train)]
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
            "kernel_sources": [args.parent] if args.parent else [], "competition_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(notebook)
    print("source", len(source), "sha256", hashlib.sha256(notebook.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
