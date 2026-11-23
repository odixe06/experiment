"""Build one private 2xT4 notebook: MobiHet L4 ablation (methods/mobihet_ablation.py) on single-backbone FeatureBERT.

The FeatureBERT L4 chain (flb-fbl-l4-top-s4x-r50, docs/research/mobihet-l4.md) ran fl_veremi_bert 49b2d88 + a 3-file overlay
through scripts/run_t3.py; its frozen source (identical manifest in all six sessions) is embedded here with
methods/mobihet_ablation.py added, and the command is that chain's with --method mobihet_ablation and the arm's hp after
L4_HP + seed. Layout as scripts/build_l4_fl.py: source snapshot + manifest in the output, --stop-after 2 = pilot,
--parent <owner>/<slug> continues the run directory of an earlier session. The v4 NPY cache is built by the snapshot's own
core/data.py, as in those notebooks (its data module differs from this repo's).
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.mobihet_ablation import HP  # noqa: E402
from scripts.build_l4_fl import DATASET, IMAGE, L4_HP  # noqa: E402

SNAPSHOT = ROOT / "runs/recovery-20261120/references/bert-l4-artifacts/flb-fbl-l4-top-s41-r50/provenance"
MANIFEST_SHA256 = "c2db040209be408e"                                    # prefix, all six flb-fbl-l4-top sessions


def code(text):
    ast.parse(text)
    return nbformat.v4.new_code_cell(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--ablation", required=True, choices=sorted(HP))
    ap.add_argument("--seed", required=True, type=int, choices=(41, 42, 43))
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--parent", default="", help="<owner>/<slug> of the previous session (session > 1)")
    ap.add_argument("--stop-after", type=int, default=50)
    ap.add_argument("--max-hours", type=float, default=11.5)
    args = ap.parse_args()
    if (args.session > 1) != bool(args.parent):
        ap.error("--parent is required exactly when --session > 1")
    run = f"bert-abl-{args.ablation}-s{args.seed}"                       # run directory: the same in every session
    slug = run if args.session == 1 else f"{run}-h{args.session}"
    raw = (SNAPSHOT / "source_manifest.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest().startswith(MANIFEST_SHA256), "not the FeatureBERT L4 source"
    frozen = json.loads(raw)
    source = {}
    for name, digest in frozen["files"].items():
        text = (SNAPSHOT / "source" / name).read_text(encoding="utf-8")
        assert hashlib.sha256(text.encode()).hexdigest() == digest, name
        source[name] = text
    assert "methods/mobihet_ablation.py" not in source
    source["methods/mobihet_ablation.py"] = (ROOT / "methods/mobihet_ablation.py").read_text(encoding="utf-8")
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in source.items()}
    hp = L4_HP + [f"seed={args.seed}"] + HP[args.ablation]
    manifest = {"experiment": f"single-backbone-featurebert-mobihet-l4-ablation-{args.ablation}", "method": "mobihet_ablation",
                "ablation": args.ablation, "seed": args.seed, "backbone": "FeatureBERT", "rounds": 50,
                "stop_after": args.stop_after, "session": args.session, "parent": args.parent or None, "run": run, "hp": hp,
                "dataset": DATASET, "client_partition": "fixed-all-5343",
                "l4_source": {"base_sha": frozen["base_sha"], "overlay": frozen["overlay"],
                              "manifest_sha256": hashlib.sha256(raw).hexdigest()}}

    setup = '''import os, subprocess, sys, time, json, hashlib, glob, shutil
from pathlib import Path
T0 = time.monotonic()
import torch
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(2)), "expected Tesla T4"
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
'''
    run_dir = f"/kaggle/working/{run}"
    resume = f'''run = Path({run_dir!r})
parent = {args.parent!r}
if parent:
    found = [Path(p).parent for p in glob.glob("/kaggle/input/**/{run}/state.pt", recursive=True)]
    assert len(found) == 1, ("parent run directory", found)
    assert json.loads((found[0].parent / "source_manifest.json").read_text()) == hashes, "parent source changed"
    s = torch.load(found[0] / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    other = torch.load(found[0] / "state_rank1.pt", map_location="cpu", weights_only=True, mmap=True)
    assert s["method"] == "mobihet_ablation" and s["hp"]["ablation"] == {args.ablation!r} and 0 < s["round"] < 50
    assert other["round"] == s["round"] and len(s["gen"]) == 2, "incomplete rank resume state"
    rows = [json.loads(x) for x in (found[0] / "rounds.jsonl").read_text().splitlines()]
    assert [x["round"] for x in rows] == list(range(1, s["round"] + 1)), "parent rows"
    print(f"resume from {{found[0]}} at round {{s['round']}}", flush=True)
    del s, other
    shutil.copytree(found[0], run, copy_function=shutil.copyfile)
else:
    assert not run.exists()
'''
    cache = '''sys.path.insert(0, "/kaggle/temp/src")
from core import data as D
t = time.monotonic()
mounted_npy = Path("/kaggle/temp/v4npy")
D.build_npy(D.find_root(), str(mounted_npy))
print(f"NPY cache {time.monotonic()-t:.1f}s", flush=True)
'''
    kernel_sources = [args.parent] if args.parent else []
    # environment of the flb-fbl-l4-top notebooks (TorchInductor cache carried across sessions in the run directory)
    cache += '''os.environ.update(TORCHINDUCTOR_COMPILE_THREADS="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", V4_NPY=str(mounted_npy),
                  TORCHINDUCTOR_CACHE_DIR="/kaggle/temp/inductor", TORCHINDUCTOR_FX_GRAPH_CACHE="1")
cache_tar = run / "inductor-cache.tar"
if cache_tar.exists():
    subprocess.run(["tar", "-xf", str(cache_tar), "-C", "/kaggle/temp"], check=False)
'''
    base = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_t3.py", "--method", "mobihet_ablation",
            "--out", run_dir, "--width", "1", "--rounds", "50", "--until", str(args.stop_after), "--save-every", "5",
            "--compile", "--cache", "", "--set", *hp]
    train = f'''cmd = {base!r}
at = cmd.index("--compile") + 1                                        # flag order of the L4 chain's command
cmd[at:at] = ["--max-hours", f"{{{args.max_hours}-(time.monotonic()-T0)/3600-0.4:.3f}}"]
print(cmd, flush=True)
rc = subprocess.run(cmd, cwd="/kaggle/temp/src").returncode
if Path("/kaggle/temp/inductor").exists():
    subprocess.run(["tar", "-cf", str(cache_tar), "-C", "/kaggle/temp", "inductor"], check=False)
assert rc == 0, f"training exited {{rc}}"
rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines()]
assert [x["round"] for x in rows] == list(range(1, rows[-1]["round"] + 1))
status = {{"method": "mobihet_ablation", "ablation": {args.ablation!r}, "backbone": "featurebert", "seed": {args.seed},
          "session": {args.session}, "last_round": rows[-1]["round"], "target_round": 50, "stop_after": {args.stop_after},
          "complete": rows[-1]["round"] == 50, "final": rows[-1]}}
Path("/kaggle/working/RUN_STATUS.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
'''
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(
        f"# {slug}\n\nFeatureBERT single-backbone FL · MobiHet L4 ablation {args.ablation} · seed {args.seed} · session "
        f"{args.session} · stop after round {args.stop_after} · all 5,343 clients."),
        code(setup), code(embed), code(resume), code(cache), code(train)]
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
