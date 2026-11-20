"""Build one private 2xT4 notebook: MobiHet FB-L configuration L4 (methods/mobihet_l4.py) on one single-model backbone.

Same data, protocol (50 rounds, all 5,343 vehicles, cosine 1e-3 -> 1e-5) and embedded-source layout as the baseline builder
(scripts/build_tabm_fl.py). L4 hp = the ../mobihet r50 chains (runs/fbl-r50/batch.json), only the seed changes.
A run longer than one session continues from the previous session's output (--parent <owner>/<slug>): the run directory
(state.pt, state_rank1.pt, *.jsonl, cms) is copied to /kaggle/working before run_fl.py resumes it.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
L4_HP = [
    "arm=none", "group_rows=4096", "synth=copula", "synth_rows=64", "q_codec=mean_fp16", "kd_weight=fixed",
    "diagnostic_ids=[506,0,1571,929,3303,2048,4752,4269,504,1,1587,935,3304,2043,4754,4266,512,29,1566,904,3306,2052,4757,4267,"
    "1547,885,3318,2357,4748,4265,1548,886,4747,4279]",
    "diagnostic_rounds=[1,2,5,10,15,20]", "q_collect=online", "q_agg=count", "kd_support=off", "synth_target=soft",
    "anchor=off", "kd_bins=1", "no_kd=false", "study_mode=class", "teacher_rows=1024", "proxy_steps=200", "real_kd=false",
    "real_kd_gate=false", "next_mode=stream", "r_kind=live", "synth_step_floor=16", "w_boot=compact", "diagnostics=false",
    "teacher_diagnostics=false", "external_diagnostics=false", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "q_bits=8",
    "synth_tier_min=[4096,16384]", "synth_late_from=25", "cop_kmeans_lite=0", "km_bytes=false", "km2=true", "cop_kmeans=6",
    "cop_bins=64", "km2_frac=0.25", "km2_tau=16", "km2_bits=6", "synth_tier_rows=[[64,64],[16,16],[16,16]]",
    "real_tier_rows=[64,37,37]", "real_select=top", "compact=true"]
MODELS = {"tabm": {"display": "TabM", "parameters": 6548, "test": "tests/test_tabm_fl.py"},
          "sae": {"display": "SAE", "parameters": 7613, "test": "tests/test_sae_fl.py"},
          "mlp": {"display": "MLP", "parameters": 7600, "test": "tests/test_dense_recurrent_fl.py"},
          "cnn-bigru-attention": {"display": "CNN-BiGRU-Attention", "parameters": 118288,
                                  "test": "tests/test_dense_recurrent_fl.py"}}


def code(text):
    ast.parse(text)
    return nbformat.v4.new_code_cell(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--backbone", required=True, choices=sorted(MODELS))
    ap.add_argument("--seed", required=True, type=int, choices=(41, 42, 43))
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--parent", default="", help="<owner>/<slug> of the previous session (session > 1)")
    ap.add_argument("--max-hours", type=float, default=11.5)
    args = ap.parse_args()
    if (args.session > 1) != bool(args.parent):
        ap.error("--parent is required exactly when --session > 1")
    model = MODELS[args.backbone]
    short = {"cnn-bigru-attention": "cnnbigru"}.get(args.backbone, args.backbone)
    run = f"{short}-l4-s{args.seed}"                      # run directory: the same in every session
    slug = run if args.session == 1 else f"{run}-h{args.session}"
    paths = sorted([*ROOT.glob("core/*.py"), *ROOT.glob("methods/*.py"), ROOT / model["test"], ROOT / "tests/test_aligned_fl.py",
                    ROOT / "scripts/run_fl.py", ROOT / "scripts/run_p21_recovery.py", *ROOT.glob("knowledge/*.json")])
    source = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths}
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in source.items()}
    hp = L4_HP + [f"seed={args.seed}"]
    manifest = {"experiment": f"single-backbone-{args.backbone}-mobihet-l4", "method": "mobihet_l4", "seed": args.seed,
                "backbone": model["display"], "parameters": model["parameters"], "rounds": 50, "session": args.session,
                "parent": args.parent or None, "run": run, "hp": hp, "dataset": DATASET,
                "client_partition": "fixed-all-5343", "protocol_revision": "dagsnet-aligned-v2",
                "l4_source": "../mobihet runs/fbl-r50/batch.json (methods a4147a1b...)"}

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
for test in ({model['test']!r}, "tests/test_aligned_fl.py"):
    subprocess.run([sys.executable, test], cwd="/kaggle/temp/src", check=True,
                   env={{**os.environ, "PYTHONPATH": "/kaggle/temp/src", "OMP_NUM_THREADS": "2"}})
'''
    cache = '''sys.path.insert(0, "/kaggle/temp/src")
from core import data as D
t = time.monotonic()
D.build_npy(D.find_root(), "/kaggle/temp/v4npy", splits=("train", "test"))
print(f"NPY cache {time.monotonic()-t:.1f}s", flush=True)
os.environ.update(V4_NPY="/kaggle/temp/v4npy", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                  TORCHINDUCTOR_COMPILE_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  WANDB_MODE="disabled", MOBIHET_RECOVERY_DIR="/kaggle/working/recovery")
'''
    run_dir = f"/kaggle/working/{run}"
    resume = f'''run = Path({run_dir!r})
parent = {args.parent!r}
if parent:
    found = [Path(p).parent for p in glob.glob("/kaggle/input/**/{run}/state.pt", recursive=True)]
    assert len(found) == 1, ("parent run directory", found)
    shutil.copytree(found[0], run)
    st = torch.load(run / "state.pt", map_location="cpu", weights_only=True)
    rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines()]
    assert rows[-1]["round"] == st["round"] and [x["round"] for x in rows] == list(range(1, st["round"] + 1)), "parent rows"
    assert (run / "state_rank1.pt").exists()
    print(f"resume from {{found[0]}} at round {{st['round']}}", flush=True)
    del st
else:
    assert not run.exists()
'''
    base = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_fl.py", "--method", "mobihet_l4",
            "--backbone", args.backbone, "--out", run_dir, "--compile", "--cache", "", "--rounds", "50", "--set", *hp]
    train = f'''cmd = {base!r} + ["--save-every", "5", "--max-hours", f"{{{args.max_hours}-(time.monotonic()-T0)/3600-0.4:.3f}}"]
print(cmd, flush=True)
subprocess.run(cmd, cwd="/kaggle/temp/src", check=True)
rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines()]
assert [x["round"] for x in rows] == list(range(1, rows[-1]["round"] + 1))
status = {{"method": "mobihet_l4", "backbone": {args.backbone!r}, "seed": {args.seed}, "session": {args.session},
          "last_round": rows[-1]["round"], "target_round": 50, "complete": rows[-1]["round"] == 50, "final": rows[-1]}}
Path("/kaggle/working/RUN_STATUS.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
'''
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(
        f"# {slug}\n\n{model['display']} single-backbone FL · MobiHet L4 · seed {args.seed} · session {args.session} · "
        "all 5,343 clients."), code(setup), code(embed), code(cache), code(resume), code(train)]
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
