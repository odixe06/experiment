"""Generate a private, frozen 2xT4 notebook for a five-model timing probe or a 50-epoch run."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("mlp","tabm","bert","cnn-lstm-feature","sae")
FILES = ["core/data.py","core/metrics.py","core/central_backbones.py","core/central_windows.py",
         "knowledge/input_order.json","knowledge/label_mapping.json","scripts/run_central_backbones.py",
         "scripts/prepare_central_cache.py","scripts/audit_central_backbones.py","tests/test_central_backbones.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--model", choices=(*MODELS,"cnn-lstm","grande","gru-attention","cnn-bigru-attention","mambacnn","probe"), required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--parent", help="attach owner/slug for same-model cross-session resume")
    ap.add_argument("--parent-artifacts", type=Path, help="version-scoped pull with frozen source/config and last epoch checkpoint")
    ap.add_argument("--parent-epoch", type=int)
    a = ap.parse_args()
    if a.parent and a.model == "probe":
        ap.error("probe has no parent")
    source = {f:(ROOT/f).read_text(encoding="utf-8") for f in [*FILES,"core/grande.py","core/dagsnet.py","core/sae.py","core/gru_attention.py","core/cnn_bigru_attention.py","core/mambacnn.py"]}
    if a.model == "mambacnn":
        source["scripts/check_mambacnn_cuda.py"] = (ROOT/"scripts/check_mambacnn_cuda.py").read_text(encoding="utf-8")
    hashes = {f:hashlib.sha256(s.encode()).hexdigest() for f,s in source.items()}
    if a.parent:
        if not a.parent_artifacts or not a.parent_epoch or not 1 <= a.parent_epoch < 50:
            ap.error("resume requires --parent-artifacts and --parent-epoch (1..49)")
        source = json.loads((a.parent_artifacts/"source_snapshot.json").read_text(encoding="utf-8"))
        hashes = json.loads((a.parent_artifacts/"source_manifest.json").read_text(encoding="utf-8"))
        if not set(FILES) <= source.keys() or {f:hashlib.sha256(s.encode()).hexdigest() for f,s in source.items()} != hashes:
            raise ValueError("parent source snapshot/manifest mismatch")
        parent_dir = a.parent_artifacts/f"central-{a.model}"
        parent_micro = json.loads((parent_dir/"config.json").read_text())["micro"]
        last_dir = f"epoch_{a.parent_epoch:03}"
        parent_hashes = {f:hashlib.sha256((parent_dir/f).read_bytes()).hexdigest()
                         for f in ("config.json",f"{last_dir}/weights.pt",f"{last_dir}/resume.pt",f"{last_dir}/metrics.json")}
    cells = [f'''import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
start = time.monotonic()
import torch
assert torch.cuda.device_count() == 2
assert all("T4" in torch.cuda.get_device_name(i) and torch.cuda.get_device_capability(i) == (7,5) for i in (0,1))
print(torch.__version__, [torch.cuda.get_device_name(i) for i in (0,1)],flush=True)
src, cache, out = Path("/kaggle/temp/src"), Path("/kaggle/temp/v4cache"), Path("/kaggle/working")
source, hashes = {source!r}, {hashes!r}
for f,s in source.items():
    p = src/f
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(s)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[f]
(out/"source_snapshot.json").write_text(json.dumps(source))
(out/"source_manifest.json").write_text(json.dumps(hashes))
os.environ.update(OMP_NUM_THREADS="2", NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1", PYTHONUNBUFFERED="1",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONPATH=str(src))
def run(*args):
    subprocess.run([str(v) for v in args], cwd=src, check=True)
''']
    if a.model == "mambacnn":
        cells.append('''assert sys.version_info[:2] == (3,12) and torch.__version__.startswith("2.10.")
assert torch._C._GLIBCXX_USE_CXX11_ABI and torch.version.cuda.startswith("12.")
wheel = "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=7a67070c1e7e99c95abd1319623f044e8a1b3fb46f774bfdea949f0a4fc79638"
run(sys.executable,"-m","pip","install","--no-deps",wheel)
run(sys.executable,"-c","from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; import selective_scan_cuda; print('MAMBA_CUDA_IMPORT_PASS', flush=True)")
''')
    cells.append('''run(sys.executable,"-m","unittest","discover","-s","tests","-p","test_central_backbones.py","-v")''')
    if a.parent:
        cells.append(f'''candidates = list(Path("/kaggle/input").rglob("central-{a.model}/config.json"))
assert len(candidates)==1, candidates
parent = candidates[0].parent
assert json.loads((parent.parent/"source_manifest.json").read_text()) == hashes
committed = sorted(parent.glob("epoch_[0-9][0-9][0-9]"))
assert [p.name for p in committed] == [f"epoch_{{e:03}}" for e in range(1,{a.parent_epoch}+1)], committed
for name,digest in {parent_hashes!r}.items():
    assert hashlib.sha256((parent/name).read_bytes()).hexdigest()==digest, name
shutil.copytree(parent, out/parent.name, copy_function=shutil.copyfile, ignore=shutil.ignore_patterns(".epoch_*"))
print("IMPORTED epoch {a.parent_epoch}; next epoch {a.parent_epoch+1}",flush=True)
''')
    cells.append(f'''run(sys.executable,"scripts/prepare_central_cache.py","--cache",cache,
    *{(["--windows"] if a.model == "cnn-lstm" else [])!r})
''')
    if a.model == "grande" and not a.parent:
        cells.append('''for name in ("dagsnet","grande"):
    run("torchrun","--standalone","--nproc_per_node=2","scripts/run_central_backbones.py",
        "--cache",cache,"--out",out/("timing-"+name),"--model",name,"--epochs","50","--probe")
''')
    if a.model in ("cnn-bigru-attention", "cnn-lstm-feature") and not a.parent:
        cells.append(f'''run("torchrun","--standalone","--nproc_per_node=2","scripts/run_central_backbones.py",
    "--cache",cache,"--out",out/"timing-{a.model}","--model","{a.model}",
    "--epochs","50","--probe")
''')
    if a.model == "mambacnn" and not a.parent:
        cells.append('''equivalence = subprocess.run([sys.executable,"scripts/check_mambacnn_cuda.py","--cache",str(cache),"--out",str(out/"mamba_cuda_equivalence.json")],cwd=src)
probes = []
for micro in (32,128,256,512):
    probe_dir = out/f"timing-mambacnn-m{micro}"
    run("torchrun","--standalone","--nproc_per_node=2","scripts/run_central_backbones.py",
        "--cache",cache,"--out",probe_dir,"--model","mambacnn","--micro",micro,"--epochs","50","--probe")
    probes.append(json.loads((probe_dir/"probe.json").read_text()))
probe = min(probes, key=lambda p: sum(p["estimated_train_eval_seconds_per_epoch"]))
selected_micro = probe["micro"]
(out/"mamba_timing_selection.json").write_text(json.dumps(dict(selected_micro=selected_micro,probes=probes),indent=2))
print("MAMBA_SELECTED_MICRO", selected_micro, flush=True)
epoch_seconds = sum(probe["estimated_train_eval_seconds_per_epoch"])
available_seconds = 11.25*3600-(time.monotonic()-start)
print("MAMBA_BUDGET", "estimated_hours_50", probe["estimated_hours_50_epochs"],
      "remaining_hours", available_seconds/3600, flush=True)
assert 0 < epoch_seconds*1.5 < available_seconds, "probe: one epoch cannot safely fit session; inspect timing before training"
assert equivalence.returncode == 0, "CUDA equivalence gate failed; timing only, no production"
''')
    if a.model == "cnn-lstm-feature" and not a.parent:
        cells.append('''probe = json.loads((out/"timing-cnn-lstm-feature"/"probe.json").read_text())
epoch_seconds = sum(probe["estimated_train_eval_seconds_per_epoch"])
available_seconds = 11.25*3600-(time.monotonic()-start)
print("CNN_LSTM_BUDGET", probe["estimated_hours_50_epochs"], "hours/50epochs", flush=True)
assert 0 < epoch_seconds*1.5 < available_seconds, "probe: one epoch cannot safely fit session"
''')
    cells.append(f'''for name in {(MODELS if a.model=="probe" else (a.model,))!r}:
    budget = 11.25-(time.monotonic()-start)/3600
    assert budget > 0.25, "session setup consumed training budget"
    print("starting",name,"remaining hours",budget,flush=True)
    run("torchrun","--standalone","--nproc_per_node=2","scripts/run_central_backbones.py",
        "--cache",cache,"--out",out/("central-"+name),"--model",name,"--epochs","50","--max-hours",budget,
        *{('["--micro",selected_micro]' if a.model == 'mambacnn' and not a.parent else repr(["--micro",parent_micro] if a.parent else []))},
        *{(["--probe"] if a.model=="probe" else ["--resume"] if a.parent else [])!r})
''')
    if a.model != "probe":
        cells.append(f'''result = out/"central-{a.model}"
complete = (result/"epoch_050").is_dir()
run(sys.executable,"scripts/audit_central_backbones.py",result,*([] if complete else ["--partial"]))
print("50 EPOCHS AUDITED" if complete else "PARTIAL; resume from latest committed epoch")''')
    nb = nbformat.v4.new_notebook()
    for cell in cells:
        ast.parse(cell)
        nb.cells.append(nbformat.v4.new_code_cell(cell))
    nb.metadata.update(kernelspec={"name":"python3","display_name":"Python 3","language":"python"},
                       kaggle={"accelerator":"nvidiaTeslaT4","isGpuEnabled":True,"isInternetEnabled":a.model == "mambacnn",
                               "language":"python","sourceType":"notebook"})
    d = ROOT/"notebooks"/a.slug
    d.mkdir(parents=True,exist_ok=True)
    path = d/(a.slug+".ipynb")
    nbformat.validate(nb)
    nbformat.write(nb,path)
    runtime = json.loads((ROOT/"knowledge/runtime.json").read_text())
    meta = dict(id=f"{a.owner}/{a.slug}",title=a.slug,code_file=path.name,language="python",kernel_type="notebook",
                is_private=True,enable_gpu=True,enable_internet=a.model == "mambacnn",machine_shape=runtime["machine_shape"],
                docker_image=runtime["docker_image"],dataset_sources=["odixe0502/veremi-nextgen-v4-scaled-per-vehicle"],
                kernel_sources=[a.parent] if a.parent else [], competition_sources=[],model_sources=[])
    (d/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
