"""Generate a private, frozen 2xT4 notebook for a multi-model timing probe or a 50-epoch run."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("mlp","tabm","bert","sae")
FILES = ["core/data.py","core/metrics.py","core/central_backbones.py",
         "knowledge/input_order.json","knowledge/label_mapping.json","scripts/run_central_backbones.py",
         "scripts/prepare_central_cache.py","scripts/audit_central_backbones.py","tests/test_central_backbones.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--model", choices=(*MODELS,"sae-ce","cnn-bigru-attention","probe"), required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--parent", help="attach owner/slug for same-model cross-session resume")
    ap.add_argument("--parent-artifacts", type=Path, help="version-scoped pull with frozen source/config and last epoch checkpoint")
    ap.add_argument("--parent-epoch", type=int)
    ap.add_argument("--seed", type=int, default=42, help="training seed passed to run_central_backbones.py")
    ap.add_argument("--max-hours", type=float, default=11.25, help="session budget; set below the account's live GPU quota")
    a = ap.parse_args()
    if a.parent and a.model == "probe":
        ap.error("probe has no parent")
    source = {f:(ROOT/f).read_text(encoding="utf-8") for f in [*FILES,"core/dagsnet.py","core/sae.py","core/tabm.py","core/cnn_bigru_attention.py"]}
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
    cells.append('''run(sys.executable,"scripts/prepare_central_cache.py","--cache",cache)
''')
    if a.model == "cnn-bigru-attention" and not a.parent:
        cells.append(f'''run("torchrun","--standalone","--nproc_per_node=2","scripts/run_central_backbones.py",
    "--cache",cache,"--out",out/"timing-{a.model}","--model","{a.model}",
    "--epochs","50","--probe")
''')
    cells.append(f'''for name in {(MODELS if a.model=="probe" else (a.model,))!r}:
    budget = {a.max_hours}-(time.monotonic()-start)/3600
    assert budget > 0.25, "session setup consumed training budget"
    print("starting",name,"remaining hours",budget,flush=True)
    run("torchrun","--standalone","--nproc_per_node=2","scripts/run_central_backbones.py",
        "--cache",cache,"--out",out/("central-"+name),"--model",name,"--epochs","50","--max-hours",budget,"--seed",{a.seed},
        *{["--micro",parent_micro] if a.parent else []!r},
        *{(["--probe"] if a.model=="probe" else ["--resume"] if a.parent else [])!r})
''')
    if a.model != "probe":
        cells.append(f'''result = out/"central-{a.model}"
complete = (result/"epoch_050").is_dir()
run(sys.executable,"scripts/audit_central_backbones.py",result,*([] if complete else ["--partial"]))
print("50 EPOCHS AUDITED" if complete else "PARTIAL; resume from latest committed epoch")''')
    # 20/11: the embedded snapshot must import on its own (a missing core/tabm.py made 11 sessions ERROR at the test cell)
    import subprocess, sys, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for f, text in source.items():
            (Path(tmp)/f).parent.mkdir(parents=True, exist_ok=True)
            (Path(tmp)/f).write_text(text, encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_central_backbones.py"],
                           cwd=tmp, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=tmp))
        if r.returncode:
            raise SystemExit("embedded source fails its own tests:\n" + r.stderr[-3000:])
    nb = nbformat.v4.new_notebook()
    for cell in cells:
        ast.parse(cell)
        nb.cells.append(nbformat.v4.new_code_cell(cell))
    nb.metadata.update(kernelspec={"name":"python3","display_name":"Python 3","language":"python"},
                       kaggle={"accelerator":"nvidiaTeslaT4","isGpuEnabled":True,"isInternetEnabled":False,
                               "language":"python","sourceType":"notebook"})
    d = ROOT/"notebooks"/a.slug
    d.mkdir(parents=True,exist_ok=True)
    path = d/(a.slug+".ipynb")
    nbformat.validate(nb)
    nbformat.write(nb,path)
    runtime = json.loads((ROOT/"knowledge/runtime.json").read_text())
    meta = dict(id=f"{a.owner}/{a.slug}",title=a.slug,code_file=path.name,language="python",kernel_type="notebook",
                is_private=True,enable_gpu=True,enable_internet=False,machine_shape=runtime["machine_shape"],
                docker_image=runtime["docker_image"],dataset_sources=["odixe0502/veremi-nextgen-v4-scaled-per-vehicle"],
                kernel_sources=[a.parent] if a.parent else [], competition_sources=[],model_sources=[])
    (d/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
