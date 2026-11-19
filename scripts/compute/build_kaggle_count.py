"""Private 2xT4 notebook that runs FLOP-count jobs of scripts/compute/run_queue.py on Kaggle (owner 20/11/2026: count on free
Kaggle slots; FLOPs do not depend on the GPU). Same harness, panel and rows as the local queue: the repo is cloned at --sha
(pushed), fold_counts.npy is embedded (sha256 checked), npy data is built only for the 200 panel vehicles, and the jobs are
split over the two GPUs (one run_queue.py worker per GPU). Output: /kaggle/working/compute/<job>.count.{json,log}.

    python scripts/compute/build_kaggle_count.py --owner ACCOUNT --slug SLUG --jobs mixed-case1-,mixed-case2-fedproto [--sha SHA]
"""
import argparse
import ast
import base64
import hashlib
import io
import json
import os
import subprocess
import zlib
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/odixe06/experiment.git"
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--jobs", required=True, help="comma-separated run_queue.py job-name prefixes")
    ap.add_argument("--sha", default="")
    ap.add_argument("--fold-counts", default=os.environ.get("V4_CACHE", "cache") + "/fold_counts.npy")
    a = ap.parse_args()
    sha = a.sha or subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    buf = io.BytesIO()
    np.save(buf, np.load(a.fold_counts))
    raw = buf.getvalue()
    fc = base64.b64encode(zlib.compress(raw, 9)).decode()
    fc_sha = hashlib.sha256(raw).hexdigest()
    jobs = a.jobs.split(",")
    cells = [f'''import base64, hashlib, io, os, subprocess, sys, time, zlib
from pathlib import Path
import numpy as np, torch
assert torch.cuda.device_count() == 2 and all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(2)), "expected 2 T4"
src, cache, npy, out = Path("/kaggle/temp/src"), Path("/kaggle/temp/cache"), Path("/kaggle/temp/v4npy"), Path("/kaggle/working/compute")
subprocess.run(["git", "clone", "-q", "{REPO}", str(src)], check=True)
subprocess.run(["git", "-C", str(src), "checkout", "-q", "{sha}"], check=True)
print(subprocess.run(["git", "-C", str(src), "log", "-1", "--format=%H %s"], capture_output=True, text=True).stdout)
cache.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
raw = zlib.decompress(base64.b64decode("{fc}"))
assert hashlib.sha256(raw).hexdigest() == "{fc_sha}", "fold_counts mismatch"
(cache / "fold_counts.npy").write_bytes(raw)
cnt = np.load(cache / "fold_counts.npy")
tr, va = cnt[:, 0].sum(1), cnt[:, 1].sum(1)                     # = scripts/compute/total_compute.py apply_panel(200)
ok = np.flatnonzero((tr > 0) & (va >= 32))
edges = np.quantile(tr[ok], np.linspace(0, 1, 11))
strata = np.clip(np.searchsorted(edges, tr[ok], side="right") - 1, 0, 9)
rng = np.random.default_rng(7)
ids = np.sort(np.concatenate([rng.choice(ok[strata == s], 20, replace=False) for s in range(10)]))
print("panel ids sha256", hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()[:16])
sys.path.insert(0, str(src))
from core import data as D
t = time.time()
D.build_npy(D.find_root(), str(npy), clients=[int(i) for i in ids])
print(f"panel npy built in {{time.time() - t:.0f}}s")''',
             f'''jobs = {jobs!r}
half = [jobs[0::2], jobs[1::2]]
env = dict(os.environ, V4_NPY=str(npy), V4_CACHE=str(cache), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2")
procs = [subprocess.Popen([sys.executable, "scripts/compute/run_queue.py", "--only", ",".join(h), "--out", str(out)], cwd=src,
                          env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=open(out / f"worker{{g}}.log", "w"), stderr=subprocess.STDOUT)
         for g, h in enumerate(half) if h]
rc = [p.wait() for p in procs]
for g in range(len(procs)):
    print(open(out / f"worker{{g}}.log").read()[-3000:])
for f in out.glob("*/state*.pt"):
    f.unlink()
print("WORKERS_RC", rc)''']
    nb = nbformat.v4.new_notebook()
    for c in cells:
        ast.parse(c)
        nb.cells.append(nbformat.v4.new_code_cell(c))
    nb.metadata.update(kernelspec={"name": "python3", "display_name": "Python 3", "language": "python"},
                       kaggle={"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True,
                               "language": "python", "sourceType": "notebook"})
    d = ROOT / "notebooks" / a.slug
    d.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, d / f"{a.slug}.ipynb")
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
    meta = dict(id=f"{a.owner}/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb", language="python", kernel_type="notebook",
                is_private=True, enable_gpu=True, enable_internet=True, machine_shape="NvidiaTeslaT4",
                docker_image=runtime["docker_image"], dataset_sources=[DATASET], kernel_sources=[], competition_sources=[],
                model_sources=[])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(d, sha)


if __name__ == "__main__":
    main()
