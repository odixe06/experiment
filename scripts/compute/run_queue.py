"""FLOP-count queue for the scenarios not measured in ../mobihet/runs/total-compute-20261119 (owner 20/11/2026): TabM, SAE, MLP,
CNN-BiGRU-Attention x 14 methods, and the three mixed-model cases x 5 heterogeneous methods. Count mode only (owner 20/11: GPU
time is hardware-specific, FLOPs are not). Panel 200, 2 rounds (round 2 = steady state), production group_rows (4096 single
model, 8192 mixed), seed 42. Sequential, resumable (a JSON with status=done is skipped). A CUDA OOM is recorded as
status=oom: run that job on a Kaggle T4 instead (same counter, same rows).

    V4_ROOT=... V4_CACHE=... python scripts/compute/run_queue.py [--only PREFIX] [--out runs/compute-20261120]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
ARGS = sys.argv[1:]
OUT = Path(ARGS[ARGS.index("--out") + 1]) if "--out" in ARGS else ROOT / "runs/compute-20261120"
ONLY = ARGS[ARGS.index("--only") + 1] if "--only" in ARGS else ""
BASE = ["fedavg", "fedprox", "afpha", "fd_ids", "lwfednids", "pfedes", "fedproto", "fedtgp", "tinyproto", "nilm",
        "perfedskd", "apfed", "v2x_htfl", "canperfl"]
BACKBONES = {"tabm": "tabm", "sae": "sae", "mlp": "mlp", "cnn-bigru": "cnn-bigru-attention"}
HETERO = ["fedproto", "fedtgp", "tinyproto", "v2x_htfl", "pfedes"]


def jobs():
    for scen, bb in BACKBONES.items():
        for m in BASE:
            yield f"{scen}-{m}", ["scripts/run_fl.py", "--backbone", bb, "--method", m, "--rounds", "2"], \
                ["group_rows=4096", "seed=42"]
    for case in ("case1", "case2", "case3"):
        for m in HETERO:
            yield f"mixed-{case}-{m}", ["scripts/run_mixed_fl.py", "--case", case, "--method", m, "--seed", "42",
                                        "--smoke-clients", "200", "--stop-after", "2"], []


def cmd(base, out, sets):
    return [*base, "--out", str(out), *(["--set", *sets] if sets else [])]


def run(name, base, sets):
    res, log, out = OUT / f"{name}.count.json", OUT / f"{name}.count.log", OUT / name
    if res.exists() and json.loads(res.read_text()).get("status") in ("done", "oom"):
        return "skip"
    if name.endswith("-canperfl"):                                    # FedAvg r1-50 measured as <scen>-fedavg; here the
        init = OUT / f"{name}-init"                                   # 10 fine-tune epochs start from a relabelled r1 state
        proxy = init / "fedavg-r50-proxy.pt"
        if not proxy.exists():
            fed = [("fedavg" if a == "canperfl" else "1" if i and base[i - 1] == "--rounds" else a) for i, a in enumerate(base)]
            r = subprocess.run([PY, str(ROOT / "scripts/compute/total_compute.py"), "count", str(OUT / f"{name}-init.count.json"),
                                "--repo", str(ROOT), "--panel", "200", "--", *cmd(fed, init, sets)], cwd=ROOT,
                               stdout=open(OUT / f"{name}-init.log", "w"), stderr=subprocess.STDOUT)
            if r.returncode:
                return f"init rc {r.returncode}"
            import torch
            s = torch.load(init / "state.pt", map_location="cpu", weights_only=False)
            s["round"] = 50
            torch.save(s, proxy)
        sets = [*sets, f'init="{proxy}"']
    argv = cmd(base, out, sets)
    t = time.time()
    with open(log, "w") as f:
        p = subprocess.run(["timeout", "5400", PY, str(ROOT / "scripts/run_local_guarded.py"), PY,
                            str(ROOT / "scripts/compute/total_compute.py"), "count", str(res), "--repo", str(ROOT),
                            "--panel", "200", "--", *argv], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode and "OutOfMemoryError" in log.read_text():
        d = json.loads(res.read_text()) if res.exists() else {}
        d["status"] = "oom"
        res.write_text(json.dumps(d, indent=1))
    for f in out.glob("state*.pt"):
        f.unlink()
    return f"rc {p.returncode} {time.time() - t:.0f}s"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, base, sets in jobs():
        if name.startswith(ONLY):
            print(time.strftime("%H:%M:%S"), name, run(name, base, sets), flush=True)
    print("QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
