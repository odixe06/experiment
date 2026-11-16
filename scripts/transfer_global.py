"""Cross-account continuation for single-global-model runs (small state, no NPY cache carried).

    python scripts/transfer_global.py fedavg:takids1909 fedprox:catbaochau ...

Per pair: last COMPLETE session of the run (below round 50) -> pull <run>/ (state 2 rank, models, inductor tar) from its owner
-> check both ranks at the same round -> private dataset <new>/flb-xfer-<method>-r<round> (zip of <run>/, Kaggle extracts it)
-> listing must show <run>/state.pt with the local size -> push next part on the new account (run_batch.push, same source/HP).
The chain then lives on the new account; later parts resume from its kernel output as usual.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_batch as B  # noqa: E402

XOUT = ROOT / "runs/transfer"


def head_of(method):
    _, rows = B.sessions()
    run = B.slug(method)
    heads = [r for r in rows.values() if r["run"] == run and (B.OUT / f"{r['slug']}.pushed").exists()]
    return max(heads, key=lambda r: r["part"])


def stage(r, new):
    d = XOUT / r["slug"]; d.mkdir(parents=True, exist_ok=True)
    ready = d / "ready.json"
    if ready.exists(): return json.loads(ready.read_text())
    pulled = d / "pull"
    subprocess.run([B.PY, str(B.SK / "kaggle_as.py"), r["account"], "--", B.PY, str(B.SK / "pull_kernel_output.py"),
                    f"{r['account']}/{r['slug']}/{r['version']}", "-p", str(pulled), "--pattern", f"^{r['run']}/"],
                   cwd=ROOT, check=True)
    src = pulled / r["run"]
    import torch
    s0 = torch.load(src / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    s1 = torch.load(src / "state_rank1.pt", map_location="cpu", weights_only=True, mmap=True)
    rr = B.last_round(r)
    assert s0["method"] == r["method"] and s0["hp"]["rounds"] == 50 and s0["round"] == s1["round"] == rr and len(s0["gen"]) == 2
    del s0, s1
    upload = d / "upload"; shutil.rmtree(upload, ignore_errors=True); upload.mkdir()
    shutil.copytree(src, upload, dirs_exist_ok=True)
    # slug == run dir and files at its root: /kaggle/input/**/<run>/state.pt matches whether Kaggle nests or not
    ref = f"{new}/{r['run']}"
    (upload / "dataset-metadata.json").write_text(json.dumps(dict(id=ref, title=f"flb xfer {r['method']} r{rr}",
                                                                  licenses=[dict(name="CC0-1.0")]), indent=1))
    intent = d / f"create-intent-{ref.split('/')[1]}.json"
    if not intent.exists():  # existing intent: dataset was created; only re-verify readiness and sizes below
        intent.write_text(json.dumps(dict(ref=ref, parent=f"{r['account']}/{r['slug']}/{r['version']}", round=rr)))
        out = B.sh([B.PY, str(B.SK / "kaggle_as.py"), new, "--", "kaggle", "datasets", "create", "-p", str(upload), "--dir-mode", "zip"])
        (d / "create-reply.txt").write_text(out)
    size = (upload / "state.pt").stat().st_size
    for _ in range(80):
        time.sleep(15)
        try:
            if B.sh([B.PY, str(B.SK / "kaggle_as.py"), new, "--", "kaggle", "datasets", "status", ref]).strip() == "ready": break
        except subprocess.CalledProcessError:
            continue
    else:
        raise RuntimeError(("dataset not ready; inspect, no push", ref))
    ls = B.sh([B.PY, str(B.SK / "kaggle_as.py"), new, "--", B.PY, str(ROOT / "scripts/list_dataset.py"), ref])
    got = dict(x.split(",") for x in ls.split())
    assert int(got["state.pt"]) == size and int(got["state_rank1.pt"]) == (upload / "state_rank1.pt").stat().st_size, ls[:300]
    st = B.sh([B.PY, str(B.SK / "kaggle_as.py"), new, "--", "kaggle", "datasets", "status", ref]).strip()
    res = dict(ref=ref, round=rr, parent=f"{r['account']}/{r['slug']}/{r['version']}", state_bytes=size, status=st, listing=ls)
    ready.write_text(json.dumps(res, indent=1))
    return res


def main():
    p, _ = B.sessions()
    for pair in sys.argv[1:]:
        method, new = pair.split(":")
        assert new not in B.EXCLUDE
        r = head_of(method)
        assert (B.OUT / f"{r['slug']}.done").read_text() == "COMPLETE" and B.last_round(r) < 50, r["slug"]
        res = stage(r, new)
        nxt = dict(r, slug=B.slug(method, r["part"] + 1), part=r["part"] + 1, account=new, resume_dataset=res["ref"],
                   transfer_from=res["parent"])
        nxt.pop("resume", None); nxt.pop("version", None)
        q = B.quota()[new]
        nxt["max_hours"] = round(min(B.CAP, q / B.MARGIN), 2)
        assert nxt["max_hours"] >= B.MIN_SESSION, (new, q)
        B.push(nxt, p["head"])


if __name__ == "__main__":
    main()
