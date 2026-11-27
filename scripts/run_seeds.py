"""Seeds 41/43 for the 15 FL baselines (owner 12/11/2026), same protocol/code path as seed 42, one notebook per (method, seed).

    python scripts/run_seeds.py plan      # live quota -> runs/seeds/plan.json (locked; never re-planned)
    python scripts/run_seeds.py launch    # generate + embed W&B + push wave 1
    python scripts/run_seeds.py collect   # one shot (owner runs it): status, pull finished outputs, push wave 2 on the SAME account
    python scripts/run_seeds.py audit     # audit every pulled run (seed/rounds checked) + 10-metric report -> runs/seeds/report.md

Code: HEAD (methods/core/run_fl identical to the TinyProto r4 SHA 55d13cc; seed-42 runs on 6617261/903ac42 differ only by
backward-compatible eval additions, the APFed final-state RAM fix and TinyProto's mu, which is passed here as in r4).
State per round: as seed 42 (run_fl: models/rNNN.pt small server tensors every round; full private state only at budget
stop / last round). Wave 2: CANPerFL after FedAvg of the same seed (input kernel, same owner); NILM session 2 resumes s1.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/seeds"
SK = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
PY = sys.executable
SEEDS = (41, 43)
# method, extra --set, measured seed-42 train+eval hours (runs/pulls rounds.jsonl)
METHODS = [("nilm", [], 14.4), ("apfed", [], 7.4), ("pfedes", [], 7.1), ("perfedskd", [], 5.2), ("v2x_htfl", [], 5.2),
           ("fedtgp", [], 5.05), ("tinyproto", ["mu=0.00015"], 5.05), ("fd_ids", [], 4.4), ("afpha", [], 3.9),
           ("fedproto", [], 3.85), ("fedprox", [], 3.7), ("fedavg", [], 3.4), ("lwfednids", [], 1.1)]
SETUP, MARGIN, CAP = 0.8, 1.15, 11.5
EXCLUDE = {"thuyanh316"}                       # reserved for CBB s41 (repo mobihet)
PULL = r"(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|models/r\d{3}\.pt)$)"


def slug(m, s, part=""):
    return f"fl2-{m.replace('_', '-')}-s{s}{part}"


def sh(args, **kw):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True, **kw).stdout


def quota():
    report = OUT / "quota-report.json"
    result = subprocess.run([PY, str(SK / "kaggle_quota.py"), "--out", str(report)], cwd=ROOT)
    if result.returncode not in (0, 3):
        result.check_returncode()
    return {
        account: row["gpu_left"]
        for account, row in json.loads(report.read_text())["accounts"].items()
        if row["status"] == "OK"
    }


def plan():
    f = OUT / "plan.json"
    assert not f.exists(), "plan locked"
    q = {a: h for a, h in quota().items() if a not in EXCLUDE}
    jobs = []                                   # job = sessions that must share one account
    for s in SEEDS:
        for m, extra, hrs in METHODS:
            if m == "nilm":
                jobs.append([dict(slug=slug(m, s), method=m, seed=s, set=extra, run=slug(m, s), hours=CAP),
                             dict(slug=slug(m, s, "-p2"), method=m, seed=s, set=extra, run=slug(m, s), hours=hrs - CAP + 2 * SETUP,
                                  wave=2, resume=slug(m, s))])
            elif m == "fedavg":
                jobs.append([dict(slug=slug(m, s), method=m, seed=s, set=extra, run=slug(m, s), hours=hrs + SETUP),
                             dict(slug=slug("canperfl", s), method="canperfl", seed=s, rounds=10, run=slug("canperfl", s),
                                  set=[f"init='\\\"/kaggle/input/**/{slug(m, s)}/state.pt\\\"'"], hours=0.83 + SETUP, wave=2,
                                  input=slug(m, s))])
            else:
                jobs.append([dict(slug=slug(m, s), method=m, seed=s, set=extra, run=slug(m, s), hours=hrs + SETUP)])
    left, slots, rows = dict(q), {a: 2 for a in q}, []
    for job in sorted(jobs, key=lambda j: -sum(x["hours"] for x in j)):
        need = sum(x["hours"] for x in job) * MARGIN
        ok = [a for a in left if left[a] >= need and slots[a] > 0]
        assert ok, ("no account for", job[0]["slug"], need)
        a = min(ok, key=lambda a: left[a] - need)          # tightest fit keeps big accounts for big jobs
        left[a] -= need
        slots[a] -= 1
        for x in job:
            rows.append(dict(x, account=a, wave=x.get("wave", 1)))
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"head": sh(["git", "rev-parse", "HEAD"]).strip(), "quota": q, "rows": rows}, indent=1))
    for r in rows:
        print(r["wave"], r["account"], r["slug"], round(r["hours"], 2))


def push(r, head):
    a = r["account"]
    args = [PY, "scripts/gen_notebook.py", "--owner", a, "--slug", r["slug"], "--method", r["method"], "--run", r["run"],
            "--rounds", str(r.get("rounds", 50)), "--max-hours", str(CAP), "--sha", head,
            "--set", f"seed={r['seed']}", *r["set"]]
    if r.get("resume"):
        args += ["--resume-kernel", f"{a}/{r['resume']}"]
    if r.get("input"):
        args += ["--input-kernel", f"{a}/{r['input']}"]
    sh(args)
    d = ROOT / "notebooks" / r["slug"]
    sh([PY, str(SK / "embed_wandb_key.py"), str(d / f"{r['slug']}.ipynb"), "--metadata", str(d / "kernel-metadata.json")])
    meta = json.loads((d / "kernel-metadata.json").read_text())
    assert meta["is_private"] and meta["id"] == f"{a}/{r['slug']}" and meta["machine_shape"] == "NvidiaTeslaT4"
    marker = OUT / f"{r['slug']}.pushed"
    assert not marker.exists(), r["slug"]
    (OUT / f"{r['slug']}.intent").write_text(time.strftime("%FT%T%z"))
    out = sh([PY, str(SK / "kaggle_as.py"), a, "--", "kaggle", "kernels", "push", "-p", str(d)])
    marker.write_text(out)
    print("PUSHED", a, r["slug"], out.strip().splitlines()[-1] if out.strip() else "", flush=True)


def status(r):
    try:
        out = sh([PY, str(SK / "kaggle_as.py"), r["account"], "--", "kaggle", "kernels", "status", f"{r['account']}/{r['slug']}"])
    except subprocess.CalledProcessError as e:
        return "UNKNOWN:" + (e.stdout or "")[-200:]
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in out.upper():
            return s
    return "UNKNOWN:" + out[-200:]


def launch():
    p = json.loads((OUT / "plan.json").read_text())
    for r in p["rows"]:
        if r["wave"] == 1 and not (OUT / f"{r['slug']}.pushed").exists():
            assert not (OUT / f"{r['slug']}.intent").exists(), ("inspect ambiguous push", r["slug"])
            push(r, p["head"])


def collect():
    """One pass, no loop. Re-run until it prints ALL_TERMINAL; wave 2 is pushed only after its parent COMPLETE + pulled."""
    p = json.loads((OUT / "plan.json").read_text())
    rows = {r["slug"]: r for r in p["rows"]}
    st = {s: status(r) for s, r in rows.items() if (OUT / f"{s}.pushed").exists() and not (OUT / f"{s}.done").exists()}
    for s, v in st.items():
        if v in ("COMPLETE", "ERROR", "CANCEL"):
            r = rows[s]
            dest = ROOT / "runs/pulls" / s
            rc = subprocess.run([PY, str(SK / "kaggle_as.py"), r["account"], "--", PY, str(SK / "pull_kernel_output.py"),
                                 f"{r['account']}/{s}", "-p", str(dest), "--pattern", PULL], cwd=ROOT).returncode
            print("PULL", s, v, "rc", rc, flush=True)
            if rc == 0:
                (OUT / f"{s}.done").write_text(v)
    for r in p["rows"]:
        parent = r.get("resume") or r.get("input")
        if r["wave"] == 2 and not (OUT / f"{r['slug']}.intent").exists() and (OUT / f"{parent}.done").exists():
            if (OUT / f"{parent}.done").read_text() == "COMPLETE":
                push(r, p["head"])

    def finished(r):
        if (OUT / f"{r['slug']}.done").exists():
            return True
        parent = r.get("resume") or r.get("input")
        return r["wave"] == 2 and (OUT / f"{parent}.done").exists() and (OUT / f"{parent}.done").read_text() != "COMPLETE"
    (OUT / "status.json").write_text(json.dumps({"at": time.strftime("%FT%T%z"), "status": st}, indent=1))
    for s, v in sorted(st.items()):
        print(f"{v:10s} {rows[s]['account']:22s} {s}")
    left = [r["slug"] for r in p["rows"] if not finished(r)]
    print("ALL_TERMINAL" if not left else f"REMAINING {len(left)}: {' '.join(left)}", flush=True)


def audit():
    """Audit each pulled run dir (NILM: the p2 pull holds rounds 1-50 of the shared run dir) + one 10-metric table."""
    p = json.loads((OUT / "plan.json").read_text())
    final = {}                                       # run dir name -> last pulled kernel holding it
    for r in p["rows"]:
        if (OUT / f"{r['slug']}.done").exists():
            final[r["run"]] = (r, ROOT / "runs/pulls" / r["slug"])
    res, dirs = {}, []
    for run, (r, pulled) in sorted(final.items()):
        found = [d.parent for d in pulled.rglob("rounds.jsonl") if d.parent.name == run]
        if not found:
            res[run] = "NO_RUN_DIR"
            continue
        out = OUT / "audit" / f"{run}.json"
        cp = subprocess.run([PY, "scripts/audit_run.py", str(found[0]), "--support", "runs/test_support.npy", "--out", str(out),
                             "--seed", str(r["seed"]), "--rounds", str(r.get("rounds", 50))], cwd=ROOT, capture_output=True, text=True)
        res[run] = (cp.stdout.strip() or cp.stderr.strip().splitlines()[-1]) if (cp.stdout or cp.stderr) else f"rc {cp.returncode}"
        print(run, "->", res[run], flush=True)
        if cp.returncode == 0:
            dirs.append(str(found[0]))
    (OUT / "audit-summary.json").write_text(json.dumps(res, indent=1))
    if dirs:
        rep = sh([PY, "scripts/report.py", *dirs])
        (OUT / "report.md").write_text(rep)
        print("REPORT", OUT / "report.md")
    missing = [r["run"] for r in p["rows"] if r["run"] not in final]
    print("NOT_PULLED", sorted(set(missing)))


if __name__ == "__main__":
    {"plan": plan, "launch": launch, "collect": collect, "audit": audit}[sys.argv[1]]()
