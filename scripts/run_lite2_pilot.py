"""Lite2 pilot on FeatureBERT (MobiHet docs/research/screening-lite2.md): B + U1, seed 42, r1-20 of the cosine-50 schedule.

    python scripts/run_lite2_pilot.py build     # lock runs/lite2-pilot/batch.json + notebook (no push)
    python scripts/run_lite2_pilot.py launch    # live quota/slot gate, then push once (intent/pushed markers)
    python scripts/run_lite2_pilot.py collect   # one shot: status; pull the terminal kernel into artifacts/lite2-pulls/<slug>
    python scripts/run_lite2_pilot.py audit     # audit_run bounded checks + source/HP + locked verdict -> audit.json

Control = audited T3 FeatureBERT seed 42 (artifacts/t3-pulls/flb-t3-s42-p3, same core/methods bytes as HEAD). Verdict locked
before push: delta Macro-F1 mean r16-20 >= -.010 and delta r20 >= -.010, bootstrap <= 382 000 B per vehicle.
"""
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

OUT, ART = ROOT / "runs/lite2-pilot", ROOT / "artifacts/lite2-pulls"
REF = ROOT / "artifacts/t3-pulls/flb-t3-s42-p3/flb-t3-s42"
SIM = ROOT.parent / "mobihet/runs/sim-kaggle-lite2/batch.json"                # audited sim: same overlay bytes required
OWNER, SLUG, CAP, UNTIL = "annhannh331", "flb-lite2-bu1-s42-r20", 8.5, 20
OVERLAYS = ["methods/mobihet_compute.py", "methods/mobihet_lite.py"]
KNOBS = ["cop_kmeans_lite=0", "km_bytes=false", "km2=true", "cop_kmeans=6", "cop_bins=64", "km2_frac=0.25", "km2_tau=16",
         "km2_bits=6", "synth_tier_rows=[[64,32],[32,16],[16,16]]"]            # B + U1
GATE, BOOT_MAX = -.010, 382_000
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
PULL = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|models/r\d{3}\.pt)$)'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    if (OUT / "batch.json").exists():
        return json.loads((OUT / "batch.json").read_text())
    ref = next(r for r in json.loads((ROOT / "runs/t3/batch.json").read_text()) if r["seed"] == 42)
    sim = json.loads(SIM.read_text())[0]["overlay_sha256"]
    assert all(sha(ROOT / p) == sim[p] for p in OVERLAYS), "overlay differs from audited sim lite2"
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()
    head = git("rev-parse", "HEAD")
    assert head == git("rev-parse", "origin/main") and not git("status", "--porcelain", "--", "core", "scripts/run_t3.py")
    assert set(git("status", "--porcelain", "--", "methods").split()[1::2]) == set(OVERLAYS)
    keys = {k.split("=")[0] for k in KNOBS}
    hp = [x for x in ref["hp"] if x.split("=")[0] not in keys] + KNOBS
    args = shlex.join(["scripts/run_t3.py", "--method", "mobihet_lite", "--out", f"/kaggle/working/{SLUG}", "--width", "1",
                       "--rounds", "50", "--until", str(UNTIL), "--save-every", "5", "--compile", "--max-hours", "BUDGET",
                       "--cache", "", "--set", *hp])
    subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", OWNER, "--slug", SLUG, "--method", "mobihet_lite",
                    "--sha", head, "--args", args, "--max-hours", str(CAP), "--overlay", *OVERLAYS],
                   cwd=ROOT, check=True, capture_output=True)
    folder = ROOT / "notebooks" / SLUG
    nb, meta = folder / (SLUG + ".ipynb"), folder / "kernel-metadata.json"
    doc = json.loads(nb.read_text())
    for i, cell in enumerate(doc["cells"]):
        cell["id"] = f"lite2-cell-{i:03d}"
        if "BEGIN INLINE WANDB CREDENTIAL" in "".join(cell["source"]):
            cell["source"] = ['print("W&B off: same policy as T3 FeatureBERT")']
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(nb), "exec")
    nb.write_text(json.dumps(doc, indent=1) + "\n")
    m = json.loads(meta.read_text())
    assert m["is_private"] and m["enable_gpu"] and m["docker_image"] and m["id"] == f"{OWNER}/{SLUG}" and not m["kernel_sources"]
    batch = [dict(arm="BU1B", seed=42, method="mobihet_lite", account=OWNER, slug=SLUG, version=1, hp=hp, base_sha=head,
                  overlay_sha256={p: sha(ROOT / p) for p in OVERLAYS}, notebook_sha256=sha(nb), metadata_sha256=sha(meta),
                  cap_hours=CAP, endpoint=UNTIL, lr_schedule_rounds=50, control="T3 FeatureBERT seed 42 (flb-t3-s42-p3)")]
    atomic_json(OUT / "batch.json", batch)
    return batch


def launch(row):
    intent, marker = OUT / (SLUG + ".intent"), OUT / (SLUG + ".pushed")
    if marker.exists():
        return print("already pushed", SLUG)
    assert not intent.exists(), "ambiguous push: inspect first"
    folder = ROOT / "notebooks" / SLUG
    assert sha(folder / (SLUG + ".ipynb")) == row["notebook_sha256"] and sha(folder / "kernel-metadata.json") == row["metadata_sha256"]
    inv = subprocess.run([sys.executable, str(SKILL / "kaggle_as.py"), OWNER, "--", sys.executable, "scripts/kaggle_inventory.py", OWNER],
                         cwd=ROOT, capture_output=True, text=True, timeout=600)
    (OUT / (SLUG + ".inventory.log")).write_text(inv.stdout + inv.stderr)
    assert inv.returncode == 0, f"inventory unavailable: see {SLUG}.inventory.log"
    q = json.loads(inv.stdout)
    assert q["gpu_left"] >= CAP * 1.15 and len(q["busy"]) < 2, (q["gpu_left"], q["busy"])
    assert SLUG not in q["versions"], "slug already exists: inspect first"
    atomic_json(OUT / (SLUG + ".gate.json"), q)
    intent.write_text(json.dumps(row, indent=2))
    r = subprocess.run([sys.executable, str(SKILL / "kaggle_as.py"), OWNER, "--", "kaggle", "kernels", "push", "-p", str(folder)],
                       capture_output=True, text=True)
    (OUT / (SLUG + ".push-reply")).write_text(r.stdout + r.stderr)
    assert r.returncode == 0 and "Kernel version 1 successfully pushed" in r.stdout, SLUG
    marker.write_text(r.stdout)
    print("LITE2_BERT_PUSHED", OWNER, SLUG, flush=True)


def collect(row):
    st = query(row)
    atomic_json(OUT / "status.json", [st])
    print(row["arm"], SLUG, st.get("status"), flush=True)
    if st.get("terminal") and not (ART / SLUG / "pulled.ok").exists():
        r = subprocess.run([sys.executable, str(SKILL / "kaggle_as.py"), OWNER, "--", sys.executable,
                            str(SKILL / "pull_kernel_output.py"), f"{OWNER}/{SLUG}/{row['version']}", "-p", str(ART / SLUG),
                            "-j", "16", "--pattern", PULL], cwd=ROOT)
        if r.returncode == 0:
            (ART / SLUG / "pulled.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        print("  pulled rc", r.returncode, flush=True)
    print("ALL_TERMINAL" if st.get("terminal") else "WAITING", flush=True)


def audit(row):
    from scripts.audit_run import audit as bounded, test_support
    read = lambda d: [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines() if x.strip()]
    d, R = ART / SLUG / SLUG, row["endpoint"]
    rs, ref, cfg = read(d), read(REF), json.loads((d / "config.json").read_text())
    try:
        rep, err = bounded(d, test_support(ROOT / "runs/test_support.npy"), 42, 50), ""
    except AssertionError as e:                       # locked HP/model, LR, test support, ten metrics from CM, finite snapshots
        rep, err = None, repr(e)[:300]
    hp = {}
    for kv in row["hp"]:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        hp[k] = v
    man = json.loads((ART / SLUG / "provenance/source_manifest.json").read_text())
    chk = {"bounded_audit": rep is not None, "rounds_exact": [r["round"] for r in rs] == list(range(1, R + 1)),
           "method": cfg["method"] == row["method"], "hp_eq_launched": all(cfg["hp"].get(k) == v for k, v in hp.items()),
           "source": man["base_sha"] == row["base_sha"] and man["overlay"] == row["overlay_sha256"]
           and all(sha(ART / SLUG / "provenance/source" / p) == h for p, h in row["overlay_sha256"].items()),
           "skip_le_1pct": all(r["skipped_frac"] <= .01 for r in rs),
           "clients_applied": all(r["clients_without_applied"] == 0 for r in rs)}
    chk["pass"] = all(chk.values())
    delta = [a["private/f1_macro"] - b["private/f1_macro"] for a, b in zip(rs, ref)]
    boot = cfg["payload_bytes"]["bootstrap"]
    res = {"checks": chk, "bounded_error": err, "macro_end": rs[-1]["private/f1_macro"], "control_end": ref[len(rs) - 1]["private/f1_macro"],
           "delta_by_round": delta, "delta_end": delta[-1], "delta_mean_last5": sum(delta[-5:]) / 5,
           "delta_p10_end": rs[-1]["private/client_f1_macro_p10"] - ref[len(rs) - 1]["private/client_f1_macro_p10"],
           "bootstrap_bytes": boot, "round_bytes": sum(r["bytes_up"] + r["bytes_down"] for r in rs),
           "control_round_bytes": sum(r["bytes_up"] + r["bytes_down"] for r in ref[:len(rs)]),
           "t_train_hours": sum(r["t_train"] for r in rs) / 3600, "control_t_train_hours": sum(r["t_train"] for r in ref[:len(rs)]) / 3600}
    ok = chk["pass"] and res["delta_mean_last5"] >= GATE and res["delta_end"] >= GATE and boot["up"] + boot["down"] <= BOOT_MAX
    res["verdict"] = "PASS" if ok else "FAIL"
    atomic_json(OUT / "audit.json", {row["arm"]: res})
    print(row["arm"], "audit", "PASS" if chk["pass"] else "FAIL", {k: v for k, v in chk.items() if not v}, err,
          "macro %.4f vs T3 %.4f dTB %+.4f dEnd %+.4f boot %.0f B" % (res["macro_end"], res["control_end"], res["delta_mean_last5"],
                                                                     res["delta_end"], boot["up"] + boot["down"]), res["verdict"], flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd, row = sys.argv[1], build()[0]
    if cmd != "build":
        {"launch": launch, "collect": collect, "audit": audit}[cmd](row)


if __name__ == "__main__":
    main()
