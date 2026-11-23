#!/usr/bin/env python3
"""Inventory of every federated training run (repo + ORICO + ORICO zip archives).

Read-only on everything except artifacts/communication-budget/work/. Stdlib only.
Usage: python3 artifacts/communication-budget/scripts/inventory.py
Outputs (artifacts/communication-budget/work/): run_inventory_raw.csv, round_comm_fields.csv,
missing_runs.csv, inventory_notes.md
"""
import csv
import glob
import hashlib
import json
import os
import re
import zipfile
from collections import Counter, defaultdict

REPO = "/home/odixe/project/experiment"
ORICO = "/run/media/odixe/ORICO/experiment-data"
ARCH = ORICO + "/experiment/archives"
OUT = REPO + "/artifacts/communication-budget/work"
SKIP_ABS = (ORICO + "/experiment/git", ORICO + "/fl_veremi_bert/dvc-cache")
K_FULL = 5343
LOC_PREF = {"repo": 0, "ORICO": 1, "zip": 2}

# ----------------------------------------------------------------------------- collection


class Run:
    """One run directory / zip folder holding rounds.jsonl and/or config.json."""

    def __init__(self, location, label, cfg_bytes, rounds_bytes, abs_dir=""):
        self.location, self.label, self.abs_dir = location, label, abs_dir
        self.cfg = None
        if cfg_bytes is not None:
            try:
                self.cfg = json.loads(cfg_bytes)
            except Exception:
                pass
        self.recs, self.bad_lines = [], 0
        self.sha = hashlib.sha256(rounds_bytes).hexdigest() if rounds_bytes is not None else ""
        if rounds_bytes is not None:
            for ln in rounds_bytes.decode("utf8", "replace").splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        self.recs.append(json.loads(ln))
                    except Exception:
                        self.bad_lines += 1
        self.has_rounds = rounds_bytes is not None
        self.has_cfg = cfg_bytes is not None
        self.rounds_path = (abs_dir + "/rounds.jsonl") if (self.has_rounds and abs_dir) else (
            label.replace("!", "!/") + "/rounds.jsonl" if self.has_rounds else "")
        self.config_path = (abs_dir + "/config.json") if (self.has_cfg and abs_dir) else (
            label.replace("!", "!/") + "/config.json" if self.has_cfg else "")


def _read(p):
    with open(p, "rb") as f:
        return f.read()


def _walk(root, skip_names):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in skip_names and not os.path.join(dp, d).startswith(SKIP_ABS)]
        if "rounds.jsonl" in fn or "config.json" in fn:
            yield dp, fn


def collect():
    runs = []
    for dp, fn in _walk(REPO, {".git", "node_modules", "__pycache__"}):
        rel = os.path.relpath(dp, REPO)
        if rel.startswith("artifacts/communication-budget"):
            continue
        if "rounds.jsonl" not in fn and not (rel.startswith("runs/") or rel.startswith("bert/runs")):
            continue
        runs.append(Run("repo", rel, _read(dp + "/config.json") if "config.json" in fn else None,
                        _read(dp + "/rounds.jsonl") if "rounds.jsonl" in fn else None, dp))
    for dp, fn in _walk(ORICO, {"dvc-cache"}):
        rel = os.path.relpath(dp, ORICO)
        runs.append(Run("ORICO", rel, _read(dp + "/config.json") if "config.json" in fn else None,
                        _read(dp + "/rounds.jsonl") if "rounds.jsonl" in fn else None, dp))
    zip_manifests = {}
    for z in sorted(os.listdir(ARCH)):
        if not z.endswith(".zip"):
            continue
        zf = zipfile.ZipFile(os.path.join(ARCH, z))
        if "MANIFEST.json" in zf.namelist():
            try:
                zip_manifests[z] = json.loads(zf.read("MANIFEST.json"))
            except Exception:
                pass
        dirs = defaultdict(dict)
        for n in zf.namelist():
            b = n.rsplit("/", 1)[-1]
            if b in ("rounds.jsonl", "config.json"):
                dirs[n.rsplit("/", 1)[0] if "/" in n else ""][b] = n
        for d, m in sorted(dirs.items()):
            runs.append(Run("zip", "%s!%s" % (z, d), zf.read(m["config.json"]) if "config.json" in m else None,
                            zf.read(m["rounds.jsonl"]) if "rounds.jsonl" in m else None))
    return runs, zip_manifests


# ----------------------------------------------------------------------------- identity

METHOD_TOKENS = [("v2x-htfl", "v2x_htfl"), ("v2x_htfl", "v2x_htfl"), ("fd-ids", "fd_ids"), ("fd_ids", "fd_ids"),
                 ("perfedskd", "perfedskd"), ("pfedes", "pfedes"), ("tinyproto", "tinyproto"), ("fedtgp", "fedtgp"),
                 ("fedproto", "fedproto"), ("fedprox", "fedprox"), ("fedavg", "fedavg"), ("lwfednids", "lwfednids"),
                 ("canperfl", "canperfl"), ("apfed", "apfed"), ("afpha", "afpha"), ("nilm", "nilm"),
                 ("scaffold", "scaffold"), ("l4", "mobihet_l4")]
MODEL2SCEN = {"dagsnet": "dagsnet", "bert": "bert", "tabm": "tabm", "sae": "sae", "mlp": "mlp",
              "cnn-bigru-attention": "cnn-bigru", "cnn-bigru": "cnn-bigru"}
NPARAMS2MODEL = {77904: "bert", 395408: "dagsnet"}
TIMING_RE = re.compile(r"(^t_|^timing/|seconds)")
COMM_RE = re.compile(r"(bytes|payload|select|particip|active|sampled|n_clients|num_clients|clients_|_clients|"
                     r"n_protos|synth_rows|synthetic_rows|real_frac|applied_steps|skipped_(frac|steps)|frac$|"
                     r"^clients$|q_forward_rows|floor_extra_steps)")
METRIC_PREFIX = re.compile(r"^(global|client|client_proto|private|private_local|proxy|local)/")


def norm_method(m):
    m = (m or "").lower().replace("-", "_")
    return {"mobihet_budget": "mobihet_l4", "mobihet": "mobihet_smoke_variant"}.get(m, m)


def method_from_label(label):
    base = label.lower().split("!")[-1]
    base = base.replace("pull", "").replace("mix128-case", "mixcase")
    for tok, m in METHOD_TOKENS:
        if tok in base:
            return m
    return ""


def identify(r):
    """Fill scenario/backbone/method/seed/K/planned/kind hints from config + label."""
    L = r.label.lower()
    cfg = r.cfg or {}
    hp = cfg.get("hp") if isinstance(cfg.get("hp"), dict) else {}
    r.raw_method = cfg.get("method") or ""
    r.is_central = ("central" in L) or ("epochs" in cfg and "hp" not in cfg and "method" not in cfg)
    m = norm_method(cfg.get("method"))
    if not m:
        mm = re.search(r"/(?:seed\d+/)?([a-z0-9_]+)/seed\d+$", L)
        m = mm.group(1) if mm and mm.group(1) in [t[1] for t in METHOD_TOKENS] else method_from_label(L)
    r.method = "centralized" if r.is_central else m
    # seed
    s = hp.get("seed", cfg.get("seed"))
    if s is None:
        ms = re.search(r"(?:-s|seed)(4[123])\b", L)
        s = int(ms.group(1)) if ms else None
    r.seed = s
    # planned rounds and K
    r.planned = hp.get("rounds", cfg.get("epochs"))
    cl = cfg.get("clients")
    r.client_ids = cl if isinstance(cl, list) else None
    r.K = len(cl) if isinstance(cl, list) else (cl if isinstance(cl, int) else None)
    # mixed or single
    case = cfg.get("case")
    if not case:
        mc = re.search(r"case([123])", L)
        case = "case" + mc.group(1) if mc else None
    r.case = case
    if case and not r.is_central:
        r.scenario = "mixed-" + case
        r.backbone = "+".join(sorted((cfg.get("groups") or {}).keys())) or ""
    else:
        inner = cfg.get("cfg") if isinstance(cfg.get("cfg"), dict) else {}
        model = inner.get("model") or cfg.get("model") or NPARAMS2MODEL.get(cfg.get("n_params"))
        if not model:
            mo = re.search(r"single-fl-artifact-supplement/([a-z-]+)/", L)
            if mo:
                model = mo.group(1)
            else:
                tail = "/".join(L.split("!")[-1].split("/")[-3:])
                for tok, mdl in (("flb-", "bert"), ("fl2-", "dagsnet"), ("mh-", "dagsnet"), ("cnn-bigru", "cnn-bigru-attention"),
                                 ("cnnbigru", "cnn-bigru-attention"), ("tabm", "tabm"), ("sae", "sae"), ("mlp", "mlp"),
                                 ("bert", "bert"), ("dagsnet", "dagsnet")):
                    if tok in tail:
                        model = mdl
                        break
        r.backbone = model or ""
        r.scenario = MODEL2SCEN.get(r.backbone, "")
    return r


def classify_kind(r, last):
    L = r.label.lower()
    base = L.split("!")[-1]
    parts = base.split("/")
    if (re.search(r"(^|/)(sm_|smoke|c_smoke|cpu_smoke)", base) or any(p.startswith("smoke") for p in parts)):
        return "smoke"
    if r.is_central:
        return "centralized"
    if re.search(r"compute-(2026|detailed|inputs|legacy)", base):
        return "compute-trace"
    if r.method == "scaffold":
        return "dropped-method"
    if r.method.startswith("mobihet") and r.method != "mobihet_l4":
        return "probe"
    if re.search(r"probe", base):
        return "probe"
    if re.search(r"pilot|dbg-", base):
        return "pilot"
    if r.method == "mobihet_l4" and not re.search(r"(fbl-l4-top-s4\d-r50|mixl4-case\d-s4\d|(cnnbigru|mlp|sae|tabm)-l4-s4\d)", base):
        return "probe"  # MobiHet L4 hyper-parameter / ablation sweeps (q4, l2, p2, n2c, lite, sim, fb-n2 ...)
    if r.K is not None and r.K != K_FULL:
        return "smoke" if (last or 0) <= 3 else "probe"
    if r.planned is not None and r.planned <= 3:
        return "smoke"
    if "l4-port" in base or "regress-" in base:
        return "probe"
    return "production-candidate"


# ----------------------------------------------------------------------------- round helpers


def rnum(rec):
    v = rec.get("round", rec.get("epoch"))
    return v if isinstance(v, int) else None


def sig(rec):
    out = []
    for k in sorted(rec):
        if TIMING_RE.search(k):
            continue
        v = rec[k]
        if isinstance(v, float):
            v = float("%.5g" % v)
        out.append((k, json.dumps(v, sort_keys=True)[:200]))
    return hashlib.md5(repr(out).encode()).hexdigest()


def session_rounds(r):
    d = {}
    dups = 0
    for rec in r.recs:
        n = rnum(rec)
        if n is None:
            continue
        if n in d:
            dups += 1
        d[n] = rec  # last occurrence wins
    r.by_round = d
    r.dup_rounds = dups
    r.first = min(d) if d else None
    r.last = max(d) if d else None
    r.contiguous = bool(d) and sorted(d) == list(range(1, r.last + 1))


def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def flat_comm(rec):
    """Flatten comm/participation-related numeric fields of one round record."""
    out = {}
    for k, v in rec.items():
        if k.startswith("model/") or METRIC_PREFIX.match(k) or TIMING_RE.search(k):
            continue
        if k.startswith("health/"):
            if COMM_RE.search(k.split("/", 2)[-1]) and numeric(v):
                out[k] = v
            continue
        if COMM_RE.search(k):
            if numeric(v):
                out[k] = v
            elif isinstance(v, dict):
                for a, b in v.items():
                    if numeric(b):
                        out["%s.%s" % (k, a)] = b
    return out


# ----------------------------------------------------------------------------- main


def load_results():
    cells, sources = {}, {}
    for f in sorted(glob.glob(REPO + "/experiments/*/results.csv")):
        scen = f.split("/")[-2]
        for row in csv.DictReader(open(f, encoding="utf-8-sig")):
            k = (scen, row["method"], int(row["seed"]))
            cells.setdefault(k, []).append(row)
            sources.setdefault(k, set()).add(row["source"])
    return cells, sources


def load_provenance():
    """round-metrics run.json: sha256 of rounds.jsonl files declared as the canonical provenance."""
    idx = json.load(open(REPO + "/runs/round-metrics/index.json"))
    prov = defaultdict(set)
    status = {}
    for e in idx["runs"]:
        k = (e["scenario"], e["method"], int(e["seed"]))
        status[k] = (e["status"], e["last_recorded"], e["target"], e["rows"])
        try:
            rj = json.load(open(REPO + "/runs/round-metrics/%s/run.json" % e["run"]))
        except Exception:
            continue
        for p in rj.get("provenance", []):
            if p.get("path", "").endswith("rounds.jsonl") and p.get("sha256"):
                prov[k].add(p["sha256"])
    return prov, status


def load_assignments():
    base = REPO + "/experiments/mixed-model-v1/"
    man = json.load(open(base + "manifest.json"))
    asg = {}
    for case, v in man["cases"].items():
        raw = open(base + v["file"], "rb").read()
        asg[case] = {"sha": hashlib.sha256(raw).hexdigest(), "manifest_sha": v["sha256"], "counts": v["counts"],
                     "map": {int(rw["client_id"]): rw["model"] for rw in csv.DictReader(raw.decode().splitlines())}}
    return asg


def source_match(r, sources):
    """Does the Kaggle ref / path named in results.csv `source` refer to this session? (exact ref or path component)"""
    if not sources:
        return False
    toks = [t for t in re.split(r"[\s,;()]+", " ".join(sources).lower()) if t]
    refs = {"/".join(t.split("/")[:2]) for t in toks if re.match(r"^[a-z0-9_.]+/[a-z0-9_.-]+(/\d+)?$", t)}
    padded = ["/" + t + "/" for t in toks]
    for c in r.label.lower().split("!")[-1].split("/")[-3:]:
        if len(c) < 8:
            continue
        if "__" in c:
            parts = c.split("__")
            if len(parts) >= 2 and (parts[0] + "/" + parts[1]) in refs:
                return True
        if any(("/" + c + "/") in p for p in padded):
            return True
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    runs, zip_manifests = collect()
    cells, results_sources = load_results()
    prov, idx_status = load_provenance()
    asg = load_assignments()

    # supplement-zip MANIFEST gives kaggle ref per (model, method, seed)
    supp_ref = {}
    for z, m in zip_manifests.items():
        for e in m.get("results", []):
            supp_ref[(e.get("model"), e.get("method"), e.get("seed"))] = e.get("ref")

    for r in runs:
        identify(r)
        session_rounds(r)
        r.kind0 = classify_kind(r, r.last)
        if (r.kind0 in ("probe", "pilot") and not r.method.startswith("mobihet")
                and source_match(r, results_sources.get((r.scenario, r.method, r.seed), set()))):
            r.kind0 = "production-candidate"   # named probe/pilot but it IS the run cited by results.csv
            r.reclassified = "name says probe/pilot but results.csv source cites this run"
        if r.scenario == "" and r.kind0 not in ("smoke", "centralized"):
            r.scenario = "unknown"

    # ---------------- group production candidates into logical runs
    groups = defaultdict(list)
    for r in runs:
        if r.kind0 == "production-candidate" and r.has_rounds and r.method and r.seed is not None:
            groups[(r.scenario, r.method, r.seed)].append(r)
    # cfg-only candidates (no rounds) are listed but never canonical; they only serve as a config donor
    cfg_pool = defaultdict(list)
    for r in runs:
        if r.kind0 == "production-candidate" and not r.has_rounds and r.cfg is not None and r.method and r.seed is not None:
            cfg_pool[(r.scenario, r.method, r.seed)].append(r)
    canon = {}      # key -> dict(...)
    notes_global = []
    for key, ss in groups.items():
        sk = (key[0], key[1], key[2])
        src = results_sources.get(sk, set())
        provs = prov.get(sk, set())
        for r in ss:
            r.src_match = source_match(r, src) or (r.sha in provs)
            ref = supp_ref.get((r.backbone, r.method, r.seed)) if "single-fl-artifact-supplement" in r.label else None
            if ref:
                refs = {"/".join(t.split("/")[:2]) for t in re.split(r"[\s,;()]+", " ".join(src).lower())}
                r.supp_ref = ref
                if ref.lower() in refs:
                    r.src_match = True

        def rank(r):
            return (-(r.last or 0), 0 if r.contiguous else 1, 0 if r.src_match else 1, LOC_PREF[r.location],
                    0 if r.has_cfg else 1, r.label)
        ss.sort(key=rank)
        best = ss[0]
        rec_map = {}
        stitched = False
        if best.contiguous:
            rec_map = dict(best.by_round)
            Rr = best.last
            canon_sessions = [best]
        else:
            stitched = True
            canon_sessions = []
            for r in ss:
                for n, rec in r.by_round.items():
                    if n not in rec_map:
                        rec_map[n] = rec
                        if r not in canon_sessions:
                            canon_sessions.append(r)
            Rr = 0
            while Rr + 1 in rec_map:
                Rr += 1
            rec_map = {n: rec_map[n] for n in range(1, Rr + 1)}
        canon[key] = {"best": best, "R": Rr, "rec": rec_map, "stitched": stitched, "sessions": ss,
                      "canon_sessions": canon_sessions}
        # relation of every other session to the canonical one
        csig = {n: sig(x) for n, x in rec_map.items()}
        for r in ss:
            r.group = key
            r.canonical = "yes" if r is best else "no"
            r.dup_of, r.rel, r.overlap = "", "", ""
            if r is best:
                r.rel = "canonical" + ("-stitched" if stitched else "")
                continue
            common = [n for n in r.by_round if n in csig]
            same = [n for n in common if sig(r.by_round[n]) == csig[n]]
            r.overlap = "%d/%d rounds identical to canonical (overlap %s)" % (
                len(same), len(common), ("1-%d" % max(common)) if common else "none")
            if r.sha == best.sha:
                r.rel, r.dup_of = "exact-duplicate", best
            elif common and len(same) == len(common) and (r.last or 0) == best.last:
                r.rel, r.dup_of = "content-duplicate", best
            elif common and len(same) == len(common):
                r.rel = "prefix-session"          # earlier session of the same chain (cumulative rounds.jsonl)
            elif not common:
                r.rel = "disjoint-session"
            else:
                r.rel = "DIVERGES"                # overlap with different content: independent re-run
        # sessions of group in non-candidate kinds are handled below

    # ---------------- attach kind/status to every run
    for r in runs:
        r.group = getattr(r, "group", None)
        r.canonical = getattr(r, "canonical", "no")
        r.dup_of = getattr(r, "dup_of", "")
        r.rel = getattr(r, "rel", "")
        r.overlap = getattr(r, "overlap", "")
    def logical_id(key):
        return "%s__%s__s%s" % key

    for r in runs:
        if r.group:
            g = canon[r.group]
            if r.canonical == "yes":
                r.kind = "production"
            elif r.rel in ("exact-duplicate", "content-duplicate"):
                r.kind = "production-duplicate"
            elif r.rel == "DIVERGES":
                r.kind = "superseded-rerun"
            else:
                r.kind = "continuation-session"
        else:
            r.kind = r.kind0 if r.kind0 != "production-candidate" else ("config-only" if not r.has_rounds else "unclassified")

    # expected R per cell
    def expected_R(key, r_planned):
        rows = cells.get(key)
        if rows:
            return int(rows[0]["round"])
        return r_planned

    # ---------------- clients per backbone
    def clients_per_backbone(r):
        if not r.case:
            return "", ""
        a = asg.get(r.case)
        if a is None:
            return "", ""
        if r.client_ids is not None:
            c = Counter(a["map"].get(i, "?") for i in r.client_ids)
            ok = (r.cfg.get("assignment_sha256") in (a["sha"], None))
            return json.dumps(dict(c), sort_keys=True), ("assignment_sha256 matches experiments/mixed-model-v1/%s.csv" % r.case
                                                        if ok else "ASSIGNMENT SHA MISMATCH")
        return json.dumps(a["counts"], sort_keys=True), "no config.json: full-K manifest counts ASSUMED (unverified)"

    # ---------------- raw inventory rows
    raw_rows = []
    for r in sorted(runs, key=lambda x: (x.scenario, x.method, str(x.seed), x.location, x.label)):
        key = r.group
        sk = (r.scenario, r.method, r.seed)
        in_results = sk in cells
        status = ""
        notes = []
        if r.kind == "production":
            g = canon[key]
            R = g["R"]
            exp = expected_R(sk, r.planned)
            status = "complete" if (exp and R >= exp) else "partial"
            if g["stitched"]:
                notes.append("canonical rounds stitched from several sessions")
        elif r.kind in ("production-duplicate", "continuation-session"):
            g = canon[key]
            status = "complete" if (r.last or 0) >= (r.planned or 10 ** 9) and r.contiguous else "partial"
        elif r.kind == "smoke":
            status = "smoke"
        elif r.kind in ("pilot", "probe"):
            status = "pilot" if r.kind == "pilot" else ("complete" if r.planned and (r.last or 0) >= r.planned else "partial")
        elif r.kind == "centralized":
            status = "n/a"
        else:
            status = "complete" if r.planned and (r.last or 0) >= r.planned else ("partial" if r.has_rounds else "no-rounds")
        if not r.has_rounds:
            notes.append("config.json only (no rounds.jsonl)")
        if r.has_rounds and not r.has_cfg:
            notes.append("rounds.jsonl without config.json (payload_bytes not in this session)")
        if r.dup_rounds:
            notes.append("%d duplicate round lines (last wins)" % r.dup_rounds)
        if r.bad_lines:
            notes.append("%d unparsable jsonl lines" % r.bad_lines)
        if r.has_rounds and not r.contiguous and r.last:
            miss = sorted(set(range(1, r.last + 1)) - set(r.by_round))
            notes.append("round gaps: %s" % miss[:10])
        if r.raw_method in ("mobihet_budget",):
            notes.append("config method name mobihet_budget = legacy name of MobiHet L4 (single-backbone repo)")
        if r.method == "canperfl":
            notes.append("CANPerFL: planned 10 = fine-tune epochs after FedAvg r50 state; FedAvg part not logged in this rounds.jsonl")
        if r.method.startswith("mobihet") and r.kind == "probe":
            notes.append("MobiHet variant/ablation probe; raw config method=%s" % r.raw_method)
        if r.rel == "DIVERGES":
            notes.append("overlapping rounds differ from canonical: independent re-run, NOT a continuation")
        cpb, cpb_note = clients_per_backbone(r) if (r.scenario or "").startswith("mixed") else ("", "")
        if cpb_note:
            notes.append(cpb_note)
        sup = ""
        if r.group:
            sup = "; ".join(sorted(results_sources.get(sk, [])))[:300]
        comm_app = "no" if r.kind == "centralized" else "yes"
        raw_rows.append({
            "logical_run_id": logical_id(key) if key else "",
            "session_id": "%s:%s" % (r.location, r.label),
            "scenario": r.scenario, "backbone": r.backbone, "method": r.method,
            "seed": "" if r.seed is None else r.seed,
            "run_kind": r.kind, "status": status,
            "n_clients": "" if r.K is None else r.K, "clients_per_backbone": cpb,
            "rounds_planned": "" if r.planned is None else r.planned,
            "round_first": "" if r.first is None else r.first, "round_last": "" if r.last is None else r.last,
            "n_round_lines": len(r.recs),
            "rounds_completed_canonical": canon[key]["R"] if key else "",
            "canonical": r.canonical,
            "duplicate_of": ("%s:%s" % (r.dup_of.location, r.dup_of.label)) if r.dup_of else "",
            "relation_to_canonical": r.rel, "overlap_with_canonical": r.overlap,
            "location": r.location, "config_path": r.config_path, "rounds_path": r.rounds_path,
            "rounds_sha256": r.sha[:16],
            "results_csv_match": ("yes" if in_results else "no") if (r.group or r.kind in ("production",)) else "",
            "results_source_matches_session": ("yes" if getattr(r, "src_match", False) else "") if r.group else "",
            "results_source": sup, "comm_applicable": comm_app,
            "in_budget_scope": "yes" if r.kind == "production" else "no",
            "notes": " | ".join(notes),
        })
    cols = list(raw_rows[0].keys())
    with open(OUT + "/run_inventory_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(raw_rows)

    # ---------------- round_comm_fields.csv (canonical runs only)
    comm_rows = []
    allkeys_by_family = defaultdict(Counter)
    for key, g in sorted(canon.items()):
        best = g["best"]
        R = g["R"]
        recs = [g["rec"][n] for n in range(1, R + 1)]
        flat = [flat_comm(x) for x in recs]
        keys = sorted({k for d in flat for k in d})
        sums, means, npres = {}, {}, {}
        for k in keys:
            vals = [d[k] for d in flat if k in d]
            sums[k] = sum(vals)
            means[k] = sum(vals) / len(vals)
            npres[k] = len(vals)
        # config (own or inherited from a sibling session of the same logical run)
        cfgrun, cfg_src = None, "missing"
        if best.cfg is not None:
            cfgrun, cfg_src = best, "own"
        else:
            for s in g["sessions"] + sorted(cfg_pool.get(key, []), key=lambda x: (LOC_PREF[x.location], x.label)):
                if s.cfg is not None and "payload_bytes" in s.cfg:
                    cfgrun, cfg_src = s, "INHERITED from %s %s:%s" % (
                        "sibling session" if s.has_rounds else "config-only dir", s.location, s.label)
                    break
        cfg = cfgrun.cfg if cfgrun else {}
        K = best.K if best.K is not None else (cfgrun.K if cfgrun else None)
        pbytes = json.dumps(cfg.get("payload_bytes"), separators=(",", ":")) if "payload_bytes" in cfg else ""
        cfg_comm = {}
        for scope, d in (("", cfg), ("hp.", cfg.get("hp") if isinstance(cfg.get("hp"), dict) else {})):
            for k, v in d.items():
                if k != "clients" and COMM_RE.search(k) and k != "payload_bytes":
                    cfg_comm[scope + k] = v if len(json.dumps(v)) < 200 else "<big>"
        # participation per round
        pk, ppr = "", []
        if any("clients_without_applied" in d for d in flat) and K:
            pk = "derived: n_clients - clients_without_applied (clients that applied >=1 step)"
            ppr = [K - d["clients_without_applied"] for d in flat if "clients_without_applied" in d]
        if any("selected_frac" in d for d in flat) and K:
            pk = "derived: selected_frac * n_clients"
            ppr = [round(d["selected_frac"] * K, 3) for d in flat if "selected_frac" in d]
        cpb = ""
        cpbn = ""
        if best.scenario.startswith("mixed"):
            cpb, cpbn = clients_per_backbone(cfgrun or best)
        fam = ("mixed-" + best.method) if best.scenario.startswith("mixed") else best.scenario
        for k in keys:
            allkeys_by_family[("mixed" if best.scenario.startswith("mixed") else
                               "bert" if best.scenario == "bert" else "single")
                              + ("/l4" if best.method == "mobihet_l4" else "")][k] += 1
        bu, bd = sums.get("bytes_up"), sums.get("bytes_down")
        notes = []
        if g["stitched"]:
            notes.append("stitched")
        if cfg_src != "own":
            notes.append("config " + cfg_src)
        if not pbytes:
            notes.append("no payload_bytes in config")
        if not recs:
            notes.append("no rounds")
        comm_rows.append({
            "logical_run_id": logical_id(key), "scenario": key[0], "method": key[1], "seed": key[2],
            "backbone": best.backbone, "canonical_session": "%s:%s" % (best.location, best.label),
            "rounds_completed": R, "n_clients": "" if K is None else K, "clients_per_backbone": cpb,
            "comm_keys_in_rounds": json.dumps(keys),
            "comm_sums_over_rounds_1_R": json.dumps(sums), "comm_means_per_round": json.dumps(means),
            "comm_rounds_present": json.dumps(npres),
            "bytes_up_sum": "" if bu is None else bu, "bytes_down_sum": "" if bd is None else bd,
            "bytes_up_mean": "" if bu is None else bu / npres["bytes_up"],
            "bytes_down_mean": "" if bd is None else bd / npres["bytes_down"],
            "participation_source": pk or "not logged (rounds.jsonl has no selection/participant key)",
            "participants_per_round": json.dumps(ppr) if ppr else "",
            "payload_bytes_config": pbytes, "payload_config_source": cfg_src,
            "config_comm_keys": json.dumps(cfg_comm), "notes": " | ".join(notes),
        })
    ccols = list(comm_rows[0].keys())
    with open(OUT + "/round_comm_fields.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ccols)
        w.writeheader()
        w.writerows(comm_rows)

    # ---------------- missing_runs.csv
    miss = []
    cent_cells = 0
    mk_cells = 0
    for sk in sorted(cells):
        scen, meth, seed = sk
        rows = cells[sk]
        if meth == "centralized":
            cent_cells += 1
            continue
        mk_cells += 1
        exp = int(rows[0]["round"])
        g = canon.get(sk)
        reason = ""
        if g is None:
            st = idx_status.get(sk)
            reason = "no rounds.jsonl found anywhere (repo, ORICO, zip archives)"
            if os.path.exists(REPO + "/runs/round-metrics/%s/%s/s%d/metrics.csv" % sk):
                reason += "; per-round METRICS exist in runs/round-metrics (metrics.csv, no comm fields)"
            # any config-only dir?
            cand = [r for r in runs if (r.scenario, r.method, r.seed) == sk and r.kind0 == "production-candidate"]
            if cand:
                reason += "; config.json-only dirs: " + ", ".join(sorted({"%s:%s" % (c.location, c.label) for c in cand})[:3])
        else:
            if g["R"] < exp:
                reason = "rounds.jsonl reaches only round %d < expected %d" % (g["R"], exp)
            elif g["stitched"]:
                reason = ""  # complete via stitching; not missing, noted elsewhere
            if not reason and not any(s.cfg is not None for s in g["sessions"] + cfg_pool.get(sk, [])):
                reason = "rounds ok but no config.json in any session: payload_bytes unknown"
            if not reason and g["best"].cfg is None:
                miss.append({"scenario": scen, "method": meth, "seed": seed, "expected_round": exp, "severity": "warning",
                             "results_status": rows[0]["status"], "results_source": rows[0]["source"][:200],
                             "reason": "per-round data complete (rounds 1..%d) but canonical session has no config.json; "
                                       "payload_bytes taken from a sibling/config-only dir (same scenario/method/seed), unverified" % g["R"]})
        if reason:
            miss.append({"scenario": scen, "method": meth, "seed": seed, "expected_round": exp, "severity": "missing",
                         "results_status": rows[0]["status"], "results_source": rows[0]["source"][:200], "reason": reason})
    with open(OUT + "/missing_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["scenario", "method", "seed", "expected_round", "severity", "results_status", "results_source", "reason"])
        w.writeheader()
        w.writerows(miss)

    # ---------------- sanity checks + notes
    problems = []
    for key, g in canon.items():
        rl = sorted(g["rec"])
        if rl != list(range(1, g["R"] + 1)):
            problems.append("canonical rounds not 1..R: %s" % (key,))
        if g["best"].last != g["R"] and not g["stitched"]:
            problems.append("canonical round_last != rounds_completed: %s" % (key,))
    per_scen = defaultdict(lambda: [0, 0])
    for sk in cells:
        if sk[1] == "centralized":
            continue
        per_scen[sk[0]][0] += 1
        if sk in canon:
            per_scen[sk[0]][1] += 1
    extra_canon = [k for k in canon if k not in cells]
    write_notes(runs, canon, cells, per_scen, extra_canon, miss, problems, cent_cells, allkeys_by_family, raw_rows,
                comm_rows)
    print("runs scanned:", len(runs), "canonical logical runs:", len(canon), "missing cells:", len(miss))
    print("problems:", problems)
    for s, (a, b) in sorted(per_scen.items()):
        print(" %-12s results cells %3d  with canonical rounds %3d" % (s, a, b))
    print("canonical groups not in results.csv:", extra_canon)


def write_notes(runs, canon, cells, per_scen, extra_canon, miss, problems, cent_cells, keysfam, raw_rows, comm_rows):
    kinds = Counter((r["scenario"], r["run_kind"]) for r in raw_rows)
    prod = Counter((r["scenario"], r["method"]) for r in raw_rows if r["run_kind"] == "production")
    sess = Counter(r["relation_to_canonical"] for r in raw_rows if r["relation_to_canonical"])
    loc = Counter((r["run_kind"], r["location"]) for r in raw_rows if r["run_kind"] == "production")
    div = [r for r in raw_rows if r["relation_to_canonical"] == "DIVERGES"]
    inh = [r for r in comm_rows if "INHERITED" in r["payload_config_source"]]
    nopay = [r for r in comm_rows if not r["payload_bytes_config"]]
    L = []
    A = L.append
    A("# Inventory notes (generated by scripts/inventory.py)\n")
    A("## Counting rules")
    A("- Scanned: repo (excluding .git), ORICO experiment-data (excluding dvc-cache, git bundle), and in-memory members of the 4 zip archives on ORICO "
      "(the repo copies of artifacts/*.zip are Git-LFS pointers, not zips). One raw row per directory holding rounds.jsonl and/or config.json.")
    A("- Identity: method/seed/backbone/K/planned rounds come from config.json (`method`, `hp.seed`, `cfg.model` or `n_params` 77904=bert / 395408=dagsnet, "
      "`clients`, `hp.rounds`, `case`); when config.json is absent they come from the path name. Legacy method name `mobihet_budget` is mapped to mobihet_l4 "
      "(single-backbone BERT/DAGSNet MobiHet L4); other `mobihet_*` methods (core, stats*, pair, rel, lite, compute) are probes.")
    A("- run_kind: smoke (sm_*, smoke*, K<5343 with <=3 rounds), centralized, compute-trace (compute-2026*/compute-detailed*), dropped-method (scaffold), probe "
      "(`probe` in name, MobiHet variants, K<5343 longer runs, l4-port, regress), pilot (`pilot`/`dbg-`), then production / production-duplicate / continuation-session.")
    A("- Logical run = (scenario, method, seed). Candidates = all non-smoke/pilot/probe FL sessions with rounds.jsonl for that key. Every session's rounds.jsonl is CUMULATIVE "
      "(a resumed Kaggle session re-emits rounds 1..N from the parent checkpoint chain), so sessions overlap on rounds 1..k; overlap is verified per round by a timing-free content signature.")
    A("- Canonical = session with the largest contiguous last round (ties: source matches results.csv `source` or round-metrics provenance sha256, then repo > ORICO > zip, "
      "then has config.json). Canonical rounds are exactly 1..R from that single session, so no round is counted twice. "
      "If no session were contiguous the rounds would be stitched (flag `canonical-stitched`).")
    A("- Other sessions: exact-duplicate (same rounds.jsonl sha256) or content-duplicate -> duplicate_of; prefix-session (earlier hop of the same chain, identical on overlap) -> continuation-session; "
      "DIVERGES (different content on overlapping rounds) -> independent re-run, listed, not used.")
    A("- rounds_completed = max round with a committed record in canonical rounds.jsonl; status complete if R >= expected round (results.csv `round`, i.e. 50, or 10 for CANPerFL), else partial.")
    A("- Mixed clients_per_backbone: from cfg.clients ids looked up in experiments/mixed-model-v1/caseN.csv (sha256 of the csv equals cfg.assignment_sha256 where verified). "
      "Sessions without config.json get the manifest counts flagged as assumed.")
    A("- results.csv expected cells: (scenario, method, seed) rows of each experiments/*/results.csv (all 'audited'). Centralized cells are excluded from missing_runs (comm_applicable=no): %d cells." % cent_cells)
    A("\n## Sanity checks")
    A("- Problems detected (canonical rounds not 1..R, or round_last != rounds_completed): %s" % (problems or "none"))
    A("- Cells vs canonical (non-centralized): " + "; ".join("%s %d/%d" % (s, b, a) for s, (a, b) in sorted(per_scen.items())))
    A("- Canonical logical runs NOT in results.csv: %s" % (extra_canon or "none"))
    A("- missing_runs.csv rows: %d" % len(miss))
    A("\n## Totals: raw rows by scenario x run_kind")
    for (s, k), n in sorted(kinds.items()):
        A("- %s | %s | %d" % (s or "(none)", k, n))
    A("\n## Canonical production runs by scenario x method")
    for (s, m), n in sorted(prod.items()):
        A("- %s | %s | %d" % (s, m, n))
    A("\n## Relations of sessions to canonical: " + ", ".join("%s=%d" % kv for kv in sorted(sess.items())))
    A("## Canonical location of production runs: " + ", ".join("%s/%s=%d" % (k[0], k[1], n) for k, n in sorted(loc.items())))
    A("\n## Ambiguities / flags")
    A("- DIVERGES sessions (same key, different content on overlap): %d" % len(div))
    for r in div[:40]:
        A("  - %s | %s | %s" % (r["logical_run_id"], r["session_id"], r["overlap_with_canonical"]))
    A("- Canonical runs whose payload_bytes had to be INHERITED from a sibling session (no config.json next to canonical rounds.jsonl): %d" % len(inh))
    for r in inh[:40]:
        A("  - %s: %s" % (r["logical_run_id"], r["payload_config_source"]))
    A("- Canonical runs with NO payload_bytes in any config: %d" % len(nopay))
    for r in nopay[:40]:
        A("  - %s (%s)" % (r["logical_run_id"], r["payload_config_source"]))
    A("- CANPerFL: planned/target 10 rounds are fine-tune epochs; initial model = FedAvg r50 state of the same seed (see LESSONS.md). Its own rounds.jsonl does not contain the FedAvg communication; later step must add it explicitly.")
    A("- Mixed `payload_bytes` accounting string says 'per-client worst-case 16 present classes; fp32' (prototype methods): it is an upper bound, not measured.")
    A("- Per-round bytes are logged ONLY for MobiHet L4 (bytes_up/bytes_down); all other methods rely on config payload_bytes x rounds x participants. "
      "Participation is not logged except perfedskd `selected_frac` and L4 `clients_without_applied`.")
    A("- MobiHet L4 `bytes_up`/`bytes_down` per round are PER-VEHICLE MEAN bytes (methods/mobihet.py: float(up[0]/up[1]); mixed: 'class-Q packets, per vehicle'), not network totals; "
      "bootstrap bytes are separate, in config.json payload_bytes.bootstrap (and codec). Verified only from code comments, not re-derived.")
    A("- DIVERGES = superseded earlier-generation re-runs (comparison-20261115 v1 SAE/TabM, mixed128 a1 v2x_htfl attempts); they have different content, cannot be cut into the canonical chain, and are not used.")
    A("- results.csv source matched the canonical session for every one of the %d canonical runs (exact Kaggle ref, path component, round-metrics provenance sha256, or supplement MANIFEST ref)." % len(comm_rows))
    A("- tabm fedavg s41: the cited source is a Kaggle run named `...-s41-probe` v3 (name says probe, results.csv cites it) -> treated as production.")
    A("- lwfednids / v2x_htfl have different n_params from the base backbone (student model / adapter): payload comes from each run's own config, do not reuse the FedAvg payload.")
    A("\n## Comm-related round keys seen in canonical runs, by family (count of runs)")
    for fam, c in sorted(keysfam.items()):
        A("- %s: %s" % (fam, json.dumps(dict(sorted(c.items())))))
    open(OUT + "/inventory_notes.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
