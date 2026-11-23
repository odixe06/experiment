"""Build work/payload_spec.csv and work/backbone_params.csv for the communication-budget audit.

    python artifacts/communication-budget/scripts/payload_specs.py [--refresh]

Pipeline (all read-only on the repo; CPU only, toy data, ~1 min):
  probe_methods.py (cwd repo)  -> work/probe_single.json   (5 backbones x 13 methods, tensors measured after init / round 1)
  probe_methods.py (cwd bert/) -> work/probe_bert.json
  probe_mixed.py               -> work/probe_mixed.json    (5 hetero methods x 3 cases)
  measured_stats.py            -> work/data_dependent_stats.*, l4_measured_bytes.csv, perfedskd_selected_frac.csv
Then assembles the CSVs. `basis`: measured = executed on toy env / parsed from run logs; calculated = formula from code with
measured shapes; estimated = needs an assumption (stated in discrepancy_note).
"""
import csv, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WORK = os.path.abspath(os.path.join(HERE, "..", "work"))
PY = sys.executable
refresh = "--refresh" in sys.argv


def ensure(path, cmd, cwd):
    if refresh or not os.path.exists(path):
        subprocess.run(cmd, cwd=cwd, check=True)
    return json.load(open(path))


single = ensure(f"{WORK}/probe_single.json", [PY, f"{HERE}/probe_methods.py", "--root", ".", "--out", f"{WORK}/probe_single.json"], ROOT)
bert = ensure(f"{WORK}/probe_bert.json", [PY, f"{HERE}/probe_methods.py", "--root", ".", "--out", f"{WORK}/probe_bert.json"], f"{ROOT}/bert")
mixed = ensure(f"{WORK}/probe_mixed.json", [PY, f"{HERE}/probe_mixed.py", "--out", f"{WORK}/probe_mixed.json"], ROOT)
if refresh or not os.path.exists(f"{WORK}/data_dependent_stats.json"):
    subprocess.run([PY, f"{HERE}/measured_stats.py"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
stats = json.load(open(f"{WORK}/data_dependent_stats.json"))
l4 = list(csv.DictReader(open(f"{WORK}/l4_measured_bytes.csv")))
sel = list(csv.DictReader(open(f"{WORK}/perfedskd_selected_frac.csv")))
PAIRS_PC = stats["pairs_per_client_mean"]                      # 13.0999 (client, class) pairs per client
for r in single + bert:
    assert "error" not in r, r
P = {(r["method"], r["backbone"]): r for r in single + bert}

COLS = ["method", "backbone", "scenario_family", "component", "direction", "frequency", "clients_rule", "elements_formula",
        "elements_value", "dtype", "bytes_per_client_per_event", "broadcast_or_unicast", "basis", "code_ref",
        "payload_bytes_method_value", "discrepancy_note", "bytes_per_client_per_event_data_avg"]
rows = []


def add(**k):
    rows.append({c: k.get(c, "") for c in COLS})


def sel_frac(backbone):
    def m(pref):
        v = [float(r["mean_selected_frac"]) for r in sel if pref in r["run"] and int(r["rounds"]) == 50]
        return (float(np.mean(v)), len(v)) if v else (None, 0)
    if backbone == "bert":
        return m("bert/runs/pulls/flb-perfedskd-s4")
    if backbone == "dagsnet":
        return m("round-metrics/evidence/dvc/fl2-perfedskd-s4")
    if backbone == "tabm":
        return m("comparison-20261115/") if True else (None, 0)
    return (None, 0)


SINGLE_BB = ["dagsnet", "tabm", "sae", "mlp", "cnn-bigru-attention"]
FAM = lambda bb: "bert" if bb == "bert" else "single"
REF = {"fedavg": "methods/fedavg.py:21-24", "fedprox": "methods/fedavg.py:21-24 (fedprox.py inherits)", "afpha": "methods/fedavg.py:21-24 (afpha.py inherits; round at afpha.py:41-67)",
       "fd_ids": "methods/fedavg.py:21-24 (fd_ids.py inherits)", "apfed": "methods/fedavg.py:21-24 (apfed.py inherits; round apfed.py:68-100)",
       "lwfednids": "methods/fedavg.py:21-24 (inherited, pruned layout lwfednids.py:140-160)", "perfedskd": "methods/perfedskd.py:36-38",
       "nilm": "methods/nilm.py:30-32", "pfedes": "methods/pfedes.py:61-62", "fedproto": "methods/fedproto.py:30-32",
       "fedtgp": "methods/fedtgp.py:50-52", "tinyproto": "methods/tinyproto.py:52-53", "v2x_htfl": "methods/v2x_htfl.py:146-147"}

for rec_list in (single, bert):
    for r in rec_list:
        m, bb = r["method"], r["backbone"]
        fam, pb = FAM(bb), r["payload_bytes"]
        p, b, d = r["P"], r["B"], r["d_cfg_head_hidden"]
        ref = REF[m] if fam == "single" else REF[m].replace("methods/", "bert/methods/")
        if m in ("fedavg", "fedprox", "afpha", "fd_ids", "apfed", "lwfednids", "perfedskd", "nilm"):
            who = {"apfed": "personalised model W~_i after local training (up) / global W^g (down)", "nilm": "unified proxy w_r only (private model never leaves)",
                   "lwfednids": "pruned (70%) narrower model", "perfedskd": "model omega_m of SELECTED devices only / global omega^t"}.get(m, "global model theta")
            rule = "all" if m != "perfedskd" else "selected"
            note = ""
            if m == "perfedskd":
                f, n = sel_frac(bb)
                note = ("payload_bytes counts every client every round; real traffic only for selected devices: round 1 = 100%, later rounds mean "
                        + (f"{f:.3f} of clients (mean of {n} local 50-round runs, work/perfedskd_selected_frac.csv)" if f else "unknown for this backbone (no local rounds.jsonl)")
                        + "; a_m accuracy scalar (4 B) uploaded by ALL clients each round and A broadcast: not counted (estimated 4+4 B)")
            if m == "afpha":
                note = ("hierarchical 8 clusters (scenario x run) simulated inside the aggregation: cluster-head<->server links and cluster ids not counted; "
                        "adaptive mu needs d_i up + d_bar down (~4+4 B/client/round) not counted; probed via inherited FedAvg layout (clients.parquet absent locally)")
            if m == "fd_ids":
                note = "teacher = the received global model (no extra payload)"
            if m == "fedprox":
                note = "proximal term local; no control variate (SCAFFOLD removed)"
            if m == "apfed":
                note = ("fusion weights theta_i (K x base) and personalised bank stay local; BN running stats are aggregated and sent like FedAvg; "
                        "whole W~_i uploaded (not only base)")
            if m == "lwfednids":
                note = (f"layout P',B' of the pruned model (full P={P[('fedavg', bb)]['P']}); pruning is one data-free server step; clients receive the pruned theta0 "
                        "(no separate mask file: pruned slices are shipped physically) = same size as one round-1 download")
            if m == "nilm":
                note = "personalised model w_s never leaves the client"
            for dr, nm in (("down", "theta params"), ("up", "theta params")):
                add(method=m, backbone=bb, scenario_family=fam, component=f"{nm} ({who})", direction=dr, frequency="round", clients_rule=rule,
                    elements_formula="P", elements_value=p, dtype="float32", bytes_per_client_per_event=4 * p,
                    broadcast_or_unicast="broadcast" if dr == "down" and m != "apfed" else ("broadcast" if dr == "down" else "unicast"),
                    basis="measured", code_ref=ref, payload_bytes_method_value=pb["down" if dr == "down" else "up"], discrepancy_note=note if nm == "theta params" else "")
            if b:
                for dr in ("down", "up"):
                    add(method=m, backbone=bb, scenario_family=fam, component="BN running_mean/running_var buffers (num_batches_tracked NOT included/sent)",
                        direction=dr, frequency="round", clients_rule=rule, elements_formula="B", elements_value=b, dtype="float32",
                        bytes_per_client_per_event=4 * b, broadcast_or_unicast="broadcast" if dr == "down" else "unicast", basis="measured", code_ref=ref,
                        payload_bytes_method_value="(included in payload value above)", discrepancy_note="")
            if m == "lwfednids":
                add(method=m, backbone=bb, scenario_family=fam, component="pruned theta0 initial broadcast (server computes mask once)", direction="down",
                    frequency="bootstrap", clients_rule="all", elements_formula="P'+B' (already the round-1 download)", elements_value=p + b, dtype="float32",
                    bytes_per_client_per_event=0, broadcast_or_unicast="broadcast", basis="calculated", code_ref="methods/lwfednids.py:1-12,140-160",
                    payload_bytes_method_value="", discrepancy_note="0 extra: counted in round-1 download; do not double count")
        elif m == "pfedes":
            for dr in ("down", "up"):
                add(method=m, backbone=bb, scenario_family=fam, component="proxy feature extractor G (Conv1d 7->8,3 ; Conv1d 8->7,3), no BN", direction=dr,
                    frequency="round", clients_rule="all", elements_formula="7*8*3+8 + 8*7*3+7 = 351", elements_value=r["P_G"], dtype="float32",
                    bytes_per_client_per_event=4 * r["P_G"], broadcast_or_unicast="broadcast" if dr == "down" else "unicast", basis="measured", code_ref=ref,
                    payload_bytes_method_value=pb[dr], discrepancy_note="identical for every backbone (G acts on the 77-d input, not on the backbone); heterogeneous F_k never exchanged" if dr == "up" else "")
        elif m in ("fedproto", "fedtgp", "v2x_htfl"):
            counts = m == "fedproto"
            avg_up = PAIRS_PC * 4 * d + (PAIRS_PC * 4 if counts else 0)
            note = {"fedproto": "payload_bytes upload = dense 16 classes + 16 fp32 counts (worst case); code uploads only classes present at the client (N_p=69,993 pairs of 85,488 dense, 13.10/client); counts needed for the n-weighted mean",
                    "fedtgp": "payload_bytes = dense 16xd worst case; only present classes uploaded (n_protos = N_p); no counts; server trains generator F(P) on uploads for S=100 epochs: server-only compute, not exchanged",
                    "v2x_htfl": "payload_bytes adds 4*16 B counts that the code never uses (server only needs class presence); prototypes live in the projected d-dim space; projection p_k (d x d), ETF classifier V (seed-shared) and generator are NOT exchanged"}[m]
            add(method=m, backbone=bb, scenario_family=fam, component="local class prototypes (mean hidden feature, present classes only)" + (" + per-class counts" if counts else ""),
                direction="up", frequency="round", clients_rule="class_present (all clients, only classes with train rows)",
                elements_formula=("n_present*d + n_present" if counts else "n_present*d") + f"; d={d}; dense bound 16*d" + ("+16" if counts else ""),
                elements_value=16 * d + (16 if counts else 0), dtype="float32", bytes_per_client_per_event=pb["up"], broadcast_or_unicast="unicast",
                basis="calculated", code_ref=ref + (f"; round fedtgp.py:72-75" if m == "fedtgp" else ""), payload_bytes_method_value=pb["up"], discrepancy_note=note,
                bytes_per_client_per_event_data_avg=avg_up)
            add(method=m, backbone=bb, scenario_family=fam, component="global prototypes (16 x d)" + (" = server-trained P^c=F(P'c)" if m != "fedproto" else " = n-weighted mean"),
                direction="down", frequency="round", clients_rule="all", elements_formula=f"16*d, d={d}", elements_value=16 * d, dtype="float32",
                bytes_per_client_per_event=pb["down"], broadcast_or_unicast="broadcast", basis="measured (tensor nbytes after toy round 1)", code_ref=ref,
                payload_bytes_method_value=pb["down"], discrepancy_note="all 16 classes have >=1 holder in v4 (classes_with_no_holder=0)" if m == "fedproto" else "")
            if m == "v2x_htfl":
                add(method=m, backbone=bb, scenario_family=fam, component="simplex-ETF classifier V (16 x d), regenerated from shared seed", direction="down", frequency="bootstrap",
                    clients_rule="all", elements_formula="16*d", elements_value=16 * d, dtype="float32", bytes_per_client_per_event=0, broadcast_or_unicast="broadcast",
                    basis="estimated", code_ref="methods/v2x_htfl.py:128", payload_bytes_method_value="",
                    discrepancy_note=f"0 if seed shared; {4 * 16 * d} B if shipped once")
        elif m == "tinyproto":
            s = r["s"]
            add(method=m, backbone=bb, scenario_family=fam, component="masked prototype coordinates n_ij*c^L (s of d dims per present class)", direction="up", frequency="round",
                clients_rule="class_present", elements_formula=f"n_present*s; s=ceil(d/10)={s}; dense bound 16*s", elements_value=16 * s, dtype="float32",
                bytes_per_client_per_event=pb["up"], broadcast_or_unicast="unicast", basis="calculated (s verified: mask true count & target nonzeros per class)",
                code_ref=ref, payload_bytes_method_value=pb["up"], discrepancy_note="dense 16-class worst case; code uploads only classes held (Cn>0); dense local prototypes (fp16) stay on the client",
                bytes_per_client_per_event_data_avg=PAIRS_PC * 4 * s)
            add(method=m, backbone=bb, scenario_family=fam, component="global target c^G on masks (16 x s)", direction="down", frequency="round", clients_rule="all",
                elements_formula=f"16*s, s={s}", elements_value=16 * s, dtype="float32", bytes_per_client_per_event=pb["down"], broadcast_or_unicast="broadcast",
                basis="calculated", code_ref=ref, payload_bytes_method_value=pb["down"],
                discrepancy_note="server tensor is dense 16xd (measured %d B) but only the mask support (s per class) must travel; mu scaling is server side" % r["tensors"]["target_dense_down"]["nbytes"])
            add(method=m, backbone=bb, scenario_family=fam, component="class masks m_j (16 x d bits)", direction="down", frequency="bootstrap", clients_rule="all",
                elements_formula="16*d bits", elements_value=16 * d, dtype="bit", bytes_per_client_per_event=pb["masks_once"], broadcast_or_unicast="broadcast",
                basis="calculated", code_ref=ref, payload_bytes_method_value=pb["masks_once"],
                discrepancy_note="reported as masks_once but cost.md tables omit it; masks are a seeded permutation (seed+13) so 0 B if seed is shared")
        elif m == "canperfl":
            pass

# CANPerFL (phase structure; payload identical to FedAvg phase 1)
for bb in SINGLE_BB + ["bert"]:
    fam, r = FAM(bb), P[("fedavg", bb)]
    for dr in ("down", "up"):
        add(method="canperfl", backbone=bb, scenario_family=fam, component="phase 1 = FedAvg theta (P+B) x 50 rounds", direction=dr, frequency="round (phase 1 only, rounds 1-50)",
            clients_rule="all", elements_formula="P+B", elements_value=r["P"] + r["B"], dtype="float32", bytes_per_client_per_event=4 * (r["P"] + r["B"]),
            broadcast_or_unicast="broadcast" if dr == "down" else "unicast", basis="calculated", code_ref="methods/canperfl.py:1-16,33-34",
            payload_bytes_method_value="0 (canperfl.payload_bytes reports phase 2 only)", discrepancy_note="canperfl payload_bytes() = 0/0; the 50 FedAvg rounds are the cost (run reuses the fedavg state)" if dr == "up" else "")
    add(method="canperfl", backbone=bb, scenario_family=fam, component="phase 2 local fine-tune 10 epochs", direction="both", frequency="phase:finetune (10 epochs)",
        clients_rule="all", elements_formula="0", elements_value=0, dtype="", bytes_per_client_per_event=0, broadcast_or_unicast="n/a", basis="calculated", code_ref="methods/canperfl.py:33-34",
        payload_bytes_method_value=0, discrepancy_note="zero communication")
    add(method="canperfl", backbone=bb, scenario_family=fam, component="final aggregated theta^50 download before fine-tune", direction="down", frequency="phase boundary (once)",
        clients_rule="all", elements_formula="P+B", elements_value=r["P"] + r["B"], dtype="float32", bytes_per_client_per_event=4 * (r["P"] + r["B"]), broadcast_or_unicast="broadcast",
        basis="calculated", code_ref="methods/canperfl.py:14-15; methods/fedavg.py:32-35",
        payload_bytes_method_value=0, discrepancy_note="canperfl.py says theta^50 download is 'counted in FedAvg's last round' but FedAvg round r downloads theta^(r-1), so theta^50 is NEVER counted: +1 download per client (conservative accounting)")

# ---------------- mixed (heterogeneous) --------------------------------------------------------------
mix = {(r["case"], r["method"]): r for r in mixed if "case" in r and "error" not in r}
case_pairs = {c: sum(v["pairs"] for v in g.values()) / sum(v["clients"] for v in g.values()) for c, g in stats["cases"].items()}
for (case, m), r in sorted(mix.items()):
    pb, d = r["payload_bytes"], r["d"]
    groups = "+".join(r["groups"])
    bb = f"{case}[{groups}]"
    ppc = case_pairs[case]
    ref = "methods/mixed.py:88-96"
    if m == "pfedes":
        for dr in ("down", "up"):
            add(method=m, backbone=bb, scenario_family="mixed", component="proxy extractor G (351 params) shared by ALL backbone groups", direction=dr, frequency="round", clients_rule="all",
                elements_formula="351", elements_value=351, dtype="float32", bytes_per_client_per_event=pb[dr], broadcast_or_unicast="broadcast" if dr == "down" else "unicast",
                basis="measured", code_ref=ref, payload_bytes_method_value=pb[dr], discrepancy_note="backbone-independent; private F_k (native cfg, projected=False) never exchanged")
        continue
    s = r.get("s")
    up_el = 16 * s if m == "tinyproto" else 16 * d
    add(method=m, backbone=bb, scenario_family="mixed", component=("masked prototype coords" if m == "tinyproto" else "local prototypes (present classes)") + (" + counts" if m == "fedproto" else ""),
        direction="up", frequency="round", clients_rule="class_present", elements_formula=(f"n_present*s, s={s}" if m == "tinyproto" else f"n_present*d, d={d} (128 for EVERY backbone via learned projection)"),
        elements_value=up_el, dtype="float32", bytes_per_client_per_event=pb["up"], broadcast_or_unicast="unicast", basis="calculated", code_ref=ref,
        payload_bytes_method_value=pb["up"], discrepancy_note=("payload_bytes labelled per-client worst-case 16 classes; v2x +64 B counts unused" if m == "v2x_htfl" else "dense 16-class worst case"),
        bytes_per_client_per_event_data_avg=ppc * 4 * (s if m == "tinyproto" else d) + (ppc * 4 if m == "fedproto" else 0))
    add(method=m, backbone=bb, scenario_family="mixed", component="global prototypes" + (" (masked 16xs)" if m == "tinyproto" else " (16 x 128)"), direction="down", frequency="round", clients_rule="all",
        elements_formula=(f"16*s, s={s}" if m == "tinyproto" else "16*128"), elements_value=up_el, dtype="float32", bytes_per_client_per_event=pb["down"], broadcast_or_unicast="broadcast",
        basis="measured (proto tensor %s after toy round 1)" % (r["tensors"]["global_proto_down"]["shape"]), code_ref=ref, payload_bytes_method_value=pb["down"],
        discrepancy_note="tinyproto server tensor dense 16x128 (8192 B) but wire = masked support" if m == "tinyproto" else "")
    if m == "tinyproto":
        add(method=m, backbone=bb, scenario_family="mixed", component="class masks (16 x 128 bits)", direction="down", frequency="bootstrap", clients_rule="all", elements_formula="16*128 bits",
            elements_value=2048, dtype="bit", bytes_per_client_per_event=pb["masks_once"], broadcast_or_unicast="broadcast", basis="calculated", code_ref=ref, payload_bytes_method_value=pb["masks_once"],
            discrepancy_note="seeded permutation: 0 B if seed shared")

# ---------------- MobiHet L4 ------------------------------------------------------------------------------
codec = None
for l in l4:
    pass
km2 = json.load(open(f"{WORK}/km2_record.json")) if os.path.exists(f"{WORK}/km2_record.json") else None
if km2 is None:
    import re, glob
    t = open(glob.glob(f"{ROOT}/runs/l4-20261120/pull/tabm-l4-s41/*.log")[0], errors="ignore").read().replace('\\"', '"')
    km2 = json.loads(re.search(r'KM2_BYTES (\{.*?\})\\n', t.replace("\\n", "\\n")).group(1)) if False else json.loads(re.search(r'KM2_BYTES (\{.*?\})(?:\\n|\n)', t).group(1))
    json.dump(km2, open(f"{WORK}/km2_record.json", "w"), indent=1)
BREF = "methods/mobihet_lite.py:44-224 (_kmeans km2 codec; bytes at 195-224)"
bsub = [("up", "class counts n_kc (4C)", km2["up"]["counts"]), ("up", "marginal histograms (64 bins x 77 x present classes, sparse varints)", km2["up"]["hist"]),
        ("up", "class moments (init/shrinkage, 25% of vehicles; int8+zlib) mean over vehicles", km2["up"]["cls"]), ("up", "Lloyd passes (4, 25% sampled each) mean over vehicles", km2["up"]["lloyd"]),
        ("up", "final pass cells (count+int8 mean+2nd moment if >=16 rows)", km2["up"]["final"]),
        ("down", "class counts", km2["down"]["counts"]), ("down", "copula CDF tables (uint16 diffs, best of lzma/bz2/zlib), identical for all", km2["down"]["cdf"]),
        ("down", "int8 k-means centres of the vehicle's own classes x 5 passes (UNICAST, mean over vehicles)", km2["down"]["centres"]),
        ("down", "kept-cluster mixture (weights, fp16 means/diag, 6-bit Cholesky; %d of 96 clusters), identical for all" % km2["kept"], km2["down"]["mixture"])]
for dr, comp, val in bsub:
    add(method="mobihet_l4", backbone="all (backbone independent)", scenario_family="mobihet_l4_single+mixed", component="BOOTSTRAP " + comp, direction=dr, frequency="bootstrap",
        clients_rule="all", elements_formula="measured bytes (LZMA/zlib/bz2 coded per vehicle, mean over 5,343)", elements_value="", dtype="coded bytes", bytes_per_client_per_event=val,
        broadcast_or_unicast="unicast" if "UNICAST" in comp else ("broadcast" if dr == "down" else "unicast"), basis="measured", code_ref=BREF,
        payload_bytes_method_value="bootstrap up=%.1f down=%.1f" % (km2["up_mean"], km2["down_mean"]) if comp.startswith("class counts n_kc") else "",
        discrepancy_note="cost.md says ~197 KB one-time bootstrap (older gauss/copula bootstrap); L4 km2 codec measures up %.0f + down %.0f = %.0f B/vehicle; if all downlink sent as one broadcast: %d B (km2['broadcast_copy'])" % (km2["up_mean"], km2["down_mean"], km2["total_mean"], km2["broadcast_copy"]) if comp.startswith("class counts n_kc") and dr == "up" else "")


def l4group(substr):
    return [r for r in l4 if substr in r["run"] and int(r["rounds"]) == 50]


def avg(rs, k):
    return float(np.mean([float(r[k]) for r in rs])) if rs else None


groups = {"dagsnet": l4group("dagsnet-l4/mh-fbl-l4-top-s4"), "bert": l4group("bert-l4-artifacts"), "tabm": l4group("tabm-l4"), "sae": l4group("sae-l4"),
          "mlp": l4group("mlp-l4"), "cnn-bigru-attention": l4group("cnnbigru-l4")}
# bert rows are listed twice by the globs in measured_stats (h2 artefacts): de-dup by run name
for k, rs in groups.items():
    groups[k] = list({r["run"]: r for r in rs}.values())
for bb, rs in groups.items():
    up, dn = avg(rs, "round_up_mean"), avg(rs, "round_down_mean")
    fam = "mobihet_l4_single"
    add(method="mobihet_l4", backbone=bb, scenario_family=fam, component="class-Q packet: per present class 16 x uint8 (p*255) + 34 B header (32 + 2 B class bitmask), raw or zlib-6, mean over vehicles",
        direction="up", frequency="round", clients_rule="class_present (all vehicles)", elements_formula="n_present*16 uint8 (+34 B); measured mean per round",
        elements_value="", dtype="uint8", bytes_per_client_per_event=round(up, 2), broadcast_or_unicast="unicast", basis="measured (rounds.jsonl bytes_up, mean of 50 rounds x %d seeds)" % len(rs),
        code_ref="methods/mobihet_sim.py:789-832 (q_bits=8 path)", payload_bytes_method_value="no payload_bytes per round (config.json: bootstrap only); mobihet.py:602 dense fp16 formula would be 32+512",
        discrepancy_note="range over rounds %.0f-%.0f B; r1 lowest" % (min(float(r["round_up_min"]) for r in rs), max(float(r["round_up_max"]) for r in rs)))
    add(method="mobihet_l4", backbone=bb, scenario_family=fam, component="global class-Q table 16x16 uint8 + 32 B header (zlib-6 or raw), single identical packet",
        direction="down", frequency="round", clients_rule="all", elements_formula="256 uint8 +32; measured", elements_value=256, dtype="uint8", bytes_per_client_per_event=round(dn, 2),
        broadcast_or_unicast="broadcast", basis="measured (rounds.jsonl bytes_down, mean of 50 rounds x %d seeds)" % len(rs), code_ref="methods/mobihet_sim.py:826-830",
        payload_bytes_method_value="", discrepancy_note="bytes_down of round r = Q produced at end of round r (consumed in r+1), so the r50 download is never used (1 of 50 events, negligible); no parameters ever exchanged")
# mixed: pilot rounds only
for l in l4:
    if "mixed-l4/pilot" in l["run"]:
        case = l["run"].split("mixl4-")[1].split("-")[0]
        gs = stats["cases"][case]
        n = sum(v["clients"] for v in gs.values())
        est_up = sum(v["clients"] * avg(groups[m], "round_up_mean") for m, v in gs.items()) / n
        est_dn = sum(v["clients"] * avg(groups[m], "round_down_mean") for m, v in gs.items()) / n
        for dr, meas, est in (("up", float(l["round_up_mean"]), est_up), ("down", float(l["round_down_mean"]), est_dn)):
            add(method="mobihet_l4", backbone=f"{case}[{'+'.join(gs)}]", scenario_family="mobihet_l4_mixed", component="class-Q packet (up) / Q table (down), Q pooled across ALL groups",
                direction=dr, frequency="round", clients_rule="all", elements_formula="as single-model", elements_value="", dtype="uint8", bytes_per_client_per_event=round(meas, 2),
                broadcast_or_unicast="unicast" if dr == "up" else "broadcast", basis="measured for rounds 1-2 only (pilot, %s); 50-round mean ESTIMATED = client-weighted single-backbone mean: %.1f B" % (l["run"].split("/")[-1], est),
                code_ref="methods/mobihet_mixed_l4.py:275-277", payload_bytes_method_value="bootstrap only", discrepancy_note="full r50 mixed rounds.jsonl not in local repo (ORICO/Kaggle); estimated_50round=%.1f" % est)

with open(f"{WORK}/payload_spec.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)

# ---------------- backbone_params.csv ---------------------------------------------------------------------
bp = []
CFG = {"dagsnet": ("core/dagsnet.py:11 width(1)", 256), "tabm": ("core/tabm.py:6", 32), "sae": ("core/sae.py:25 fl_config (head_hidden=93=77+16)", 93),
       "mlp": ("core/mlp.py:6", 32), "cnn-bigru-attention": ("core/cnn_bigru_attention.py:9 (context = 2*gru_hidden=128)", 128)}
for bb in SINGLE_BB:
    r = P[("fedavg", bb)]
    bp.append(dict(backbone=bb, variant="single-FL native (FedAvg-family layout)", n_params=r["P"], n_buffers=r["B"], bytes_fp32=4 * (r["P"] + r["B"]), proto_dim_d=CFG[bb][1], code_ref=CFG[bb][0]))
    bp.append(dict(backbone=bb, variant="LwFedNIDS 70% pruned (exchanged)", n_params=P[("lwfednids", bb)]["P"], n_buffers=P[("lwfednids", bb)]["B"],
                   bytes_fp32=4 * (P[("lwfednids", bb)]["P"] + P[("lwfednids", bb)]["B"]), proto_dim_d=P[("lwfednids", bb)]["d_cfg_head_hidden"], code_ref="methods/lwfednids.py"))
    v = P[("v2x_htfl", bb)]
    bp.append(dict(backbone=bb, variant="V2X-HtFL local model (+d x d projection, NOT exchanged)", n_params=v["P"], n_buffers=v["B"], bytes_fp32=4 * (v["P"] + v["B"]), proto_dim_d=CFG[bb][1], code_ref="methods/v2x_htfl.py:48-80"))
r = P[("fedavg", "bert")]
bp.append(dict(backbone="bert (FeatureBERT)", variant="FeatureBERT FL native", n_params=r["P"], n_buffers=r["B"], bytes_fp32=4 * r["P"], proto_dim_d=64, code_ref="bert/core/bert.py:11 FULL d=64"))
for k, lab in (("lwfednids", "LwFedNIDS 70% pruned"), ("v2x_htfl", "V2X-HtFL local (+projection)")):
    q = P[(k, "bert")]
    bp.append(dict(backbone="bert (FeatureBERT)", variant=lab, n_params=q["P"], n_buffers=q["B"], bytes_fp32=4 * (q["P"] + q["B"]), proto_dim_d=q["d_cfg_head_hidden"], code_ref="bert/methods/" + k + ".py"))
bp.append(dict(backbone="ALL", variant="pFedES proxy extractor G (exchanged)", n_params=351, n_buffers=0, bytes_fp32=1404, proto_dim_d="", code_ref="methods/pfedes.py:33-45; measured P_G=351"))
mm = {(r["case"], r["method"]): r for r in mixed if "case" in r}
seen = {}
for case in ("case1", "case2", "case3"):
    for name, variant in (("fedproto", "mixed MixedAdapter (projection 128 + classifier), stored"), ("v2x_htfl", "mixed V2X (+128x128 proj), stored, not exchanged"), ("pfedes", "mixed native (no adapter), stored, not exchanged")):
        for g, v in mm[(case, name)]["groups"].items():
            if (g, name) in seen:
                continue
            seen[(g, name)] = 1
            bp.append(dict(backbone=g, variant=variant, n_params=v["P"], n_buffers=v["B"], bytes_fp32=4 * (v["P"] + v["B"]), proto_dim_d=128 if name != "pfedes" else v["cfg_head_hidden"],
                           code_ref="core/mixed_backbones.py:41-70; cross-check runs/mixed128-parameters.json"))
with open(f"{WORK}/backbone_params.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["backbone", "variant", "n_params", "n_buffers", "bytes_fp32", "proto_dim_d", "code_ref"]); w.writeheader(); w.writerows(bp)
print("payload_spec rows", len(rows), "backbone rows", len(bp))
