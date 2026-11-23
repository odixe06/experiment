# Communication formulas per method

Scope: simulated client↔server **training** payload of every executed FL run. **Not network traffic**: no run has a
packet capture or socket counter, so no number in this folder is network traffic. Evaluation, checkpoints, Kaggle
artifacts and W&B logging are excluded.

Reproduce: `~/Tools/miniconda3/envs/nckh/bin/python artifacts/communication-budget/scripts/compute_budget.py`
(it needs ORICO mounted). Inputs come from `scripts/inventory.py` (runs) and `scripts/payload_specs.py` (tensor sizes, probes).

## Basis labels

| label | meaning | used for |
|---|---|---|
| measured | byte length of an actual encoded packet (LZMA/zlib/bz2/varint) computed for every vehicle during the run and logged | MobiHet L4 bootstrap and per-round packets |
| calculated | exact element count × dtype from the implementation, with tensor shapes from instantiating the model on CPU and data counts from the dataset fold table | all baselines |
| estimated | needs an assumption or extrapolation | none in the production totals; see the bounds in audit_report.md |
| network traffic | measured on a network link | **none exists** |

## Notation

| symbol | value | source |
|---|---|---|
| K | clients = 5,343 (all participate every round, except PerFed-SKD) | config.json `clients`, manifests |
| R | rounds actually executed (50; CANPerFL 10 fine-tune epochs) | rounds.jsonl, counted contiguously 1..R |
| P, B | parameters, BN running buffers (num_batches_tracked is not exchanged) | backbone_params.csv |
| M = 4(P+B) | fp32 model bytes: DAGSNet 1,594,688; BERT 311,616; TabM 26,192; SAE 30,452; MLP 30,400; CNN-BiGRU 473,664 | instantiation = config `payload_bytes` |
| M′ | LwFedNIDS pruned model: 162,000; 40,776; 8,328; 12,280; 7,784; 48,612 (same order) | config `payload_bytes` |
| d | prototype dimension: 256, 64, 32, 93, 32, 128 (same order); mixed = 128 for every backbone | config `cfg.head_hidden`, `core/mixed_backbones.py` |
| s | TinyProto kept coordinates = ⌈d/10⌉: 26, 7, 4, 10, 4, 13; mixed 13 | `methods/tinyproto.py` |
| C | classes = 16 | |
| N_p | (client, class) pairs with ≥ 1 train+val row = **69,993** of 85,488 (13.10 per client) | `../mobihet/cache/fold_counts.npy`, row sum 37,361,273 = protocol rows |
| N_p,g, K_g | the same per mixed backbone group (e.g. case1 dagsnet 1,603 clients, 20,913 pairs) | data_dependent_stats.csv |
| P_G | pFedES proxy extractor = 351 params → 1,404 B | probe |
| n_r | PerFed-SKD devices selected in round r = selected_frac_r × K | rounds.jsonl |

Downlink convention: a download is counted once **per receiving client** (unicast simulation). The column
`server_down_unique_B` gives the size of one copy per round for content that is the same for every client, which is
what a true broadcast or multicast would send.

## Per-run totals (over all clients and rounds 1..R)

| method | bootstrap (once) | recurring uplink | recurring downlink | metadata | basis |
|---|---|---|---|---|---|
| FedAvg, FedProx, FD-IDS, APFed, NILM-FL | 0 | K·R·M | K·R·M | 0 | calculated |
| AFPHA | 0 | K·R·M | K·R·M | down 4·K·(R−1): personal μ_i (fp32) computed on the server | calculated |
| LwFedNIDS | 0 (the pruned θ⁰ is the round-1 download) | K·R·M′ | K·R·M′ | 0 | calculated |
| PerFed-SKD | 0 | M·Σ_r n_r | M·Σ_r n_r | up 4·K·(R−1): accuracy a_m of every device for selection | calculated from logged selection |
| CANPerFL (own run) | down K·M: one copy of θ⁵⁰ | 0 | 0 | 0 | calculated |
| CANPerFL phase 1 | FedAvg run of the same scenario and seed, reused rather than re-run; reported in `canperfl_phase1_attributed_B` | | | | calculated |
| pFedES | 0 | K·R·4P_G | K·R·4P_G | 0 | calculated |
| FedProto | 0 | R·N_p·(4d+4): prototype and count of present classes | K·R·4Cd | 0 | calculated |
| FedTGP | 0 | R·N_p·4d | K·R·4Cd: server-trained global prototypes | 0 | calculated |
| V2X-HtFL | 0 | R·N_p·4d: projected prototypes, no counts | K·R·4Cd | 0 | calculated |
| TinyProto | 0 (class masks come from a seeded permutation rebuilt on every client) | R·N_p·4s | K·R·4Cs | 0 | calculated |
| MobiHet L4 | K·(b̄_up + b̄_down), the measured per-vehicle mean over all 5,343 | K·Σ_r bytes_up_r: mean over all vehicles of the measured class-Q packet | K·Σ_r bytes_down_r: one measured Q table to every vehicle | 0 | measured |

Mixed cases (FedProto, FedTGP, V2X-HtFL, TinyProto, pFedES) use the same formulas with d = 128 and s = 13. The
payload does not depend on the backbone because each `MixedAdapter` projects to 128. Per group g, the uplink is
R·N_p,g·(per-class bytes), the downlink is K_g·R·(per-client bytes), and the groups sum exactly to the case total
(`per_backbone_breakdown`). In mixed L4 every backbone group is a virtual rank of one allreduce, so `bytes_up` is a
single mean over all vehicles of the case. The byte split by backbone was not logged.

Derived per run: total = bootstrap + recurring + metadata (up + down); per client = total / K; recurring per client
per round = recurring / (K·R).

## Why these choices

- **Present classes only on the uplink.** The server ignores a class with n_kc = 0: weight 0 in FedProto, and
  `holders` count only present classes in TinyProto. A client therefore sends only N_p rows. The dense 16-class value
  that `payload_bytes()` returns is an upper bound; it is kept as `dense_upper_bound_up_B`.
- **Counts on the uplink.** Only FedProto weights prototypes by n_kc. V2X `payload_bytes()` adds 64 B of counts that
  `methods/v2x_htfl.py` never uses, so they are dropped here.
- **The class-presence index is not counted.** A sparse upload needs ≤ 2 B (a 16-bit bitmap) per client per round,
  ≤ 0.53 MB per run in total. Bounds are in audit_report.md.
- **TinyProto masks are 0 B** (owner decision, 21/11). Every client rebuilds them from the shared seed
  (`methods/tinyproto.py:34-38`, seed + 13). The value `masks_once` = C·d/8 B that `payload_bytes()` reports and the
  docstring's "sent once" are not counted.
- **Seeded objects are not counted.** These are the V2X-HtFL ETF anchor V, the MobiHet transfer inputs and θ⁰ of the
  seeded initialisation. The θ⁰ download is still counted as round 1's download, as `payload_bytes()` does.
- **AFPHA is algebraically FedAvg.** The 8-cluster hierarchy is computed on the server. A real RSU tier would add
  8·M up and 8·M down per round between the clusters and the cloud; that backhaul is outside client↔server and not
  counted.
- **Bootstrap is counted once.** Only MobiHet L4 (statistics) and CANPerFL (θ⁵⁰) have a bootstrap. Continuation sessions re-emit cumulative logs, so each logical run is taken from a single canonical
  session that covers rounds 1..R.
- **Precision and serialization.** Every baseline tensor is fp32 (4 B/element) with no compression or framing.
  MobiHet L4 per-round packets are uint8 with a zlib/raw choice per packet and a 32 or 34 B header. Its bootstrap
  uses varint, int8 or fp16 and the smallest of lzma, bz2 and zlib (`methods/mobihet_lite.py`, `methods/mobihet_sim.py`).
  These are encoded byte lengths, not tensor sizes.
