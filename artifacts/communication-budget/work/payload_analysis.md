# Payload analysis (communication-budget audit, worker: communication-analysis)

Reproduce: `python artifacts/communication-budget/scripts/payload_specs.py [--refresh]` (env nckh, CPU, ~1 min). Basis tags: measured =
executed on toy env or parsed from run logs; calculated = code formula with measured shapes; estimated = needs assumption.
Wire dtype: every model/prototype tensor is fp32 (flat layout `core/flat.py`, 4 B/element) except MobiHet (uint8 Q + lossless coders).
BN `num_batches_tracked` is never exchanged; BN buffers = running_mean/var only (`Layout.b`, core/flat.py:29). No protocol headers counted.

## Backbone sizes (work/backbone_params.csv, all measured by instantiation)
| backbone | P | B | 4(P+B) B | d (prototype dim) |
|---|---:|---:|---:|---:|
| DAGSNet W=1 | 395,408 | 3,264 | 1,594,688 | 256 |
| FeatureBERT (bert/) | 77,904 | 0 (LayerNorm) | 311,616 | 64 |
| TabM k=4 | 6,548 | 0 | 26,192 | 32 (member-mean) |
| SAE (fl_config, CE-only) | 7,613 | 0 | 30,452 | 93 (=77+16) |
| MLP | 7,600 | 0 | 30,400 | 32 |
| CNN-BiGRU | 118,288 | 128 | 473,664 | 128 |
LwFedNIDS pruned (exchanged): DAGSNet 39,494+1,006 (162,000 B); BERT 10,194 (40,776); TabM 2,082; SAE 3,070; MLP 1,946; BiGRU 12,113+40.
pFedES proxy G: 351 params = 1,404 B for every backbone. Mixed (d=128 for every backbone, `core/mixed_backbones.py:29-31`; stored P/B equal
runs/mixed128-parameters.json): dagsnet 430,368+3,264; bert 88,288; cnn-bigru 136,864+128; tabm 19,028; sae 21,709; mlp 13,888 (none of these is exchanged).

## Single-model and BERT FL (per client per round, fp32)
| method | up | down | who / rule | evidence |
|---|---|---|---|---|
| FedAvg, FedProx, AFPHA, FD-IDS | θ (P) + BN (B) | same, broadcast | all 5,343 | probe tensors; payload_bytes matches |
| APFed | W̃_i whole model P+B | W^g P+B | all | payload matches (θ_i fusion weights, bank stay local) |
| NILM-FL | proxy w_r P+B | same | all | private w_s never sent |
| LwFedNIDS | P'+B' | P'+B' | all | pruned layout; mask = physically dropped weights, no extra file |
| PerFed-SKD | P+B | P+B | **selected only**: r1 100%, later ≈0.28 (DAGSNet), ≈0.44 (TabM), ≈0.37-0.39 (BERT) | perfedskd_selected_frac.csv; SAE/MLP/BiGRU: only final-round value for SAE (0.444 s41) |
| CANPerFL | FedAvg ×50 then 0 | same | all | phase 2 = 0 B |
| pFedES | 351 params | 351 | all | 1,404 B each way |
| FedProto | n_present·d + counts | 16·d broadcast | class_present | payload dense 16d+16 |
| FedTGP | n_present·d | 16·d (server-trained P̂) broadcast | class_present | server generator never sent |
| TinyProto | n_present·s, s=ceil(d/10) | 16·s broadcast | class_present | masks 16·d bits once |
| V2X-HtFL | n_present·d (projected space) | 16·d broadcast | class_present | projection d×d, ETF V, generator not sent |
Example DAGSNet (B/client/round): FedAvg 1,594,688 / 1,594,688; LwFedNIDS 162,000; pFedES 1,404; FedProto ≤16,448 up (avg 13.10 pairs: 13.0999*(4*256+4)=13,425) / 16,384
down; FedTGP ≤16,384 / 16,384; TinyProto ≤1,664 (avg 1,362) / 1,664; V2X ≤16,448 / 16,384 (d=256). Average sparse upload:
FedProto 13.0999·(4d+4); FedTGP/V2X 13.0999·4d; TinyProto 13.0999·4s (column `bytes_per_client_per_event_data_avg`).

Downlink nature: every downlink above is the same content for all clients (global θ, global prototypes, proxy G, TinyProto target): broadcast.
In simulation count it per receiving client (as cost.md does); single-broadcast size = the same number once. No personalised unicast downlink
exists except MobiHet's per-vehicle bootstrap centres (below).

## Data-dependent (work/data_dependent_stats.*)
Source: `/home/odixe/project/mobihet/cache/fold_counts.npy` (sibling repo, read-only; shape 5343×2×16, sum 37,361,273 = train+val rows). N_p = 69,993
(client,class) pairs with rows, of 85,488 dense (81.9%); 13.10/client (min 2, median 15, max 16); 0 clients with zero rows; every class has ≥1 holder.
Per mixed group pairs are in the CSV (e.g. case1 dagsnet 1,603 clients 20,913; bert 21,143; cnn-bigru 2,137 clients 27,937). Test support is
`runs/test_support.npy` (60,515 pairs) - not needed for uplink. Mixed assignment counts (core/model_assignment.py:15-18, computed): case1 1603/1603/2137; case2
1603/1603/2137; case3 1069/1069/1069/1068/1068.

## Mixed (FedProto, FedTGP, TinyProto, V2X-HtFL, pFedES)
Exchange is backbone-independent: prototypes d=128 (learned projection per backbone, `MixedAdapter`), pFedES proxy 351 params, TinyProto s=13.
Per-client bytes (all groups alike): FedProto ≤8,256 up/8,192 down; FedTGP ≤8,192/8,192; V2X ≤8,256 (64 B unused counts)/8,192; TinyProto ≤832/832 + 256 B masks once;
pFedES 1,404/1,404. Measured on toy run (`probe_mixed.py`): global proto tensor (16,128) fp32 = 8,192 B; pFedES proxy 351 floats.
mixed.py:88-96 payload_bytes self-describes as "per-client worst-case 16 present classes".

## MobiHet L4 (single and mixed): no parameter/weight exchange
Bootstrap (once, measured by the codec in `methods/mobihet_lite.py:195-224`, identical across all backbones/seeds, per vehicle mean over 5,343):
up 98,943 B = counts 64 + histograms 28,746 + class moments 6,694 + Lloyd 4,599 + final cells 58,839; down 241,720 B (unicast) = counts 64 + CDF 59,361 +
own-class centres 12,794 + mixture 169,501 (of which Cholesky codes 124,765). If the whole downlink is one broadcast: 267,806 B. Total 340,663 B.
Per round (`methods/mobihet_sim.py:789-832`, q_bits=8): up = 34 B header + present-class rows of 16 uint8 (raw or zlib), mean over vehicles; down = one 16×16 uint8
Q table + 32 B, broadcast. Measured 50-round means (B/vehicle/round up / down): DAGSNet 132.8/145.8, BERT 175.8/187.6, TabM 201.4/227.0, SAE 203.2/239.1,
MLP 192.5/212.2, BiGRU 191.5/206.8 (spread across seeds < 1 B; range over rounds 102-209 up, 145-241 down). Total 50 rounds ≈ 7-12 KB + bootstrap ≈ 340 KB
→ bootstrap dominates (97%+). Mixed: only 2-round pilots local (up 177-194, down 232-236); 50-round mean ESTIMATED by client-weighted single means
(case1 169/183, case2 198/225, case3 181/201). Mixed L4 pools Q and bootstrap over all groups (one network): bootstrap bytes same as single-model
(case2 pilot 98,942.8 / 241,696; others identical to single-model).
Q aggregation weights n_kc are sent only in the bootstrap (counts); none per round.

## Discrepancies vs docs/research/cost.md, bert cost.md, payload_bytes()
1. cost.md MobiHet row "~117 KB/~114 KB per round + 197 KB bootstrap" is the old 3-bit proxy-delta design; L4 sends ~0.13-0.24 KB/round and 340.7 KB bootstrap. `mobihet.py:324` payload_bytes reports dense_reference + bootstrap only; per-round bytes exist only in rounds.jsonl (`mobihet_mixed_l4.py:275`).
2. FedProto/V2X/FedTGP/TinyProto payload_bytes are dense 16-class worst cases; actual uploads cover only present classes (avg 13.10/16 = 81.9%). Downlink = full 16 classes (correct).
3. V2X `payload_bytes` up includes 64 B counts the algorithm does not send (fedproto needs them: weighted mean).
4. TinyProto `masks_once` (16·d/8 B) is returned but absent from both cost.md tables; masks are seed-derived (0 B if seed shared).
5. PerFed-SKD payload_bytes = all-client, all-round (4(P+B)); real = selected fraction (round 1 all).
6. CANPerFL payload_bytes = 0; its phase-1 cost = FedAvg ×50 and the docstring claim "θ^50 download counted in FedAvg's last round" is wrong (round r downloads θ^{r-1}): +1 download 4(P+B)/client (conservative).
7. FedAvg-family do count BN running buffers but not num_batches_tracked: consistent with what is aggregated.
8. AFPHA: 8-cluster hierarchy and adaptive-μ scalars (≈4 B up, 4 B down) are not counted; PerFed-SKD a_m scalar (4 B) from all clients not counted.
9. LwFedNIDS: pruned θ⁰ is shipped once but equals the round-1 download (do not double count). cost.md number (0.162 MB) matches measured.
10. bert/cost.md numbers (311,616; 40,776; 1,404; 4,160/4,096; 448; 4,096) all reproduced by the probe; v2x P=82,128 includes unsent projection.
11. SAE d: cost.md d=256 notation is DAGSNet-only; SAE d=93, TabM 32, MLP 32, BiGRU 128, BERT 64, mixed 128 (cost.md table "4Cd = 16.4 KB" holds for DAGSNet only).

## Not verifiable from local code/data (candidates for Kaggle / owner)
- PerFed-SKD selected_frac per round for SAE, MLP, CNN-BiGRU (only final-round 0.444 SAE s41 in RUN_STATUS) and mixed N/A: need `rounds.jsonl` of those runs (script: parse `selected_frac`).
- Mixed L4 full 50-round `bytes_up/bytes_down` (rounds.jsonl) for 9 runs: only 2-round pilots local.
- Whether TinyProto masks / V2X ETF seed count as communicated (policy choice).
- fold_counts.npy lives in sibling repo `../mobihet/cache`; the v4 parquet is not local (DVC pointers only), so AFPHA could not be instantiated (clients.parquet) - its tensors equal FedAvg's.
- No on-wire serialization exists for model/prototype methods (tensor-size counts only); L4 bytes are real coded sizes from per-vehicle simulation (LZMA/zlib/bz2), computed on the whole 5,343 population for bootstrap, all vehicles for per-round up.
