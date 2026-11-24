# Audit: communication budget of executed FL runs (21/11/2026)

**Update 24/11/2026:** MobiHet is now L4 without logit adjustment (method `mobihet`). The `mobihet_l4` rows below were
replaced in the three CSVs by `mobihet` rows measured from the audited MobiHet R50 runs (`scripts/mobihet_update.py`, same
formula); the L4 rows are archived in `../mobihet/archive/l4-la-20261124/experiment/communication-budget/`. Statements about
`mobihet_l4` in this report describe that archived L4 set.

Verdict: the 324 canonical production runs are complete; none is partial. Every run's total is calculated or measured
from its own logs and config. None of these numbers is network traffic: they are simulated payload sizes. The
formulas are in [method_formulas.md](method_formulas.md).

## Scope

| set | runs | in summary | how counted |
|---|---:|---|---|
| canonical production: 6 single-model scenarios × 15 methods × 3 seeds, plus 3 mixed cases × 6 methods × 3 seeds | 324 | yes | one canonical session per logical run, rounds 1..R contiguous |
| pilots: mixed L4 r2 | 6 | no | the 3 pilots with K = 5,343 are byte-identical prefixes of the canonical s42 chain and must not be added; the 3 dbg runs (K = 100/100/20) are independent |
| superseded re-runs: SAE/TabM comparison v1, mixed-case2 V2X a1 | 48 computed, 3 skipped | no | the v1 SAE prototype methods use their own config (head_hidden 16 ≠ 93); the 3 mixed-case2 V2X a1 sessions have no config, so K is unknown and they are skipped |
| MobiHet probes (older variants, K = 200 panels, BERT L4 probes) | 53 | no | measured `bytes_*` from logs; old variants are `low` confidence because their byte semantics were not re-audited |
| excluded | 1,041 rows | — | continuation sessions and duplicates (subsumed), smoke, compute-trace FLOP panels, centralized, config-only, SCAFFOLD |

Inventory: 1,472 run directories across the repo, ORICO and 3 ORICO zip archives
([run_inventory.csv](run_inventory.csv), columns `in_budget` and `budget_exclusion_reason`). For every
non-centralized cell in `experiments/*/results.csv`, the canonical session matches the cited source. No expected
cell lacks per-round data. 306 runs have R = 50; CANPerFL has R = 10 fine-tune epochs. K = 5,343 for all.

## Verification

| check | result |
|---|---|
| Dense model size: own `config.payload_bytes` vs CPU instantiation, all production runs | 0 mismatches (12 expected mismatches, all in SAE v1 superseded runs, which use their own config) |
| Hand recomputation, DAGSNet FedAvg | 2·5,343·50·1,594,688 = 852,041,798,400 B ✓ |
| Hand recomputation, DAGSNet FedProto | up 50·69,993·1,028 ✓, down 5,343·50·16,384 ✓ |
| MobiHet L4 recurring vs an independent sum of the inventory worker | max \|Δ\| 1.5e-8 B over 27 runs ✓ |
| PerFed-SKD: selected devices per round re-read from rounds.jsonl vs inventory | Δ = 0 ✓ |
| Mixed per-backbone breakdown sums to the case total | 0 mismatches; group sizes and pairs partition 5,343 clients / 69,993 pairs ✓ |
| Bootstrap counted once | bootstrap only in L4 (27) and CANPerFL (18); recurring excludes it ✓ |
| Double counting across sessions | cumulative logs, one session per logical run, round sequence 1..R with no gaps (0 gaps) ✓ |
| L4 `bytes_up` scope | `methods/mobihet_sim.py:789-832` accumulates every vehicle's packet each round. Mixed L4 groups are virtual ranks of one allreduce (`methods/mobihet_mixed_l4.py:192`), so the mean is over all 5,343 vehicles; K·mean is exact ✓ |
| L4 bootstrap scope | `methods/mobihet_lite.py:211-224` sums every vehicle's encoded bytes and divides by N = 5,343 ✓ |
| Broadcast vs unicast | every baseline downlink is identical for all clients (one copy in `server_down_unique_B`); the only per-client downlinks are L4 bootstrap centres and AFPHA μ_i |
| Mean ± std | sample SD (ddof = 1) over seeds 41/42/43. SD = 0 for deterministic formulas; it is non-zero only for PerFed-SKD (selection) and L4 (measured packets) |

## Results (mean over 3 seeds, total = up + down over all clients and rounds)

| scenario | FedAvg family | LwFedNIDS | PerFed-SKD | FedProto / FedTGP / V2X | TinyProto | pFedES | MobiHet L4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DAGSNet | 852.04 GB | 86.56 GB | 240.02 ± 5.72 GB | 7.97 / 7.96 / 7.96 GB | 0.809 GB | 0.750 GB | 1.895 ± 0.000084 GB |
| BERT | 166.50 GB | 21.79 GB | 62.54 ± 1.80 GB | 2.00 / 1.99 / 1.99 GB | 0.218 GB | 0.750 GB | 1.917 ± 0.00005 GB |
| TabM | 13.99 GB | 4.45 GB | 6.17 ± 0.04 GB | 1.01 / 1.00 / 1.00 GB | 0.124 GB | 0.750 GB | 1.935 ± 0.0003 GB |
| SAE | 16.27 GB | 6.56 GB | 7.95 ± 0.08 GB | 2.91 / 2.89 / 2.89 GB | 0.311 GB | 0.750 GB | 1.938 ± 0.0006 GB |
| MLP | 16.24 GB | 4.16 GB | 6.92 ± 0.02 GB | 1.01 / 1.00 / 1.00 GB | 0.124 GB | 0.750 GB | 1.928 ± 0.0001 GB |
| CNN-BiGRU | 253.08 GB | 25.97 GB | 98.07 ± 0.79 GB | 3.99 / 3.98 / 3.98 GB | 0.404 GB | 0.750 GB | 1.927 ± 0.0002 GB |
| mixed case1/2/3 | N/A | N/A | N/A | 3.99 / 3.98 / 3.98 GB | 0.404 GB | 0.750 GB | 1.914 / 1.932 / 1.925 GB |

- **FedAvg family.** FedAvg, FedProx, FD-IDS, APFed and NILM-FL are identical. AFPHA adds 4 B per client per round
  for μ_i (+0.0001%).
- **CANPerFL.** Its own run moves only θ⁵⁰ once (8.52 GB DAGSNet, 1.66 GB BERT, ...). With the reused FedAvg phase it
  totals 1.01× FedAvg.
- **MobiHet L4.** It sends 0.13–0.24 KB per vehicle per round each way, but bootstrap is 340.7 KB per vehicle once,
  94–96% of its total. L4 is therefore cheaper than FedProto, FedTGP and V2X-HtFL on DAGSNet, CNN-BiGRU, BERT, SAE
  and all three mixed cases, and more expensive on TabM and MLP. TinyProto and pFedES are cheaper than L4 everywhere.
- **Totals.** All 324 production runs together: 25,838 GB as executed, or 29,793 GB with CANPerFL's phase 1
  attributed.

Full tables: [communication_summary.csv](communication_summary.csv) (mean/std per component) and
[communication_ranking.csv](communication_ranking.csv) (rank within scenario, × FedAvg).

## Differences from the existing communication tables

1. `docs/research/cost.md`, MobiHet row (~117/114 KB per round, 197 KB bootstrap) described the older Q3 proxy-delta
   MobiHet. It was replaced on 21/11 by a MobiHet L4 row (DAGSNet: 133 B up / 146 B down per round, bootstrap
   98.9 KB / 241.7 KB, compute 0.917× FedAvg). The stale `methods.mobihet` entry was removed from `cost.json`.
2. `cost.md` and the bert cost table: per-client values for FedAvg, LwFedNIDS, pFedES, TinyProto and the FedTGP
   downlink match exactly. Their prototype uplinks (4Cd, + counts) are dense 16-class values. The real uplink is 81.9%
   of dense (69,993 / 85,488 pairs), and `dense_upper_bound_up_B` keeps the dense value.
3. `cost.md` totals rule "byte/xe/round × K × 50": this overstates PerFed-SKD, whose real total is 0.28–0.49× FedAvg
   because only selected devices transfer after round 1.
4. `payload_bytes()` of V2X-HtFL includes 64 B of counts that the method never sends; they are dropped.
5. CANPerFL docstring ("θ⁵⁰ download counted in FedAvg's last round"): FedAvg's round r downloads θ^(r−1), so θ⁵⁰
   needs one more download. It is counted as CANPerFL bootstrap.
6. Worker interim figures superseded: per-client FedProto average "13,425 B" (correct value 13.0999·1,028 = 13,467 B)
   and the "estimated" mixed L4 50-round means. Full 50-round mixed L4 logs exist on ORICO and are used as measured.

## Limitations and uncounted components

- **Network traffic.** None exists. Totals exclude protocol framing, TLS/TCP/IP headers, retransmission and
  serialization containers. A `torch.save` container would add ~1.6 KB per tensor (see `work/probe_single.json`).
- **Class-presence index** for sparse prototype uploads: not counted, at most 2 B per client per round (≤ 0.53 MB
  per run).
- **TinyProto masks** cost 0 B (owner decision 21/11: seed-derived on every client). `payload_bytes()` still reports
  `masks_once`; that value is not counted.
- **AFPHA backhaul.** The cluster↔cloud tier (8·M per round each way) is not client↔server and is not counted.
- **Data-dependent N_p.** It comes from `/home/odixe/project/mobihet/cache/fold_counts.npy`, a sibling repo, because
  the v4 parquet is DVC-only here. Its row total equals the protocol's 37,361,273. The per-backbone mixed breakdown
  also assumes that the file's client order matches `core/model_assignment.py`; the case totals do not depend on that.
- **24 canonical runs without their own config.json.** These are all of mixed-case2, mixed-case3 fedtgp s41 and
  tinyproto s41/s43, and mlp/tabm V2X. Payload is formula-based and K = 5,343 comes from the manifest, so their
  totals do not depend on the missing file. mixed-case2 V2X inherited a superseded a1 config, which is only used as a
  label.
- **Evaluation, checkpoint upload/download, Kaggle artifact transfer and W&B** are excluded by scope.
- **Probes of old MobiHet variants** use logged `bytes_*` whose sampling scope (e.g. `BYTES_SAMPLE` = 64 per rank in
  `methods/mobihet.py`) was not re-audited. They are `low` confidence and outside the summary.

## Files

| file | content |
|---|---|
| run_inventory.csv | all 1,472 run dirs/sessions, canonical flag, budget inclusion and reason, source location |
| communication_per_run.csv | 431 computed runs: components, totals, formula, basis, confidence, sources, per-backbone breakdown |
| communication_summary.csv | 108 (scenario, method) cells, mean ± std of every component |
| communication_ranking.csv | rank within scenario and × FedAvg |
| scripts/inventory.py, scripts/payload_specs.py (+ probe_*.py, measured_stats.py), scripts/compute_budget.py | reproduction |
| work/ | intermediate tables, probes, cross-check JSON |
