# Mixed-model: fixed vehicle assignments — 17/11/2026

Owner confirmed: preserve each vehicle's existing data; randomize only its model.
This supersedes the earlier 25%/25%/remaining allocation in heterogeneous-backbones.md.

## Frozen assignment

Dataset: `odixe0502/veremi-nextgen-v4-scaled-per-vehicle`, natural receiver-vehicle
clients 0..5342. No repartition, row shuffle across clients, scaling change or split change.
The existing train+val / test boundary remains intact.

| Case | Models in declared order | Requested percentages | Exact client counts |
|---|---|---|---|
| 1 | DAGSNet, BERT, CNN-BiGRU-Attention | 30 / 30 / 40 | 1603 / 1603 / 2137 |
| 2 | TabM, SAE, MLP | 30 / 30 / 40 | 1603 / 1603 / 2137 |
| 3 | DAGSNet, BERT, CNN-BiGRU-Attention, TabM, SAE | 20 / 20 / 20 / 20 / 20 | 1069 / 1069 / 1069 / 1068 / 1068 |

Assignment seed **2026**, separate RNG per case (`Random("2026:case1")`, etc.).
Shuffle sorted global IDs once; allocate contiguous shuffled blocks by largest remainder.
Break equal remainders by the declared model order. No labels, sample counts or scores
enter assignment. These are client percentages, not sample or compute percentages.

All methods, all 50 rounds and training seeds **41/42/43** reuse exactly the same
CSV for a given case. Three seeds measure training variation conditional on this one
assignment; they do not measure sensitivity to different model assignments.

Frozen files:
- [Manifest and SHA256 hashes](../../../experiments/mixed-model-v1/manifest.json)
- [Case 1](../../../experiments/mixed-model-v1/case1.csv)
- [Case 2](../../../experiments/mixed-model-v1/case2.csv)
- [Case 3](../../../experiments/mixed-model-v1/case3.csv)
- [Local validation evidence](../../../experiments/mixed-model-v1/validation.json)

Reproduce or verify (standard library only):

```powershell
python scripts/build_mixed_assignments.py
python scripts/build_mixed_assignments.py --check
python -m unittest discover -s tests -p test_model_assignment.py -v
```

Builder refuses to overwrite differing frozen files. The loader validates the locked
specification and CSV bytes, then selects by **global client ID**, never local GPU row.
Future runners must verify global train/test ID coverage, record assignment hash in
config/checkpoint and reject resume with a different hash. The loader exists but is
**not yet wired into a mixed-model training runner**.

## Validation and implementation boundary

Three local tests PASS: exact counts/coverage and independence from training RNG;
sharding/reordering invariance; count-preserving assignment tampering rejected and
existing files not overwritten. Client universe independently checked against `ids.npy`
inside a collected full-client round50 CM artifact: exactly 0..5342. This does not
re-audit the source dataset's full contents or the old result metrics.

Repository-wide documentation link check still reports existing broken relative links
in archived handoffs and copied run-source documents. These unrelated snapshots were
left untouched. Existing trailing blank-line warnings in operations documents remain.

Current `scripts/run_fl.py` accepts one backbone per run; BERT is not in its FL choices.
`core/fl.py` owns one cfg/Layout; prototype methods use its single hidden dimension.
No mixed-model training, notebook generation or Kaggle push was performed in this step.

Next implementation work:
1. Add model-group execution and grouped BERT reference/gradient tests, preserving
   global client identity and each client's BN/optimizer/checkpoint state.
2. Specify and implement shared representation adapters for FedProto/FedTGP/TinyProto/
   V2X-HtFL, and the shared proxy for pFedES. Dimension and learning rules are not locked
   by these assignment files. Do not silently change the old single-model baselines.
3. Owner excluded the proposal on17/11: run only FedProto, FedTGP, TinyProto,
   V2X-HtFL and pFedES. No automatic 70/30 real/synthetic modification.
4. Preserve the [aligned FL protocol](../aligned-fl-v2.md): SAE CE-only/representation93,
   TabM k4, 50 rounds, seeds41/42/43; extend evaluation to model-group metrics and cost.
5. Validate resume/aggregation, then real-data GPU smoke before long production runs.

Current authorized scope: 3 cases × 5 baselines × seeds41/42/43 = **45 runs**, 50 rounds.
Owner approved **128 dimensions** on17/11. The design uses a trainable projection before
the16-class classifier so CE and prototype loss both train that representation.
pFedES retains its existing shared77→77 proxy mechanism; it does not need a common
private representation dimension. Launch is authorized after scientific choices,
reference/resume tests and GPU smoke gates are satisfied.

## Build and pilots — 17/11/2026

Latest15:40ICT:15/15pilots passed GPU/resume/CM audit. Production case2:
14/15COMPLETE50, V2X seed42 ERROR at r18 (SAE AMP skip1.15867%, threshold1%).
Case1/3 production not submitted. Production completion is confirmed by live
status/logs; independent final-CM audit still pending. See live handoff for evidence.

16:20ICT: all14completed final-CM audits now PASS; [results](mixed128-case2-results.md).
V2X SAE failure localized on real checkpoint batches to fp16 encoder gradients at
loss scale4096. a2 usesFP32 local training for SAE only (same math/parameters/steps),
other backbones retain AMP. All3V2X seeds repushed fresh to keep numerical policy common.
GPU fixture proof is not yet proof of full50round a2 completion. Old a1 results stay labeled.

- Implemented `core/mixed_backbones.py`, `methods/mixed.py`, `scripts/run_mixed_fl.py`:
  backbone groups share one previous-round server state and aggregate once across all groups/ranks.
  All five mechanisms retained; source method server routines reused for FedTGP/V2X.
- CPU forward/gradient/masked-BN checks PASS for six native/projected backbones;
  five-method six-backbone two-round exact state/CM resume checks PASS; joint FedProto
  aggregation and real CLI checkpoint/history smoke PASS. CPU evidence is not GPU proof.
- Adapter keeps native output layers in layout with explicit zero gradient; they are
  excluded from prediction but included in stored parameter count. TabM retains four
  member classifiers/CEs and averages projected features for prototypes.
- 45 production notebooks built locally (provisional owner, not pushed).
  Ledger `runs/mixed128-build.json`. Rebuild after any corrective source change.
- 15 bounded pilots (case×method, seed42) submitted v1, initially all RUNNING.
  Each uses30clients and at most128 rows/client, two rounds with boundary resume;
  this cap is diagnostic-only, never production. Explicit2×T4/compile/AMP gates.
  Ledgers `runs/mixed128-pilot-launch.json`, `runs/mixed128-pilots-live.json`.
  Production must wait for exact-source GPU evidence and measured budget.

13:46 ICT update: all5case2 pilots v1 COMPLETE with2round/resume/AMP/compile PASS,
zero skipped steps, independent final-CM10metric audit PASS. Case1/3 pilots still RUNNING.
Preparing first5case2 production runs seed42. Production includes new per-backbone
health counters and per-sample predictions/CSV; each exact notebook re-runs its GPU
smoke before full-data training. No pilot row cap carries into production.

Stored adapter parameters (includes inactive native output layers): DAGSNet430368,
BERT88288, TabM19028, SAE21709, MLP13888, CNN-BiGRU136864. V2X adds16640parameters
for its128D weight-normalized projection; its inherited classifier is inactive.
Do not label these as the original centralized parameter counts.
