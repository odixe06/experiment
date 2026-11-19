# TabM single-backbone baseline matrix

## Locked scope — 14/11/2026

- Backbone: TabM k=4, two hidden blocks of width 32, 6,548 trainable parameters, 77 inputs, 16 classes.
- Data/client partition: VeReMi v4, fixed 5,343 natural clients; train+val for training, official test for evaluation.
- Methods: FedAvg, FedProx, AFPHA, FD-IDS, APFed, LwFedNIDS, CANPerFL, TinyProto, pFedES, NILM-FL,
  FedTGP, PerFed-SKD, FedProto, V2X-HtFL. SCAFFOLD, centralized and proposal excluded.
- Seeds: 41, 42, 43. Only initialization, dropout and shuffle RNG change; data and client partition stay fixed.
- Shared protocol: 50 FL rounds, all clients each round, one local epoch, batch 64, AdamW, cosine 1e-3→1e-5,
  weight decay 1e-4, gradient clip 1, AMP. No centralized warm-start.
- Reporting: test confusion matrix and all metrics saved every round. Primary comparison = round-50 mean ± standard
  deviation over three seeds, not best-seed selection.
- CANPerFL is the paper's two-stage procedure: same-seed FedAvg round-50 parent, then 10 local fine-tuning epochs.
  Its fine-tune epochs are stored individually; it is not mislabeled as an independent 50-round federation.

## TabM adapters

- Supervised loss: mean cross-entropy over four TabM members.
- Prediction and KD distribution: arithmetic mean of four member probabilities.
- Prototype representation: mean of four post-block member hidden vectors, dimension 32.
- APFed classifier: TabM member heads (`weight`, `bias`); both ensemble blocks are base parameters.
- LwFedNIDS: one dependency-aware L1 structured mask at initialization; 70% of each 32-wide hidden layer is
  physically removed (10 units retained). Input adapters and output member heads are sliced consistently.
- V2X-HtFL: trainable 32→32 weight-normalized projection is added after the member-mean representation.
- pFedES: paper proxy remains a shared 77→77 two-convolution enhancer; local classifier is TabM.

Implementation keeps DAGSNet path unchanged. `scripts/run_fl.py --backbone tabm` selects TabM. CPU smoke covers all
14 methods, reference/grouped equality, CANPerFL parent load, finite one-round training and evaluation shape/counts.

## Launch contract

Each method/seed is a separate private 2×T4 Kaggle notebook. Notebook embeds exact source and hashes, asserts two Tesla
T4 GPUs, builds one NPY cache, runs exact round-1→round-2 resume smoke, then production with a session budget guard.
Artifacts include source/experiment manifests, `rounds.jsonl`, per-round confusion matrices, per-round small model state,
resume checkpoints and `RUN_STATUS.json`. Incomplete sessions stop cleanly and must resume from their own checkpoint.

## Interim audited round-50 result

Sample standard deviation over seeds 41/42/43. APFed headline uses personalized `client` view; others use `global`.

| Method | Seed 41 | Seed 42 | Seed 43 | Macro-F1 mean ± sample std |
|---|---:|---:|---:|---:|
| FedAvg | 0.842643 | 0.847424 | 0.843763 | 0.844610 ± 0.002501 |
| FedProx | 0.830465 | 0.833102 | 0.825901 | 0.829823 ± 0.003643 |
| AFPHA | 0.826838 | 0.828985 | 0.821977 | 0.825933 ± 0.003591 |
| FD-IDS | 0.804815 | 0.805610 | 0.805849 | 0.805425 ± 0.000541 |
| APFed (`client`) | 0.855935 | 0.861721 | 0.857185 | 0.858280 ± 0.003044 |

All 15 runs passed local audit of source hashes, config, contiguous 50 rounds, all per-client confusion matrices over
9,347,679 test rows per view/round, recomputed metrics, finite/skipped guard and 50 non-empty model checkpoints.
Executed notebooks are preserved beside each source notebook. Tensor deserialization was covered by remote resume smoke;
the Windows local audit environment has no PyTorch and therefore did not repeat strict tensor loading.
