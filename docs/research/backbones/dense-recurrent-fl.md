# Single-model MLP and CNN-BiGRU-Attention

Scope: 14 existing baselines, seeds 41/42/43, each backbone separately; 84 notebooks.
Dataset, clients, local epoch, optimizer, aggregation and evaluation follow protocol.md.
Each direct run targets 50 rounds. CANPerFL uses same-backbone/same-seed FedAvg round50,
then 10 personalization epochs. No centralized warm start.

MLP retains centralized 77→64→32→16, dropout0.1, 7,600 parameters. Renamed modules
for FL: features.0/features.3/head correspond to centralized sequential 0/3/6.
CNN-BiGRU-Attention retains Conv64→BN→MaxPool→Conv128→BiGRU64→Attention128,
dropout0.3, 118,288 parameters. Hidden representation is attention context128;
MLP uses hidden32 before final dropout. APFed shares body and personalizes classifier.
V2X adds projection and ETF as in existing adapter.

LwFedNIDS physically prunes 70% per dependency group by summed absolute parameter slices.
MLP widths become20/10. CNN widths become20/39, GRU20 per direction, attention39.
GRU reset/update/candidate rows and recurrent columns are sliced together; the same selected
hidden coordinates are used in both directions. This is an explicit structured-pruning adaptation.
BN running buffers are sliced but excluded from parameter-importance scores.

Build: scripts/build_dense_recurrent_matrix.py (no push). Account assignment is provisional;
check live quota/slots before launch. Generated notebooks include mandatory reference and method
smokes, then two-GPU real-data smoke with round-boundary resume before production.
Grouped GRU uses explicit PyTorch reset-after recurrence; throughput and AMP equivalence require
GPU validation. No T4 speed claim follows from parameter count or CPU tests.

Validation 14/11/2026: CPU PyTorch2.15.0 passed eval forward, masked train forward,
parameter gradients and BN running statistics against reference modules. All28 model/method
one-round smokes passed, including CANPerFL parent loading. For26 direct model/method pairs,
round1 serialize/restore then round2 matched uninterrupted per-view confusion matrices.
This is CM equality on small CPU fixtures, not bit-exact CUDA weight equivalence.
V2X-HtFL server shuffle RNG was missing from state; now saved/restored as gen_s.
84 notebooks generated and static audited; not pushed. Mandatory two-T4 smoke remains pending.
