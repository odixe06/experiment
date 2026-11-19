# MambaCNN supervised VeReMi adaptation

**Closed by owner,14November2026:** drop MambaCNN and retain earlier models. No further optimization, push or resume authorized by the current task. Keep source and diagnostic artifacts. No production epoch was trained.

Owner explicitly approved on 13 November 2026: retain three Mamba branches + CNN + self/cross-attention, all 77 features and 16 classes, choose dimensions in 100k–150k parameters, train supervised with the established centralized protocol. Name `mambacnn`; not `mambacnn-kd`, because there is no teacher or KD objective.

## Fixed architecture

[Model](../../core/mambacnn.py), built by `build_model("mambacnn")`:

| Component | Selected configuration | Trainable parameters |
|---|---|---:|
| Shared scalar embedding | Linear 1→40 with bias, 77 feature positions | 80 |
| Three independent branches | Two residual Mamba-1 blocks each; width40, expansion2, state16, dt-rank3, causal depthwise kernel4, pre-RMSNorm | 87,120 |
| Self-attention | Third branch as Q/K/V, width40, 4 heads | 6,560 |
| Cross-attention | First branch Q, second branch K/V, width40, 4 heads | 6,560 |
| CNN | Conv1d 1→32→64→64, kernel3, stride1, padding1, SiLU after each layer | 18,688 |
| Classifier | Concatenate three mean-pooled outputs: 40+40+64; Linear144→64, SiLU, dropout0.1, Linear64→16 | 10,320 |
| **Total** | **All parameters used at inference** | **129,328** |

These numerical dimensions, pooling, normalization and nonlinearity choices are explicit adaptations, not recovered paper settings. No parameter count tuning against test accuracy. Feature order is the existing frozen 77-column order; sequence positions are features within one row, not chronological messages. CNN consumes the same row. Mamba causality does not imply the complete self/cross-attention model is causal across feature positions.

Mamba input/output projections omit bias; depthwise convolution and delta projection include bias. B/C/delta depend on the input. A = -exp(A_log), initialized from negative integers1..16; diagonal skip starts at1; delta initialized log-uniformly in[0.001,0.1]. RMSNorm epsilon1e-5. Scan, delta, exponentials and gating accumulate in fp32 under AMP. CPU/double precision uses a PyTorch prefix reference without unstable division by cumulative decay products. Since version2, GPU fp32 uses the official Mamba selective-scan CUDA extension. Both implement the same recurrence; numerical equivalence is tolerance-based, not bit equality.

## Training and reproducibility

Full train+val37,361,273 rows, full test9,347,679 rows, seed42, logical batch4096, AdamW weight decay1e-4, cosine1e-3→1e-5 over50 epochs, clip1.0, plain16-class CE. Keep already scaled v4 features. No ADASYN, IEHO, extra scaling, stacking, teacher pretraining, embedding transfer or distillation. These exclusions follow the approved supervised scope and shared protocol.

Runner/fingerprint and notebook builder support `mambacnn`, per-epoch ten metrics, weights, optimizer/scaler and epoch resume. Initial version1 used microbatch32/GPU. Version2 benchmarks32/128/256/512 on the same two-T4 real-data pipeline and chooses the lowest measured train+eval time, retaining logical batch4096. Different microbatch/dropout random-number consumption can change the trajectory; there is no earlier completed MambaCNN production result to preserve. Parent resumes inherit the selected microbatch and frozen source.

## Verification

Local Python3.13/PyTorch2.10 CPU: **10 tests passed** in8.052seconds in `tests/test_central_backbones.py`:

- Independent sequential selective recurrence vs prefix outputs and gradients, lengths1/7/77, float64 atol/rtol1e-10 and float32 atol/rtol2e-5. Length1 correctly has zero sensitivity to transition A from a zero initial state.
- Exact parameter count, finite forward/backward, strict state reconstruction, sharded evaluation, causal Mamba block prefix, CPU bf16 autocast gradients.
- Existing runner fixture: odd train tail, saved predictions→CM→all10metrics, stopped/resumed two epochs vs uninterrupted weights bit-exact, including optimizer restoration and epoch-local RNG.

CUDA fp16 test is included and activates on GPU workers, but has not run locally. CUDA/DDP, full-data quality and timing remain pending.

Initial local notebook SHA256 was `288154b895080b89223e8f430cc5ffa611c3b93d355a673087cbb59c8d3524a5`. On14November the owner explicitly requested launch. Quota/identity checks found qwerasdf3004 with30hours free and zero reserved; previous CNN-BiGRU version1 was COMPLETE, and no existing MambaCNN appeared in the owner-scoped listing. A redundant fresh dataset listing timed out; dataset access was already demonstrated by the completed CNN-BiGRU run on this same account.

Regenerated for qwerasdf3004, adding a probe-budget gate: projected epoch time×1.5 must fit the remaining session before production starts. **Version1 ERROR at that gate**, before production. Notebook SHA256 `def07ee423a5b457f89295bab8d82f1c2468ecbfbdfc0461e68f5b2eeb58a457`. Worker confirmed PyTorch2.10.0+cu128, two TeslaT4 and all10tests PASS in20.586seconds. Exact-version pull: four JSON files plus log at `runs/pulls/central-mambacnn-v1`, zero missing. Measured estimate: train72,659.171875s + eval5,784.8515625s per epoch, total21.79001h/epoch or1,089.50033h/50; peak8.42649GiB, zero skips in9 probe steps. No production checkpoint exists.

## CUDA correction, 14 November 2026

Evidence → mechanism → falsifiable sign: pure-PyTorch prefix scan materializes large B×L×H×N tensors and launches many operations per block; six blocks and micro32 multiply the cost. Replace only scan with the fused official CUDA kernel, then measure larger microbatches. Accept only finite output/gradient equivalence and materially lower real-data timing, with epoch×1.5 fitting the session. Architecture,129,328 parameters, data, objective, optimizer and logical batch stay fixed. Owner explicitly requested the correction.

Dependency pinned to official [Mamba v2.3.2.post1](https://github.com/state-spaces/mamba/releases/tag/v2.3.2.post1), wheel `mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`, SHA256 `7a67070c1e7e99c95abd1319623f044e8a1b3fb46f774bfdea949f0a4fc79638`. Pip validates that digest with `--no-deps`; torch/image stay pinned. Notebook Internet enabled for this download. Missing CUDA extension is fatal on GPU, never silently falls back to the slow reference.

Predeclared gates: scan forward atol3e-5/rtol1e-4 and gradients atol/rtol3e-4 at lengths1/7/77;20 paired real-data AMP optimizer steps with identical initialization, rows and RNG, max logit error<0.005, relative gradient L2<0.01, all20 updates applied on both paths, final max weight error<0.002. This is bounded arithmetic/trajectory evidence, not proof of identical full-training metrics. [Gate script](../../scripts/check_mambacnn_cuda.py). Existing CPU exact-resume tests remain applicable to CPU only.

Local11tests:10PASS, CUDA scan test skipped (CPU-only torch),18.376s.17 embedded files, source/hash/AST/schema and metadata validation PASS. Pushed private [version2](https://www.kaggle.com/code/qwerasdf3004/veremi-central-mambacnn-e50/versions/2), cap12h; SHA256 `7a8cc7dd33d144880bf2f6ee69bfe740fde0fa24b141eb3859ef5d8f6197e646`. Quota29.888h, reserved0 before push; version1 terminal ERROR confirmed. Worker2T4, official CUDA import and11/11tests PASS17.569s. All20paired steps applied; maximum logit error0.001953125, relative gradient L2≤0.000568634. **Final max weight error0.004817676 exceeded0.002; version2 ERROR before timing/production.** Exact-version source/log pulled (2JSON, no missing) to `runs/pulls/central-mambacnn-v2`. Cause not yet localized; do not silently relax this gate.

Pushed [version3](https://www.kaggle.com/code/qwerasdf3004/veremi-central-mambacnn-e50/versions/3), SHA256 `8389edfda1a64740d208264361bf893838ff32393041205333a101e08a709df3`, adding per-parameter/index/gradient diagnostics and evaluation comparison before the same final weight assertion. Timing probes can run for diagnosis after equivalence exits nonzero; production still requires both successful equivalence and budget gates. Numerical thresholds unchanged. No global credential switch, Git push or W&B.

Version3 terminal ERROR confirmed through SDK `version_label=v3`;12JSON files and log pulled to `runs/pulls/central-mambacnn-v3`, zero missing. Weight difference localized to attention key biases: self index54 max0.006034181, cross index65 max0.005550618, gradients around1e-8 or smaller. Largest difference in other parameter tensors0.0001487583. Eval diagnostic on64 previously used training rows: max logit error0.001953125, zero prediction disagreements. Key bias contributes a constant across key positions and cancels algebraically through softmax; floating-point arithmetic can leave tiny gradients which AdamW amplifies. Local follow-up added a nullspace test and a gate excluding only those80coordinates while retaining raw differences, plus held-out functional checks. **Owner cancelled before validating or pushing that follow-up; do not claim the revised gate passed.**

Real-data two-T4 probe estimates (all logicalbatch4096, zero skipped probe updates): micro32=243.35951h/50,128=63.29112h/50,256=66.97319h/50,512=68.56993h/50. Fastest128 estimated train3838.29102s + eval718.67010s per epoch, about75.94935minutes/epoch and17.21× faster than version1 total. These are short-probe extrapolations, not completed epochs. Despite fewer parameters, this model remains computationally costly. No version4 was pushed; no production checkpoint exists.
