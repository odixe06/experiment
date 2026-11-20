# Bàn giao — experiment, 21/11/2026 (+07)

Dùng `/home/odixe/project/experiment` trên nhánh `main`. Trạng thái phiên Claude e360305d đã nhập vào lịch sử Git và artifacts bên dưới. Protocol, seed 41/42/43, HP L4 và cosine 50 giữ nguyên. [Context](../../CONTEXT.md) · [Protocol](../research/protocol.md) · [Compute](../research/compute.md)

## Cập nhật lúc 04:28 +07

- Mixed L4 Case3 đủ 3/3 seed ở R50. Phiên seed 43 H2 khớp history và CM R1–R49 với parent. Audit đủ 50 CM, 10 metric gộp và theo backbone, SHA-256 checkpoint hai rank. Biên bản `runs/round-metrics/audit/mixed-l4-case3-s43-r50-independent.json`. [Bảng case3](../../experiments/mixed-case3/results.csv)
- Mixed L4 Case1 đủ 3/3 seed R50. Audit 50 CM và SHA-256 hai shard checkpoint mỗi seed. History và CM của seed 41/42/43 khớp parent tới R49/R44/R47. [Bảng case1](../../experiments/mixed-case1/results.csv)
- Centralized BERT P2 đã audit PASS cả ba seed tại E17/E16/E17 (41/42/43). Kiểm lại 50 CM, metrics CSV, dự đoán epoch cuối, weights và optimizer/scaler. Biên bản độc lập ở `runs/round-metrics/audit/central-bert-p2-independent.json`.
- BERT seed 41 P3 audit E21. Kiểm lại dự đoán → CM, 10 metric từng epoch, checkpoint và prefix E1–E17 với P2. Chọn `dinhan20t` có slot trống và quota 7,67 giờ cho phiên nối dài hơn. Dataset riêng tư `dinhan20t/exp-central-bert-s41-e21-state` ready và khớp 151 tên, kích thước file. P4 nối từ E22 với cache NPY, budget 7,50 giờ. Receipts `runs/recovery-20261120/after-boot/bert-s41-e21-*.json`.
- BERT seed 42 P3 audit E19. Kiểm lại dự đoán → CM và đủ 10 metric E1–E19, checkpoint, source thực thi và prefix E1–E16 với P2. Owner cũ còn 1,26 giờ, sát 72–73 phút mỗi epoch. Dataset riêng tư `dinhan25/exp-central-bert-s42-e19-state` đã ready và khớp 137 tên, kích thước file. P4 nối từ E20, budget 7,05 giờ với cache NPY. Receipts `runs/recovery-20261120/after-boot/bert-s42-e19-*.json`.
- BERT seed 43 P3 audit E19, prefix E1–E17 khớp P2. Owner cũ còn 1,03 giờ, dưới thời gian một epoch đo được. Dataset riêng tư `hoangdang20t/exp-central-bert-s43-e19-state` đã ready, 137 tên và kích cỡ khớp ZIP, cấu hình và checkpoint đã kiểm SHA-256. P4 tiếp tục từ E20. Receipts `runs/recovery-20261120/after-boot/bert-s43-e19-*.json`.
- Codex `9b576f7` bổ sung đủ sáu history Mixed128. Kiểm độc lập 300 round, đủ 10 metric cho mọi view gộp và theo backbone, 30.000 giá trị hữu hạn. R50 khớp CM và CSV; case3 khớp prefix parent, case2 khớp macro-F1 log. ZIP trên ORICO, receipt `runs/round-metrics/audit/mixed-history-codex-verification.json`.
- Đã nhập Codex `aceefc7`. ZIP CNN-LSTM-feature trên ORICO khớp SHA-256 của LFS và 258 mục trong manifest. Cả 50 epoch khớp prediction → CM → 10 metric. Đã chuẩn hóa run lịch sử ở `runs/round-metrics/legacy/cnn-lstm-feature/centralized/s42/`. [Hồ sơ backup](backup.md#github-centralized-cnn-lstm-feature-supplement--20112026)
- Mixed L4 case2 đủ 3/3 seed tại R50. [Chỉ mục kết quả](../../experiments/README.md)
- Đã gỡ hai symlink tên checkout cũ. Source BERT nằm trong `bert/`, DAGSNet ở gốc repo. Năm script MobiHet local dùng đường dẫn mới. [Hồ sơ backup](backup.md)

| Phiên nối | Mốc parent audit PASS | Budget | Trạng thái |
|---|---:|---:|---|
| `lelovo/mixl4-case1-s42-h3/1` | R44 | 2,90 h | COMPLETE, R50 audit PASS |
| `goilaaiudi/mixl4-case3-s43-h2/1` | R49 | 2,50 h | COMPLETE, R50 audit PASS |
| `dinhan20t/mixl4-case1-s41-h2/1` | R49 | 1,60 h | COMPLETE, R50 audit PASS |
| `dinhan20t/mixl4-case1-s43-h2/1` | R47 | 2,00 h | COMPLETE, R50 audit PASS |
| `abbafan/exp-central-bert-s41-p3/1` | E17 | 5,70 h | COMPLETE, E21 audit PASS |
| `dinhan20t/exp-central-bert-s41-p4/1` | E21 | 7,50 h | RUNNING, gắn cache NPY |
| `chiencosieuhang/exp-central-bert-s42-p3/1` | E16 | 4,25 h | COMPLETE, E19 audit PASS |
| `dinhan25/exp-central-bert-s42-p4/1` | E19 | 7,05 h | RUNNING, gắn cache NPY |
| `bestyorn/exp-central-bert-s43-p3/1` | E17 | 3,35 h | COMPLETE, E19 audit PASS |
| `hoangdang20t/exp-central-bert-s43-p4/1` | E19 | 11,25 h | RUNNING, gắn cache NPY |

Collector `experiment-continuations-2324-20261120` hoàn tất lúc 03:12 +07, exit 0, cả ba output audit PASS trên ORICO. Log và ledger ở `runs/recovery-20261120/after-boot/{continuation-2324-watch.log,continuation-2324-watch.json,continuation-launch-2324.json}`. State ở `~/.codex/long-jobs/jobs/experiment-continuations-2324-20261120/state.json`.

Collector `experiment-central-bert-s41-p3-20261120` hoàn tất lúc 04:24 +07, exit 0. State ở `~/.codex/long-jobs/jobs/experiment-central-bert-s41-p3-20261120/state.json`. Log `runs/recovery-20261120/after-boot/central-bert-s41-p3-watch.log`, ledger `continuation-launch-2340.json` cùng thư mục.

Collector Case3 seed 43 H2 `experiment-mixed-case3-s43-h2-20261121` kết thúc lúc 01:16 +07, exit code 0. State ở `~/.codex/long-jobs/jobs/experiment-mixed-case3-s43-h2-20261121/state.json`. Log `runs/recovery-20261120/after-boot/mixed-case3-s43-h2-watch.log`, ledger `continuation-launch-0021.json` cùng thư mục. Output trên ORICO.

Nhóm collector thu output trên ORICO, so history và CM với parent, kiểm receipt cache. Delivery `queue`. State ở `~/.codex/long-jobs/jobs/<name>/state.json`.

| Collector | PID | Log và receipt trong `runs/recovery-20261120/after-boot/` |
|---|---:|---|
| `experiment-mixed-case1-s41-h2-20261121` | 150112 | `mixed-case1-s41-h2-watch.{log,json}` |
| `experiment-mixed-case1-s43-h2-20261121` | 148826 | `mixed-case1-s43-h2-watch.{log,json}` |
| `experiment-central-bert-s41-p4-20261121` | 196673 | `central-bert-s41-p4-watch.{log,json}` |
| `experiment-central-bert-s42-p4-20261121` | 169217 | `central-bert-s42-p4-watch.{log,json}` |
| `experiment-central-bert-s43-p4-20261121` | 149875 | `central-bert-s43-p4-watch.{log,json}` |

Ledger `continuation-launch-{case1-s41-h2,case1-s43-h2,bert-s41-p4,bert-s42-p4,bert-s43-p4}.json` trong cùng thư mục.

### Cache cho phiên nối

- 02:15 +07: Đọc được 12 output cache từ exact ref `odixe0502/mobihet-v4npy-cache/1` qua tất cả owner đang chạy. Manifest được pin SHA-256. Mười NPY khớp cả ba config BERT P2. Đã tải hai metadata v4 gốc, đủ **12/12 SHA-256 khớp** identity của parent. Mixed case1 giữ cùng thứ tự 77 feature, fold, cách decode và đọc các mảng. Receipt `runs/recovery-20261120/after-boot/mobihet-npy-preflight.json`.
- Hai builder nhận `--npy-cache runs/recovery-20261120/after-boot/mobihet-npy-preflight.json` cho phiên nối. Gate kiểm mount, SHA-256 manifest và từng mảng trước train. Mixed dùng `V4_NPY` trỏ mount. BERT tạo cache view ở `/kaggle/temp` với 10 NPY read-only và `identity.json` từ 12 SHA đã kiểm, đối chiếu `config.data` của parent. BERT giữ parquet v4 để đọc metadata, dùng `--parent-dataset` khi chuyển tài khoản. Source và checkpoint giữ nguyên.
- Log Mixed ghi build NPY 193–249 giây. BERT P2 ghi 228–305 giây từ import xong tới bắt đầu train. Bốn log L4 ở repo MobiHet ghi kiểm cache 40–47 giây trên T4. Ước tính giảm khoảng 2,5–3,5 phút và 3–4,5 phút cho mỗi phiên nối tương ứng. Tính thời gian mount thực vào budget. Một log MobiHet cũ mất 208 giây để đọc và kiểm cache. Mixed H3 seed 42 có round đầu 30,2 phút và các round sau khoảng 11,7 phút. Notebook hiện tại chưa xuất Inductor tar. Chỉ xét tar cùng source/runtime/GPU khi đã đo và kiểm tương đương.
- Mixed Case1 H2 ghi kiểm SHA-256 cache trên Kaggle **41,77 giây** ở seed 41 và **38,48 giây** ở seed 43 với cùng manifest đã pin. Receipts `runs/mixed-l4/sessions/mixl4-case1-s{41,43}-h2/npy_cache_receipt.json`.

## Trạng thái kiểm ngày 20/11 lúc 20:47 +07

- Đã query cả **27 exact ref** trong bàn giao. 18 COMPLETE, 9 RUNNING. Kiểm Mixed L4 trước. Listing output exact version trả 0 file cho cả chín phiên đang chạy (`after-boot/active-output-listings.json`). Hai kernel FLOP đã COMPLETE và 27 trace mới đã audit.
- **Mixed L4** tiếp tục từ các phiên case1/case3 dưới đây. Chỉ nối khi chứng thực được biên checkpoint và file state của cả hai rank. Chỉ công bố endpoint r50 sau audit tất cả round.
- **Centralized BERT** P2 đã nối từ epoch 9, log ba seed đã ghi tới commit epoch 14. P3 chờ output P2 đã audit. Dựng P3 bằng `scripts/build_central_backbones.py --owner <owner> --model bert --seed <seed> --slug exp-central-bert-s<seed>-p3 --parent <owner>/<P2-slug> --parent-artifacts <pull-dir> --parent-epoch <E> --max-hours <budget>`. Giữ cùng owner khi đủ quota/slot để mount parent. Kết thúc chain khi epoch 50 audit PASS.

Ba collector của đợt này dùng `nckh`, account-scoped auth và đường API `www`. Từng collector tự pull/audit các phiên terminal trong nhóm, ghi receipt sau từng phiên và xếp thông báo về thread Codex khi cả nhóm kết thúc. Collector ghi artifact và receipt. Phiên Codex đọc receipt, công bố endpoint hoặc triển khai nối khi có checkpoint dưới đích.

| Collector | PID | Log và receipt trong `runs/recovery-20261120/after-boot/` |
|---|---:|---|
| `experiment-mixed-case1-recovery-20261120` | 17827 | `mixed-case1-watch.{log,json}` |
| `experiment-mixed-case3-recovery-20261120` | 17850 | `mixed-case3-watch.{log,json}` |
| `experiment-central-bert-p2-recovery-20261120` | 17873 | `central-bert-p2-watch.{log,json}` |

Script collector `runs/recovery-20261120/after-boot/collect_terminal.py`. State supervisor tương ứng nằm ở `~/.codex/long-jobs/jobs/<name>/state.json`. Nếu reboot, xác minh state và receipt trước khi khởi động lại. Kernel remote tiếp tục qua reboot. Đích output của 9 kernel đang chạy đã trỏ vào ORICO, xem `experiment-data/experiment/local-artifacts/pending-collectors.json`.

## Kaggle recovery

Quota buổi tối xác minh 31/31 account, snapshot `runs/recovery-20261120/after-boot/quota.json`. Kaggle báo reset 21/11 00:00 UTC. Đọc lại quota và slot trước khi chọn phiên nối. Ba notebook mixed trước khi tắt máy dùng source pilot và cùng target r50. Hash notebook và exact ref ở `runs/recovery-20261120/pre-shutdown-launch.json`. Evidence live và kết quả query chính xác nằm trong `runs/recovery-20261120/{all-kernels-after-boot.json,after-boot/}`.

| Exact ref | Trạng thái và audit |
|---|---|
| `gaming3/cnnbigru-l4-s41/1` | COMPLETE, PASS r50, macro-F1 .863450 |
| `hoangdang20t/cnnbigru-l4-s42/2` | COMPLETE, PASS r50, .863215 |
| `bestyorn/cnnbigru-l4-s43/1` | COMPLETE, PASS r50, .861055 |
| `abbafan/exp-central-bert-s41/2` | COMPLETE phiên 1, checkpoint epoch 9 PASS, .945419 |
| `chiencosieuhang/exp-central-bert-s42/1` | COMPLETE phiên 1, checkpoint epoch 9 PASS, .947931 |
| `bestyorn/exp-central-bert-s43/2` | COMPLETE phiên 1, checkpoint epoch 9 PASS, .946489 |
| `lelovo/mixl4-case1-s42/1` | COMPLETE pilot r2 PASS, .758572 |
| `gaming3/mixl4-case2-s42/1` | COMPLETE pilot r2 PASS, .676016 |
| `gaming3/mixl4-case2-s41/1` | COMPLETE r50, PASS 50 round và checkpoint hai rank, .853143 |
| `gaming3/mixl4-case2-s42-h2/1` | COMPLETE r50, PASS 50 round và checkpoint hai rank, .854014 |
| `abbafan/mixl4-case2-s43/1` | COMPLETE r50, PASS 50 round và checkpoint hai rank, .854455 |
| `lelovo/mixl4-case3-s42/1` | COMPLETE pilot r2 PASS, .717384 |
| `goilaaiudi/exp-count-l4-g4096/1` | COMPLETE, 3 JSON done và 3 log khớp các job CNN-BiGRU L4, mixed L4 case1/case3 |
| `goilaaiudi/exp-flops-v3-bert/1` | COMPLETE, pull/audit PASS 15 trace |
| `goilaaiudi/exp-flops-v3-dagsnet/1` | COMPLETE, pull/audit PASS 15 trace |
| `hoangdang20t/exp-flops-v3-cnnbigru/1` | COMPLETE, pull/audit PASS 15 trace |
| `hoangdang20t/exp-flops-v3-mixed13/1` | COMPLETE, pull/audit PASS 12 trace |
| `gaming3/exp-central-cnnbigru-s41/1` | COMPLETE, epoch 50 đã audit và ghi ở phiên Claude |

CNN-BiGRU L4 đủ 10 metric r50 trong [results.csv](../../experiments/cnn-bigru/results.csv). Kiểm seed, L4 HP, source, LR, đủ 5.343 xe, CM 9.347.679 dòng, hội tụ hữu hạn và checkpoint cả hai rank. Audit chạy lại idempotent. Pilot mixed dùng CM và health riêng từng backbone, scope theo đúng assignment. Chỉ endpoint 50 vào bảng kết quả. [Chi tiết L4](../research/mobihet-l4.md)

Ba BERT epoch 9 dùng source đóng băng riêng theo seed. Seed 41 và 43 có cùng hash, seed 42 có sửa thứ tự barrier/config DDP và bật find_unused cho SAE-CE. Cấu hình BERT dùng cùng HP. Đã kiểm full state của model, optimizer và AMP scaler, fingerprint data, history liên tục, prediction → CM và 10 metric. Byte nguồn, log, prediction và checkpoint ở `runs/central-20261120/pulls/` và ledger recovery.

| Exact ref đang chạy | Log Kaggle lúc 20:39 +07 |
|---|---|
| `abbafan/exp-central-bert-s41-p2/1` | RUNNING, commit e14, đang e15 bước 5.750 / 9.122 |
| `chiencosieuhang/exp-central-bert-s42-p2/1` | RUNNING, commit e14, đang e15 bước 750 / 9.122 |
| `bestyorn/exp-central-bert-s43-p2/1` | RUNNING, commit e14, đang e15 bước 7.250 / 9.122 |
| `dinhan20t/mixl4-case1-s41/1` | RUNNING, log r21 |
| `lelovo/mixl4-case1-s42-h2/1` | RUNNING, restore r2, log r32 |
| `dinhan20t/mixl4-case1-s43/1` | RUNNING, log r20 |
| `lelovo/mixl4-case3-s42-h2/1` | COMPLETE r50, audit PASS cả 50 CM và checkpoint hai rank |
| `gaming3/mixl4-case3-s41/1` | COMPLETE R50, audit PASS |
| `goilaaiudi/mixl4-case3-s43/1` | COMPLETE R49, audit PASS và đã nối H2 |

H2 mixed kiểm receipt và kích thước hai rank từ mount trước khi decode dataset. `scripts/audit_mixed_l4.py` đối chiếu source, LR, history, CSV và CM gộp/của từng backbone ở mọi round đã commit. Nó chỉ nhận `--csv` ở r50. Với phiên terminal bị ngắt trước marker cuối, `--interrupted` audit tới biên `checkpoint.json`. Nếu endpoint checkpoint dưới r50, dùng `scripts/build_mixed_l4.py --owner <owner> --case <case> --seed <seed> --session <n+1> --parent <owner>/<slug> --parent-artifacts <pull-dir> --max-hours <budget>` sau khi hai file state đã audit. Giữ source, assignment, config hash và checkpoint của parent.

`experiment-mixed-l4-terminal-20261120` đã thu case2 seed 42 r50 lúc 15:12 +07. Kiểm lại source, config và lịch cosine với pilot, tính lại đủ 10 metric trên CM gộp và từng backbone ở cả 50 round, kiểm support từng xe/lớp và SHA-256 của 530.729.210 byte checkpoint hai rank. Receipt `runs/mixed-l4/sessions/mixl4-case2-s42-h2/independent-verification.json`. [results.csv](../../experiments/mixed-case2/results.csv)

`experiment-mixed-l4-terminal-next-20261120` thu case2 seed 43 và `experiment-mixed-l4-shutdown-20261120` thu seed 41. Cả hai đã tính lại đủ 10 metric cho cả 50 round và kiểm hai checkpoint. Receipts nằm trong `runs/mixed-l4/sessions/mixl4-case2-s{41,43}/independent-verification.json`. Buổi tối chạy lại auditor mới trên cả ba seed, PASS 150 round và CSV khớp từng dòng. [Kết quả case2](../../experiments/mixed-case2/results.csv)

## Mixed128 baseline

Import hai commit Codex `5d2648b` và `66c1e9c` qua merge `3281ad6`. Khôi phục 1.324 file mixed128 từ backup LFS đã xác minh. Audit local lại 45/45 cấu hình r50, 378 dòng theo backbone/view, đủ 10 metric mỗi dòng. Khớp config hash, assignment, seed, lịch sử r50 và support từng lớp của 5.343 xe trên 9.347.679 bản tin. 342 dòng đã có giữ nguyên. Script đối chiếu và receipt SHA-256 nằm trong `runs/recovery-20261120/{audit_mixed128_archive.py,mixed128-local-audit.json}`. [Bảng đầy đủ](../research/mixed/mixed128-all-results.md) · [CSV](../research/mixed/mixed128-all-results.csv)

| Kết quả mới | Exact ref | Macro-F1 r50 |
|---|---|---:|
| Case 1 V2X-HtFL seed 41 | `sonlebaodac/mix128-case1-v2x-htfl-s41-a2-r10-r44/1` | .84538 |
| Case 1 V2X-HtFL seed 42 | `minhgia2026/mix128-case1-v2x-htfl-s42-a2-r04-r05-r06-r07-r42/1` | .84564 |
| Case 1 V2X-HtFL seed 43 | `nguyntngquckhnh/mix128-case1-v2x-htfl-s43-a2-r08-r44/1` | .84411 |
| Case 3 pFedES seed 41 | `nguynkhacao/mix128-case3-pfedes-s41-a1-r06-r16-r17-r18/1` | .80720 |
| Case 3 pFedES seed 43 | `nguynkhacao/mix128-case3-pfedes-s43-a1-r06-r08-r09-r10-r43/1` | .80866 |

### Baseline bốn backbone

Khôi phục đủ 10 metric và exact ref cho 168 cấu hình FL của TabM, SAE, MLP và CNN-BiGRU từ bản sao Kaggle. Hai ZIP có đủ config, history và CM cuối. Audit tính lại 10 metric cho cả 168 run, khớp CSV và support từng xe/lớp. Đã nâng 132 dòng từ `marker_metrics` lên `audited`. [Bảng bốn backbone](../research/backbones/four-backbone-seed-results-live.md)

**339 dòng endpoint có đủ 10 metric và nhãn `audited`**. Ba ô endpoint còn thiếu thuộc Centralized BERT. [Chỉ mục kết quả](../../experiments/README.md)

## Lịch sử từng round và epoch

- `runs/round-metrics/` trỏ tới `ORICO/experiment-data/experiment/round-metrics/`. Mỗi run có đúng hai file trong `<scenario>/<method>/s<seed>/`. `metrics.csv` dùng cùng schema `step,view` và 10 metric cho mỗi bước/view. `run.json` ghi seed, đích, exact ref khi xác minh được, source/config, CM, log, checkpoint, SHA-256 và danh sách dữ liệu thiếu. `index.json` tra cả 342 run. CANPerFL có 10 epoch fine-tune và trỏ về FedAvg 50 round cùng seed.
- **339/342 run đã hoàn tất kiểm chứng**, gồm 45/45 baseline Mixed128. Chỉ mục có **40.169 dòng bước/view**. Khôi phục history DAGSNet từ object DVC xác minh MD5. Kéo 550 CM của 11 run Centralized, đối chiếu lại đủ 10 metric cho từng epoch. CNN-BiGRU Centralized seed 42 dùng bảng audit 50 epoch ở độ chính xác 5 chữ số do Kaggle trả 403 với artifact gốc. [Bảng CNN-BiGRU](../../papers/mnkash-2026-cnn-bigru/results.md)
- Ba endpoint chờ xác minh cuối là Centralized BERT seed 41/42/43. E21/E19/E19 là các epoch BERT đã audit. Sau khi audit và cập nhật CSV endpoint, chạy lại `PYTHONPATH=. python runs/round-metrics/audit/audit.py` với env `nckh`, rồi `verify.py` cùng thư mục. Biên bản pull và kiểm nằm ở `runs/round-metrics/audit/`.

## Git và dữ liệu

- Local main chứa merge `ee262da` (central + kết quả), `dd1dba1` (L4), `2bf7680` (Codex, backup Git LFS) và `3281ad6` (Codex, mixed128 r50). Source, docs, tests và thay đổi đã bảo toàn đều nằm trong lịch sử main. GitHub main chưa push.
- Kiểm checksum 71 mapping artifact. `runs/worktree-consolidation/copy-map.json` cho biết vị trí gốc và đích. Collisions, biến thể cache và Git admin được giữ trong cùng thư mục. Notebooks, run cũ, checkpoints, metadata và DVC pointers đã có bản canonical.
- Worktree `experiment-wt` đã gỡ sau khi kiểm Git và đối chiếu 1.872 file ngoài Git với bản canonical. Commit `a8cb691` và hai commit Codex tới `5cf5795` đều nằm trong `main`. Chỉ còn một checkout thật. Đã xác minh SHA-256 và chuyển khoảng 75 GB artifact, checkpoint và cache sang ORICO. [Hồ sơ backup](backup.md)
- Backup LFS remote 903.962.154 byte có bản local `runs/worktree-consolidation/runs-no-checkpoints-2026-11-20.zip`, SHA-256 `9e5d8eb4573785f418ff710c38de0af0b238cdac0bcfd31714a871ee42f2c012`.
- Kết quả khoa học chạy trước đây lưu trong `experiments/`. Evidence DAGSNet/BERT L4, source native và lịch LR nằm trong `runs/recovery-20261120/references/`. Đã pull đủ cả hai rank cho sáu run CNN-BiGRU và mixed pilot. Kiểm kích thước object, đọc metadata state weights-only, đối chiếu round/config và ghi SHA-256 trong `runs/recovered-checkpoints-20261120/`.

Kiểm local trên main buổi tối. Suite portable 46 PASS (thêm 4 subtest), FeatureBERT 39 PASS và 1 skip. Dense recurrent CLI PASS 28 method smokes và 26 kiểm tra resume CM. Compute CPU/CUDA 3 PASS. Mixed L4 CLI CPU PASS lockstep, pooling, assignment, identity và resume r1→r2. Auditor chấp nhận biên commit khi thiếu marker cuối và bắt được tamper metric r1. Test NPY đối chiếu parquet thiếu shard v4 local. Link check giữ nguyên 142 lỗi link đã có từ trước. Log kiểm thử ở `runs/recovery-20261120/after-boot/`.

21/11 chạy lại portable 45 PASS và 1 skip (thêm 4 subtest), BERT 39 PASS và 1 skip, dense recurrent CLI PASS 28 smoke và 26 resume CM. Mixed L4 CLI CPU PASS từ cache panel trên ORICO. Gate NPY PASS trên fixture mmap, từ chối file sai SHA và sai identity. Notebook BERT P4 kiểm import E19 và SHA của toàn bộ cây parent khi mô phỏng mount local.

## Compute

Đủ 9 bảng lịch sử và 9 bảng `compute-detail.csv`. Cả **114/114 hàng** có SHA-256 khớp trace. BERT lịch sử ước tính scaling 2.048 → 4.096, trace mới đếm trực tiếp. Tổng FL 50 round ngoại suy từ hai round eager trên panel 200 xe. Mixed dùng FedProto cùng case. [Quy ước, số liệu và cách tái lập](../research/compute.md)

- `experiment-flops-small-20261120` PASS lúc 14:39 +07. Đủ trace cho TabM, SAE, MLP (16 hàng mỗi backbone) và mixed case2 (6 hàng). Kiểm lại SHA-256 của 57 trace đã công bố. Sinh lại cả 9 bảng và `activity.csv` cho kết quả khớp từng byte. Log và receipt nằm trong `runs/compute-detailed-v3/{local-matrix.log,local-matrix-verification.json}`.
- Hai kernel của `hoangdang20t` bổ sung đủ 27 trace. Audit SHA-256 notebook exact version, manifest source, log, panel, backward, packing và reconciliation coarse/detail. Receipt độc lập `runs/recovery-20261120/after-boot/flop-independent.json`. Receipt thu bốn kernel `runs/compute-detailed-v3/kaggle/collection.json`.
- CSV detailed chỉ xếp hạng hàng có trace đã kiểm. Trace baseline cũ, phép đo seed 42, hai round mới và extrapolation 50 đều có nhãn riêng.

## Việc còn lại

1. Khi phiên BERT hiện tại kết thúc, đọc receipt và mốc epoch thật. Nếu dưới 50, nối cùng seed, source, config và optimizer/scaler đã audit. Cập nhật CSV khi có E50.
2. Sinh lại chỉ mục, kiểm bảng và commit local các endpoint mới sau khi audit. Làm việc tiếp trên main canonical.
