# Bàn giao — experiment, 20/11/2026 chiều (+07)

Dùng `/home/odixe/project/experiment` trên nhánh `main`. Trạng thái phiên Claude e360305d đã nhập vào lịch sử Git và artifacts bên dưới. Protocol, seed 41/42/43, HP L4 và cosine 50 giữ nguyên. [Context](../../CONTEXT.md) · [Protocol](../research/protocol.md) · [Compute](../research/compute.md)

## Kaggle recovery

Kiểm quota và danh sách 30/30 account. Cửa sổ kernel gần handoff có 62 ref, gồm 49 COMPLETE, 10 RUNNING và 3 ERROR thuộc project vrepair. Kiểm slot theo kernel gần nhất của từng account. Owner-scoped receipts nằm trong `runs/recovery-20261120/{quota,current-status,priority,ledger}.json`. Không đổi global login.

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
| `gaming3/mixl4-case2-s42-h2/1` | COMPLETE r50, PASS 50 round và checkpoint hai rank, .854014 |
| `lelovo/mixl4-case3-s42/1` | COMPLETE pilot r2 PASS, .717384 |
| `goilaaiudi/exp-count-l4-g4096/1` | COMPLETE, 3 JSON done và 3 log khớp các job CNN-BiGRU L4, mixed L4 case1/case3 |
| `gaming3/exp-central-cnnbigru-s41/1` | COMPLETE, epoch 50 đã audit và ghi ở phiên Claude |

CNN-BiGRU L4 đủ 10 metric r50 trong [results.csv](../../experiments/cnn-bigru/results.csv). Kiểm seed, L4 HP, source, LR, đủ 5.343 xe, CM 9.347.679 dòng, hội tụ hữu hạn và checkpoint cả hai rank. Audit chạy lại idempotent. Pilot mixed dùng CM và health riêng từng backbone, scope theo đúng assignment. Chỉ endpoint 50 vào bảng kết quả. [Chi tiết L4](../research/mobihet-l4.md)

Ba BERT epoch 9 dùng source đóng băng riêng theo seed. Seed 41 và 43 có cùng hash, seed 42 có sửa thứ tự barrier/config DDP và bật find_unused cho SAE-CE. Cấu hình BERT dùng cùng HP. Đã kiểm full state của model, optimizer và AMP scaler, fingerprint data, history liên tục, prediction → CM và 10 metric. Byte nguồn, log, prediction và checkpoint ở `runs/central-20261120/pulls/` và ledger recovery.

| Exact ref đang chạy | Quan sát gần nhất |
|---|---|
| `abbafan/exp-central-bert-s41-p2/1` | RUNNING, import epoch 9, epoch 10 đã qua step 5.250 / 9.122 |
| `chiencosieuhang/exp-central-bert-s42-p2/1` | RUNNING, import epoch 9, epoch 10 đã qua step 4.750 / 9.122 |
| `bestyorn/exp-central-bert-s43-p2/1` | RUNNING, import epoch 9, epoch 10 đã qua step 5.500 / 9.122 |
| `lelovo/mixl4-case1-s42-h2/1` | RUNNING, hai rank xác nhận restore r2 |
| `lelovo/mixl4-case3-s42-h2/1` | RUNNING, hai rank xác nhận restore r2 |
| `gaming3/mixl4-case2-s41/1` | RUNNING, seed mới sau pilot PASS, source đóng băng của pilot, mục tiêu r50 |
| `abbafan/mixl4-case2-s43/1` | RUNNING, seed mới sau pilot PASS, source đóng băng của pilot, mục tiêu r50 |
| `gaming3/mixl4-case3-s41/1` | RUNNING, 2×T4 và lockstep PASS, source pilot case3, seed 41, ngân sách 9,25 giờ |
| `goilaaiudi/exp-flops-v3-bert/1` | RUNNING, trace mới FeatureBERT và native BERT L4 |
| `goilaaiudi/exp-flops-v3-dagsnet/1` | RUNNING, trace mới DAGSNet |
| `hoangdang20t/exp-flops-v3-cnnbigru/1` | RUNNING, trace mới CNN-BiGRU |
| `hoangdang20t/exp-flops-v3-mixed13/1` | RUNNING, trace mới mixed case1/3 |

H2 mixed kiểm receipt và kích thước hai rank từ mount trước khi decode dataset. `scripts/audit_mixed_l4.py` nhận pull, notebook đã chạy, exact ref và inventory checkpoint. Script xuất receipt cho phiên nối và chỉ nhận `--csv` khi đủ r50. Tiếp theo, đọc `RUN_STATUS.json` và checkpoint đã commit trước khi dựng H3. BERT tiếp tục bằng `scripts/build_central_backbones.py --parent-artifacts ... --parent-epoch E`. Mixed dùng `scripts/build_mixed_l4.py --parent-artifacts ...`, giữ source, assignment và checkpoint hash của parent. Không chạy orchestrator MobiHet cũ.

`experiment-mixed-l4-terminal-20261120` đã thu case2 seed 42 r50 lúc 15:12 +07. Kiểm lại source, config và lịch cosine với pilot, tính lại đủ 10 metric trên CM gộp và từng backbone ở cả 50 round, kiểm support từng xe/lớp và SHA-256 của 530.729.210 byte checkpoint hai rank. Receipt `runs/mixed-l4/sessions/mixl4-case2-s42-h2/independent-verification.json`. [results.csv](../../experiments/mixed-case2/results.csv)

Case3 seed 41 dùng slot vừa giải phóng trên `gaming3`. Quota kiểm trước push 14,33 giờ, giữ phần ngân sách cho case2 seed 41. Source và HP lấy từ pilot case3, đổi seed 42 → 41. `experiment-mixed-l4-terminal-next-20261120` theo dõi 5 phiên mixed hiện tại, thu phiên terminal kế và gửi Codex queue. Log và trạng thái nằm trong `runs/recovery-20261120/mixed-l4-session-watch-next.{log,json}`.

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

Khôi phục đủ 10 metric và exact ref cho 168 cấu hình FL của TabM, SAE, MLP và CNN-BiGRU từ bản sao Kaggle. Tính lại 36 cấu hình TabM từ CM, đối chiếu config, toàn bộ vòng liên tục và support từng lớp. 132 cấu hình được đối chiếu marker cuối, manifest và smoke của đúng phiên bản, trạng thái `marker_metrics` trong results.csv. Không có CM của các phiên bản đó trong backup. Macro-F1 giữ nguyên ở độ chính xác công bố. Các checksum và script audit nằm trong `runs/recovery-20261120/{backbone-archive-audit.json,audit_backbone_archive.py}`. [Bảng bốn backbone](../research/backbones/four-backbone-seed-results-live.md)

## Git và dữ liệu

- Local main chứa merge `ee262da` (central + kết quả), `dd1dba1` (L4), `2bf7680` (Codex, backup Git LFS) và `3281ad6` (Codex, mixed128 r50). Source, docs, tests và thay đổi đã bảo toàn đều nằm trong lịch sử main. GitHub main chưa push.
- Kiểm checksum 71 mapping artifact. `runs/worktree-consolidation/copy-map.json` cho biết vị trí gốc và đích. Collisions, biến thể cache và Git admin được giữ trong cùng thư mục. Notebooks, run cũ, checkpoints, metadata và DVC pointers đã có bản canonical.
- Checkout `experiment-l4` và checkout detached dùng hồi quy đã gỡ. `experiment-wt` còn được phiên Codex và shell mở làm cwd. Dùng main cho mọi công việc mới. Gỡ sau khi các tiến trình đó chuyển sang main hoặc đóng. Commit `a8cb691` đã nằm trong main.
- Backup LFS remote 903.962.154 byte có bản local `runs/worktree-consolidation/runs-no-checkpoints-2026-11-20.zip`, SHA-256 `9e5d8eb4573785f418ff710c38de0af0b238cdac0bcfd31714a871ee42f2c012`.
- Kết quả khoa học chạy trước đây lưu trong `experiments/`. Evidence DAGSNet/BERT L4, source native và lịch LR nằm trong `runs/recovery-20261120/references/`. Đã pull đủ cả hai rank cho sáu run CNN-BiGRU và mixed pilot. Kiểm kích thước object, đọc metadata state weights-only, đối chiếu round/config và ghi SHA-256 trong `runs/recovered-checkpoints-20261120/`.

Kiểm local trên main. 47 pytest PASS trong suite portable và 38 PASS trong FeatureBERT. Dense recurrent PASS 28 method smokes và 26 kiểm tra resume CM. Mixed L4 PASS lockstep, pooling, assignment, identity và resume r1→r2. Compute regression CPU/CUDA PASS 3 kiểm tra. Test NPY đối chiếu parquet cần dataset v4 local đầy đủ; máy hiện có cache panel và DVC pointers.

## Compute

Đã cập nhật đủ 9 bảng lịch sử và xuất 9 bảng `compute-detail.csv`, kèm trace source, loại đo, FLOP client/server/other/bootstrap, eval riêng, Big-O và ranking. Có 57/114 hàng có trace chi tiết, gồm 6 probe centralized. BERT lịch sử có ước tính scaling 2.048 → 4.096, trace mới đo trực tiếp. Mixed dùng FedProto cùng case. [Quy ước, số liệu và cách tái lập](../research/compute.md)

- `experiment-flops-small-20261120` PASS lúc 14:39 +07. Đủ trace cho TabM, SAE, MLP (16 hàng mỗi backbone) và mixed case2 (6 hàng). Kiểm lại SHA-256 của 57 trace đã công bố. Sinh lại cả 9 bảng và `activity.csv` cho kết quả khớp từng byte. Log và receipt nằm trong `runs/compute-detailed-v3/{local-matrix.log,local-matrix-verification.json}`.
- `experiment-flops-kaggle-20261120` chờ bốn kernel count ở trên, pull/audit output exact version và sinh lại bảng. Log `runs/compute-detailed-v3/remote-collect.log`.
- CSV detailed chỉ xếp hạng hàng có trace đã kiểm. Trace baseline cũ, phép đo seed 42, hai round mới và extrapolation 50 đều có nhãn riêng.

## Việc còn lại

1. Audit các phiên BERT và mixed khi terminal. Nối từ checkpoint được kiểm cho tới epoch/round 50. Case1 seed 41/43 và case3 seed 43 chờ slot/quota. Case2 seed 42 đã đủ r50, seed 41/43 còn chạy. Chọn slot/quota mới ở lúc push.
2. Thu 57 trace chi tiết còn thiếu từ bốn kernel Kaggle. Xử lý từng trường hợp OOM và xác minh số backward, padding và provenance trước khi đưa vào bảng. So các trace mới với lịch sử theo nguồn và quy ước.
3. Teammate bổ sung CM cho 132 cấu hình còn ở mức kiểm marker. Account teammate không có credential ở đây.
4. Đóng các phiên dùng cwd `experiment-wt`, mở workspace `experiment` và gỡ worktree cũ sau khi kiểm trạng thái lần cuối. Chỉ phát triển trên main.
