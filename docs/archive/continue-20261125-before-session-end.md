# Snapshot bàn giao — 25/11/2026 09:22 (+07)

Trạng thái hiện tại ở [bàn giao sống](../operations/continue.md).

Snapshot trước compact: [continue-20261124-before-compact](../archive/continue-20261124-before-compact.md).

## Batch MobiHet (L4 bỏ LA) + ablation A1/A2/A4 — bắt đầu 24/11 07:50

Owner 24/11: MobiHet = L4 `la_mode=off` ([MobiHet](../research/mobihet.md), [ablation](../research/mobihet-ablation.md)).
24 hàng `mobihet` đã ghi từ run A3 cũ (kiểm lại, `runs/mobihet-promote/receipt.json`). L4 + ablation L4 → `../mobihet/archive/l4-la-20261124/`.

- **Lịch:** `runs/ablation-mh/plan.json` (quota `runs/ablation-mh/quota-20261124.json`; `plan.py`, margin 1 phút, run nhỏ ≤ 2 h vào
  account ít quota). 83 notebook ≈ 371 GPU-h: 3 seed MobiHet thiếu (`bert-mh-s43`, `cnnbigru-mh-s43`, `mixmh-case1-s41`), A1/A2/A4 ×
  9 kịch bản × 3 seed trừ 6 run chờ, 5 kernel đếm FLOP `exp-count-mh-*`. Run chờ quota ghi bên dưới.
- **Push:** `runs/ablation-mh/slot_queue.py`, chu kỳ 1200 s. Kiểm quota và slot theo owner trước từng lượt, một notebook
  mỗi owner mỗi lượt. `slot-queue.json` giữ intent và exact version, `push-ledger.jsonl` giữ biên nhận. `PUSH_UNCLEAR` cần
  đối chiếu Kaggle trước retry. Push đi qua guard dùng chung của `kaggle-push`.
- **Collector:** `runs/ablation-mh/watch_collect.py`, chu kỳ 1200 s, 8 luồng status, 2 pull/audit đồng thời, `-j 64` mỗi pull.
  Đọc exact ref từ `active.txt` và `slot-queue.json`. Kết quả r50 PASS cập nhật CSV kịch bản. Trace FLOP phải đủ file theo plan
  và khớp SHA nguồn. Phiên partial/audit FAIL ghi `handoff.jsonl`, waiter đánh thức phiên Codex để xử lý.
- **Sau khi đủ:** `python3 artifacts/communication-budget/scripts/mobihet_update.py`, `python scripts/compute/report_detail.py`,
  `python scripts/compute/report.py`, `python scripts/experiments_status.py`; cập nhật [compute](../research/compute.md) (bảng L4 →
  MobiHet), [MobiHet](../research/mobihet.md). Local đếm TabM/SAE/MLP/mixed-case2 (`scripts/compute/run_queue.py --only …-mobihet`, `--ablation`).
- Theo dõi: `tail -n 200 -F -- runs/ablation-mh/{slot-queue,watch-collect,codex-handoff}.log`.
- **Tài khoản, 25/11:** quota xác minh 26/34 sau khi thử lại qua endpoint dự phòng. Owner yêu cầu gỡ cấu hình vận hành và
  credential local của 8 account còn `UNVERIFIED`. Đã gỡ khỏi hàng đợi, roster và cache health. Giữ biên nhận audit.
  Trong các entry gỡ khỏi queue, 11 đã audit R50, 7 đã push và cần chạy lại do tài khoản cũ mất khả năng vận hành, 3 chưa push.
  Exact ref và quyết định ghi trong `runs/ablation-mh/account-retirement-20261125.json`.
- **Chạy lại:** `tamnp2807/mixmh-abl-a1-case1-s41/1` được Kaggle xác nhận 09:01, sau đó báo RUNNING.
  Phiên cũ thuộc đợt owner xác nhận chưa chạy ngày 24/11. Giữ nguyên SHA notebook, seed, cache input và cấu hình 2×T4.
  Tài khoản mới có 30 h trước push, dự toán run 9,3 h. Arm A1, seed 41 train từ round 1.
- **Chạy lại 7 kernel theo owner, 25/11:** đổi đúng owner trong metadata. Cả 7 notebook khớp SHA từ ledger cũ.
  Sáu run train bắt đầu round 1 và một run đếm FLOP. Kaggle đã nhận 5 phiên version 1.
  Exact ref: `trietbackup/mixmh-case1-s41/1`, `khunglongtrex/exp-count-mh-main2/1`,
  `telannass/cnnbigru-mhabl-a2-s43/1`, `yellowbalm/mixmh-abl-a2-case2-s43/1`, `tamnp2807/mixmh-abl-a4-case3-s42/1`.
  Biên nhận và ánh xạ exact ref cũ → mới: `runs/ablation-mh/rerun-wave-20261125.json`.
  `mixmh-abl-a4-case3-s43` trên `tamnp2807` chờ slot và quota. `mixmh-abl-a1-case3-s42` trên `shippertocdo`
  chờ đủ 7 giờ sau khi trừ tải đang chạy. Reset quota đọc từ API là 28/11.
- **Chờ bố trí quota:** mixed-case1 a2/a4 × 3 seed, `mixmh-abl-a1-case1-s42`, `mixmh-abl-a1-case1-s43`, `mlp-mhabl-a4-s43`.
  Ba notebook chưa push bị gỡ khỏi queue cũng chờ gán owner. Mốc reset cũ 28/11 là thời điểm kiểm lại quota.
- **Audit 25/11:** 8 kernel train PASS R50, 2 kernel FLOP PASS với 9 trace. Ba pull timeout hoàn tất sau khi đổi endpoint.
  Kiểm lại SHA notebook, support mỗi lớp/xe, 10 metric và CSV. Receipt `runs/ablation-mh/support-audit-20261125.json`.
  Không còn session đã audit nào cần nối. BERT và CNN-BiGRU đủ ba seed MobiHet. Còn mixed case1 seed 41.
- **Đang chạy:** controller `mobihet-ablation-collect-20261125` (supervisor 12022, collector PID 12045) và
  `mobihet-ablation-handoff-r2-20261125` (supervisor 12240). Delivery `queue` về thread hiện tại. Theo dõi
  `watch-collect.log`, `codex-collector-20261125.log`, `codex-handoff-r2-20261125.log` dưới `runs/ablation-mh/`.
  Dispatcher `mobihet-ablation-rerun-r2-20261125` (supervisor 14237) chỉ xử lý 7 slug owner vừa duyệt, kiểm lại mỗi 20 phút.
  Log `codex-rerun-r2-20261125.log`. State controller tại `~/.codex/long-jobs/jobs/mobihet-ablation-rerun-r2-20261125/state.json`.
  Bản dispatcher đầu dừng vì 2 lỗi trước khi vào guard ghi. Đã kiểm nhật ký guard và ghi lại ở `rerun-prewrite-failures-20261125.json`.
  Waiter cũ ngày 25/11 thoát 3 trước khi collector ghi PID. Waiter r2 đã mở sau khi xác nhận PID sống.
- **Backup checkpoint 25/11:** owner yêu cầu pull đa luồng và push DVC. Job `experiment-checkpoint-dvc-20261125`
  (supervisor 17555, delivery `queue`) kiểm kê exact version của batch MobiHet và 24 run A3 đã đưa vào bảng chính.
  Bỏ qua 15 account theo danh sách owner trong script `inventory.py`. Hai pull kernel đồng thời, 64 luồng mỗi kernel.
  Checkpoint nằm trong `pulls/<owner>/<slug>/checkpoints/vN/` của chính run, giữ nguyên đường dẫn Kaggle bên trong.
  Mỗi run có `checkpoint-index.json` và DVC pointer theo owner/slug/version trên OneDrive. Log và receipt tại
  `runs/ablation-mh/checkpoint-backup-20261125/`. Run đầu đã PASS và có pointer. [Biên bản backup](../operations/backup.md)#checkpoint-mobihet--25112026).
- **Bằng chứng đợt 24/11:** `runs/ablation-mh/watch-collect.json` và audit tại `pulls/<owner>/<slug>/ablation-audit.json`.
  Có 37 kernel PASS tính đến cuối lượt 24/11. Các job controller 24/11 đã dừng sau lần tắt máy.
- Đếm FLOP local TabM/SAE/MLP/mixed-case2 đã xong cả MobiHet và các arm. Log `runs/ablation-mh/count-local-{main,abl}.log`.
- Việc tay: 5 account có lỗi token trong bàn giao 24/11 (`chiencosieuhang`, `abbafan`, `gaming3`, `goilaaiudi`, `hoangdang20t`).
  Commit `803a84f`, `288628e` có trailer `Co-Authored-By:` (trái quy tắc owner 30/10); sửa cần force-push: owner quyết.
- Git: `main` = `7375539` khớp `origin/main` local lúc bàn giao. Hook Git LFS đã gỡ khỏi `.git/hooks` (owner 24/11: DVC thay LFS; bản cũ ở
  `.git/lfs-hooks-removed-20261124/`). `.gitattributes` vẫn khai báo `artifacts/*.zip` LFS cho các ZIP lịch sử.

Dùng `/home/odixe/project/experiment` trên nhánh `main`. Trạng thái phiên Claude e360305d đã nhập vào lịch sử Git và artifacts bên dưới. Protocol, seed 41/42/43, HP L4 và cosine 50 giữ nguyên. [Context](../../CONTEXT.md) · [Protocol](../research/protocol.md) · [Compute](../research/compute.md)

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
- Ba endpoint chờ xác minh cuối là Centralized BERT seed 41/42/43. E27/E24/E25 là các epoch BERT đã audit (22/11). Sau khi audit và cập nhật CSV endpoint, chạy lại `PYTHONPATH=. python runs/round-metrics/audit/audit.py` với env `nckh`, rồi `verify.py` cùng thư mục. Biên bản pull và kiểm nằm ở `runs/round-metrics/audit/`.

## Git và dữ liệu

- Local main chứa merge `ee262da` (central + kết quả), `dd1dba1` (L4), `2bf7680` (Codex, backup Git LFS) và `3281ad6` (Codex, mixed128 r50). Lịch sử main giữ source, docs, tests và dữ liệu đã ghi.
- Kiểm checksum 71 mapping artifact. `runs/worktree-consolidation/copy-map.json` cho biết vị trí gốc và đích. Collisions, biến thể cache và Git admin được giữ trong cùng thư mục. Notebooks, run cũ, checkpoints, metadata và DVC pointers đã có bản canonical.
- Worktree `experiment-wt` đã gỡ sau khi kiểm Git và đối chiếu 1.872 file ngoài Git với bản canonical. Commit `a8cb691` và hai commit Codex tới `5cf5795` đều nằm trong `main`. Chỉ còn một checkout thật. Đã xác minh SHA-256 và chuyển khoảng 75 GB artifact, checkpoint và cache sang ORICO. [Hồ sơ backup](../operations/backup.md))
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

1. Batch MobiHet (mục đầu): collector xử lý 7 kernel chạy lại. Theo dõi `handoff.jsonl` và quota các notebook chờ.
   Khi đủ, chạy các lệnh "Sau khi đủ". Backup theo yêu cầu owner.
2. BERT tập trung P4 đã audit đến E27/E24/E25. Giữ P5 theo yêu cầu owner.
3. Ablation L4 cũ: không thu tiếp (archive `../mobihet/archive/l4-la-20261124/`). Lịch sử đến 23/11: [snapshot](../archive/continue-20261124-before-compact.md).

