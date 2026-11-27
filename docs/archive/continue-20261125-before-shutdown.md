Snapshot trước khi dừng máy. Bàn giao sống tại [continue](../operations/continue.md).

# Bàn giao — experiment, 25/11/2026 20:15 (+07)

Mốc cuối phiên trên `/home/odixe/project/experiment`, local `main` tại `7375539`. Có thay đổi chưa stage, gồm DVC pointer mới.
ORICO đang mount tại `/run/media/odixe/ORICO`. [CONTEXT](../../CONTEXT.md) · [Bản đồ tài liệu](../README.md) · [Snapshot trước phiên kết thúc](../archive/continue-20261125-before-session-end.md)

## Tiếp tục từ đây

1. Đọc receipt backup và job cập nhật CSV bên dưới. Đối chiếu exact version trong `slot-queue.json`,
   `push-ledger.jsonl` và `watch-collect.json`.
2. MobiHet đủ **27/27** hàng R50. `trietbackup/mixmh-case1-s41/1` PASS, macro-F1 0,895531.
   Ma trận chính đủ **339/342** endpoint, gồm Mixed128 **45/45** và FL của TabM/SAE/MLP/CNN-BiGRU **168/168**.
   Centralized BERT seed 41/42/43 mới audit P4 tới E27/E24/E25. Giữ P5 chờ owner.
   [Kết quả](../../experiments/README.md) · [MobiHet](../research/mobihet.md) · [Ablation](../../experiments/ablation/README.md)
3. Ablation đã audit **51/81** ô R50. Đợt 20:14 phân bổ thêm notebook ngắn cho `tamnp2807`,
   `trietbackup`, `chuchuotdaubep` và `heykewtiie`. `khunglongtrex/mlp-mhabl-a2-s41` vào lượt kế tiếp.
   Dispatcher vẫn kiểm slot và quota live, các cấu hình chưa đủ quota tiếp tục chờ reset API 28/11.
4. Khi đủ artifact đã audit, cập nhật byte, compute và bảng kết quả bằng `artifacts/communication-budget/scripts/mobihet_update.py`,
   `scripts/compute/report_detail.py`, `scripts/compute/report.py` và `scripts/experiments_status.py`, rồi cập nhật
   [MobiHet](../research/mobihet.md) và [compute](../research/compute.md).
   L4 có LA và ablation L4 đời trước lưu ở `../mobihet/archive/l4-la-20261124/`.

## Job nền

- `experiment-checkpoint-dvc-direct` (supervisor 37819) chuyển đợt checkpoint đang chạy sang rclone trực tiếp.
  Đợi lệnh DVC đang ghi hoàn tất và VFS đồng bộ, rồi tiếp tục từ receipt và file tải dở.
  Log `runs/ablation-mh/checkpoint-backup-20261125/direct-backup.log`. Worker cập nhật catalog sau đợt chuyển.
- `mobihet-ablation-collect-resume` (supervisor 33649) pull/audit và ghi R50 cho batch đang chạy.
- `mobihet-ablation-completion-wave-rebalance` (supervisor 68185) dispatch 33 cấu hình qua Kaggle push guard.
  Log push ở `runs/ablation-mh/completion-wave-rebalance.log`, log quyết định quota/slot ở `slot-queue.log`.
- `experiment-dvc-rename-after-direct` (supervisor 37870) chờ backup trực tiếp xác minh xong trước khi đổi
  `veremi_build/fl_dagsnet_veremi_v2/dvc-storage` thành `veremi_build/experiment/dvc-storage`.
- `experiment-dvc-verified-local-cleanup` (supervisor 38533) đợi hai job DVC kết thúc.
  Chỉ dọn payload checkpoint của run audit R50, có pointer hợp lệ, đủ object và MD5 đọc từ OneDrive khớp.
  Giữ metadata, audit, cache chung và mọi file thiếu bằng chứng backup.
  Log `runs/ablation-mh/checkpoint-backup-20261125/cleanup-verified-local.log`.

## Batch MobiHet, audit và rerun

- Lượt kiểm 17:43 thấy thêm 6 phiên COMPLETE. Audit R50 PASS cho BERT A4 seed 42/43, CNN-BiGRU A2 seed 43,
  mixed case2 A2 seed 43, mixed case3 A2 seed 41 và mixed case3 A4 seed 42. Đã ghi đủ 10 metric vào
  `experiments/ablation/<scenario>/results.csv`. Receipt nằm trong `watch-collect.json` và `ablation-audit.json` từng phiên.
- `trietbackup/mixmh-case1-s41/1` và `tamnp2807/mixmh-abl-a1-case1-s41/1` đã PASS R50.
- Audit 25/11 có 8 kernel train PASS R50 và 2 kernel FLOP PASS với 9 trace trong `support-audit-20261125.json`.
  Sau đó `khunglongtrex/exp-count-mh-main2/1` PASS lúc 09:55, bổ sung 2 trace mixed case1/3 MobiHet. Đã kiểm SHA-256 với
  `runs/compute-detailed-v3/kaggle/exp-count-mh-main2/count-audit.json`.
- Đã gỡ 8 account `UNVERIFIED` khỏi config, roster và credential local sau khi xác minh được 26/34 account.
  Giữ biên nhận audit và provenance, gồm 11 endpoint đã audit. Quyết định và exact ref ở `account-retirement-20261125.json`.
- Bảy rerun dùng notebook khớp SHA cũ, sáu cấu hình train từ round 1 và một cấu hình đếm FLOP.
  Năm version 1 đã được nhận. Exact ref cũ và mới nằm trong `rerun-wave-20261125.json`.
  Hai lượt còn chờ gồm `mixmh-abl-a4-case3-s43` chờ slot `tamnp2807` và `mixmh-abl-a1-case3-s42` chờ đủ 7 giờ quota
  `shippertocdo` (2,60 giờ lúc 10:02). Rerun A1 case1 seed 41 thuộc đợt riêng của 24/11, Kaggle nhận lại lúc 09:01 ngày 25/11.

## Backup checkpoint

[Tra checkpoint theo run](../operations/checkpoint-runs.csv) · [Danh sách file](../operations/checkpoint-files.csv) ·
[Phạm vi và khôi phục](../operations/backup.md#checkpoint-mobihet--25112026).

## Lịch sử từng round và epoch

`runs/round-metrics/` trỏ tới `ORICO/experiment-data/experiment/round-metrics/`. Mỗi run có `metrics.csv` và `run.json`
trong `<scenario>/<method>/s<seed>/`. `index.json` tra 342 run và 40.169 dòng bước/view. CANPerFL trỏ về 50 round FedAvg
cùng seed trước 10 epoch fine-tune. Sau audit endpoint BERT, chạy `PYTHONPATH=. python runs/round-metrics/audit/audit.py`
bằng env `nckh`, rồi `verify.py` cùng thư mục. [Hồ sơ backup](../operations/backup.md)

Việc tay còn mở từ phiên trước gồm quyết định của owner về trailer `Co-Authored-By` ở `803a84f`, `288628e`.
Snapshot bàn giao giữ hồ sơ khôi phục artifact, Git, test và kết quả lịch sử.
