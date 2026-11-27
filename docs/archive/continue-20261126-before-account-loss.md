# Bàn giao — experiment, 26/11/2026 15:29 (+07)

Repo `main` tại `7375539`. Backup checkpoint trên DVC OneDrive đã xong.
[CONTEXT](../../CONTEXT.md) · [Snapshot trước khi dừng](../archive/continue-20261125-before-shutdown.md)

## Tiếp tục thí nghiệm

1. ORICO và OneDrive đã mount. Ba job dưới `codex-long-jobs` đang chạy: `experiment-ablation-collect-20261126-afternoon`,
   `experiment-ablation-dispatch-20261126-afternoon` và `experiment-ablation-handoff-20261126-afternoon`. Collector kiểm exact ref
   mỗi 20 phút. Dispatcher kiểm slot và quota trước mỗi push. Waiter báo audit fail hoặc checkpoint cần nối.
   [Trạng thái audit](../../runs/ablation-mh/watch-collect.json) · [Biên nhận push](../../runs/ablation-mh/push-ledger.jsonl) ·
   [Log dispatcher](../../runs/ablation-mh/slot-queue.log) · [Log collector](../../runs/ablation-mh/watch-collect.log)
2. **Ablation 66/81** ô đã audit R50. Ba lượt mới PASS đủ 10 metric từ CM mọi round:
   `shippertocdo/mixmh-abl-a2-case2-s42/1` (macro-F1 0,853202),
   `shippertocdo/tabm-mhabl-a2-s42/1` (0,830667) và `nocknnock/sae-mhabl-a4-s41/1` (0,728566).
   Pull ban đầu lỗi do ORICO chưa cắm. Audit lại trên SSD rồi chép sang ORICO, đã so SHA-256 toàn bộ file.
   [Kết quả](../../experiments/ablation/README.md)
3. Kaggle đã nhận `nocknnock/mlp-mhabl-a4-s42/1` và `tamnp2807/mlp-mhabl-a4-s43/1`, với quota trước push 1,45 h
   và 1,60 h, ước tính 1,15 h mỗi run. Hai MLP giữ nguyên byte notebook. Kaggle đã nhận `shippertocdo/cnnbigru-mhabl-a1-s42/1`
   (quota 2,93 h), trần phiên đầu 2,7 h, giữ lịch cosine 50 round. Cả ba phiên mới QUEUED lúc kiểm 15:29. Seed 41 đã đo 4,29 h train + eval, nên S42 dự kiến cần nối.
   Khi parent COMPLETE, audit exact ref, pull và kiểm state của cả hai rank, giữ frozen source, chờ đủ 15 phút và kiểm quota/slot
   trước khi build/push h2. CNN-BiGRU A2 S42 đang RUNNING, log live đã tới R45. Sáu notebook dài còn chờ dispatcher. `cacaosua` và `preyta` còn 30 h nhưng mỗi account đã
   có hai phiên QUEUED. Mốc reset API báo 28/11.
   [Quota](../../runs/ablation-mh/quota-live-user-20261126.json) · [Status exact version](../../runs/ablation-mh/user-wave-status-20261126.json)
4. MobiHet đối chứng **27/27**. Centralized BERT đã audit P4 E27/E24/E25, còn thiếu ba endpoint.
   P5 chờ quyết định riêng. [Ma trận chính](../../experiments/README.md) · [MobiHet](../research/mobihet.md)

## Backup DVC

- Ngày 26/11, BERT đã gộp vào remote OneDrive của experiment, đã fetch thử từ thư mục bert bằng cấu hình repo cha. Biên nhận kiểm kê và dọn file tạm tại runs/onedrive-dvc-merge-20261126/.
- Đã dọn cache BERT cũ và 65 file staging trùng trên ORICO, thu hồi 28,5 GiB. 66 file staging còn lại đã kiểm và xóa ngày 27/11, BERT trên ORICO chuyển sang `experiment-data/experiment/bert/`. [Hồ sơ backup](../operations/backup.md#dọn-cache-bert-và-bản-staging-trùng-trên-orico--26112026)
- Batch MobiHet đã kiểm trên OneDrive đủ **52/52 phiên, 1.854 checkpoint**, tổng 157.189.865.329 byte gồm metadata.
  `cloud-audit-final.json` xác nhận MD5 toàn bộ manifest, danh mục exact ref và kích thước mọi object khớp.
  Năm object 0 byte đã phục hồi và so checksum. Hai CSV tra cứu đã cập nhật.
- Phần local ngoài batch đã kiểm 10.830 đường dẫn. 8.886 khớp DVC BERT theo MD5 và byte.
  Đã backup 1.944 đường dẫn còn lại, 960 object riêng biệt (83,4 GiB). `rclone check` báo 960 khớp, 0 sai lệch.
  Hai job `experiment-checkpoint-dvc-audit-20261126` và `experiment-checkpoint-dvc-all-local-20261126` đã thành công.
  Biên nhận `local-supplement-receipt.json` dưới `runs/ablation-mh/checkpoint-backup-20261125/`.
  [DVC pointer bổ sung](../../artifacts/checkpoints-local-2026-11-26.dvc)
- [Checkpoint theo run](../operations/checkpoint-runs.csv) · [Danh sách file](../operations/checkpoint-files.csv) · [Sổ backup](../operations/backup.md)

## Lịch sử từng round và epoch

`runs/round-metrics/` trỏ tới `ORICO/experiment-data/experiment/round-metrics/`. Mỗi run có `metrics.csv` và `run.json`
trong `<scenario>/<method>/s<seed>/`. `index.json` tra 342 run và 40.169 dòng bước/view. CANPerFL trỏ về 50 round FedAvg
cùng seed trước 10 epoch fine-tune. Sau audit endpoint BERT, chạy `PYTHONPATH=. python runs/round-metrics/audit/audit.py`
bằng env `nckh`, rồi `verify.py` cùng thư mục. [Hồ sơ backup](../operations/backup.md)

Việc tay còn mở từ phiên trước gồm quyết định của owner về trailer `Co-Authored-By` ở `803a84f`, `288628e`.
Snapshot bàn giao giữ hồ sơ khôi phục artifact, Git, test và kết quả lịch sử.
