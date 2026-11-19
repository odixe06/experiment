> Snapshot16/11/2026; trạng thái sống tại docs/operations/continue.md của repo.

# CONTEXT: fl_veremi_bert / 14 baseline FL với client FeatureBERT trên VeReMi NextGen v4 (5.343 xe)

Đọc theo thứ tự: CONTEXT → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) → [bàn giao sống](docs/operations/continue.md).

## Bài toán

IDS V2X 16 lớp, học cộng tác, **mỗi xe nhận là một client** (5.343 client, dataset v4).
Repo này chạy lại các phương pháp của [`fl_dagsnet_veremi_v2`](https://github.com/odixe06/fl_dagsnet_veremi_v2) với **cấu hình hoàn toàn giống v2**.
Khác biệt duy nhất: model client đổi từ DAGSNet sang **FeatureBERT** (77.904 tham số), lấy từ repo
[`dangvanhoang1711/veremi-bert-77`](https://github.com/dangvanhoang1711/veremi-bert-77).
Code `core/` và `methods/` chép từ v2. Chỉ phần phụ thuộc kiến trúc được sửa ([methods](docs/research/methods.md)).

## Quyết định owner (13/11/2026)

- Cấu hình giống v2: 50 round, mọi client mỗi round, 1 epoch local trên fold train + val, batch 64, AdamW cosine 1e-3 → 1e-5,
  clip 1, AMP, gộp theo n_k. Đánh giá giữ như v2 (global trên toàn test; client trên test của chính xe; 10 metric). Chi tiết: [protocol](docs/research/protocol.md).
- **14 phương pháp FL**: các baseline của v2 **bỏ SCAFFOLD**. **Owner 15/11 bổ sung T3 (nền CBB), 3 seed41/42/43**, cùng code/cấu hình T3 DAGSNet; chỉ đổi model sang FeatureBERT. **Bỏ centralized.**
- **3 seed: 42, 41, 43** (seed 41/43 thêm 14/11; `FLB_SEED=<seed>` → `runs/batch-s<seed>/`, slug hậu tố `-s<seed>`).
- LwFedNIDS tỉa 70 % theo các nhóm d / head-dim / FFN của BERT ([knowledge/bert](knowledge/bert.md) §5).
- Kaggle: đẩy trên mọi account còn quota + slot trống (1 slot trống là đủ); **chừa account mobihet B21** (`EXCLUDE` trong
  `scripts/run_batch.py`). Phần còn thiếu chờ trong `deferred`, watcher tự đẩy khi có slot.
- Repo GitHub **public** `odixe06/fl_veremi_bert`. Không đẩy data, PDF bài báo, credential. Env conda `nckh`. W&B project `fl-veremi-bert`.
- Skill và AGENTS chép từ `../mobihet`; cấu trúc tài liệu giống mobihet / v2.

## Trạng thái

Kiểm 14/11/2026 01:50 VN: cả 13 phiên (APFed, NILM v1 và 11 phiên p2) đều COMPLETE (budget-stop sạch sẽ).
Đã pull và audit 13/13 PASS_BOUNDED; rounds đạt r14–r37 (chưa đủ 50 round). Provenance SHA cb193e6 verified 100%.
CANPerFL tiếp tục chờ FedAvg r50. Quota các account đã cạn; chờ reset thứ Bảy 00:00 UTC (07:00 VN) để nối p3.
Status chính xác: `runs/batch/status.json`. Bảng 10 metric: `runs/batch/report.md`.
Xem [bàn giao sống](docs/operations/continue.md). Kết quả: [results](docs/research/results.md).

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Đang làm gì, job nào chạy, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Cấu hình chung, cách chấm | [protocol](docs/research/protocol.md) |
| Danh sách phương pháp, chỗ đổi cho BERT | [methods](docs/research/methods.md) |
| Kết quả 50 round | [results](docs/research/results.md) |
| Model FeatureBERT, engine grouped, bản tỉa | [knowledge/bert](knowledge/bert.md) |
| Dataset, cột, số liệu | [knowledge/dataset](knowledge/dataset.md) |
| Máy local, Kaggle, lỗi kernel | [local-env](knowledge/local-env.md), [kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md) |
| Bản DAGSNet (kết quả 3 seed để so) | `../fl_dagsnet_veremi_v2/docs/research/results.md` |

## Bổ sung 15/11 — T3 và collect baseline

T3: `scripts/run_t3.py` giữ driver/resume/RNG của mobihet, dùng `core/local_t3.py` (floor16, hỗ trợ buffer rỗng) và engine FeatureBERT hiện có. Baseline giữ driver cũ. Hp T3 chép từ `../mobihet/runs/t3-s41/batch.json`, chỉ thay seed. Không suy FLOPs DAGSNet sang BERT.
Build/launch riêng: `python scripts/run_t3_batch.py build|launch`; plan `runs/t3/batch.json`, source overlay có hash. Trước launch: smoke/compile/round/resume phải PASS. Status job tại [continue](docs/operations/continue.md).

T3 đã push v1 cả3seed15/11: minhtran0601/flb-t3-s41, annhtnt/flb-t3-s42, trietbackup/flb-t3-s43. Smoke100xe + resume2→3 auditPASS; chưa r50/T4 artifact audit. Baseline42 đã pull47session, seed41 đã push20session/pull10, seed43 đã push13session (counts session không là số method hoàn tất). Watcher3seed collect/nối/audit đang chạy; chi tiết tại bàn giao.

Kiểm 2026-11-15T21:21:01.161052+07:00: T3 BERT41p3/DAGSNet41 có local r50; BERT43 local r32. Exact kernels COMPLETE; job `t3-collect-all-20261115-evening` pull đầy đủ/audit hai rank trước nối. Watcher cũ đã chết; trạng thái hiện hành tại bàn giao. Script BERT đã tách helper/HP tham chiếu khỏi repo mobihet; xem [scripts](scripts/README.md).

Kiểm 2026-11-15T21:58:19.503742+07:00: T3 BERT41 r50 audit CM+state hai rank PASS, macro .914839434; BERT42 r35/43 r32 đã audit và push p2 nối tới50 (annhtnt/trietbackup, v1). Controller `flb-t3-resume-42-43-20261115` watch1200s. DAGSNet T3 cả3seed r50 audit xong. [10metric BERT](runs/t3/report.md); job/receipt tại bàn giao.

2026-11-16T00:31:37.017251+07:00: owner cancel T3 BERT42/43 p2 queue, đang retry p3 giữ parent/source/HP. pFedES41p3 exact COMPLETE; job `flb-t3-retry-p3-pfedes41-20261116` push retry + pull/audit pFedES41 + watch1200s. Receipts/trạng thái tại bàn giao, không launch trùng.
