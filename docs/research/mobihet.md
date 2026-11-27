# MobiHet — phương pháp đề xuất (L4 bỏ logit adjustment, owner 24/11/2026)

MobiHet = cấu hình **L4** của FB-L ([../mobihet FB-L](../../../mobihet/docs/research/screening-fbl.md)) **không có logit adjustment**
(`la_mode=off`, trước đây là ablation A3). Owner 24/11: nhận A3 làm phương pháp chính, giữ tên `mobihet`. Lý do: bỏ LA dương ở
24/24 cặp seed R50 trên 9 kịch bản, và [screening LA](../../../mobihet/docs/research/screening-la.md) không tìm được biến thể LA
nào tốt hơn. L4 có LA cùng mọi số liệu của nó đã chuyển sang `../mobihet/archive/l4-la-20261124/`.

Mỗi xe giữ model riêng, chỉ trao thống kê bootstrap (copula/KM) một lần và gói Q lớp mỗi round, không trao tham số. Giữ nguyên
ngân sách dòng thật theo tầng cỡ xe 64/37/37, synthetic 64,64/16,16/16,16, chọn dòng thật theo CE cũ (`real_select=top`) và
engine `compact=true`. Loss dòng thật là CE thường `CE(z, y)`.

## Code

- hp: `scripts/build_l4_fl.py::MOBIHET_HP` = `L4_HP` (44 khóa, = `../mobihet/runs/fbl-r50/batch.json`) + `la_mode=off`.
  `methods/mobihet_budget.py` mặc định vẫn `la_mode=full` để code lịch sử tái lập được. Mọi recipe MobiHet ghi rõ `la_mode=off`.
- Đơn model: `methods/mobihet_l4.py` (TabM/SAE). Mixed: `methods/mobihet_mixed_l4.py`, `scripts/run_mixed_l4.py` (dùng `MOBIHET_HP`).
  FeatureBERT: source L4 đóng băng + `methods/mobihet_budget.py` của `../mobihet` (thêm đúng khối `la_mode`, pin SHA-256 trong
  `scripts/build_bert_l4_ablation.py`).
- Notebook: `scripts/build_l4_fl.py --ablation none`, `scripts/build_mixed_l4.py --ablation none`,
  `scripts/build_bert_l4_ablation.py --ablation none`. Thư mục run `<backbone>-mh-s<seed>`, `mixmh-<case>-s<seed>`, `bert-mh-s<seed>`.
- Tương đương: `tests/test_mobihet_ablation.py::test_mobihet_is_l4_without_la` và test BERT tương ứng. MobiHet (`la_mode=off`)
  bằng bit-exact L4 có `la_shift` đặt 0 bằng tay (đúng code A3 cũ) qua 2 round: state, Q, CE dòng, CM, byte, bootstrap.

## Kết quả R50 (`experiments/<kịch bản>/results.csv`, method `mobihet`)

24 hàng là run A3 R50 đã audit. `scripts/promote_mobihet_rows.py` kiểm lại từng run trước khi ghi: receipt audit PASS r50,
10 metric tính lại từ CM r50 khớp bảng, byte tính lại từ `config.json` + `rounds.jsonl` khớp bảng. Receipt
`runs/mobihet-promote/receipt.json`. Batch 24/11 bổ sung BERT seed 43 và CNN-BiGRU seed 43. Mixed case1 seed 41 đã audit R50, macro-F1 0,895531 ([bàn giao](../operations/continue.md)).

| kịch bản | seed | Macro-F1 mean ± SD | Δ vs L4 cùng seed (số seed dương) | byte/xe (KB) | hạng byte |
|---|---|---:|---:|---:|---:|
| dagsnet | 3/3 | 0.90572 ± 0.00016 | +0.00021 (3/3) | 354.3 | 3/15 |
| bert | 3/3 | 0.91281 ± 0.00046 | +0.00021 (3/3) | 358.5 | 3/15 |
| tabm | 3/3 | 0.83734 ± 0.00385 | +0.00462 (3/3) | 362.0 | 6/15 |
| sae | 3/3 | 0.72812 ± 0.00707 | +0.00865 (3/3) | 362.8 | 3/15 |
| mlp | 3/3 | 0.87592 ± 0.00107 | +0.00311 (3/3) | 360.7 | 6/15 |
| cnn-bigru | 3/3 | 0.86405 ± 0.00129 | +0.00147 (3/3) | 360.3 | 3/15 |
| mixed-case1 | 3/3 | 0.89520 ± 0.00030 | +0.00023 (2/2) | 357.9 | 3/6 |
| mixed-case2 | 3/3 | 0.85807 ± 0.00070 | +0.00420 (3/3) | 361.4 | 3/6 |
| mixed-case3 | 3/3 | 0.87617 ± 0.00012 | +0.00173 (3/3) | 360.0 | 3/6 |

Bảng đủ 10 metric từng seed: `experiments/<kịch bản>/README.md` (sinh bằng `scripts/experiments_status.py`).
SD là sample SD. Hai seed không phải kiểm định ý nghĩa.

## Chi phí

- **Communication:** đo từ log, không phải traffic mạng. Bootstrap 98,9 KB lên + 241,7 KB xuống mỗi xe một lần (như L4),
  cộng gói Q mỗi round. Tổng 354–363 KB/xe/50 round. Bảng `artifacts/communication-budget/communication_{per_run,summary,ranking}.csv`
  do `artifacts/communication-budget/scripts/mobihet_update.py` sinh lại từ chính các run MobiHet.
- **Compute:** đếm op cùng harness với baseline (panel 200, group_rows 4.096, r1 + 49 × r2). Bỏ LA không đổi shape tensor nào,
  vì phép cộng shift vẫn chạy với shift 0. Trace `<prefix>-mobihet` chạy lại cho cả 9 kịch bản bằng `scripts/compute/run_queue.py`.
  Số ở [compute](compute.md).
