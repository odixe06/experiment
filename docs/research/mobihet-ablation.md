# MobiHet — ablation study (owner 24/11/2026)

Ba ablation độc lập, mỗi cái xuất phát thẳng từ MobiHet đầy đủ ([MobiHet](mobihet.md) = L4 bỏ logit adjustment, `MOBIHET_HP`).
Không cộng dồn thay đổi. Protocol, dataset v4, 5.343 xe, 50 round, seed 41/42/43 giữ nguyên. Ablation "bỏ LA" (A3 cũ) không còn,
vì đó chính là MobiHet. Ablation từ L4 có LA (21/11, A1–A4) đã chuyển sang `../mobihet/archive/l4-la-20261124/`.
Kết quả: [experiments/ablation/](../../experiments/ablation/README.md), mỗi kịch bản một thư mục con.

## Component → code

| | component | vị trí |
|---|---|---|
| C1 | Federated Gaussian copula + K-means mixture (bootstrap 1 lần) | `mobihet.py:_copula_bootstrap`, `mobihet_sim.py:_kmeans`, codec km2 `mobihet_lite.py:_kmeans` |
| C2 | Class-level KD (ma trận Q) | `_csum` ở `mobihet_budget.py:_loss_tier`, gộp theo count ở `mobihet.py:_round_private`, codec 8-bit `mobihet_sim.py` |
| C4 | Synthetic data (CE r1, KL(Q[y] ‖ p) từ r2) | `mobihet.py:_synth`, liều theo tầng/round `mobihet_compute.py:_rows_for`, loss `_loss_tier` |
| C5 | Chọn dòng thật theo CE lưu (`real_select=top`) | `mobihet_budget.py:_real_rows`, `_score` |
| C6 | Ngân sách theo tầng + bước synthetic-only (≥ 16) | `real_tier_rows`, `synth_tier_*`, `synth_step_floor` ở `core/local.py` |

## Ba arm

Code: [methods/mobihet_ablation.py](../../methods/mobihet_ablation.py) (mixin; hp `ablation`, bảng `HP`; assert `la_mode=off`),
mixed [methods/mobihet_ablation_mixed.py](../../methods/mobihet_ablation_mixed.py) + `scripts/run_mixed_ablation.py`.
`ablation=none` = MobiHet nguyên vẹn (đối chứng, cũng là run MobiHet chính).

| arm | bỏ | thay đổi | byte |
|---|---|---|---|
| A1 | synthetic (C4 + C6) | CE(z) thường trên **mọi** dòng thật, batch 64, `synth_step_floor=0`. Không dòng synthetic nên không dựng copula/KM, không Q, không CE dòng. Không LA nên không cần count toàn cục: chỉ train local | 0 B bootstrap, 0 B/round |
| A2 | C5 | `real_select=random` (chỉ HP). Giữ ngân sách dòng thật/synthetic, mọi loss | như MobiHet |
| A4 | nén Q | Q raw fp32 dense 16 × 16 hai chiều, không uint8/fp16/bitmask/zlib, giữ chuẩn hóa hàng khi giải mã | 1.024 B lên + 1.024 B xuống mỗi xe mỗi round |

## Kiểm chứng local

`tests/test_mobihet_ablation.py` (panel 20 xe, 2 round) và `tests/test_mobihet_ablation_bert.py` (snapshot BERT + budget có `la_mode`):

- MobiHet = L4 với `la_shift` đặt 0 bằng tay (A3 cũ), bit-exact; bootstrap và byte như L4.
- `none` = MobiHet bit-exact trên DAGSNet, TabM, SAE, MLP, CNN-BiGRU, BERT và mixed case2.
- A1: bootstrap 0 B, không bảng generator, Q None, 0 B/round, số bước = Σ⌈n/64⌉, loss và gradient = CE(z).
- A2: không CE dòng, `real_frac`/số bước/byte như MobiHet. A4: round 1 = MobiHet, Q fp32 lệch ≤ 2/255, 1.024 B.
- Resume r1 → r2 = chạy liền cho cả ba arm. Mixed: mọi arm 2 round, health từng backbone.

## Chạy và thu

Builder: `scripts/build_l4_fl.py --ablation <arm>`, `scripts/build_mixed_l4.py --ablation <arm>`,
`scripts/build_bert_l4_ablation.py --ablation <arm>`. Thư mục run `<backbone>-mhabl-<arm>-s<seed>`,
`mixmh-abl-<arm>-<case>-s<seed>`, `bert-mhabl-<arm>-s<seed>`. Batch, quota, hàng đợi push và collector:
`runs/ablation-mh/` ([bàn giao](../operations/continue.md)). Audit `scripts/audit_ablation.py` (hp = MOBIHET_HP + arm, 50 CM
→ 10 metric, hợp đồng byte), ghi `experiments/ablation/<kịch bản>/{results,cost}.csv`; arm `none` ghi thêm hàng `mobihet`
vào `experiments/<kịch bản>/results.csv`. FLOP: `scripts/compute/run_queue.py --ablation` + `scripts/compute/report_ablation.py`
→ `experiments/ablation/<kịch bản>/compute.csv`.

L4 trên 2×T4 không tái lập bit-exact giữa các phiên (atomics CUDA, ~1e-4 macro ở r2). Mọi chênh lệch ablation − MobiHet đọc cùng
dải nhiễu này và SD giữa seed.
