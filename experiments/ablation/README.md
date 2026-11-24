# Ablation MobiHet

Ba ablation độc lập từ MobiHet đầy đủ (= L4 bỏ logit adjustment, owner 24/11/2026). Thiết kế, code và kiểm chứng:
[mobihet-ablation](../../docs/research/mobihet-ablation.md). Ablation cũ từ L4 có LA (A1–A4, 21/11) nằm ở
`../mobihet/archive/l4-la-20261124/experiment/ablation-l4/`.

| arm | bỏ |
|---|---|
| `mobihet` | không bỏ gì (đối chứng = hàng `mobihet` của `experiments/<kịch bản>/results.csv`) |
| a1 | synthetic data + tier budget (kéo theo copula/KM, class-Q, chọn dòng theo CE); không gửi gì |
| a2 | chọn dòng thật theo CE (`real_select=random`) |
| a4 | nén Q (raw fp32 dense, 1.024 B mỗi chiều) |

Mỗi thư mục con là một kịch bản (`dagsnet`, `bert`, `tabm`, `sae`, `mlp`, `cnn-bigru`, `mixed-case1..3`):

- `results.csv`: arm, seed, view, round 50, 10 metric, client F1-macro mean/p10, F1 từng lớp (JSON), status, exact ref.
  Ghi bằng `scripts/audit_ablation.py --record` sau khi audit đủ 50 CM. Hàng `mobihet` ban đầu là run A3 cũ đã kiểm lại
  (`scripts/promote_mobihet_rows.py`).
- `cost.csv`: byte mỗi xe (bootstrap lên/xuống, lên/xuống cộng 50 round, tổng) và tổng mạng (× 5.343), theo từng seed.
- `compute.csv`: Tops client/server/bootstrap/khác/tổng (không gồm eval), panel 200, 50 round ngoại suy, `x_mobihet` =
  tổng/MobiHet. Sinh bằng `python scripts/compute/report_ablation.py`.

So chênh lệch arm − `mobihet` theo cùng seed. Lịch sử từng round, CM và manifest nằm trong bản pull
`runs/ablation-mh/pulls/<owner>/<slug>/` (ORICO; hàng `mobihet` cũ: `runs/ablation-l4/pulls/`).
