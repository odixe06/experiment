# Compute — FLOP mọi kịch bản

Owner 19/11/2026 (repo `../mobihet`, [harness gốc](../../../mobihet/docs/research/compute-harness.md)): đo compute **cùng một cách**
cho mọi phương pháp. Owner 20/11: **chỉ dùng FLOP** (số op), bỏ GPU-time vì phụ thuộc phần cứng; job không vừa GPU local (4 GB)
thì đếm trên Kaggle T4 (cùng counter ⇒ cùng số op).

## Cách đo

- Harness `scripts/compute/total_compute.py count OUT.json --repo . --panel 200 -- <script> ...` (chép từ `../mobihet/scripts/` 20/11,
  thêm wrap `methods.mixed`). Mỗi aten op dispatch trong một pha được đếm một lần, forward + backward (`scripts/compute/op_counter.py`):
  matmul/conv theo công thức `torch.utils.flop_counter`, op khác = 1 op/phần tử. GRU của CNN-BiGRU viết bằng `baddbmm` nên được đếm như matmul.
- Pha: **bootstrap** = constructor phương pháp; **train** = mọi `local_epoch` của round (forward/backward/optimizer của client);
  **ngoài train** = phần còn lại của round (server, gộp, codec, prototype); **eval** báo riêng, không cộng.
- Panel 200 xe (fold train 1.476.064 dòng, chọn theo 10 tầng kích thước, seed 7), 2 round eager, round 2 = trạng thái ổn định;
  `group_rows` như production (4.096 đơn model; 8.192 mixed). Engine grouped đệm client theo nhóm ⇒ số dòng executed > dòng thật;
  cùng engine cho mọi phương pháp của một kịch bản nên tỷ lệ × FedAvg so sánh được.
- **Tổng FL** = bootstrap + 50 × (train + ngoài train). CANPerFL = tổng FedAvg + 10 epoch fine-tune (đếm bằng state FedAvg 1 round gắn nhãn
  r50; FLOP/epoch không phụ thuộc trọng số).
- FeatureBERT: count eager ở `group_rows` 2.048 (OOM 4 GB ở 4.096), op train quy đổi về dòng executed của engine 4.096 bằng time run.
- Panel 200 đệm nặng hơn full-scale (FedAvg DAGSNet 3,77 dòng executed/dòng thật so với 1,087 full-scale) ⇒ dùng **tỷ lệ**, không nhân
  tuyệt đối lên 5.343 xe.

Số liệu từng phương pháp: `experiments/<kịch bản>/compute.csv` (Tops = 1e12 op mỗi pha, tổng, × FedAvg, dòng thật/executed). Nguồn
JSON: `../mobihet/runs/total-compute-20261119/` (DAGSNet, FeatureBERT) và `runs/compute-20261120/` (còn lại; gitignored).
Sinh lại: `python scripts/compute/report.py`.

## Bảng tổng

<!-- compute:begin (scripts/compute/report.py) -->

Tổng FL 50 round, đơn vị: **× FedAvg cùng kịch bản** (đơn model); mixed: **Tops = 1e12 op** (không có FedAvg). Panel 200 xe.

| Phương pháp | dagsnet | bert |
|---|---:|---:|
| FedAvg | 1.000 | 1.000 |
| FedProx | 1.001 | 1.000 |
| AFPHA_EDL_CMSO | 1.001 | 1.000 |
| FD-IDS | 1.325 | 1.319 |
| APFed_LDwCBN | 1.993 | 2.000 |
| LwFedNIDS | 0.107 | 0.205 |
| V2X-HtFL | 1.139 | 1.098 |
| CANPerFL | 1.200 | 1.200 |
| TinyProto | 1.102 | 1.097 |
| pFedES | 2.638 | 2.751 |
| NILM-FL | 2.000 | 2.000 |
| FedTGP | 1.103 | 1.097 |
| PerFed-SKD | 1.426 | 1.415 |
| FedProto | 1.000 | 1.000 |
| *FedAvg tổng (Tops)* | 3944.8 | 11785.5 |

<!-- compute:end -->
