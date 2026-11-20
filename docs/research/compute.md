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
  **`group_rows` 4.096 cho mọi phương pháp/kịch bản** (owner 20/11, cả mixed, dù production mixed chạy 8.192). Engine grouped đệm client theo nhóm ⇒ số dòng executed > dòng thật;
  cùng engine cho mọi phương pháp của một kịch bản nên tỷ lệ × FedAvg so sánh được.
- **Tổng FL** = bootstrap + 50 × (train + ngoài train). CANPerFL = tổng FedAvg + 10 epoch fine-tune (đếm bằng state FedAvg 1 round gắn nhãn
  r50; FLOP/epoch không phụ thuộc trọng số).
- Job OOM trên GPU local 4 GB (FeatureBERT, mixed, DAGSNet NILM, nửa CNN-BiGRU) đếm trên Kaggle T4 cùng harness ở 4.096
  (`scripts/compute/build_kaggle_count.py`, clone `5b9d0e3`): `odixeuit/exp-count-bert-g4096`, `nguyetquedo/exp-count-mixed-g4096`,
  `bestyorn/exp-count-cnnbigru-a`, `accanniez/exp-count-cnnbigru-b` (pull 20/11 06:20, bảng dưới sinh lại cùng ngày).
- Panel 200 đệm nặng hơn full-scale (FedAvg DAGSNet 3,77 dòng executed/dòng thật so với 1,087 full-scale) ⇒ dùng **tỷ lệ**, không nhân
  tuyệt đối lên 5.343 xe.

Số liệu từng phương pháp: `experiments/<kịch bản>/compute.csv` (Tops = 1e12 op mỗi pha, tổng, × FedAvg, dòng thật/executed). Nguồn
JSON: `../mobihet/runs/total-compute-20261119/` (DAGSNet, FeatureBERT) và `runs/compute-20261120/` (còn lại; gitignored).
Sinh lại: `python scripts/compute/report.py`.

- MobiHet L4 (20/11): DAGSNet/BERT = `../mobihet` `dags-mh-l4` / `bert-mh-l4` (cùng harness; BERT đếm 2048, quy về executed của
  time run như FedAvg BERT); TabM/SAE/MLP/mixed case2 đếm local từ worktree `../experiment-l4` (`run_queue.py --only …mobihet_l4`);
  CNN-BiGRU + mixed case1/3: Kaggle `goilaaiudi/exp-count-l4-g4096` (source nhúng). Model nhỏ đắt hơn FedAvg vì chi phí synthetic cố
  định (LESSONS 20/11); mixed panel 200 đệm L4 gấp đôi FedProto (độ rộng compact = max nhóm grouped), chưa đo full-scale.
  Sinh lại: `python scripts/compute/report.py <abs ../mobihet/runs/total-compute-20261119> <abs runs/compute-20261120> <abs runs/compute-20261120/kaggle/*/compute>`.

## Bảng tổng

<!-- compute:begin (scripts/compute/report.py) -->

Tổng FL 50 round, đơn vị: **× FedAvg cùng kịch bản** (đơn model); mixed: **Tops = 1e12 op** (không có FedAvg). Panel 200 xe.

| Phương pháp | dagsnet | bert | tabm | sae | mlp | cnn-bigru | mixed-case1 | mixed-case2 | mixed-case3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | N/A | N/A | N/A |
| FedProx | 1.001 | 1.000 | 1.003 | 1.013 | 1.009 | 1.000 | N/A | N/A | N/A |
| AFPHA_EDL_CMSO | 1.001 | 1.000 | 1.003 | 1.013 | 1.009 | 1.000 | N/A | N/A | N/A |
| FD-IDS | 1.325 | 1.319 | 1.335 | 1.380 | 1.394 | 1.321 | N/A | N/A | N/A |
| APFed_LDwCBN | 1.993 | 2.000 | 1.981 | 1.919 | 1.948 | 1.999 | N/A | N/A | N/A |
| LwFedNIDS | 0.107 | 0.154 | 0.278 | 0.512 | 0.244 | 0.108 | N/A | N/A | N/A |
| V2X-HtFL | 1.139 | 1.102 | 1.206 | 3.853 | 1.394 | 1.107 | 11653.0 | 187.8 | 7243.0 |
| CANPerFL | 1.200 | 1.200 | 1.200 | 1.200 | 1.200 | 1.200 | N/A | N/A | N/A |
| TinyProto | 1.102 | 1.101 | 1.108 | 1.144 | 1.130 | 1.101 | 11596.5 | 100.3 | 7157.9 |
| pFedES | 2.638 | 2.751 | 2.892 | 3.642 | 3.383 | 2.675 | 28754.3 | 115.3 | 18042.4 |
| NILM-FL | 2.000 | 2.000 | 2.010 | 2.008 | 2.005 | 2.000 | N/A | N/A | N/A |
| FedTGP | 1.103 | 1.101 | 1.110 | 1.178 | 1.134 | 1.101 | 11596.9 | 100.7 | 7158.3 |
| PerFed-SKD | 1.426 | 1.419 | 1.433 | 1.480 | 1.505 | 1.422 | N/A | N/A | N/A |
| FedProto | 1.000 | 1.000 | 1.006 | 1.033 | 1.010 | 1.000 | 10831.8 | 96.4 | 6818.6 |
| MobiHet L4 | 0.920 | 0.917 | 1.310 | 2.445 | 1.952 |  |  | 212.2 |  |
| *FedAvg tổng (Tops)* | 3944.8 | 11785.5 | 29.4 | 7.5 | 11.1 | 7219.4 |  |  |  |

<!-- compute:end -->
