# Compute — FLOP mọi kịch bản

Owner 19/11/2026 (repo `../mobihet`, [harness gốc](../../../mobihet/docs/research/compute-harness.md)): đo compute **cùng một cách**
cho mọi phương pháp. Owner 20/11: **chỉ dùng FLOP** (số op), bỏ GPU-time vì phụ thuộc phần cứng; job không vừa GPU local (4 GB)
thì đếm trên Kaggle T4 (cùng counter ⇒ cùng số op).

## Đọc bảng

- `compute.csv` lưu phép đo lịch sử theo quy ước **FLOP công thức + phép toán trên mỗi phần tử**, gồm cả thao tác bộ nhớ. `compute-detail.csv` lưu tổng FLOP công thức, phần số học ước tính, công việc phần tử, attribution, nguồn và trạng thái kiểm chứng.
- Tổng chi tiết dùng FLOP công thức cộng ước tính số học, đơn vị 10¹². Cột `total_element_work_tops` ghi sao chép, index, sắp xếp, RNG và các op không nằm trong nhóm số học có công thức. Đọc `unattributed` để thấy phần chưa có biên client/server rõ. Các nhãn đều gán theo hoạt động đang chạy.
- Phép đo mới ghi ATen dispatch theo pha, scope, thiết bị và tên op. Tích lũy mỗi dispatch vào đúng một scope. Trình lập bảng kiểm tổng trace chi tiết bằng tổng coarse cho từng pha và thiết bị. TorchDispatchMode và scope được đưa vào từng luồng Lockstep. CUDA autograd dùng nhãn được chụp ở lệnh backward.
- Scope con bao gồm forward model, backward, optimizer, teacher, KD, proxy, prototype, server training, model aggregation, global statistics, synthetic data và codec khi có biên riêng trong code. Backward chung của NILM-FL được giữ trong nhóm private/proxy joint. Phép toán chuẩn bị checkpoint, hoạt động NumPy/zlib và network nằm ngoài công thức FLOP. Tensor ops preprocessing đi vào cột riêng. Evaluation báo riêng và được loại khỏi tổng huấn luyện.
- Tổng FL chi tiết = preprocessing + bootstrap + r1 + 49 × r2. Cosine của hai round đếm giữ horizon 50. CANPerFL cộng đường FedAvg 50 round và 10 vòng fine-tune (r1 + 9 × r2), mỗi stage có init riêng. Đây là phép chiếu từ panel, chất lượng phụ thuộc các đường hoạt động về sau.
- Centralized dùng probe thật 9 bước đầy đủ với batch mỗi rank 2.048, micro theo từng model và mỗi bước được áp dụng. Chiếu lên 2 rank và 50 epoch trên cùng panel train. Chi phí bước cuối dùng batch đầy đủ, sai số biên tối đa một global batch/epoch. Eval dựa trên mẫu validation riêng. Phép đo CPU dùng `no_grad` và SDPA math để dispatcher thấy linear, GRU và attention. Communications DDP nằm ngoài bảng.
- FedAvg là chuẩn chia trong từng backbone đơn. Mixed chia theo FedProto cùng case, cùng assignment và panel. `rank` chỉ xếp các hàng có trace mới. `legacy_rank` xếp riêng quy ước lịch sử. Cột legacy minus detailed ghi độ lệch tổng. Cột detailed with element work cho thấy ảnh hưởng của phép toán phần tử lên phép chiếu mới. Chỗ trống nghĩa là thiếu phép đo tương ứng.

| Kịch bản | Bảng chi tiết |
|---|---|
| DAGSNet | [compute-detail.csv](../../experiments/dagsnet/compute-detail.csv) |
| FeatureBERT | [compute-detail.csv](../../experiments/bert/compute-detail.csv) |
| TabM | [compute-detail.csv](../../experiments/tabm/compute-detail.csv) |
| SAE | [compute-detail.csv](../../experiments/sae/compute-detail.csv) |
| MLP | [compute-detail.csv](../../experiments/mlp/compute-detail.csv) |
| CNN-BiGRU | [compute-detail.csv](../../experiments/cnn-bigru/compute-detail.csv) |
| Mixed case1 | [compute-detail.csv](../../experiments/mixed-case1/compute-detail.csv) |
| Mixed case2 | [compute-detail.csv](../../experiments/mixed-case2/compute-detail.csv) |
| Mixed case3 | [compute-detail.csv](../../experiments/mixed-case3/compute-detail.csv) |

## Phép đo lịch sử

- Harness `scripts/compute/total_compute.py count OUT.json --repo . --panel 200 -- <script> ...` (chép từ `../mobihet/scripts/` 20/11,
  thêm wrap `methods.mixed`). Mỗi aten op dispatch trong một pha được đếm một lần, forward + backward (`scripts/compute/op_counter.py`):
  matmul/conv theo công thức `torch.utils.flop_counter`, op khác = 1 op/phần tử. GRU của CNN-BiGRU viết bằng `baddbmm` nên được đếm như matmul.
- Pha: **bootstrap** = constructor phương pháp; **train** = mọi `local_epoch` của round (forward/backward/optimizer của client);
  **ngoài train** = phần còn lại của round (server, gộp, codec, prototype); **eval** báo riêng, không cộng.
- Panel 200 xe (fold train 1.476.064 dòng, chọn theo 10 tầng kích thước, seed 7), 2 round eager, round 2 = trạng thái ổn định;
  **`group_rows` yêu cầu 4.096** (owner 20/11). Mixed áp cap riêng 256 cho nhóm BERT trong cả baseline và L4. Một số trace BERT lịch sử đếm ở 2.048 rồi quy đổi theo số dòng executed. Engine grouped đệm client theo nhóm ⇒ số dòng executed > dòng thật;
  cùng engine cho mọi phương pháp của một kịch bản nên tỷ lệ × FedAvg so sánh được.
- **Tổng FL** = bootstrap + 50 × (train + ngoài train). CANPerFL = tổng FedAvg + 10 epoch fine-tune (đếm bằng state FedAvg 1 round gắn nhãn
  r50; FLOP/epoch không phụ thuộc trọng số).
- Job OOM trên GPU local 4 GB (FeatureBERT, mixed, DAGSNet NILM, nửa CNN-BiGRU) đếm trên Kaggle T4 cùng harness ở 4.096
  (`scripts/compute/build_kaggle_count.py`, clone `5b9d0e3`): `odixeuit/exp-count-bert-g4096`, `nguyetquedo/exp-count-mixed-g4096`,
  `bestyorn/exp-count-cnnbigru-a`, `accanniez/exp-count-cnnbigru-b` (pull 20/11 06:20, bảng dưới sinh lại cùng ngày).
- Panel 200 đệm nặng hơn full-scale (FedAvg DAGSNet 3,77 dòng executed/dòng thật so với 1,087 full-scale) ⇒ dùng **tỷ lệ**, không nhân
  tuyệt đối lên 5.343 xe.

Số liệu từng phương pháp: `experiments/<kịch bản>/compute.csv` (Tops = 1e12 op mỗi pha, tổng, × FedAvg, dòng thật/executed). Nguồn
JSON nằm trong `runs/compute-legacy/20261119/` và `runs/compute-20261120/` (gitignored).
Sinh lại: `python scripts/compute/report.py`.

- MobiHet L4 (20/11): DAGSNet/BERT dùng trace lịch sử `dags-mh-l4` / `bert-mh-l4` (cùng harness; BERT đếm 2048, quy về executed của
  time run như FedAvg BERT); TabM/SAE/MLP/mixed case2 đếm local trong `runs/compute-20261120/` (`run_queue.py --only …mobihet_l4`);
  CNN-BiGRU + mixed case1/3: Kaggle `goilaaiudi/exp-count-l4-g4096` (source nhúng). Model nhỏ đắt hơn FedAvg vì chi phí synthetic cố
  định (LESSONS 20/11); mixed panel 200 đệm L4 gấp đôi FedProto (độ rộng compact = max nhóm grouped), chưa đo full-scale.
  Sinh lại từ repo chính bằng `python scripts/compute/report.py`.

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
| MobiHet L4 | 0.920 | 0.917 | 1.310 | 2.445 | 1.952 | 0.917 | 20353.7 | 212.2 | 12776.1 |
| *FedAvg tổng (Tops)* | 3944.8 | 11785.5 | 29.4 | 7.5 | 11.1 | 7219.4 |  |  |  |

<!-- compute:end -->

## MobiHet L4 và độ nhạy theo backbone

Ước tính lịch sử dùng cùng panel cho từng cặp. CNN-BiGRU L4 đạt 0,917× FedAvg. Mixed case1/case2/case3 lần lượt đạt 1,879×, 2,202× và 1,874× FedProto. BERT L4 lịch sử nhân **phần train** với 5.092.288 / 3.045.792 = 1,672 khi chuyển từ 2.048 lên 4.096. Bootstrap, eval và phần round ngoài train vẫn giữ phép đo ở layout nguồn. Tỷ lệ 0,917× là ước tính theo dòng. Job BERT chi tiết đo trực tiếp 4.096 trên source native đã đóng băng.

MLP L4 trên trace mới khoảng 13,42 TFLOPs chiếu 50 round, gồm khoảng 3,65 TFLOPs ở phạm vi synthetic. FedAvg khoảng 10,52 TFLOPs. Tỷ lệ mới 1,276× thấp hơn 1,952× quy ước cũ do cột chi phí dữ liệu và các op phần tử không số học được tách khỏi tổng FLOP. Số nhân ma trận forward/backward FedAvg ở r2 khớp trace cũ, tổng 195,08 GFLOPs. Chênh lệch quy ước đặc biệt lớn khi lấy mẫu copula/KM chứa indexing, copy, RNG và `searchsorted`.

Mixed case2 L4 đạt khoảng 187,26 TFLOPs, FedProto khoảng 91,82 TFLOPs, tỷ lệ 2,039×. Nhóm grouped chọn chiều rộng compact theo xe có quota lớn nhất còn gộp trong nhóm. Panel 200 có ít xe mỗi backbone. Nhiều bước vẫn chạy chiều rộng 64 thật + 64 synthetic dù mask chỉ kích hoạt ít hơn. Mixed còn nhân chi phí theo từng nhóm và phân phối model. So sánh giới hạn trong panel. Cần đo panel rộng hơn hoặc full-scale để định lượng padding ở 5.343 xe.

Big-O trong CSV là chi phí mỗi round trừ khi biểu thức ghi số vòng. `E` là dòng thực thi kể cả đệm, `F` chi phí backbone mỗi dòng, `U` số bước grouped thực thi tính theo số client, `P` tham số, `K` client, `C` lớp, `H` chiều prototype, `D` số feature, `B` số bin CDF và `S` epoch server. Mixed dùng tổng qua các nhóm backbone. Bootstrap L4 xét riêng, gồm thống kê feature và KM với số sample, chiều feature, số cụm và số vòng Lloyd.

## Tái lập

```bash
export V4_ROOT="$PWD/runs/compute-inputs/v4-cache"
export V4_CACHE="$V4_ROOT"
export MOBIHET_RECOVERY_DIR="$PWD/runs/compute-detailed-v3/recovery"
python scripts/compute/run_queue.py --only mlp-fedavg,mlp-mobihet_l4 --out "$PWD/runs/compute-detailed-v3"
python scripts/compute/count_central.py --model mlp --panel-cache "$V4_CACHE" --out runs/compute-detailed-v3/centralized-mlp-cpu-math --cpu
python scripts/compute/report.py
python scripts/compute/report_detail.py
```

Raw cho trace mới nằm trong `runs/compute-detailed-v3`. File `activity.csv` cung cấp tổng theo scope của các round đo thật. Mỗi hàng mới trong bảng giữ SHA-256 của file JSON nguồn. Notebook count Kaggle đóng gói cùng mã, lưu manifest thực thi và dùng bộ cache panel có hash. Bản chạy local giữ corpus panel và fold counts trong `runs/compute-inputs/v4-cache`.
