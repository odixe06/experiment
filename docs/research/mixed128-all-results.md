# Mixed-model — kết quả 3 case

Đã audit round 50: **45/45** cấu hình.

Macro-F1 round 50; mọi số đã kiểm tra confusion matrix: 5.343 xe, 9.347.679 dòng test. Giữ dữ liệu từng xe và mapping model cố định; seed 41/42/43. Ô trống = chưa có kết quả round 50.

Case 1: 30% DAGSNet / 30% BERT / 40% CNN-BiGRU-Attention. Case 2: 30% TabM / 30% SAE / 40% MLP. Case 3: mỗi model DAGSNet/BERT/CNN-BiGRU-Attention/TabM/SAE 20%.

FedProto/FedTGP/TinyProto: client_proto; V2X-HtFL/pFedES: client. Projection 128D cho prototype; pFedES giữ head native.

## case1

| Phương pháp | Seed 41 | Seed 42 | Seed 43 |
|---|---:|---:|---:|
| fedproto | 0.84434 | 0.84180 | 0.84298 |
| fedtgp | 0.83459 | 0.83296 | 0.83413 |
| tinyproto | 0.84943 | 0.84849 | 0.84808 |
| v2x_htfl | 0.84538 | 0.84564 | 0.84411 |
| pfedes | 0.81583 | 0.81513 | 0.82276 |

## case2

| Phương pháp | Seed 41 | Seed 42 | Seed 43 |
|---|---:|---:|---:|
| fedproto | 0.75111 | 0.76196 | 0.75632 |
| fedtgp | 0.65899 | 0.63780 | 0.65815 |
| tinyproto | 0.78550 | 0.79992 | 0.79041 |
| v2x_htfl | 0.83862 | 0.83839 | 0.83917 |
| pfedes | 0.79253 | 0.79390 | 0.79630 |

## case3

| Phương pháp | Seed 41 | Seed 42 | Seed 43 |
|---|---:|---:|---:|
| fedproto | 0.78312 | 0.78986 | 0.79201 |
| fedtgp | 0.71413 | 0.70963 | 0.71734 |
| tinyproto | 0.80497 | 0.81310 | 0.80725 |
| v2x_htfl | 0.84148 | 0.84053 | 0.84073 |
| pfedes | 0.80720 | 0.80690 | 0.80866 |

## Chưa đủ 50 round

| Case | Phương pháp | Seed | Round đã lưu | Trạng thái |
|---|---|---:|---:|---|

CSV kèm đủ 10 metric, từng seed/model/view. Kết quả dở dang không dùng thay round 50.
