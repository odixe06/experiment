# Macro-F1 cuối theo model và seed — live

Đủ 10 metric cho 168 cấu hình từ backup Kaggle. Đã tính lại 36 cấu hình TabM từ final confusion matrix. 132 cấu hình còn lại có đủ 10 metric trong marker cuối và qua kiểm seed, phiên bản, completion và smoke. Xem cột trạng thái và ref chính xác trong [TabM](../../../experiments/tabm/results.csv), [SAE](../../../experiments/sae/results.csv), [MLP](../../../experiments/mlp/results.csv) và [CNN-BiGRU](../../../experiments/cnn-bigru/results.csv).

TabM TinyProto dùng aligned-v2 với μ=0,00015. SAE dùng aligned-v2 CE-only/prototype93. TabM và MLP V2X-HtFL dùng fp32proj-a4.

**Cấu hình chung:** 77 feature/bản tin, 16 lớp; 5.343 client theo xe; 14 phương pháp, seed 41/42/43. Train 50 round, 1 epoch local/round, batch 64; AdamW, LR cosine 1e-3 → 1e-5. CANPerFL fine-tune thêm 10 round sau FedAvg50. Số tham số dưới đây là backbone gốc, chưa tính module riêng của phương pháp hoặc pruning.

## TabM

**Model:** 77 → 32 → 32 → 16, ensemble k=4 dùng chung trọng số; **6.548 tham số**, dropout 0,1. Representation 32 chiều; dự đoán bằng trung bình xác suất của 4 thành viên.

| Phương pháp | Seed 41 | Seed 42 | Seed 43 |
|---|---:|---:|---:|
| FedAvg | 0.84264 | 0.84742 | 0.84376 |
| FedProx | 0.83046 | 0.83310 | 0.82590 |
| AFPHA | 0.82684 | 0.82899 | 0.82198 |
| FD-IDS | 0.80482 | 0.80561 | 0.80585 |
| APFed | 0.85593 | 0.86172 | 0.85719 |
| LwFedNIDS | 0.73120 | 0.74075 | 0.71341 |
| CANPerFL | 0.85767 | 0.86297 | 0.86095 |
| TinyProto | 0.79722 | 0.80019 | 0.79742 |
| pFedES | 0.82226 | 0.82528 | 0.81947 |
| NILM-FL | 0.82142 | 0.82318 | 0.81878 |
| FedTGP | 0.75675 | 0.74327 | 0.75346 |
| PerFed-SKD | 0.82992 | 0.83183 | 0.82893 |
| FedProto | 0.71452 | 0.74804 | 0.73581 |
| V2X-HtFL | 0.84733 | 0.84772 | 0.84721 |

## SAE

**Model:** encoder 77 → 32 → 16; nối input77 với code16 → classifier 93 → 16. **7.613 tham số lưu**, trong đó 4.528 ảnh hưởng dự đoán; decoder 3.085 tham số không dùng trong loss FL. Bản FL CE-only, representation 93 chiều, dropout 0.

| Phương pháp | Seed 41 | Seed 42 | Seed 43 |
|---|---:|---:|---:|
| FedAvg | 0.75734 | 0.76008 | 0.75591 |
| FedProx | 0.72478 | 0.72084 | 0.71691 |
| AFPHA | 0.71116 | 0.70110 | 0.69518 |
| FD-IDS | 0.75361 | 0.74925 | 0.75434 |
| APFed | 0.77434 | 0.79206 | 0.77561 |
| LwFedNIDS | 0.70805 | 0.74172 | 0.73713 |
| CANPerFL | 0.79183 | 0.79606 | 0.79139 |
| TinyProto | 0.60258 | 0.58502 | 0.60631 |
| pFedES | 0.70590 | 0.70520 | 0.70679 |
| NILM-FL | 0.71277 | 0.71123 | 0.71078 |
| FedTGP | 0.53451 | 0.55382 | 0.54116 |
| PerFed-SKD | 0.72029 | 0.71865 | 0.72067 |
| FedProto | 0.49925 | 0.50489 | 0.49220 |
| V2X-HtFL | 0.79172 | 0.79229 | 0.79114 |

## MLP

**Model:** 77 → 64 → 32 → 16, ReLU; **7.600 tham số**, dropout 0,1. Representation 32 chiều; nhận trực tiếp vector feature của một bản tin.

| Phương pháp | Seed 41 | Seed 42 | Seed 43 |
|---|---:|---:|---:|
| FedAvg | 0.87526 | 0.88122 | 0.87629 |
| FedProx | 0.86805 | 0.87589 | 0.87069 |
| AFPHA | 0.86496 | 0.87266 | 0.86738 |
| FD-IDS | 0.82049 | 0.83187 | 0.81975 |
| APFed | 0.87869 | 0.88551 | 0.88117 |
| LwFedNIDS | 0.80014 | 0.80204 | 0.80608 |
| CANPerFL | 0.88179 | 0.88673 | 0.88436 |
| TinyProto | 0.84476 | 0.84541 | 0.84682 |
| pFedES | 0.84076 | 0.84217 | 0.84352 |
| NILM-FL | 0.84096 | 0.84709 | 0.84509 |
| FedTGP | 0.80387 | 0.79344 | 0.80704 |
| PerFed-SKD | 0.85015 | 0.85227 | 0.85138 |
| FedProto | 0.84544 | 0.84672 | 0.84928 |
| V2X-HtFL | 0.84634 | 0.84664 | 0.84695 |

## CNN-BiGRU-Attention

**Model:** input 77×1 → Conv64 + BN + MaxPool → Conv128 → BiGRU64 mỗi chiều → Attention128 → 16 lớp. **118.288 tham số**, representation 128 chiều, dropout 0,3. Chuỗi chạy trên 77 feature của một bản tin, không phải cửa sổ nhiều bản tin.

| Phương pháp | Seed 41 | Seed 42 | Seed 43 |
|---|---:|---:|---:|
| FedAvg | 0.88175 | 0.88271 | 0.86853 |
| FedProx | 0.85047 | 0.83744 | 0.83903 |
| AFPHA | 0.83747 | 0.82222 | 0.82928 |
| FD-IDS | 0.82556 | 0.84579 | 0.82847 |
| APFed | 0.88658 | 0.88483 | 0.87370 |
| LwFedNIDS | 0.74788 | 0.73904 | 0.75716 |
| CANPerFL | 0.88294 | 0.88235 | 0.87476 |
| TinyProto | 0.79906 | 0.80083 | 0.79847 |
| pFedES | 0.73922 | 0.74134 | 0.75814 |
| NILM-FL | 0.82624 | 0.82469 | 0.82286 |
| FedTGP | 0.73200 | 0.73011 | 0.73545 |
| PerFed-SKD | 0.80377 | 0.80460 | 0.80327 |
| FedProto | 0.75186 | 0.74912 | 0.75506 |
| V2X-HtFL | 0.79896 | 0.80091 | 0.79914 |
