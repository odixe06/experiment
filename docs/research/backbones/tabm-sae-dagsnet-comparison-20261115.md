# DAGSNet – TabM – SAE: 14 baseline FL

Ngày15/11/2026. Lịch cloud giữ tắt; chỉ collect/audit kết quả.

Round50; CANPerFL = FedAvg50 + fine-tune10. Cùng dataset v4,5343client,9347679 dòng test.
DAGSNet: seed42, số từ bảng đã audit trên GitHub main commit68a01b29cb3dd47d221d48a16289a8c9f1dfa155; nguồn đã làm tròn4số lẻ.
TabM/SAE: mean ± sample standard deviation (ddof=1), seeds41/42/43;5số lẻ. Không chọn best round/seed.

## Macro-F1 cuối — tổng hợp3seed cho TabM/SAE

| Phương pháp | View | DAGSNet seed42 | TabM mean ± SD | SAE mean ± SD |
|---|---|---:|---:|---:|
| FedAvg | global | 0.9042 | 0.84461 ± 0.00250 | 0.69881 ± 0.00621 |
| FedProx | global | 0.8998 | 0.82982 ± 0.00364 | 0.66801 ± 0.00081 |
| AFPHA | global | 0.8990 | 0.82593 ± 0.00359 | 0.65854 ± 0.00074 |
| FD-IDS | global | 0.8992 | 0.80542 ± 0.00054 | 0.68303 ± 0.00176 |
| APFed | client | 0.9119 | 0.85828 ± 0.00304 | 0.71195 ± 0.01085 |
| LwFedNIDS | global | 0.8904 | 0.72845 ± 0.01387 | 0.68438 ± 0.00057 |
| CANPerFL | client | 0.9088 | 0.86053 ± 0.00268 | 0.73251 ± 0.00917 |
| TinyProto | client_proto | 0.8534 | 0.29646 ± 0.01197 | 0.35965 ± 0.01957 |
| pFedES | client | 0.8511 | 0.82234 ± 0.00290 | 0.63032 ± 0.00401 |
| NILM-FL | client | 0.8846 | 0.82113 ± 0.00221 | 0.66041 ± 0.00451 |
| FedTGP | client_proto | 0.8719 | 0.75116 ± 0.00703 | 0.08473 ± 0.02322 |
| PerFed-SKD | client | 0.8710 | 0.83023 ± 0.00148 | 0.67771 ± 0.00307 |
| FedProto | client_proto | 0.8698 | 0.73279 ± 0.01696 | 0.18821 ± 0.01825 |
| V2X-HtFL | client | 0.8661 | 0.47895 ± 0.39971 | 0.17401 ± 0.00522 |

## Cùng seed42 — Macro-F1

| Phương pháp | DAGSNet | TabM | SAE |
|---|---:|---:|---:|
| FedAvg | 0.9042 | 0.84742 | 0.69492 |
| FedProx | 0.8998 | 0.83310 | 0.66882 |
| AFPHA | 0.8990 | 0.82899 | 0.65939 |
| FD-IDS | 0.8992 | 0.80561 | 0.68472 |
| APFed | 0.9119 | 0.86172 | 0.70087 |
| LwFedNIDS | 0.8904 | 0.74075 | 0.68374 |
| CANPerFL | 0.9088 | 0.86297 | 0.72719 |
| TinyProto | 0.8534 | 0.28756 | 0.33812 |
| pFedES | 0.8511 | 0.82528 | 0.63056 |
| NILM-FL | 0.8846 | 0.82318 | 0.65847 |
| FedTGP | 0.8719 | 0.74327 | 0.06875 |
| PerFed-SKD | 0.8710 | 0.83183 | 0.67683 |
| FedProto | 0.8698 | 0.74804 | 0.16768 |
| V2X-HtFL | 0.8661 | 0.01951 | 0.17227 |

## Giới hạn so sánh

- DAGSNet hiện dùng1seed theo owner; chưa khôi phục đủ artifacts3seed từ máy cũ. Không suy ý nghĩa thống kê từ bảng này.
- TinyProto: DAGSNet r4 dùng μ=0.00015; bản TabM/SAE thiếu overrideμ và lấy heuristic từ train counts. Đây không phải phép so sánh chỉ đổi backbone.
- Headline của FedProto/FedTGP/TinyProto là client_proto; các phương pháp personalized còn lại là client, không đổi sang view có điểm cao hơn.
- Audit mới chỉ kiểm CM cuối,10metric/per-class/client distribution,config/source và history đủround; chưa replay weights hoặc audit mọi CM từnground.
- Không chạy lại hoặc sửa model/hyperparameter trong lượt báo cáo.

## Đủ10metric cuối — mỗi model/phương pháp

Single-label multiclass: accuracy = precision_micro = recall_micro = recall_weighted = f1_micro.

| Model | Method | Seeds | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DAGSNet | fedavg | 42 | 0.9228 | 0.9354 | 0.9228 | 0.9147 | 0.8957 | 0.9228 | 0.9228 | 0.9042 | 0.9228 | 0.9085 |
| tabm | fedavg | 41,42,43 | 0.90147 ± 0.00109 | 0.88160 ± 0.00193 | 0.90147 ± 0.00109 | 0.86721 ± 0.00031 | 0.83338 ± 0.00276 | 0.90147 ± 0.00109 | 0.90147 ± 0.00109 | 0.84461 ± 0.00250 | 0.90147 ± 0.00109 | 0.87925 ± 0.00091 |
| sae | fedavg | 41,42,43 | 0.84013 ± 0.00331 | 0.84307 ± 0.03703 | 0.84013 ± 0.00331 | 0.83326 ± 0.03166 | 0.69134 ± 0.00625 | 0.84013 ± 0.00331 | 0.84013 ± 0.00331 | 0.69881 ± 0.00621 | 0.84013 ± 0.00331 | 0.81269 ± 0.00348 |
| DAGSNet | fedprox | 42 | 0.9214 | 0.9307 | 0.9214 | 0.9112 | 0.8941 | 0.9214 | 0.9214 | 0.8998 | 0.9214 | 0.9049 |
| tabm | fedprox | 41,42,43 | 0.89699 ± 0.00155 | 0.87683 ± 0.00084 | 0.89699 ± 0.00155 | 0.86314 ± 0.00099 | 0.81953 ± 0.00354 | 0.89699 ± 0.00155 | 0.89699 ± 0.00155 | 0.82982 ± 0.00364 | 0.89699 ± 0.00155 | 0.87434 ± 0.00152 |
| sae | fedprox | 41,42,43 | 0.82682 ± 0.00089 | 0.75977 ± 0.00046 | 0.82682 ± 0.00089 | 0.80154 ± 0.00044 | 0.65809 ± 0.00107 | 0.82682 ± 0.00089 | 0.82682 ± 0.00089 | 0.66801 ± 0.00081 | 0.82682 ± 0.00089 | 0.79611 ± 0.00085 |
| DAGSNet | afpha | 42 | 0.9211 | 0.9300 | 0.9211 | 0.9106 | 0.8932 | 0.9211 | 0.9211 | 0.8990 | 0.9211 | 0.9045 |
| tabm | afpha | 41,42,43 | 0.89553 ± 0.00164 | 0.87553 ± 0.00095 | 0.89553 ± 0.00164 | 0.86186 ± 0.00103 | 0.81536 ± 0.00354 | 0.89553 ± 0.00164 | 0.89553 ± 0.00164 | 0.82593 ± 0.00359 | 0.89553 ± 0.00164 | 0.87282 ± 0.00159 |
| sae | afpha | 41,42,43 | 0.82334 ± 0.00081 | 0.75876 ± 0.00065 | 0.82334 ± 0.00081 | 0.79930 ± 0.00057 | 0.64864 ± 0.00120 | 0.82334 ± 0.00081 | 0.82334 ± 0.00081 | 0.65854 ± 0.00074 | 0.82334 ± 0.00081 | 0.79140 ± 0.00079 |
| DAGSNet | fd_ids | 42 | 0.9220 | 0.9373 | 0.9220 | 0.9155 | 0.8914 | 0.9220 | 0.9220 | 0.8992 | 0.9220 | 0.9054 |
| tabm | fd_ids | 41,42,43 | 0.89560 ± 0.00092 | 0.87177 ± 0.00042 | 0.89560 ± 0.00092 | 0.86090 ± 0.00055 | 0.80388 ± 0.00131 | 0.89560 ± 0.00092 | 0.89560 ± 0.00092 | 0.80542 ± 0.00054 | 0.89560 ± 0.00092 | 0.87123 ± 0.00079 |
| sae | fd_ids | 41,42,43 | 0.83303 ± 0.00149 | 0.80397 ± 0.03551 | 0.83303 ± 0.00149 | 0.81009 ± 0.00422 | 0.67357 ± 0.00210 | 0.83303 ± 0.00149 | 0.83303 ± 0.00149 | 0.68303 ± 0.00176 | 0.83303 ± 0.00149 | 0.80382 ± 0.00154 |
| DAGSNet | apfed | 42 | 0.9278 | 0.9377 | 0.9278 | 0.9217 | 0.9032 | 0.9278 | 0.9278 | 0.9119 | 0.9278 | 0.9169 |
| tabm | apfed | 41,42,43 | 0.90853 ± 0.00103 | 0.88608 ± 0.00128 | 0.90853 ± 0.00103 | 0.87276 ± 0.00075 | 0.84658 ± 0.00336 | 0.90853 ± 0.00103 | 0.90853 ± 0.00103 | 0.85828 ± 0.00304 | 0.90853 ± 0.00103 | 0.88638 ± 0.00102 |
| sae | apfed | 41,42,43 | 0.84873 ± 0.00471 | 0.85623 ± 0.03015 | 0.84873 ± 0.00471 | 0.84725 ± 0.02496 | 0.70430 ± 0.00920 | 0.84873 ± 0.00471 | 0.84873 ± 0.00471 | 0.71195 ± 0.01085 | 0.84873 ± 0.00471 | 0.82172 ± 0.00543 |
| DAGSNet | lwfednids | 42 | 0.9162 | 0.9320 | 0.9162 | 0.9111 | 0.8830 | 0.9162 | 0.9162 | 0.8904 | 0.9162 | 0.8986 |
| tabm | lwfednids | 41,42,43 | 0.86323 ± 0.00527 | 0.83168 ± 0.04840 | 0.86323 ± 0.00527 | 0.83601 ± 0.01019 | 0.71252 ± 0.00965 | 0.86323 ± 0.00527 | 0.86323 ± 0.00527 | 0.72845 ± 0.01387 | 0.86323 ± 0.00527 | 0.83546 ± 0.00714 |
| sae | lwfednids | 41,42,43 | 0.83043 ± 0.00029 | 0.82349 ± 0.00018 | 0.83043 ± 0.00029 | 0.81123 ± 0.00015 | 0.67373 ± 0.00058 | 0.83043 ± 0.00029 | 0.83043 ± 0.00029 | 0.68438 ± 0.00057 | 0.83043 ± 0.00029 | 0.80234 ± 0.00033 |
| DAGSNet | canperfl | 42 | 0.9233 | 0.9281 | 0.9233 | 0.9171 | 0.8986 | 0.9233 | 0.9233 | 0.9088 | 0.9233 | 0.9159 |
| tabm | canperfl | 41,42,43 | 0.90869 ± 0.00088 | 0.91326 ± 0.00220 | 0.90869 ± 0.00088 | 0.89810 ± 0.00261 | 0.85120 ± 0.00256 | 0.90869 ± 0.00088 | 0.90869 ± 0.00088 | 0.86053 ± 0.00268 | 0.90869 ± 0.00088 | 0.88675 ± 0.00096 |
| sae | canperfl | 41,42,43 | 0.85300 ± 0.00393 | 0.84377 ± 0.00075 | 0.85300 ± 0.00393 | 0.83926 ± 0.00171 | 0.71539 ± 0.00891 | 0.85300 ± 0.00393 | 0.85300 ± 0.00393 | 0.73251 ± 0.00917 | 0.85300 ± 0.00393 | 0.82704 ± 0.00455 |
| DAGSNet | tinyproto | 42 | 0.8866 | 0.8702 | 0.8866 | 0.8759 | 0.8457 | 0.8866 | 0.8866 | 0.8534 | 0.8866 | 0.8774 |
| tabm | tinyproto | 41,42,43 | 0.52500 ± 0.00828 | 0.30882 ± 0.01034 | 0.52500 ± 0.00828 | 0.58488 ± 0.00776 | 0.32898 ± 0.01171 | 0.52500 ± 0.00828 | 0.52500 ± 0.00828 | 0.29646 ± 0.01197 | 0.52500 ± 0.00828 | 0.53368 ± 0.00877 |
| sae | tinyproto | 41,42,43 | 0.47014 ± 0.01211 | 0.37310 ± 0.01797 | 0.47014 ± 0.01211 | 0.56946 ± 0.01241 | 0.43824 ± 0.01744 | 0.47014 ± 0.01211 | 0.47014 ± 0.01211 | 0.35965 ± 0.01957 | 0.47014 ± 0.01211 | 0.46586 ± 0.01508 |
| DAGSNet | pfedes | 42 | 0.8864 | 0.8692 | 0.8864 | 0.8800 | 0.8405 | 0.8864 | 0.8864 | 0.8511 | 0.8864 | 0.8803 |
| tabm | pfedes | 41,42,43 | 0.88515 ± 0.00142 | 0.88451 ± 0.00072 | 0.88515 ± 0.00142 | 0.87486 ± 0.00028 | 0.80497 ± 0.00294 | 0.88515 ± 0.00142 | 0.88515 ± 0.00142 | 0.82234 ± 0.00290 | 0.88515 ± 0.00142 | 0.86415 ± 0.00149 |
| sae | pfedes | 41,42,43 | 0.78016 ± 0.00214 | 0.84966 ± 0.00540 | 0.78016 ± 0.00214 | 0.81342 ± 0.00382 | 0.58179 ± 0.00392 | 0.78016 ± 0.00214 | 0.78016 ± 0.00214 | 0.63032 ± 0.00401 | 0.78016 ± 0.00214 | 0.75165 ± 0.00278 |
| DAGSNet | nilm | 42 | 0.9156 | 0.9291 | 0.9156 | 0.9146 | 0.8717 | 0.9156 | 0.9156 | 0.8846 | 0.9156 | 0.9020 |
| tabm | nilm | 41,42,43 | 0.88987 ± 0.00109 | 0.91277 ± 0.00194 | 0.88987 ± 0.00109 | 0.89275 ± 0.00186 | 0.80457 ± 0.00212 | 0.88987 ± 0.00109 | 0.88987 ± 0.00109 | 0.82113 ± 0.00221 | 0.88987 ± 0.00109 | 0.86778 ± 0.00110 |
| sae | nilm | 41,42,43 | 0.79864 ± 0.00322 | 0.84227 ± 0.00268 | 0.79864 ± 0.00322 | 0.81348 ± 0.00151 | 0.61515 ± 0.00555 | 0.79864 ± 0.00322 | 0.79864 ± 0.00322 | 0.66041 ± 0.00451 | 0.79864 ± 0.00322 | 0.77198 ± 0.00328 |
| DAGSNet | fedtgp | 42 | 0.9020 | 0.8924 | 0.9020 | 0.8945 | 0.8610 | 0.9020 | 0.9020 | 0.8719 | 0.9020 | 0.8947 |
| tabm | fedtgp | 41,42,43 | 0.82038 ± 0.01066 | 0.74699 ± 0.00323 | 0.82038 ± 0.01066 | 0.81902 ± 0.00427 | 0.76531 ± 0.00901 | 0.82038 ± 0.01066 | 0.82038 ± 0.01066 | 0.75116 ± 0.00703 | 0.82038 ± 0.01066 | 0.81765 ± 0.00767 |
| sae | fedtgp | 41,42,43 | 0.21885 ± 0.07150 | 0.13663 ± 0.00503 | 0.21885 ± 0.07150 | 0.19439 ± 0.04618 | 0.11394 ± 0.03373 | 0.21885 ± 0.07150 | 0.21885 ± 0.07150 | 0.08473 ± 0.02322 | 0.21885 ± 0.07150 | 0.14982 ± 0.01555 |
| DAGSNet | perfedskd | 42 | 0.8994 | 0.8903 | 0.8994 | 0.8931 | 0.8598 | 0.8994 | 0.8994 | 0.8710 | 0.8994 | 0.8928 |
| tabm | perfedskd | 41,42,43 | 0.89217 ± 0.00122 | 0.89473 ± 0.00141 | 0.89217 ± 0.00122 | 0.88465 ± 0.00124 | 0.81710 ± 0.00154 | 0.89217 ± 0.00122 | 0.89217 ± 0.00122 | 0.83023 ± 0.00148 | 0.89217 ± 0.00122 | 0.87054 ± 0.00123 |
| sae | perfedskd | 41,42,43 | 0.81650 ± 0.00147 | 0.83890 ± 0.00118 | 0.81650 ± 0.00147 | 0.81928 ± 0.00128 | 0.64060 ± 0.00277 | 0.81650 ± 0.00147 | 0.81650 ± 0.00147 | 0.67771 ± 0.00307 | 0.81650 ± 0.00147 | 0.78756 ± 0.00180 |
| DAGSNet | fedproto | 42 | 0.8978 | 0.8778 | 0.8978 | 0.8923 | 0.8651 | 0.8978 | 0.8978 | 0.8698 | 0.8978 | 0.8932 |
| tabm | fedproto | 41,42,43 | 0.74962 ± 0.01534 | 0.73646 ± 0.01589 | 0.74962 ± 0.01534 | 0.81993 ± 0.01442 | 0.75890 ± 0.01659 | 0.74962 ± 0.01534 | 0.74962 ± 0.01534 | 0.73279 ± 0.01696 | 0.74962 ± 0.01534 | 0.76447 ± 0.01675 |
| sae | fedproto | 41,42,43 | 0.28315 ± 0.04689 | 0.22233 ± 0.00730 | 0.28315 ± 0.04689 | 0.40686 ± 0.01869 | 0.23589 ± 0.02619 | 0.28315 ± 0.04689 | 0.28315 ± 0.04689 | 0.18821 ± 0.01825 | 0.28315 ± 0.04689 | 0.28617 ± 0.02854 |
| DAGSNet | v2x_htfl | 42 | 0.8974 | 0.8865 | 0.8974 | 0.8910 | 0.8554 | 0.8974 | 0.8974 | 0.8661 | 0.8974 | 0.8907 |
| tabm | v2x_htfl | 41,42,43 | 0.53043 ± 0.42505 | 0.57052 ± 0.34973 | 0.53043 ± 0.42505 | 0.59210 ± 0.36728 | 0.49424 ± 0.36810 | 0.53043 ± 0.42505 | 0.53043 ± 0.42505 | 0.47895 ± 0.39971 | 0.53043 ± 0.42505 | 0.52508 ± 0.43361 |
| sae | v2x_htfl | 41,42,43 | 0.46560 ± 0.02172 | 0.24855 ± 0.02061 | 0.46560 ± 0.02172 | 0.39166 ± 0.02362 | 0.18979 ± 0.00400 | 0.46560 ± 0.02172 | 0.46560 ± 0.02172 | 0.17401 ± 0.00522 | 0.46560 ± 0.02172 | 0.38731 ± 0.01390 |

## Nguồn TabM/SAE từngseed

| Model | Method | Seed | Kernel/version | Final macro-F1 |
|---|---|---:|---|---:|
| sae | fd_ids | 43 | [dangthiet/sae-fl-fd-ids-s43/1](https://www.kaggle.com/code/dangthiet/sae-fl-fd-ids-s43/1) | 0.68122 |
| sae | perfedskd | 41 | [dangthiet/sae-fl-perfedskd-s41/1](https://www.kaggle.com/code/dangthiet/sae-fl-perfedskd-s41/1) | 0.68112 |
| sae | tinyproto | 43 | [dangthiet/sae-fl-tinyproto-s43/1](https://www.kaggle.com/code/dangthiet/sae-fl-tinyproto-s43/1) | 0.36448 |
| sae | v2x_htfl | 43 | [dangthiet/sae-fl-v2x-htfl-s43/1](https://www.kaggle.com/code/dangthiet/sae-fl-v2x-htfl-s43/1) | 0.17988 |
| tabm | canperfl | 41 | [dangthiet/tabm-fl-canperfl-s41/1](https://www.kaggle.com/code/dangthiet/tabm-fl-canperfl-s41/1) | 0.85767 |
| tabm | canperfl | 42 | [dangthiet/tabm-fl-canperfl-s42/1](https://www.kaggle.com/code/dangthiet/tabm-fl-canperfl-s42/1) | 0.86297 |
| tabm | fedavg | 41 | [dangthiet/tabm-fl-fedavg-s41-probe/3](https://www.kaggle.com/code/dangthiet/tabm-fl-fedavg-s41-probe/3) | 0.84264 |
| tabm | fedavg | 42 | [dangthiet/tabm-fl-fedavg-s42/2](https://www.kaggle.com/code/dangthiet/tabm-fl-fedavg-s42/2) | 0.84742 |
| tabm | lwfednids | 41 | [dangthiet/tabm-fl-lwfednids-s41/1](https://www.kaggle.com/code/dangthiet/tabm-fl-lwfednids-s41/1) | 0.73120 |
| tabm | lwfednids | 42 | [dangthiet/tabm-fl-lwfednids-s42/1](https://www.kaggle.com/code/dangthiet/tabm-fl-lwfednids-s42/1) | 0.74075 |
| sae | apfed | 41 | [hoangdang1711/sae-fl-apfed-s41/1](https://www.kaggle.com/code/hoangdang1711/sae-fl-apfed-s41/1) | 0.72254 |
| sae | apfed | 42 | [hoangdang1711/sae-fl-apfed-s42/1](https://www.kaggle.com/code/hoangdang1711/sae-fl-apfed-s42/1) | 0.70087 |
| sae | canperfl | 43 | [hoangdang1711/sae-fl-canperfl-s43/1](https://www.kaggle.com/code/hoangdang1711/sae-fl-canperfl-s43/1) | 0.72724 |
| sae | fedavg | 43 | [hoangdang1711/sae-fl-fedavg-s43/1](https://www.kaggle.com/code/hoangdang1711/sae-fl-fedavg-s43/1) | 0.69553 |
| sae | fedtgp | 42 | [hoangdang1711/sae-fl-fedtgp-s42/1](https://www.kaggle.com/code/hoangdang1711/sae-fl-fedtgp-s42/1) | 0.06875 |
| sae | pfedes | 41 | [hoangdang1711/sae-fl-pfedes-s41/1](https://www.kaggle.com/code/hoangdang1711/sae-fl-pfedes-s41/1) | 0.63420 |
| sae | v2x_htfl | 41 | [hoangdang1711/sae-fl-v2x-htfl-s41/1](https://www.kaggle.com/code/hoangdang1711/sae-fl-v2x-htfl-s41/1) | 0.16987 |
| tabm | canperfl | 43 | [hoangdang1711/tabm-fl-canperfl-s43/1](https://www.kaggle.com/code/hoangdang1711/tabm-fl-canperfl-s43/1) | 0.86095 |
| tabm | fedavg | 43 | [hoangdang1711/tabm-fl-fedavg-s43/2](https://www.kaggle.com/code/hoangdang1711/tabm-fl-fedavg-s43/2) | 0.84376 |
| tabm | fedprox | 41 | [hoangdang1711/tabm-fl-fedprox-s41/2](https://www.kaggle.com/code/hoangdang1711/tabm-fl-fedprox-s41/2) | 0.83046 |
| tabm | lwfednids | 43 | [hoangdang1711/tabm-fl-lwfednids-s43/1](https://www.kaggle.com/code/hoangdang1711/tabm-fl-lwfednids-s43/1) | 0.71341 |
| tabm | tinyproto | 41 | [hoangdang1711/tabm-fl-tinyproto-s41/1](https://www.kaggle.com/code/hoangdang1711/tabm-fl-tinyproto-s41/1) | 0.29176 |
| sae | apfed | 43 | [hoangdangnhi/sae-fl-apfed-s43/1](https://www.kaggle.com/code/hoangdangnhi/sae-fl-apfed-s43/1) | 0.71243 |
| sae | canperfl | 41 | [hoangdangnhi/sae-fl-canperfl-s41/1](https://www.kaggle.com/code/hoangdangnhi/sae-fl-canperfl-s41/1) | 0.74310 |
| sae | canperfl | 42 | [hoangdangnhi/sae-fl-canperfl-s42/1](https://www.kaggle.com/code/hoangdangnhi/sae-fl-canperfl-s42/1) | 0.72719 |
| sae | fedavg | 41 | [hoangdangnhi/sae-fl-fedavg-s41/1](https://www.kaggle.com/code/hoangdangnhi/sae-fl-fedavg-s41/1) | 0.70597 |
| sae | fedavg | 42 | [hoangdangnhi/sae-fl-fedavg-s42/1](https://www.kaggle.com/code/hoangdangnhi/sae-fl-fedavg-s42/1) | 0.69492 |
| sae | fedproto | 43 | [hoangdangnhi/sae-fl-fedproto-s43/1](https://www.kaggle.com/code/hoangdangnhi/sae-fl-fedproto-s43/1) | 0.20260 |
| sae | pfedes | 42 | [hoangdangnhi/sae-fl-pfedes-s42/1](https://www.kaggle.com/code/hoangdangnhi/sae-fl-pfedes-s42/1) | 0.63056 |
| tabm | fedproto | 41 | [hoangdangnhi/tabm-fl-fedproto-s41/1](https://www.kaggle.com/code/hoangdangnhi/tabm-fl-fedproto-s41/1) | 0.71452 |
| tabm | fedprox | 42 | [hoangdangnhi/tabm-fl-fedprox-s42/3](https://www.kaggle.com/code/hoangdangnhi/tabm-fl-fedprox-s42/3) | 0.83310 |
| tabm | fedprox | 43 | [hoangdangnhi/tabm-fl-fedprox-s43/3](https://www.kaggle.com/code/hoangdangnhi/tabm-fl-fedprox-s43/3) | 0.82590 |
| tabm | tinyproto | 42 | [hoangdangnhi/tabm-fl-tinyproto-s42/1](https://www.kaggle.com/code/hoangdangnhi/tabm-fl-tinyproto-s42/1) | 0.28756 |
| tabm | tinyproto | 43 | [hoangdangnhi/tabm-fl-tinyproto-s43/1](https://www.kaggle.com/code/hoangdangnhi/tabm-fl-tinyproto-s43/1) | 0.31007 |
| sae | fedprox | 41 | [hoangdv1711/sae-fl-fedprox-s41/1](https://www.kaggle.com/code/hoangdv1711/sae-fl-fedprox-s41/1) | 0.66802 |
| sae | fedprox | 42 | [hoangdv1711/sae-fl-fedprox-s42/1](https://www.kaggle.com/code/hoangdv1711/sae-fl-fedprox-s42/1) | 0.66882 |
| sae | perfedskd | 42 | [hoangdv1711/sae-fl-perfedskd-s42/1](https://www.kaggle.com/code/hoangdv1711/sae-fl-perfedskd-s42/1) | 0.67683 |
| tabm | afpha | 41 | [hoangdv1711/tabm-fl-afpha-s41/2](https://www.kaggle.com/code/hoangdv1711/tabm-fl-afpha-s41/2) | 0.82684 |
| tabm | afpha | 42 | [hoangdv1711/tabm-fl-afpha-s42/2](https://www.kaggle.com/code/hoangdv1711/tabm-fl-afpha-s42/2) | 0.82899 |
| tabm | perfedskd | 41 | [hoangdv1711/tabm-fl-perfedskd-s41/1](https://www.kaggle.com/code/hoangdv1711/tabm-fl-perfedskd-s41/1) | 0.82992 |
| tabm | perfedskd | 42 | [hoangdv1711/tabm-fl-perfedskd-s42/1](https://www.kaggle.com/code/hoangdv1711/tabm-fl-perfedskd-s42/1) | 0.83183 |
| tabm | pfedes | 41 | [hoangdv1711/tabm-fl-pfedes-s41/1](https://www.kaggle.com/code/hoangdv1711/tabm-fl-pfedes-s41/1) | 0.82226 |
| tabm | pfedes | 42 | [hoangdv1711/tabm-fl-pfedes-s42/1](https://www.kaggle.com/code/hoangdv1711/tabm-fl-pfedes-s42/1) | 0.82528 |
| sae | afpha | 41 | [minhiw/sae-fl-afpha-s41/1](https://www.kaggle.com/code/minhiw/sae-fl-afpha-s41/1) | 0.65820 |
| sae | fedprox | 43 | [minhiw/sae-fl-fedprox-s43/1](https://www.kaggle.com/code/minhiw/sae-fl-fedprox-s43/1) | 0.66720 |
| sae | lwfednids | 41 | [minhiw/sae-fl-lwfednids-s41/1](https://www.kaggle.com/code/minhiw/sae-fl-lwfednids-s41/1) | 0.68453 |
| sae | perfedskd | 43 | [minhiw/sae-fl-perfedskd-s43/1](https://www.kaggle.com/code/minhiw/sae-fl-perfedskd-s43/1) | 0.67518 |
| tabm | afpha | 43 | [minhiw/tabm-fl-afpha-s43/2](https://www.kaggle.com/code/minhiw/tabm-fl-afpha-s43/2) | 0.82198 |
| tabm | fd_ids | 41 | [minhiw/tabm-fl-fd-ids-s41/2](https://www.kaggle.com/code/minhiw/tabm-fl-fd-ids-s41/2) | 0.80482 |
| tabm | fedproto | 43 | [minhiw/tabm-fl-fedproto-s43/1](https://www.kaggle.com/code/minhiw/tabm-fl-fedproto-s43/1) | 0.73581 |
| tabm | nilm | 41 | [minhiw/tabm-fl-nilm-s41/1](https://www.kaggle.com/code/minhiw/tabm-fl-nilm-s41/1) | 0.82142 |
| tabm | pfedes | 43 | [minhiw/tabm-fl-pfedes-s43/1](https://www.kaggle.com/code/minhiw/tabm-fl-pfedes-s43/1) | 0.81947 |
| sae | fedtgp | 41 | [nguynt298/sae-fl-fedtgp-s41/1](https://www.kaggle.com/code/nguynt298/sae-fl-fedtgp-s41/1) | 0.11136 |
| sae | fedtgp | 43 | [nguynt298/sae-fl-fedtgp-s43/1](https://www.kaggle.com/code/nguynt298/sae-fl-fedtgp-s43/1) | 0.07407 |
| sae | nilm | 42 | [nguynt298/sae-fl-nilm-s42/1](https://www.kaggle.com/code/nguynt298/sae-fl-nilm-s42/1) | 0.65847 |
| sae | nilm | 43 | [nguynt298/sae-fl-nilm-s43/1](https://www.kaggle.com/code/nguynt298/sae-fl-nilm-s43/1) | 0.65721 |
| sae | v2x_htfl | 42 | [nguynt298/sae-fl-v2x-htfl-s42/1](https://www.kaggle.com/code/nguynt298/sae-fl-v2x-htfl-s42/1) | 0.17227 |
| sae | afpha | 42 | [phamquang0606/sae-fl-afpha-s42/1](https://www.kaggle.com/code/phamquang0606/sae-fl-afpha-s42/1) | 0.65939 |
| sae | afpha | 43 | [phamquang0606/sae-fl-afpha-s43/1](https://www.kaggle.com/code/phamquang0606/sae-fl-afpha-s43/1) | 0.65803 |
| sae | fedproto | 41 | [phamquang0606/sae-fl-fedproto-s41/1](https://www.kaggle.com/code/phamquang0606/sae-fl-fedproto-s41/1) | 0.19434 |
| sae | lwfednids | 42 | [phamquang0606/sae-fl-lwfednids-s42/1](https://www.kaggle.com/code/phamquang0606/sae-fl-lwfednids-s42/1) | 0.68374 |
| tabm | fd_ids | 42 | [phamquang0606/tabm-fl-fd-ids-s42/2](https://www.kaggle.com/code/phamquang0606/tabm-fl-fd-ids-s42/2) | 0.80561 |
| tabm | fd_ids | 43 | [phamquang0606/tabm-fl-fd-ids-s43/2](https://www.kaggle.com/code/phamquang0606/tabm-fl-fd-ids-s43/2) | 0.80585 |
| tabm | fedproto | 42 | [phamquang0606/tabm-fl-fedproto-s42/1](https://www.kaggle.com/code/phamquang0606/tabm-fl-fedproto-s42/1) | 0.74804 |
| tabm | nilm | 42 | [phamquang0606/tabm-fl-nilm-s42/1](https://www.kaggle.com/code/phamquang0606/tabm-fl-nilm-s42/1) | 0.82318 |
| tabm | nilm | 43 | [phamquang0606/tabm-fl-nilm-s43/1](https://www.kaggle.com/code/phamquang0606/tabm-fl-nilm-s43/1) | 0.81878 |
| tabm | perfedskd | 43 | [phamquang0606/tabm-fl-perfedskd-s43/1](https://www.kaggle.com/code/phamquang0606/tabm-fl-perfedskd-s43/1) | 0.82893 |
| sae | fd_ids | 42 | [qwerasdf3004/sae-fl-fd-ids-s42/1](https://www.kaggle.com/code/qwerasdf3004/sae-fl-fd-ids-s42/1) | 0.68472 |
| sae | nilm | 41 | [qwerasdf3004/sae-fl-nilm-s41/1](https://www.kaggle.com/code/qwerasdf3004/sae-fl-nilm-s41/1) | 0.66556 |
| sae | pfedes | 43 | [qwerasdf3004/sae-fl-pfedes-s43/1](https://www.kaggle.com/code/qwerasdf3004/sae-fl-pfedes-s43/1) | 0.62619 |
| sae | tinyproto | 41 | [qwerasdf3004/sae-fl-tinyproto-s41/1](https://www.kaggle.com/code/qwerasdf3004/sae-fl-tinyproto-s41/1) | 0.37635 |
| sae | tinyproto | 42 | [qwerasdf3004/sae-fl-tinyproto-s42/1](https://www.kaggle.com/code/qwerasdf3004/sae-fl-tinyproto-s42/1) | 0.33812 |
| tabm | apfed | 43 | [qwerasdf3004/tabm-fl-apfed-s43/2](https://www.kaggle.com/code/qwerasdf3004/tabm-fl-apfed-s43/2) | 0.85719 |
| tabm | fedtgp | 43 | [qwerasdf3004/tabm-fl-fedtgp-s43/1](https://www.kaggle.com/code/qwerasdf3004/tabm-fl-fedtgp-s43/1) | 0.75346 |
| tabm | v2x_htfl | 43 | [qwerasdf3004/tabm-fl-v2x-htfl-s43/1](https://www.kaggle.com/code/qwerasdf3004/tabm-fl-v2x-htfl-s43/1) | 0.74679 |
| sae | fd_ids | 41 | [ttalgies/sae-fl-fd-ids-s41/1](https://www.kaggle.com/code/ttalgies/sae-fl-fd-ids-s41/1) | 0.68315 |
| sae | fedproto | 42 | [ttalgies/sae-fl-fedproto-s42/1](https://www.kaggle.com/code/ttalgies/sae-fl-fedproto-s42/1) | 0.16768 |
| sae | lwfednids | 43 | [ttalgies/sae-fl-lwfednids-s43/1](https://www.kaggle.com/code/ttalgies/sae-fl-lwfednids-s43/1) | 0.68486 |
| tabm | apfed | 41 | [ttalgies/tabm-fl-apfed-s41/2](https://www.kaggle.com/code/ttalgies/tabm-fl-apfed-s41/2) | 0.85593 |
| tabm | apfed | 42 | [ttalgies/tabm-fl-apfed-s42/2](https://www.kaggle.com/code/ttalgies/tabm-fl-apfed-s42/2) | 0.86172 |
| tabm | fedtgp | 41 | [ttalgies/tabm-fl-fedtgp-s41/1](https://www.kaggle.com/code/ttalgies/tabm-fl-fedtgp-s41/1) | 0.75675 |
| tabm | fedtgp | 42 | [ttalgies/tabm-fl-fedtgp-s42/1](https://www.kaggle.com/code/ttalgies/tabm-fl-fedtgp-s42/1) | 0.74327 |
| tabm | v2x_htfl | 41 | [ttalgies/tabm-fl-v2x-htfl-s41/1](https://www.kaggle.com/code/ttalgies/tabm-fl-v2x-htfl-s41/1) | 0.67054 |
| tabm | v2x_htfl | 42 | [ttalgies/tabm-fl-v2x-htfl-s42/1](https://www.kaggle.com/code/ttalgies/tabm-fl-v2x-htfl-s42/1) | 0.01951 |

Raw results/audit: `runs/comparison-20261115/audit_summary.json`.
Bảng từnground đủ10metrics: [histories](tabm-sae-round-history-20261115.md).

DAGSNet source: [results.md tại commit cố định](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/68a01b29cb3dd47d221d48a16289a8c9f1dfa155/docs/research/results.md).
