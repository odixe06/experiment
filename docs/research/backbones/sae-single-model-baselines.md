# SAE single-model baseline matrix

Owner chốt 14/11/2026: chạy 14 baseline (bỏ SCAFFOLD), mỗi baseline với một backbone SAE, seed 41/42/43.
Phạm vi này độc lập với bảng DAGSNet seed42 và ma trận TabM.

## Cấu hình khóa

- Dataset/client/evaluation/optimizer/LR/batch/50 round: giữ [protocol chung](../protocol.md).
- Backbone: supervised sparse autoencoder TAS-VANET đã dùng trong centralized, đầu vào 77, 16 lớp.
- Encoder `77→32→16`, decoder `16→32→77`, classifier `[77 input đã unit-scale ; code16]→16`.
- Tham số backbone: **7.613**. Prototype/hidden representation: bottleneck 16 chiều.
- Objective local: reconstruction `1,0` + KL sparsity `1,2` (`rho=0,05`) + classification CE `1,5`.
  Không thay bằng CE thuần vì khi đó decoder không nhận gradient.
- Seed: 41, 42, 43; seed chỉ chi phối initialization/shuffle. Client partition và dữ liệu cố định.
- Không centralized warm-start. Headline round50; lưu CM và đủ metric từng round như protocol.

## Adapter phương pháp

- FedAvg/FedProx/CANPerFL: dùng nguyên objective SAE ở local epoch.
- FD-IDS/PerFed-SKD/NILM: objective SAE thay phần supervised CE; KD vẫn dùng phân phối logits 16 lớp.
- FedProto/FedTGP/TinyProto: prototype trên bottleneck 16 chiều; objective SAE giữ nguyên trước regularizer prototype.
- APFed: classifier riêng là `head`; encoder và decoder thuộc shared body.
- LwFedNIDS: structured-prune 70% dependency groups, `32→10`, code `16→5`, decoder hidden `32→10`;
  model vật lý còn 3.070 tham số.
- pFedES: cả enhanced/raw branch giữ reconstruction+sparsity+classification; bước train proxy vẫn dùng classifier SAE đã freeze.
- V2X-HtFL: ETF/projection trên code16; reconstruction+sparsity cộng vào local objective.

## Build status

- 42/42 notebook local đã build: 39 direct + 3 CANPerFL.
- Mỗi notebook private, 2×T4, dataset v4, nhúng source/hash/manifest, smoke 2 round có resume trước production.
- Static audit PASS: nbformat, AST mọi code cell, metadata, seed/method uniqueness, source wiring và CAN parent wiring.
- CPU smoke đã PASS với runtime riêng tmp/fl-test-runtime (PyTorch2.15CPU). Notebook bắt buộc chạy lại `tests/test_sae_fl.py` trên Kaggle trước smoke/train.
- Wave1 đang launch theo `scripts/launch_tabm_final_sae_wave.py`: FedAvg41/42 chuyển sang hoangdangnhi,
  FedAvg43 hoangdang1711. CANPerFL SAE41/42 sau này phải rebuild parent theo owner mới này;
  metadata local cũ dangthiet không còn là parent đúng. Chỉ push sau FedAvg cùng seed round50.
