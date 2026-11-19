# Audit đồng nhất code 4 backbone với DAGSNet — 16/11/2026

Phạm vi: TabM, SAE CE-v2, MLP, CNN-BiGRU-Attention; 14 baseline; chuẩn là DAGSNet và
protocol chung. “PASS” nghĩa cùng driver, dữ liệu/client, optimizer/lịch train, HP method,
aggregation/evaluation semantics; khác biệt chỉ là adapter representation/head/pruning cần thiết.

## Kết luận

| Phương pháp | Kết luận | Ghi chú |
|---|---|---|
| FedAvg | PASS | Cùng local loop và weighted aggregation |
| FedProx | PASS | Cùng `mu=0.01` |
| AFPHA | PASS | Cùng clustering/aggregation; representation không đổi nghĩa |
| FD-IDS | PASS | Cùng `T=3`, `beta=0.1`, `lam=0.5`, `mu=0.01` |
| APFed | PASS | Personalize đúng classifier cuối của từng backbone |
| LwFedNIDS | PASS-adapter | Cùng prune70%; dependency-aware slicing riêng từng kiến trúc |
| CANPerFL | PASS | FedAvg50 cùng backbone/seed rồi fine-tune10 |
| TinyProto | PASS-current | Đã đồng nhất `mu=0.00015`; không dùng kết quả TabM cũ sai HP |
| pFedES | PASS | Cùng hai bước local và proxy77→77 |
| NILM-FL | PASS | Cùng private/proxy semantics |
| FedTGP | PASS-adapter | Cùng HP/server; prototype dùng hidden representation tương ứng |
| PerFed-SKD | PASS | Cùng client selection/KD; TabM KD trên mean probability |
| FedProto | PASS-adapter | Cùng loss/aggregation; prototype dimension theo backbone |
| V2X-HtFL | PASS completion/telemetry a4 TabM/MLP; audit rộng còn chờ | Cả6 run a4 đủ50round, max skip <1%; chưa audit CM/weights độc lập hoặc rerun DAGSNet cùng telemetry |

## Bằng chứng

- Cập nhật19:36 ngày16/11: cả3seed TabM và3seed MLP a4 đã COMPLETE50, smoke/history PASS.
  Max skip MLP toàn3seed0,14683%, TabM0,20089%, dưới gate1%; không còn trạng thái lỗi smoke của a2.
  Đây chưa phải audit độc lập CM/weights hay bằng chứng health của DAGSNet cũ.

- Cập nhật17:04 ngày16/11: TabM V2X a4 seed42 đủ50round, max skip0,16524% <1%; còn2seed chờ.
  MLP a2 nhúng projection cũ: regression loss thật tái hiện seed42 proj_b overflow FP16/scale4096.
  Shared projection FP32 PASS fixture41/42/43, đã rebuild/push3 MLP a4; chưa có GPU-smoke/full-run proof.
  Vì vậy chưa nâng kết luận toàn bộ V2X thành PASS.

- `runs/aligned-v2-hp-audit.json`: 14/14 method có cùng HP defaults với source DAGSNet.
- Cùng `scripts/run_fl.py`: 50 round, batch64, AdamW, cosine1e-3→1e-5, decay1e-4,
  clip1, AMP, toàn bộ5.343 client, cùng split/evaluation.
- CPU reference/smoke PASS: TabM14 method; SAE CE-v2 14 method; MLP/CNN 28 cặp method-model;
  26 direct cặp MLP/CNN resume round-boundary CM PASS.
- Aligned artifacts SAE/MLP/CNN: mọi final marker hiện có đều có telemetry; không final nào có
  `skipped_frac >= 1%`. V2X MLP bị smoke chặn trước production.
- V2X rerun a3: TabM seed42 skip100% r1/r2; seed43 skip100%/98,3%. Normalize eps1e-2
  không chữa được. Kết quả cũ seed42 F1=0,01951 không hợp lệ để so backbone.

## Khác biệt hợp lệ, không phải lệch protocol

- TabM: CE trung bình4 member; inference/KD trung bình probability; prototype là mean hidden4 member.
- SAE CE-v2: classifier/prototype representation93D = input77 + code16; decoder không ảnh hưởng prediction.
- MLP: hidden32; CNN-BiGRU: attention context128; APFed chọn đúng head tương ứng.
- LwFedNIDS phải slice dependency theo cấu trúc mỗi model; số parameter sau prune khác nhau.

## Việc còn lại trước khi tuyên bố 14/14 PASS

1. Diagnostic V2X cùng batch: eager-FP32 vs eager-AMP vs compiled-AMP.
2. Sửa nguyên nhân non-finite, khóa smoke `skipped_frac < 1%`.
3. Chạy lại V2X cho cả backbone cần so sánh bằng cùng source; không dùng run TabM/MLP cũ.
