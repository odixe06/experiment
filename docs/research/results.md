# Kết quả — 5.343 xe, DAGSNet W = 1, 50 round (30/10/2026)

Tính lại từ confusion matrix đã pull (`scripts/report.py runs/pulls/*/`), không chép từ log. Protocol: [protocol](protocol.md).
Headline mỗi phương pháp = model được đánh giá theo luật của bài (global; model client trên test của chính nó; prototype cho
FedProto/FedTGP). Byte = lên + xuống mỗi xe mỗi round. Một seed (42). Chi phí: [cost](cost.md).

Audit các run trong bảng chính: đủ 50 round (CANPerFL: 10 epoch fine-tune sau FedAvg r50), CM/metric hữu hạn, HP chung đúng protocol. Train health và khả năng replay model có giới hạn riêng bên dưới.

Chưa có / lưu ý:
- **TinyProto**: r2 sai prototype suy luận; r3 bất ổn scale (prototype F1 0,2904, target L2 bùng nổ 2.943). R4 đã sửa μ=1,5e-4 hoàn tất đủ 50 round, target L2 ổn định (0,2657), prototype F1 đạt 0,8534 ([chẩn đoán](../methods/tinyproto.md)).
- **NILM-FL**: nối phiên thành công (s1 r1–36 + s2 r37–50), preflight parent r1–36 byte-identical, hoàn tất 50 round.
- **APFed**: metric r50 đầy đủ; bank model client r50 mất (rank 0 hết RAM host khi ghi state cuối; sửa `bda3c07`).
- **SCAFFOLD**: không ổn định giai đoạn đầu (train_loss NaN r15–16, F1 r5 0,23, r15–16 0,25/0,23), hồi phục từ r18.
- **MobiHet**: repo riêng `../mobihet`, đang sàng lọc H0–H4 10 round; owner chưa cho đưa lên Notion.
- FedAvg/FedProx/SCAFFOLD/FedProto chạy SHA `6617261` (không có snapshot model từng round; state r50 trên Kaggle).

| method | view | round | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted | client F1 mean | p10 | min | byte/client/round |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| centralized (train+val → test) | — | ep50 | 0,9580 | 0,9684 | 0,9580 | 0,9560 | 0,9531 | 0,9580 | 0,9580 | 0,9593 | 0,9580 | 0,9556 | — | — | — | — |
| afpha | global (headline) | 50 | 0,9211 | 0,9300 | 0,9211 | 0,9106 | 0,8932 | 0,9211 | 0,9211 | 0,8990 | 0,9211 | 0,9045 | 0,8652 | 0,7523 | 0,1786 | 3.189.376 |
| apfed | client (headline) | 50 | 0,9278 | 0,9377 | 0,9278 | 0,9217 | 0,9032 | 0,9278 | 0,9278 | 0,9119 | 0,9278 | 0,9169 | 0,8663 | 0,7533 | 0,1500 | 3.189.376 |
|  | global | 50 | 0,9216 | 0,9333 | 0,9216 | 0,9132 | 0,8947 | 0,9216 | 0,9216 | 0,9031 | 0,9216 | 0,9075 | 0,8645 | 0,7487 | 0,1806 | 3.189.376 |
| canperfl | client (headline) | 10 | 0,9233 | 0,9281 | 0,9233 | 0,9171 | 0,8986 | 0,9233 | 0,9233 | 0,9088 | 0,9233 | 0,9159 | 0,8779 | 0,7684 | 0,1545 | 0 |
|  | global | 10 | 0,9228 | 0,9354 | 0,9228 | 0,9147 | 0,8957 | 0,9228 | 0,9228 | 0,9042 | 0,9228 | 0,9085 | 0,8667 | 0,7540 | 0,1500 | 0 |
| fd_ids | global (headline) | 50 | 0,9220 | 0,9373 | 0,9220 | 0,9155 | 0,8914 | 0,9220 | 0,9220 | 0,8992 | 0,9220 | 0,9054 | 0,8664 | 0,7552 | 0,2176 | 3.189.376 |
| fedavg | global (headline) | 50 | 0,9228 | 0,9354 | 0,9228 | 0,9147 | 0,8957 | 0,9228 | 0,9228 | 0,9042 | 0,9228 | 0,9085 | 0,8667 | 0,7540 | 0,1500 | 3.189.376 |
| fedproto | client_proto (headline) | 50 | 0,8978 | 0,8778 | 0,8978 | 0,8923 | 0,8651 | 0,8978 | 0,8978 | 0,8698 | 0,8978 | 0,8932 | 0,7873 | 0,5911 | 0,0000 | 32.832 |
|  | client | 50 | 0,9013 | 0,8958 | 0,9013 | 0,8947 | 0,8588 | 0,9013 | 0,9013 | 0,8726 | 0,9013 | 0,8942 | 0,7788 | 0,5745 | 0,0000 | 32.832 |
| fedprox | global (headline) | 50 | 0,9214 | 0,9307 | 0,9214 | 0,9112 | 0,8941 | 0,9214 | 0,9214 | 0,8998 | 0,9214 | 0,9049 | 0,8658 | 0,7524 | 0,1786 | 3.189.376 |
| fedtgp | client_proto (headline) | 50 | 0,9020 | 0,8924 | 0,9020 | 0,8945 | 0,8610 | 0,9020 | 0,9020 | 0,8719 | 0,9020 | 0,8947 | 0,7794 | 0,5712 | 0,0000 | 32.768 |
|  | client | 50 | 0,9007 | 0,8989 | 0,9007 | 0,8950 | 0,8594 | 0,9007 | 0,9007 | 0,8740 | 0,9007 | 0,8933 | 0,7782 | 0,5691 | 0,0000 | 32.768 |
| lwfednids | global (headline) | 50 | 0,9162 | 0,9320 | 0,9162 | 0,9111 | 0,8830 | 0,9162 | 0,9162 | 0,8904 | 0,9162 | 0,8986 | 0,8476 | 0,7291 | 0,2139 | 324.000 |
| nilm | client (headline) | 50 | 0,9156 | 0,9291 | 0,9156 | 0,9146 | 0,8717 | 0,9156 | 0,9156 | 0,8846 | 0,9156 | 0,9020 | 0,7854 | 0,5777 | 0,0000 | 3.189.376 |
|  | proxy | 50 | 0,9227 | 0,9421 | 0,9227 | 0,9192 | 0,8925 | 0,9227 | 0,9227 | 0,8993 | 0,9227 | 0,9051 | 0,8638 | 0,7500 | 0,2963 | 3.189.376 |
| perfedskd | client (headline) | 50 | 0,8994 | 0,8903 | 0,8994 | 0,8931 | 0,8598 | 0,8994 | 0,8994 | 0,8710 | 0,8994 | 0,8928 | 0,8133 | 0,6573 | 0,1241 | 3.189.376 |
| pfedes | client (headline) | 50 | 0,8864 | 0,8692 | 0,8864 | 0,8800 | 0,8405 | 0,8864 | 0,8864 | 0,8511 | 0,8864 | 0,8803 | 0,7449 | 0,5264 | 0,0000 | 2.808 |
| scaffold | global (headline) | 50 | 0,9225 | 0,9011 | 0,9225 | 0,8842 | 0,8859 | 0,9225 | 0,9225 | 0,8911 | 0,9225 | 0,9003 | 0,8744 | 0,7738 | 0,2963 | 6.352.640 |
| tinyproto | client_proto (headline) | 50 | 0,8866 | 0,8702 | 0,8866 | 0,8759 | 0,8457 | 0,8866 | 0,8866 | 0,8534 | 0,8866 | 0,8774 | 0,7607 | 0,5521 | 0,0000 | 3.328 |
|  | client | 50 | 0,8994 | 0,8902 | 0,8994 | 0,8921 | 0,8579 | 0,8994 | 0,8994 | 0,8693 | 0,8994 | 0,8922 | 0,7731 | 0,5649 | 0,0000 | 3.328 |
| v2x_htfl | client (headline) | 50 | 0,8974 | 0,8865 | 0,8974 | 0,8910 | 0,8554 | 0,8974 | 0,8974 | 0,8661 | 0,8974 | 0,8907 | 0,7642 | 0,5519 | 0,0000 | 32.832 |
|  | client_proto | 50 | 0,8947 | 0,8701 | 0,8947 | 0,8877 | 0,8597 | 0,8947 | 0,8947 | 0,8627 | 0,8947 | 0,8892 | 0,7576 | 0,5468 | 0,0000 | 32.832 |


## Lịch sử run / audit trước đó (bằng chứng provenance)

NILM r36 là interim session 1 trước khi nối phiên r37–50 thành công. TinyProto r3 bất ổn scale do μ heuristic cũ, xem [chẩn đoán](../methods/tinyproto.md).
Audit lưu tại [NILM r50](audits/2026-10-30/fl2-nilm-r2.json), [TinyProto r4](audits/2026-10-30/fl2-tinyproto-r4.json) và [TinyProto r3 cũ](audits/2026-10-30/fl2-tinyproto-r3.json).

| method | view | round | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted | client F1 mean | p10 | min | byte/client/round | ghi chú |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nilm (s1) | client | 36 | 0,9152 | 0,9235 | 0,9152 | 0,9127 | 0,8731 | 0,9152 | 0,9152 | 0,8858 | 0,9152 | 0,9032 | 0,7853 | 0,5778 | 0,0000 | 3.189.376 | interim s1, đã nối tiếp s2 đủ r50 |
|  | proxy | 36 | 0,9181 | 0,9414 | 0,9181 | 0,9178 | 0,8865 | 0,9181 | 0,9181 | 0,8914 | 0,9181 | 0,8984 | 0,8549 | 0,7325 | 0,2963 | 3.189.376 | |
| tinyproto (r3) | client_proto | 50 | 0,5954 | 0,3357 | 0,5954 | 0,6164 | 0,3207 | 0,5954 | 0,5954 | 0,2904 | 0,5954 | 0,5790 | 0,3459 | 0,2171 | 0,0000 | 3.328 | bất ổn scale μ heuristic; r4 sửa μ=1,5e-4 đã đạt 0,8534 |
|  | client | 50 | 0,8234 | 0,8480 | 0,8234 | 0,8477 | 0,7498 | 0,8234 | 0,8234 | 0,7881 | 0,8234 | 0,8323 | 0,6790 | 0,4734 | 0,0667 | 3.328 | |


Báo cáo owner: [Notion](https://app.notion.com/p/409730cee68d81f19259d14bea45ea61), 14 hàng hoàn tất ban đầu + chuẩn bị bổ sung 2 hàng NILM-FL & TinyProto r4; quy trình bổ sung tại [notion-results](../operations/notion-results.md).
