# Case 2 — kết quả mixed128 round 50

30% TabM, 30% SAE, 40% MLP; giữ nguyên dữ liệu từng xe. 15 run hoàn tất; V2X dùng bản sửa a2, cả3seed final-CM audit PASS. CM audit kiểm đủ5343xe/9347679dòng và10metric.

| Phương pháp | View | Seed41 | Seed42 | Seed43 |
|---|---|---:|---:|---:|
| fedproto | client_proto | 0.75111 | 0.76196 | 0.75632 |
| fedtgp | client_proto | 0.65899 | 0.63780 | 0.65815 |
| tinyproto | client_proto | 0.78550 | 0.79992 | 0.79041 |
| v2x_htfl | client | 0.83862 | 0.83839 | 0.83917 |
| pfedes | client | 0.79253 | 0.79390 | 0.79630 |

Macro-F1 headline theo view của từng phương pháp; không đổi sang classifier head để chọn điểm cao hơn.
CSV kèm đủ10metric cho mỗi seed, mỗi model và cả hai view nếu có. V2X dùng a2 (SAE trainFP32); các phương pháp khác dùng a1.
