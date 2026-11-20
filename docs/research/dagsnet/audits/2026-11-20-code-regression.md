# Hồi quy code DAGSNet: `36ac7ff` (code sinh kết quả 14 PP) vs `827a053` (hiện tại) — 20/11/2026

Câu hỏi: code hiện tại (sau khi gộp TabM/SAE/MLP/CNN-BiGRU/mixed vào `core/` và `methods/`) có còn cho đúng kết quả DAGSNet
đã ghi trong [results.md](../results.md) không.

**Cách chạy.** Cùng lệnh hai checkout: `scripts/run_fl.py --method <m> --subset 20 --rounds 2 --set group_rows=1024
eval_rows=4096 seed=42` (TinyProto thêm `mu=0.00015`), eager, GPU local RTX 3050. So `rounds.jsonl` (mọi khóa trừ `t_*`) và
mọi tensor trong `state.pt`. Script ở `runs/regress-20261120/{regress.sh,cmp.py}` (gitignored), checkout cũ
`git worktree add --detach runs/regress-20261120/old 36ac7ff`.

| phương pháp | kết quả |
|---|---|
| FedAvg, FedProx, AFPHA, FD-IDS, LwFedNIDS, pFedES, FedProto, FedTGP, TinyProto, NILM-FL, PerFed-SKD, APFed | **bit-exact** (metric + mọi tensor state) |
| V2X-HtFL | lệch nhỏ, **có chủ đích**: commit `c398d56` chạy projection ở fp32 (tránh tràn fp16 của grad `proj_b`), thêm generator riêng `gen_s`, cosine `eps = 1e-12`. 20 xe r2: client macro-F1 0,80792 (cũ) vs 0,80817 (mới) |
| CANPerFL | lệch, nhưng **chạy cùng code mới hai lần cũng lệch** (GPU không deterministic khi fine-tune từ state FedAvg); `methods/canperfl.py` không đổi, đường train của `core/local.py` với DAGSNet tương đương. Init: state FedAvg 2 round gắn `round = 50` (`reg/fedavg-r50-patched.pt`) cho cả hai bên |

**Kết luận.** 12/14 phương pháp tái lập bit-exact. Số V2X-HtFL DAGSNet trong `results.csv` sinh bằng code trước `c398d56`
(khác biệt số học nhỏ, cùng thuật toán). CANPerFL không tái lập bit-exact được trên GPU kể cả cùng code.

**Khác biệt đã biết giữa các kịch bản (không phải lỗi):** centralized DAGSNet dùng `scripts/run_central.py`, centralized backbone
khác dùng `scripts/run_central_backbones.py`; mixed có projection 128 chiều cho prototype (pFedES giữ head gốc).
