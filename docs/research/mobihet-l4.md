# MobiHet L4 trên mọi kịch bản — port 20/11/2026

Phương pháp đề xuất MobiHet FB-L cấu hình **L4** (owner 19/11, repo `../mobihet`: [FB-L](../../../mobihet/docs/research/screening-fbl.md)):
mỗi xe giữ model riêng, chỉ trao thống kê bootstrap (copula/KM) + gói Q lớp, không trao tham số; FB N4 (dòng thật 64/37/37 theo
tầng cỡ xe, synthetic 64,64/16,16/16,16), chọn dòng thật theo CE cũ (`real_select=top`), engine `compact=true`.

## Quyết định owner 20/11

- **DAGSNet, BERT:** dùng lại chain r50 × 3 seed của `../mobihet` / `../fl_veremi_bert` (đúng protocol, audit ở repo đó); không chạy lại.
- **TabM, SAE, MLP, CNN-BiGRU:** chạy mới ở repo này, hp L4 khóa y nguyên (chỉ đổi seed), 3 seed. TabM: mọi số hạng loss tính theo
  từng member rồi lấy trung bình (như baseline TabM), điểm chọn dòng = CE trung bình member, Q = xác suất trung bình member.
- **Mixed case1–3:** làm sau đơn model (cần Q/bootstrap gộp xuyên các nhóm model).
- Push tự động khi có slot/quota (không dùng account MobiHet đang giữ slot, không đụng account teammate).

## Code

- `methods/mobihet*.py` (15 file) = byte copy `../mobihet/methods` (method `mobihet_budget` sha `a4147a1b…`);
  `scripts/run_p21_recovery.py`, `scripts/sim_fold_val.py` chép theo (import / harness local).
- `methods/mobihet_l4.py`: subclass `mobihet_budget`, chỉ đổi loss bước cho TabM (logit 4D) và SAE (thêm số hạng 0 cho decoder
  không dùng, như baseline SAE CE-only). DAGSNet/MLP/CNN-BiGRU đi thẳng `mobihet_budget`.
- `core/local.py`: nhánh MobiHet bật khi hp có `synth_step_floor` (bước floor chỉ synthetic; BN buffer + loss log chỉ từ bước có
  dòng thật; loss chia `ceil(n/b)`) — đúng `../mobihet/core/local.py`; baseline không có key này ⇒ không đổi.
- `scripts/run_fl.py`: hp không phải JSON giữ dạng chuỗi (`arm=none`); `--clients`.
- `scripts/build_l4_fl.py`: notebook 2×T4, nhúng source, hp L4 (= `../mobihet/runs/fbl-r50/batch.json`, 44 khóa) + seed,
  `--parent` nối phiên (copy thư mục run từ output phiên trước, kiểm round liền).

## Kiểm trước push (local, `runs/l4-port/`)

| kiểm | kết quả |
|---|---|
| L4 DAGSNet repo này vs `../mobihet` (panel 200 xe fold train→val, 2 round, compiled) | bit-exact: rounds.jsonl, CM r1/r2, 10/11 tensor state; `row_score` chỉ khác ô sink (vị trí đổ của dòng không train, không bao giờ đọc) |
| FedAvg DAGSNet/TabM/SAE trước vs sau port (20 xe, 2 round) | bit-exact |
| L4 TabM resume (stop r1 → resume r2) vs chạy liền | bit-exact (trừ ô sink) |
| Smoke L4 TabM/SAE/MLP (CPU) + CNN-BiGRU (GPU), compile gate | PASS, loss hữu hạn |
| `tests/test_{tabm,sae,dense_recurrent,aligned}_fl.py` | PASS |

Kernel, kết quả: [continue](../operations/continue.md); số ghi `experiments/<kịch bản>/results.csv` (method `mobihet_l4`).
