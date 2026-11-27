# Bàn giao — experiment, 27/11/2026 18:20 (+07)

[CONTEXT](../../CONTEXT.md) · [Snapshot 26/11 trước khi mất account](../archive/continue-20261126-before-account-loss.md)

## Tiếp tục thí nghiệm

1. Owner mất quyền **`cacaosua`, `preyta`, `nocknnock`, `shippertocdo`** ngày 27/11. Sáu phiên dưới đây coi như mất, không collect:
   `cacaosua/mixmh-abl-a1-case3-s42/1`, `cacaosua/mixmh-abl-a2-case1-s41/1`, `preyta/mixmh-abl-a1-case1-s42/1`,
   `preyta/mixmh-abl-a1-case1-s43/1`, `nocknnock/mlp-mhabl-a4-s42/1`, `shippertocdo/cnnbigru-mhabl-a1-s42/1`.
   Collector, dispatcher và waiter của 26/11 đã dừng khi máy tắt. Chưa mở lại.
2. Đã collect `tamnp2807`: cả 7 kernel ablation của account này đều audit PASS R50. Lượt cuối là `tamnp2807/mlp-mhabl-a4-s43/1`,
   macro-F1 0,876583. Ngày 27/11 collect thêm `chuchuotdaubep/mlp-mhabl-a1-s42/1`, macro-F1 0,848535, PASS đủ 50 CM.
   **Ablation 69/81.** [Kết quả](../../experiments/ablation/README.md) · [Trạng thái audit](../../runs/ablation-mh/watch-collect.json)
3. **27/11 đã push lại toàn bộ 12 notebook ablation còn thiếu.** Mỗi account chỉ dùng tối đa 2 slot. Mọi push đi qua guarded CLI.
   - Lượt 1 (18:00): 7 kernel đang RUNNING trên `datle211206`, `datninhz`, `fuocs2k6`, `gialut`, `hunhtnduytn`, `lcsinh`, `levanloc1806`
     (`runs/ablation-mh/wave-20261127.txt`). `cnnbigru-mhabl-a1-s42` dùng lại bản 11.5 h `runs/ablation-mh/pre-budget-…`.
   - Lượt 2 (18:40), slot thứ hai: `datle211206/mixmh-abl-a2-case1-s43`, `datninhz/mixmh-abl-a4-case1-s42`, `fuocs2k6/mixmh-abl-a4-case1-s43`
     (`wave2-20261127.txt`). `mixmh-abl-a1-case3-s42` và `mlp-mhabl-a4-s42` đứng đầu hàng đợi rerun (mục Rerun).
   Collector `runs/ablation-mh/watch_collect.py` chạy nền, đọc `active.txt`. Sau khi audit PASS: `apply.py file-new`, rồi backup checkpoint.
4. MobiHet đối chứng đủ **27/27**. Centralized BERT đã audit P4 E27/E24/E25, còn thiếu ba endpoint. P5 chờ owner quyết.
   [Ma trận chính](../../experiments/README.md)

## Checkpoint — kiểm kê 27/11

Bảng phủ từng run chính thức: [checkpoint-coverage.csv](checkpoint-coverage.csv). Sinh lại bằng
`runs/checkpoint-coverage-20261127/run.sh` (cần mount ORICO).

- 27/11 đã pull và đẩy **19 bundle** MobiHet/ablation hoàn tất sau batch 26/11 (19,4 GB) lên DVC `onedrive` bằng
  `checkpoint-backup-20261125/direct-backup.py`. Kết quả 19/19 `DVC_VERIFIED` (rclone check), tổng **71 bundle**.
  Pointer nằm ở `artifacts/mobihet-checkpoints-2026-11-25/<owner>/<slug>/v1.dvc`. Đã thêm 19 run vào
  [checkpoint-runs.csv](checkpoint-runs.csv) và [checkpoint-files.csv](checkpoint-files.csv).
- Kiểm remote: cả 127 pointer (tracked và trước đây untracked) có manifest trên OneDrive, đủ `nfiles`, mẫu 300/29.033 object không thiếu.

- **Ma trận chính, 342 run** (15 phương pháp + centralized × 3 seed × 6 kịch bản, cộng 6 phương pháp × 3 seed × 3 mixed):
  - Có trên DVC: 114 run, gồm DAGSNet 44/48, BERT 47/48 (thiếu MobiHet s43) và centralized của bốn backbone.
    MobiHet có 16/27 trên DVC, tính cả `trietbackup/mixmh-case1-s41/1` đẩy ngày 27/11.
  - **Chỉ còn trên Kaggle nhánh codex: 209 run, khoảng 186 GB.** Gồm FL baseline của TabM, SAE, MLP và CNN-BiGRU (42 run mỗi
    backbone), cùng 41 run mixed baseline. Repo này không có credential của 26 account đó.
    Danh sách exact ref, file và byte: [checkpoint-request-codex.csv](checkpoint-request-codex.csv) (214 dòng).
  - Thiếu hẳn 8 run. DAGSNet seed 42 FedAvg, FedProx, FedProto chỉ có metrics và CM trong `runs/pulls/fl2-*.dvc`.
    Mixed case1 V2X-HtFL 3 seed và mixed case3 pFedES seed 41/43 chưa có receipt kiểm kê checkpoint. Năm run mixed này
    và CNN-BiGRU centralized s42 đã nằm trong danh sách yêu cầu.
  - Mất account: 11 run MobiHet. Chi tiết theo exact ref nằm trong CSV, gồm `anhtruong1909`, `lelovo`, `bestyorn` và 6 account khác.
- **Ablation, 69 ô**: 55 ô có trên DVC (37 ô từ batch 26/11 và 18 ô đẩy ngày 27/11), 14 ô mất account.
- Không có dấu hiệu xóa nhầm. 14 bundle MobiHet trên ORICO `ablation-l4/pulls/` khớp đúng 14 pointer đã đổi tên.
  Checkpoint của bản có logit adjustment (đã bỏ) xóa ngày 26–27/11 là có chủ đích. 53 pointer DVC trước đây chưa có trong Git, nay đã commit.

## Dữ liệu local theo kịch bản

Ngày 27/11, dữ liệu ORICO đã sắp lại thành `experiment-data/experiment/<kịch bản>/<phương pháp>/s<seed>/` và `ablation/<kịch bản>/<arm>/s<seed>/`.
Đã dời 389 đơn vị (76.795 file, 312,7 GB) bằng rename. Số file và byte từng đơn vị khớp trước và sau, 974/974 file round-metrics và 3.028/3.028 file checkpoint (71 bundle) khớp SHA-256.
Đường dẫn SSD cũ giữ dạng link. Sau mỗi lần collector pull, chạy `python3 runs/orico-reorg-20261127/apply.py file-new`.
[Layout và quy trình](backup.md#dữ-liệu-local-theo-kịch-bản--27112026)

## Lịch sử từng round và epoch

`runs/round-metrics/` trỏ tới `ORICO/experiment-data/experiment/_shared/round-metrics/`. Mỗi run có `metrics.csv` và `run.json`
trong `<scenario>/<method>/s<seed>/`. `index.json` tra 342 run. **Còn cũ**: vẫn ghi khóa cũ `mobihet_l4` (bản có logit adjustment, đã bỏ) thay cho
`mobihet`, nên trường `checkpoint` của 27 run đó trỏ vào thư mục đã xóa. Dựng lại bằng `PYTHONPATH=. python runs/round-metrics/audit/audit.py`
(env `nckh`), rồi chạy `verify.py` cùng thư mục.

## Kiểm sau khởi động 27/11

- Journal vẫn khớp sau khi khởi động lại máy, không còn link SSD nào gãy. `verify.py bundles` cho 3.028/3.028 file của 71 bundle khớp SHA-256, 0 lỗi.
- Không còn job nền nào chạy. Collector, dispatcher và waiter chưa mở lại, đợi lượt push ngày 28/11.

## Account 27/11

Owner cấp 20 token KGAT. Chỉ repo này và repo 10k dùng chúng (không dùng toàn cục): token nằm ở
`~/.kaggle/accounts-veremi/<user>.mcp-token` (thư mục 0700). Chỉ `kaggle_as.py`, `kaggle_account.py` và script `kaggle-mcp` của 2 repo này đọc
thư mục đó. Helper của repo khác (ví dụ `mobihet`) không thấy token.
- 6 account không có GPU (kernel chỉ nhận 0 hoặc 1 GPU). Theo yêu cầu owner, đã xóa token, lock, state và ledger của chúng.
- 13 account đã xác nhận 2×T4 (chạy qua assert 2×T4, `tkayyy` qua probe): `datle211206`, `datninhz`, `fuocs2k6`, `gialut`,
  `hunhtnduytn`, `lcsinh`, `levanloc1806`, `nlphk252`, `ohkazu`, `sunny0703`, `sunny070306`, `tuntrnhvminh`, `tkayyy`.
  `tkayyy` có kernel khác chạy lúc 03:51 ngày 27/11 (`tkayyy/the-last`), nên account này có người khác cùng dùng.
- `trtai0111` đã dùng hết quota tuần, kiểm GPU sau 07:00 ngày 28/11. Chưa đưa vào danh sách dispatch.
Quota reset lúc 07:00 ngày 28/11. Kaggle MCP xem skill `kaggle-mcp`.

## Rerun 27/11 — checkpoint mất

**Phương pháp chỉ có MobiHet. Ablation chỉ gồm A1, A2, A4.** Không dùng tên A3/L4 cho kết quả, bảng hay tài liệu.
Notebook MobiHet mới build bằng `--ablation none`, slug `<backbone>-mh-s<seed>` / `mixmh-<case>-s<seed>`.

| ô MobiHet | exact ref đang chạy (push 27/11) |
|---|---|
| bert s43 | `gialut/bert-mh-s43/1` |
| tabm s41 / s42 / s43 | `hunhtnduytn/tabm-abl-a3-s41/1`, `lcsinh/tabm-abl-a3-s42/1` (+ phiên 2), `levanloc1806/tabm-abl-a3-s43/1` |
| sae s42 / s43 | `nlphk252/sae-abl-a3-s42/1` (+ phiên 2), `ohkazu/sae-abl-a3-s43/1` |
| mlp s41 / s43 | `sunny0703/mlp-abl-a3-s41/1`, `sunny070306/mlp-abl-a3-s43/1` |
| dagsnet s42 | `tuntrnhvminh/dagsnet-abl-a3-s42/1` (+ phiên 2) |
| mixed-case3 s41 / s43 | trong hàng đợi, notebook `mixmh-case3-s41/43` |

- 8 kernel có slug `*-abl-a3-*` push trước khi đổi tên. Kaggle không đổi tên được kernel đã push, nên owner chọn giữ chúng chạy.
  Slug cũ chỉ còn trong exact ref. Kết quả ghi là `mobihet`. Chúng chạy source MobiHet cũ, bit-exact với `ablation=none`.
  Vì vậy phải audit bằng worktree `runs/ablation-l4/legacy-audit-803a84f` (commit `803a84f^`), truyền qua `ABLATION_AUDIT` cho
  `runs/ablation-l4/collect.sh`, vì audit hiện hành chỉ nhận `none/a1/a2/a4`.
- Seed 42 của 3 kernel đó chạy 2 phiên: phiên 1 dừng ở R2, phiên 2 (`-h2`, cùng owner) chạy tới R50.
  Danh sách nằm ở `runs/ablation-mh/rerun-continue.txt`. `lcsinh/tabm-abl-a3-s42/1` đã PASS R2.
- Collector `runs/ablation-l4/watch_collect.py` (biến môi trường `ABLATION_NO_RECORD=1`) chỉ pull và audit. Sau khi PASS R50, thay dòng
  `mobihet` của ô đó bằng tay và archive dòng cũ kèm exact ref. `bert-mh-s43` và `mixmh-case3-*` đi qua collector ablation-mh, nên sẽ tự ghi thêm
  dòng `mobihet`: khi đó xóa dòng cũ và archive nó.
- Dispatcher `runs/ablation-mh/rerun_dispatch.py` chạy nền (log `rerun-dispatch.log`). Khi kernel xong, collector pull và audit.
  Dispatcher đợi 15 phút rồi push phiên nối hoặc mục kế tiếp trong `rerun-pending.txt`
  (2 ablation thiếu, 2 mixed-case3 MobiHet, 14 ô ablation A1/A2/A4), lên account trong `rerun-accounts.txt`.
  Không giữ slot trước. Owner nào còn chờ phiên nối (MobiHet hoặc 10k) thì dispatcher để slot trống đó cho phiên nối.
- **DAGSNet s42 FedAvg/FedProx/FedProto chưa rerun.** Bản gốc lấy từ pipeline cũ `bert/runs/maintenance-20261116/dagsnet42/fl2-*`,
  repo không còn notebook của pipeline này nên phải dựng lại.

## Việc tay

- Gửi [checkpoint-request-codex.csv](checkpoint-request-codex.csv) cho người giữ nhánh codex. Nhờ họ pull exact ref, đẩy lên remote
  `onedrive` của repo (`NCKH/veremi_build/experiment/dvc-storage`) và commit pointer. Gợi ý layout:
  `artifacts/baseline-checkpoints/<owner>/<slug>/v<N>.dvc`.
- Owner 27/11 quyết định bỏ 25 checkpoint `LOST_ACCESS` và chạy lại các ô đó, xem mục Rerun.
- Trailer `Co-Authored-By` ở `803a84f`, `288628e` vẫn chờ owner quyết.
