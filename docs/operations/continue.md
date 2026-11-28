# Bàn giao — experiment, 28/11/2026 (+07)

[CONTEXT](../../CONTEXT.md) · [Snapshot 26/11 trước khi mất account](../archive/continue-20261126-before-account-loss.md)

## Tiếp tục thí nghiệm — chốt 28/11 07:15

**Đang làm 28/11 13:05 (phiên Claude): backup checkpoint DVC hai repo, 5k trước** — trạng thái và cách chạy tiếp ở
[backup.md](backup.md#backup-28112026-đang-chạy-nền). Chuỗi nền: driver `backup-20261128.py run` (pid 101795) → `finish-5k-20261128.sh`
(lần 2 + FedProto + `5k-backup-done`) và `upload_watchdog.sh`, đều trong `runs/ablation-mh/checkpoint-backup-20261125/`;
chuỗi 10k `run-endpoints-20261128.sh` chờ `5k-backup-done`. Khi xong: commit pointer, push, làm lại bundle Git hai repo.
Bản ORICO của 80 bundle 5k đã xóa (chỉ còn trên DVC): bước dựng độ phủ checkpoint/`verify.py bundles` phải dựa vào pointer.
Tiến trình nền độc lập: collector ablation-mh (pid trong `watch-collect.pid`), loop 10k (`runs/mixed-10k/mobihet-20261128/loop.pid` ở repo 10k),
collector BERT P5 (`runs/recovery-20261120/after-boot/central-bert-p5-watch.log`).

**Còn chạy trên Kaggle (5k), không còn mục nào chờ push:**

| ô | exact ref | ghi chú |
|---|---|---|
| (hết — chỉ còn BERT centralized P5 bên dưới) | | |

- Ablation **81/81** ô R50 (A1/A2/A4 mỗi arm 27). 08:14: `hunhtnduytn/mixmh-abl-a2-case1-s41-h2/1` (macro 0,895455) và
  `levanloc1806/mixmh-abl-a4-case1-s41-h2/1` (0,895569) PASS R50, đủ hai shard `checkpoint-r050/state_rank{0,1}.pt` trên Kaggle,
  chưa có trong `backup-targets-20261128.json` → sau driver hiện tại chạy `backup.py only <ref>` cho hai ref này.
  09:15: `nlphk252/dagsnet-mhabl-a4-s41/1` PASS R50 (macro 0,905777, `state.pt` + `state_rank1.pt`), thay dòng `truantfu`;
  pull `truantfu` đã xóa bằng `purge_replaced.sh`. Ba ref này đã thêm vào danh sách (37 mục), driver đang chạy chỉ nạp 34 mục đầu.
  `datninhz/dagsnet-mhabl-a2-s42/1` PASS 07:13. MobiHet đủ **27/27** dòng; cả 11 ô MobiHet rerun 27/11 đã PASS R50,
  đủ hai shard và đã thay dòng. Dispatcher 5k đã thoát `DONE: nothing pending` (03:47).
- Collector ablation-mh chạy độc lập (`setsid`, pid trong `runs/ablation-mh/watch-collect.pid`). Collector legacy
  (codex-long-jobs `experiment-5k-legacy-collector-20261127`) không còn ref nào, có thể dừng.
  Không có waiter: phiên mới tự đọc log rồi xử lý theo các bước dưới.
- **Khi một ref ở bảng trên xong:** collector audit và (nếu PASS R50) tự ghi, `--record` tự xóa dòng cũ cùng ô.
  Rồi: kiểm hai shard (`pull_kernel_output.listing`), xóa pull của phiên bị thay bằng
  `runs/ablation-mh/purge_replaced.sh ablation-mh/<owner cũ>/<slug>` (xem dòng `replaced` trong `runs/ablation-mh/watch-logs/<owner>__<slug>.log`).
  Nếu dừng trước R50 (`NEEDS_CONTINUATION`): dispatcher không nối được loại này (chỉ nhận parent `audit == PASS`); build
  `scripts/build_mixed_l4.py --session N+1 --parent <owner>/<slug> --npy-cache runs/recovery-20261120/after-boot/mobihet-npy-preflight.json …`
  và push bằng `runs/ablation-mh/push.sh` sau ≥ 15 phút, thêm exact ref vào `active.txt`, ack `handoff.jsonl`.
- **Owner 28/11:** bỏ hẳn kết quả của phiên thiếu checkpoint (không archive) và xóa pull của chúng khi đã có bản thay PASS.
  [Log xóa](../../runs/ablation-mh/purge-no-checkpoint-20261128.log). MobiHet legacy: `scripts/replace_mobihet_rows.py <exact ref> --write`.
- DAGSNet FedProto s42: rerun `tuntrnhvminh/fl2-fedproto-s42-checkpoint/1` PASS R50, đủ hai shard, đã thay dòng
  (macro-F1 0,869656; bản gốc chỉ rank 0: 0,869797). **Còn làm:** trỏ `docs/bert/research/comparison.json` và
  `artifacts/communication-budget/run_inventory*.csv` sang rerun rồi xóa `bert/runs/maintenance-20261116/dagsnet42/fl2-fedproto`.
- **Centralized BERT P5** (owner 28/11 cho chạy trên account MCP), push 07:18, budget 11,25 giờ, nối từ checkpoint P4 đã audit
  qua dataset private mới (đủ file + byte so với bản staged trên ORICO `_shared/staging/bert-s4x-eNN/<owner>`):
  `tuntrnhvminh/exp-central-bert-s41-p5/1` (từ E27), `tkayyy/exp-central-bert-s42-p5/1` (từ E24), `nlphk252/exp-central-bert-s43-p5/1` (từ E25).
  Mỗi epoch ~67–98 phút, nên một phiên chỉ ~7–9 epoch: cần thêm P6 (và có thể P7) cho tới E50.
  Collector độc lập `runs/recovery-20261120/after-boot/collect_bert_child.py --name central-bert-p5-watch --all …`
  (log `central-bert-p5-watch.log`, receipt `central-bert-p5-watch.json`, pull vào `runs/central-20261120/pulls/exp-central-bert-s4x-p5`).
  P6: stage `central-bert/epoch_001..E` + `source_manifest.json` của P5 thành dataset `…-eE-state`, build
  `scripts/build_central_backbones.py --model bert --slug exp-central-bert-s4x-p6 --parent <P5 exact owner/slug> --parent-dataset … --parent-artifacts <pull P5> --parent-epoch E --npy-cache runs/recovery-20261120/after-boot/mobihet-npy-preflight.json --seed 4x`,
  thêm `continuation-launch-bert-s4x-p6.json`. [Ledger P5](../../runs/central-bert-p5-20261128/push-ledger.jsonl)
- `trtai0111`: owner cho probe 2×T4 sau 07:00 rồi thêm vào `rerun-accounts.txt`; chưa làm.
- **Kiểm 28/11:** `experiments_status.py --check` chỉ thiếu BERT centralized (P5 đang chạy). Compute: `compute-detail.csv` đủ mọi
  phương pháp 9 kịch bản; ablation `compute.csv` + `cost.csv` đủ mọi ô có kết quả. Byte: `artifacts/communication-budget/communication_summary.csv`
  đủ 15 phương pháp × 9 kịch bản. Còn: checkpoint coverage, round-metrics index, backup DVC (bước cuối dưới).
- Experiment 10k (cùng account): MobiHet chính thức đang chạy, xem [bàn giao 10k](../../../experiment_10k_clients/docs/operations/continue.md).

Theo dõi:

```bash
tail -n 50 -F /home/odixe/project/experiment/runs/ablation-mh/watch-collect.log \
  /home/odixe/project/experiment_10k_clients/runs/mixed-10k/mobihet-20261128/loop.log
```

## Checkpoint — kiểm kê 27/11

- Teammate đã đưa 7 pointer lên `origin/main@1cc630f`. Audit từ OneDrive xác minh **213/214** mục yêu cầu,
  **1.813** file và **205.765.586.145 byte**. Cả 213 biên nhận khớp exact ref, seed, hai shard và endpoint trong bảng chính.
  Có 201 R50 và 12 R10 fine-tune CANPerFL sau FedAvg R50. Cả ba case mixed đủ 45/45.
  [Layout và kiểm tra](backup.md#checkpoint-baseline-5k--27112026) · [Biên nhận](../../runs/checkpoint-codex-20261127/audit-summary.json)
- Ma trận chính có **327/342** checkpoint DVC. DAGSNet FedAvg/FedProx s42 đã lấy từ exact ref gốc, audit 50 CM và
  xác minh state R50 khớp config; lưu theo `dagsnet/<method>/s42/<owner>__<slug>__v1` trên ORICO.
  FedProto s42 thiếu rank 1 ở bản gốc và có rerun ở trên. 11 ô MobiHet còn chờ checkpoint thay thế.
- Ô duy nhất ngoài batch teammate là CNN-BiGRU tập trung s42, exact ref
  `qwerasdf3004/veremi-central-cnn-bigru-attention-e50/1`. Cần người có quyền kiểm tra và bổ sung file checkpoint E50 vào DVC.
- Ablation có 55 bản backup DVC từ batch trước. Các lượt audit mới giữ evidence và checkpoint đang có cho tới khi backup tương ứng đã xác minh.
- Batch teammate chỉ tải manifest và biên nhận để kiểm tra. Dữ liệu model giữ trên DVC theo yêu cầu owner.
  [Phủ theo ô](checkpoint-coverage.csv) · [Tra theo exact ref](checkpoint-runs.csv) · [Tra từng shard](checkpoint-files.csv)

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

| ô MobiHet | exact ref (push 27/11) — cả 11 PASS R50, đã ghi, đủ hai shard |
|---|---|
| bert s43 | `gialut/bert-mh-s43/1` |
| tabm s41 / s42 / s43 | `hunhtnduytn/tabm-abl-a3-s41/1`, `lcsinh/tabm-abl-a3-s42-h2/1`, `levanloc1806/tabm-abl-a3-s43/1` — cả ba PASS, đã ghi |
| sae s42 / s43 | `nlphk252/sae-abl-a3-s42-h2/1`, `ohkazu/sae-abl-a3-s43/1` — PASS, đã ghi |
| mlp s41 / s43 | `sunny0703/mlp-abl-a3-s41/1`, `sunny070306/mlp-abl-a3-s43/1` — PASS, đã ghi |
| dagsnet s42 | `tuntrnhvminh/dagsnet-abl-a3-s42-h2/1` (`replace_mobihet_rows.py`) |
| mixed-case3 s41 / s43 | `hunhtnduytn/mixmh-case3-s41/1`, `levanloc1806/mixmh-case3-s43/1` |

- 8 kernel có slug `*-abl-a3-*` push trước khi đổi tên. Kaggle không đổi tên được kernel đã push, nên owner chọn giữ chúng chạy.
  Slug cũ chỉ còn trong exact ref. Kết quả ghi là `mobihet`. Chúng chạy source MobiHet cũ, bit-exact với `ablation=none`.
  Vì vậy phải audit bằng worktree `runs/ablation-l4/legacy-audit-803a84f` (commit `803a84f^`), truyền qua `ABLATION_AUDIT` cho
  `runs/ablation-l4/collect.sh`, vì audit hiện hành chỉ nhận `none/a1/a2/a4`.
- Seed 42 của 3 kernel đó chạy 2 phiên: phiên 1 dừng ở R2, phiên 2 (`-h2`, cùng owner) chạy tới R50.
  Ledger giữ exact ref của hai phiên cho từng backbone. Cả ba parent đã PASS R2.
- Collector `runs/ablation-l4/watch_collect.py` (biến môi trường `ABLATION_NO_RECORD=1`) chỉ pull và audit. Sau khi PASS R50, thay dòng
  `mobihet` của ô đó bằng `scripts/replace_mobihet_rows.py`. `bert-mh-s43` và `mixmh-case3-*` đi qua collector ablation-mh: record sẽ báo
  `conflicting existing result` vì dòng cũ còn đó. Khi ấy chạy cùng script cho exact ref mới (receipt arm `none`).
- Dispatcher `runs/ablation-mh/rerun_dispatch.py` chạy nền (log `rerun-dispatch.log`). Khi kernel xong, collector pull và audit.
  Dispatcher đợi 15 phút rồi push phiên nối hoặc mục kế tiếp trong `rerun-pending.txt`
  (2 ablation thiếu, 2 mixed-case3 MobiHet, 14 ô ablation A1/A2/A4), lên account trong `rerun-accounts.txt`.
  Không giữ slot trước. Owner nào còn chờ phiên nối (MobiHet hoặc 10k) thì dispatcher để slot trống đó cho phiên nối.
## Sau khi collector hoàn thành — sửa index và tên

Điều kiện bắt đầu: `runs/ablation-mh/active.txt` và `runs/ablation-l4/active.txt` đều rỗng, `rerun-pending.txt` và `rerun-continue.txt` cũng rỗng,
và `rerun_dispatch.py` đã ghi `DONE`. Làm theo thứ tự sau:

1. **results.csv**: với mỗi ô trong bảng rerun ở trên, chỉ giữ dòng `mobihet` của exact ref mới (cả `experiments/<kịch bản>/results.csv` và
   `experiments/ablation/<kịch bản>/results.csv`). Xóa dòng cũ, không archive (owner 28/11). Các ô ablation A1/A2/A4 rerun cũng làm như vậy.
   Cuối cùng chạy `python scripts/experiments_status.py` và `--check`.
2. **Dời output vào layout kịch bản**: `python3 runs/orico-reorg-20261127/apply.py file-new`. Đặt output của kernel slug cũ
   `*-abl-a3-*` vào `<kịch bản>/mobihet/s<seed>/`, không đặt dưới tên A3/L4.
3. **Backup checkpoint** của các run mới (`checkpoint-backup-20261125/backup.py`, `BACKUP_JOBS`), rồi sinh lại
   `checkpoint-coverage.csv` (`runs/checkpoint-coverage-20261127/run.sh`) và cập nhật `checkpoint-runs.csv`, `checkpoint-files.csv`.
4. **Dựng lại round-metrics index**: `PYTHONPATH=. python runs/round-metrics/audit/audit.py` (env `nckh`), rồi `verify.py` cùng thư mục.
   Đổi khóa cũ `mobihet_l4` thành `mobihet`, và trỏ trường `checkpoint` của 27 run MobiHet vào bundle mới hoặc bundle còn trên DVC.
5. **Bỏ tên cũ ở local**: sau khi bước 2–4 đã xong và đã kiểm:
   - gỡ worktree audit cũ bằng `git worktree remove runs/ablation-l4/legacy-audit-803a84f`;
   - bỏ `ABLATION_AUDIT` và `ABLATION_NO_RECORD` khỏi cách chạy collector;
   - tìm lại mọi chỗ trong tài liệu còn ghi `abl-a3` hoặc `mobihet_l4`, chỉ giữ lại trong exact ref.

## Việc tay

- Nhờ người có quyền trên `qwerasdf3004` kiểm tra output E50 của `qwerasdf3004/veremi-central-cnn-bigru-attention-e50/1`.
  Nếu có checkpoint, bổ sung vào remote `onedrive` của experiment và gửi pointer hoặc commit.
- Trailer `Co-Authored-By` ở `803a84f`, `288628e` vẫn chờ owner quyết.
