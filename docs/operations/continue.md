# Continue — bàn giao sống, 20/11/2026 (giờ VN)

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md) → trang này.
Repo local: `~/project/experiment` (symlink `fl_dagsnet_veremi_v2` → `experiment`); GitHub `odixe06/experiment` (URL cũ redirect).
Worktree tạm `~/project/experiment-wt` (nhánh `fix/central-ddp-compute`) giữ `runs/central-20261120/`, `runs/compute-20261120/`,
`notebooks/` của phiên này — dời sang `experiment/` rồi `git worktree remove` khi queue compute xong.
Bàn giao trước khi gộp repo: [continue-20261120-before-reorg](../archive/continue-20261120-before-reorg.md) (ledger mixed/backbone của teammate).

## Đang chạy

### Centralized (phiên này, private 2×T4, builder `scripts/build_central_backbones.py --seed S --max-hours H`)

Ledger: `runs/central-20261120/ledger.json` (ref, notebook SHA256). Version 1 cả 11 ERROR ở cell test: snapshot nhúng thiếu
`core/tabm.py` (lỗi builder có sẵn; đã sửa + builder tự chạy test trên snapshot trước khi ghi notebook). Version 2: SAE-CE ×3 ERROR
(DDP: decoder không vào loss) và TabM s42 ERROR (race `config.json` rank 1) ⇒ sửa runner + test 2 rank, push **version 3**.

| kịch bản | seed | kernel | budget h | ghi chú |
|---|---:|---|---:|---|
| bert | 41 | `abbafan/exp-central-bert-s41/2` | 11,25 | ≈ 9 epoch/phiên ⇒ cần ~5 phiên nối (`--parent`) |
| bert | 43 | `bestyorn/exp-central-bert-s43/2` | 11,25 | như trên |
| tabm | 41/42/43 | `anhtruong1909/…-tabm-s41/2`, `minhtran0601/…-tabm-s42/3`, `nguyenminhtran060113/…-tabm-s43/2` | 3,4 / 3,05 / 2,8 | budget = quota account − lề |
| mlp | 41/42/43 | `odixeuit/…-mlp-s41/2`, `nguyetquedo/…-mlp-s42/2`, `annhtnt/…-mlp-s43/2` | 2,75 / 2,65 / 2,5 | |
| sae (sae-ce) | 41/42/43 | `thuyanh316/…-saece-s41/3`, `takids1909/…-saece-s42/3`, `keocaosu/…-saece-s43/3` | 2,4 / 2,2 / 1,45 | |

Thêm 20/11 ~02:00 (3 account mới + slot thứ 2 của account đang chạy chain orchestrator — nó không chọn lại account đó):
`chiencosieuhang/exp-central-bert-s42/1`, `gaming3/exp-central-cnnbigru-s41/1`, `goilaaiudi/exp-central-cnnbigru-s43/1`,
`lelovo/exp-central-dagsnet-s41/1`, `hoangdang20t/exp-central-dagsnet-s43/1` ⇒ **16/16 centralized đã push**. DAGSNet dùng
`scripts/gen_notebook.py --sha 6617261 --no-wandb --args "scripts/run_central.py --width 1 --epochs 50 --compile --seed S
--out /kaggle/working/fl2-central-w1-sS"` để trùng source run s42 `minhtran0601/fl2-central-w1`).
Account còn trống sau đợt này: slot thứ 2 của chiencosieuhang, gaming3, lelovo, abbafan, bestyorn (để dành cho hop orchestrator / phiên nối BERT).
Run chưa đủ 50 epoch: nối bằng `build_central_backbones.py --parent OWNER/SLUG --parent-artifacts <pull> --parent-epoch E` cùng seed.

### MobiHet FB-L r50 (repo `../mobihet`, không phải bảng kịch bản)

Orchestrator `../mobihet/runs/fbl-r50/orchestrate.py` (PID 38919) tự hop; đã dùng `hoangdang20t` (DAGS s41h2) và `goilaaiudi`
(BERT s41h2). Nó chọn account quota lớn nhất, không biết slot của job khác ⇒ phiên này chỉ chiếm 1 slot ở `abbafan`/`bestyorn`.
File đã vá (đọc thêm `runs/fbl-r50/exclude.json` mỗi lần hop). 02:17 hop BERT s43 → `gaming3` **BLOCKED** vì ổ đầy
(Errno 28 khi copy state vào `artifacts/fbl-hop/…-s43-r50/upload`; chưa có `create-intent.json` ⇒ chạy lại sẽ làm lại sạch).
Owner 20/11 cho restart khi ổ có chỗ: `kill -TERM -38919`; `cd ~/project/mobihet && setsid python runs/fbl-r50/orchestrate.py >> runs/fbl-r50/orchestrate.log 2>&1 &`
(ghi PID mới vào `runs/fbl-r50/orchestrate.pid`); trước đó ghi `exclude.json` = account phiên này đang chiếm 2 slot.

### Mixed (account teammate, không có credential ở máy này)

case1 V2X-HtFL s41/42/43 (`sonlebaodac`, `minhgia2026`, `nghuay`), case3 pFedES s41/43 (`nguynkhacao`, `nghuay`).

### Ổ đĩa (20/11 ~02:15 đầy 100 %)

Ổ ORICO (exFAT) mount `/run/media/odixe/ORICO`; `scripts`: dời `../fl_veremi_bert/runs/{maintenance-20261116,pulls,transfer,t3}` (~80 GB,
artifact baseline/T3 đã audit) sang `/run/media/odixe/ORICO/experiment-data/fl_veremi_bert/runs/`, symlink về chỗ cũ (kiểm số file + byte
trước khi xoá nguồn). Không đụng `artifacts/fbl-*`, `runs/fbl-r50` (orchestrator). exFAT: không symlink/quyền/hardlink bên trong.

### Compute FLOP

Owner 20/11: **mọi job đếm ở group_rows 4096** (một packing cho mọi phương pháp/kịch bản). `scripts/compute/run_queue.py` → `runs/compute-20261120/`.
Local: TabM/SAE/MLP xong 14/14; CNN-BiGRU nửa đầu đang chạy (task nền). Kaggle `scripts/compute/build_kaggle_count.py` (2 T4, 1 worker/GPU):
`accanniez/exp-count-cnnbigru-b` (CNN nửa sau, 4096 — dùng được); `minhtrit06/exp-count-mixed-c12`, `minhtriethihi/exp-count-mixed-c3`
chạy mixed ở 8192 (trước khi có `--group-rows`) ⇒ **không dùng**. Còn phải đếm lại ở 4096 trên Kaggle: BERT 14 PP (`bert-*`, trước đếm 2048),
DAGSNet NILM (`dags-base-nilm`, trước fallback 2048), mixed 15 (`run_mixed_fl.py --group-rows 4096`); commit `fix/central-ddp-compute`
chưa push được lúc ổ đầy — push trước khi build notebook (notebook clone SHA).
OOM ⇒ status `oom` ⇒ đếm trên Kaggle T4 cùng harness. CANPerFL các model: FedAvg r1 relabel r50 làm điểm bắt đầu fine-tune.
DAGSNet/BERT đã đo ở `../mobihet/runs/total-compute-20261119/` (+ `canperfl.py` phiên này).

### Dọn file bỏ (owner 20/11, commit `9ff5c8b`)

Đã gỡ khỏi Git: code centralized TCNN/CNN-LSTM/GRANDE/GRU-Attention/MambaCNN + windows, toàn bộ MobiHet/T3/Lite2/P21 (`methods/`,
`bert/`, `scripts/build_mobihet_*`), script launch/check wave đã xong, probe, Notion one-off. Khôi phục: `git show 5b9d0e3:<path>`.
Giữ: tool mixed production + `scripts/cloud_dispatch/` (5 run teammate còn chạy; `audit_mixed_results.py` import `dispatcher`).
Local đã xoá: `../fl_veremi_bert/runs/sm_*` (770 MB smoke), 26 log ImportError trong `runs/compute-20261120/`,
`../mobihet/runs/total-compute-20261119/bert-nilm.time.perturbed-by-pytest-1957.json`.
**Còn xoá khi xong việc:** `scratchpad/reg/` + `scratchpad/old/` sau khi so hồi quy; worktree `experiment-wt` sau khi dời
`runs/{central,compute,count-kaggle}-20261120/` + `notebooks/` về `experiment/` (bước 6).

## Bước tiếp

1. Khi centralized COMPLETE: pull (`pull_kernel_output.py` qua `kaggle_as.py`), `scripts/audit_central_backbones.py`, thêm hàng vào
   `experiments/<kịch bản>/results.csv` (status `audited`), chạy `python scripts/experiments_status.py`. PARTIAL ⇒ nối phiên.
2. MLP centralized s41/42/43 COMPLETE (02:0x) — pull + audit. Nối BERT s41/s42/s43 (≈ 9 epoch/phiên).
3. Queue FLOP xong: `python scripts/compute/report.py` → `experiments/*/compute.csv`, cập nhật [compute](../research/compute.md).
4. Khi `../mobihet/runs/fbl-r50/orchestrate.log` có `ORCH_DONE` (orchestrator còn gọi `../fl_veremi_bert/scripts/run_fbl_r50.py`):
   dời artifact baseline BERT (`../fl_veremi_bert/runs/batch*`, `artifacts/*`, ~80 GB, gitignored) vào `bert/runs`, `bert/artifacts`
   (cùng ổ ⇒ `mv`), đối chiếu DVC `bert/artifacts/*.dvc`; giữ repo GitHub `fl_veremi_bert` (notebook cũ clone SHA của nó).
   Thay đổi chưa commit ở `../fl_veremi_bert` (FB-L MobiHet) thuộc `../mobihet`, không nhập.
5. Mixed 5 run teammate xong: thêm hàng audited vào `experiments/mixed-case{1,3}/results.csv`.
6. Queue local + 3 notebook đếm Kaggle xong: dời `runs/{central,compute,count-kaggle}-20261120/`, `notebooks/` sang
   `~/project/experiment/`, mở + merge PR nhánh `fix/central-ddp-compute`, `git worktree remove ~/project/experiment-wt`, pull main.

## Việc tay

- **Teammate (dangvanhoang1711):** chuyển sang nhánh `main` của repo `odixe06/experiment` (đường dẫn tài liệu đã đổi, xem
  [docs/README](../README.md)); gửi CM/10 metric cuối (hoặc audit JSON) của TabM/SAE/MLP/CNN-BiGRU 14 PP × 3 seed để điền đủ
  `experiments/{tabm,sae,mlp,cnn-bigru}/results.csv` (repo hiện chỉ có macro-F1); báo khi 5 run mixed xong.
- Hồi quy DAGSNet code cũ `36ac7ff` vs hiện tại (14 PP, 20 xe, 2 round, eager) đang chạy: `scratchpad/regress.sh`, kết quả `scratchpad/reg/`.
