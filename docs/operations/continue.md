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

Chưa push (chờ slot/quota reset **21/11 00:00 +07**): BERT s42, CNN-BiGRU s41/s43 (`build_central_backbones.py --model bert|cnn-bigru-attention --seed S`), DAGSNet s41/s43 (DAGSNet dùng `scripts/gen_notebook.py --sha 6617261 --no-wandb --args "scripts/run_central.py --width 1
--epochs 50 --compile --seed S --out /kaggle/working/fl2-central-w1-sS"` để trùng source run s42 `minhtran0601/fl2-central-w1`).
Run chưa đủ 50 epoch: nối bằng `build_central_backbones.py --parent OWNER/SLUG --parent-artifacts <pull> --parent-epoch E` cùng seed.

### MobiHet FB-L r50 (repo `../mobihet`, không phải bảng kịch bản)

Orchestrator `../mobihet/runs/fbl-r50/orchestrate.py` (PID 38919) tự hop; đã dùng `hoangdang20t` (DAGS s41h2) và `goilaaiudi`
(BERT s41h2). Nó chọn account quota lớn nhất, không biết slot của job khác ⇒ phiên này chỉ chiếm 1 slot ở `abbafan`/`bestyorn`.
File đã vá (đọc thêm `runs/fbl-r50/exclude.json` mỗi lần hop) nhưng **chưa restart** (bị chặn quyền) — xem việc tay.

### Mixed (account teammate, không có credential ở máy này)

case1 V2X-HtFL s41/42/43 (`sonlebaodac`, `minhgia2026`, `nghuay`), case3 pFedES s41/43 (`nguynkhacao`, `nghuay`).

### Compute FLOP local

`scripts/compute/run_queue.py` (count, panel 200, 2 round, group_rows 4096 / mixed 8192) → `runs/compute-20261120/`; log `queue.log`.
OOM ⇒ status `oom` ⇒ đếm trên Kaggle T4 cùng harness. CANPerFL các model: FedAvg r1 relabel r50 làm điểm bắt đầu fine-tune.
DAGSNet/BERT đã đo ở `../mobihet/runs/total-compute-20261119/` (+ `canperfl.py` phiên này).

## Bước tiếp

1. Khi centralized COMPLETE: pull (`pull_kernel_output.py` qua `kaggle_as.py`), `scripts/audit_central_backbones.py`, thêm hàng vào
   `experiments/<kịch bản>/results.csv` (status `audited`), chạy `python scripts/experiments_status.py`. PARTIAL ⇒ nối phiên.
2. Sau reset quota 21/11: push BERT s42, CNN-BiGRU s41/s43, DAGSNet s41/s43; nối BERT s41/s43.
3. Queue FLOP xong: `python scripts/compute/report.py` → `experiments/*/compute.csv`, cập nhật [compute](../research/compute.md).
4. Khi `../mobihet/runs/fbl-r50/orchestrate.log` có `ORCH_DONE` (orchestrator còn gọi `../fl_veremi_bert/scripts/run_fbl_r50.py`):
   dời artifact baseline BERT (`../fl_veremi_bert/runs/batch*`, `artifacts/*`, ~80 GB, gitignored) vào `bert/runs`, `bert/artifacts`
   (cùng ổ ⇒ `mv`), đối chiếu DVC `bert/artifacts/*.dvc`; giữ repo GitHub `fl_veremi_bert` (notebook cũ clone SHA của nó).
   Thay đổi chưa commit ở `../fl_veremi_bert` (FB-L MobiHet) thuộc `../mobihet`, không nhập.
5. Mixed 5 run teammate xong: thêm hàng audited vào `experiments/mixed-case{1,3}/results.csv`.

## Việc tay

- **Teammate (dangvanhoang1711):** chuyển sang nhánh `main` của repo `odixe06/experiment` (đường dẫn tài liệu đã đổi, xem
  [docs/README](../README.md)); gửi CM/10 metric cuối (hoặc audit JSON) của TabM/SAE/MLP/CNN-BiGRU 14 PP × 3 seed để điền đủ
  `experiments/{tabm,sae,mlp,cnn-bigru}/results.csv` (repo hiện chỉ có macro-F1); báo khi 5 run mixed xong.
- **Owner (tùy chọn):** cho phép restart orchestrator MobiHet để `exclude.json` có hiệu lực:
  `kill -TERM -38919` rồi `cd ~/project/mobihet && setsid python runs/fbl-r50/orchestrate.py >> runs/fbl-r50/orchestrate.log 2>&1 &`.
