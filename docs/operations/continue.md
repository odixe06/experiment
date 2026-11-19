# Continue — bàn giao sống, 20/11/2026 ≈ 03:15 +07 (máy tắt sau phiên này)

**Phiên sau chỉ cần đọc file này để tiếp tục.** Khi cần chi tiết: [CONTEXT](../../CONTEXT.md) (quyết định owner),
[LESSONS](../../LESSONS.md), [bài học vận hành](lessons.md), [kaggle](kaggle.md), [compute](../research/compute.md).
Repo MobiHet (phương pháp đề xuất, chain FB-L r50 + orchestrator) có bàn giao riêng: `~/project/mobihet/docs/operations/continue.md`.

## 0. Bối cảnh tối thiểu

- Repo `odixe06/experiment` (public), local `~/project/experiment` (nhánh `main`). Mỗi `experiments/<kịch bản>/` = một kịch bản:
  `dagsnet`, `bert`, `tabm`, `sae`, `mlp`, `cnn-bigru`, `mixed-case1..3`; mỗi kịch bản 14 PP + centralized × seed 41/42/43
  (mixed: chỉ 5 PP dị thể, còn lại N/A). Số liệu ghi `results.csv`, README sinh bằng `python scripts/experiments_status.py`
  (`--check` liệt kê ô thiếu). Code FeatureBERT FL ở `bert/` (cwd `bert/`). Env `nckh`: `/home/odixe/Tools/miniconda3/envs/nckh/bin`.
- **Worktree `~/project/experiment-wt`, nhánh `fix/central-ddp-compute`** (đã push, HEAD = commit cuối phiên này) chứa toàn bộ việc
  phiên 20/11 + thư mục gitignored `runs/{central,compute,count-kaggle,regress}-20261120/`, `notebooks/`. **Làm việc trong worktree
  cho tới bước 5.** Env đo/chạy local: `V4_ROOT=~/project/mobihet/knowledge V4_CACHE=~/project/mobihet/cache`.
- Kaggle: credential `~/.kaggle/accounts`, luôn qua `.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py <owner> -- …`;
  quota `kaggle_quota.py --out <json>`; pull `pull_kernel_output.py <owner>/<slug>[/<ver>] -p <dir> -j 8 [--pattern REGEX]`.
  Quota reset **21/11 00:00 +07**. Mỗi account 2 slot GPU; orchestrator MobiHet chọn account theo quota, không biết slot ⇒
  account bị job phiên này giữ slot phải nằm trong `~/project/mobihet/runs/fbl-r50/exclude.json`.
- Commit: **không** `Co-Authored-By`. Không commit credential, data, PDF.

## 1. Kernel Kaggle đang chạy (20/11 03:00)

### Centralized (ledger `runs/central-20261120/ledger.json`)

| kịch bản | seed | kernel | ghi chú |
|---|---:|---|---|
| bert | 41 / 43 / 42 | `abbafan/exp-central-bert-s41/2`, `bestyorn/exp-central-bert-s43/2`, `chiencosieuhang/exp-central-bert-s42/1` | ≈ 9 epoch/phiên ⇒ cần ~5 phiên nối |
| cnn-bigru | 41 / 43 | `gaming3/exp-central-cnnbigru-s41/1`, `goilaaiudi/exp-central-cnnbigru-s43/1` | builder `build_central_backbones.py --model cnn-bigru-attention` |
| dagsnet | 41 / 43 | `lelovo/exp-central-dagsnet-s41/1`, `hoangdang20t/exp-central-dagsnet-s43/1` | `gen_notebook.py --sha 6617261 … scripts/run_central.py` (không resume) |

Đã xong + ghi `results.csv` (status `audited`, epoch 50, audit trong kernel): MLP, SAE-CE, TabM × 3 seed; DAGSNet s42, CNN-BiGRU s42 có sẵn.

**Khi COMPLETE:** pull nhẹ để biết epoch cuối:
`--pattern '^(source_snapshot\.json|source_manifest\.json|central-[^/]+/(config\.json|epoch_[0-9]+/metrics\.json))$'` vào
`runs/central-20261120/pulls/<slug>`. Có `epoch_050` ⇒ ghi hàng `centralized` từ `epoch_050/metrics.json` (như 9 hàng đã ghi).
Thiếu ⇒ pull thêm `central-<model>/epoch_<E>/(weights.pt|resume.pt)` rồi nối phiên (account có slot + quota, cùng seed):

```bash
python scripts/build_central_backbones.py --owner NEW --model bert --slug exp-central-bert-s41-p2 --seed 41 \
  --parent abbafan/exp-central-bert-s41 --parent-artifacts runs/central-20261120/pulls/exp-central-bert-s41 --parent-epoch E --max-hours H
python .agents/skills/kaggle-training-notebook/scripts/kaggle_as.py NEW -- kaggle kernels push -p notebooks/exp-central-bert-s41-p2
```

`--max-hours` = quota account − lề (≤ 11,25). DAGSNet centralized không resume được; nếu PARTIAL phải chạy lại trên account đủ quota.

### Đếm FLOP (owner: **chỉ FLOP, mọi job ở group_rows 4096**, seed 42; ledger `runs/count-kaggle-20261120/ledger.json`)

| kernel | job | ghi chú |
|---|---|---|
| `accanniez/exp-count-cnnbigru-b` | CNN-BiGRU apfed, nilm, perfedskd, tinyproto, v2x_htfl, canperfl | |
| `bestyorn/exp-count-cnnbigru-a` | CNN-BiGRU afpha, fd_ids, lwfednids, pfedes, fedproto, fedtgp | push 03:00 thay queue local |
| `odixeuit/exp-count-bert-g4096` | BERT 14 PP | trước đo ở 2048 ⇒ thay |
| `nguyetquedo/exp-count-mixed-g4096` | mixed case1–3 × 5 PP + `dags-base-nilm` | |
| `minhtrit06/exp-count-mixed-c12`, `minhtriethihi/exp-count-mixed-c3` | mixed ở 8192 | **không dùng** (khác group_rows) |

Notebook do `scripts/compute/build_kaggle_count.py` sinh (clone SHA `5b9d0e3`, panel 200 xe seed 7, 2 worker/GPU). Output
`compute/<job>.count.json`. Local đã có 48 JSON trong `runs/compute-20261120/` (TabM/SAE/MLP 14 PP, CNN-BiGRU fedavg/fedprox).

**Khi COMPLETE:** pull `--pattern '^compute/.*\.count\.(json|log)$'` vào `runs/compute-20261120/kaggle/<slug>`, rồi
`python scripts/compute/report.py ../mobihet/runs/total-compute-20261119 runs/compute-20261120 runs/compute-20261120/kaggle/*/compute`
(thư mục sau ghi đè trước) ⇒ `experiments/*/compute.csv` + khối sinh trong [compute](../research/compute.md). Gỡ `bestyorn`
khỏi `exclude.json` của MobiHet khi `exp-count-cnnbigru-a` xong.

### Mixed (account teammate, máy này không có credential)

case1 V2X-HtFL s41/42/43 (`sonlebaodac`, `minhgia2026`, `nghuay`), case3 pFedES s41/43 (`nguynkhacao`, `nghuay`). Không đụng.

Slot lúc tắt máy (cả hai repo): `chiencosieuhang`, `gaming3`, `bestyorn`, `lelovo`, `goilaaiudi`, `hoangdang20t` 2/2 slot;
`abbafan` còn 1 slot (dành nối phiên BERT centralized / hop MobiHet). Quota reset 21/11 00:00 ⇒ account cũ có lại 30 h.

## 2. Việc dở khác

- **Hồi quy code** DAGSNet cũ `36ac7ff` vs hiện tại (20 xe, 2 round, eager, `group_rows=1024 eval_rows=4096`): FedAvg, FedProx, AFPHA
  **bit-exact**; 11 PP còn lại chưa chạy (dừng vì tắt máy). Script `runs/regress-20261120/{regress.sh,cmp.py}`: tạo checkout cũ
  `git -C ~/project/experiment worktree add runs/regress-20261120/old 36ac7ff`, sửa đường dẫn "new" trong `regress.sh` nếu đã dời worktree,
  `regress.sh fd_ids lwfednids pfedes fedproto fedtgp tinyproto nilm perfedskd apfed v2x_htfl canperfl`, rồi `python cmp.py reg <methods>`.
  Sau đó viết doc audit đồng nhất (HP, engine, method parity); khác biệt đã biết: centralized DAGSNet dùng `run_central.py`, còn lại
  `run_central_backbones.py`; mixed có projection 128D.
- **Dọn file (owner 20/11, commit `9ff5c8b`):** đã gỡ code centralized TCNN/CNN-LSTM/GRANDE/GRU-Attention/MambaCNN + windows, MobiHet/T3/
  Lite2/P21, script wave/probe/Notion one-off (khôi phục `git show 5b9d0e3:<path>`); giữ tool mixed production + `scripts/cloud_dispatch/`.
  Local đã xoá `../fl_veremi_bert/runs/sm_*`, log lỗi cũ trong `runs/compute-20261120/`. Còn xoá sau bước 5: worktree.
- **Ổ đĩa:** ORICO exFAT `/run/media/odixe/ORICO`; `../fl_veremi_bert/runs/{maintenance-20261116,pulls,transfer,t3}` đã dời sang
  `/run/media/odixe/ORICO/experiment-data/fl_veremi_bert/runs/` + symlink về (cần cắm ORICO khi đọc chúng). Giữ ≥ 30 GB trống.
- **Artifact BERT baseline** (`../fl_veremi_bert/runs/batch*`, `artifacts/*`) dời vào `bert/runs`, `bert/artifacts` khi MobiHet `ORCH_DONE`
  (orchestrator còn gọi `../fl_veremi_bert/scripts/run_fbl_r50.py`); giữ repo GitHub `fl_veremi_bert` (notebook cũ clone SHA của nó).

## 3. Bước tiếp (thứ tự)

1. Bật máy: khởi động lại orchestrator MobiHet (lệnh trong bàn giao mobihet), cắm ORICO.
2. Kiểm status 7 centralized + 4 count notebook; xử lý COMPLETE như mục 1 (nối phiên BERT ngay khi phiên trước COMPLETE).
3. Hồi quy 11 PP còn lại (mục 2), local GPU.
4. Teammate báo 5 mixed xong ⇒ thêm hàng audited vào `experiments/mixed-case{1,3}/results.csv`; `experiments_status.py`.
5. Khi không còn job phiên này: dời `runs/*-20261120/`, `notebooks/` của worktree sang `~/project/experiment/`, mở + merge PR
   `fix/central-ddp-compute` (`gh pr create --base main`, body không attribution AI), `git worktree remove ~/project/experiment-wt`, `git pull`.

## 4. Việc tay

- **Teammate (dangvanhoang1711):** chuyển sang `main` của `odixe06/experiment`; gửi CM / 10 metric (hoặc audit JSON) TabM/SAE/MLP/CNN-BiGRU
  14 PP × 3 seed (repo chỉ có macro-F1, status `marker_f1`); báo khi 5 run mixed xong.
