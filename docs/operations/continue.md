# Continue — bàn giao sống, 20/11/2026 ≈ 09:35 +07 (máy tắt sau phiên này)

**Phiên sau chỉ cần đọc file này.** Chi tiết khi cần: [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [bài học vận hành](lessons.md),
[kaggle](kaggle.md), [sổ lỗi kernel](notebook-errors.md), [compute](../research/compute.md),
L4: `~/project/experiment-l4/docs/research/mobihet-l4.md`. MobiHet có bàn giao riêng (`~/project/mobihet/docs/operations/continue.md`),
**nhưng không còn kernel MobiHet nào chạy** — mọi việc sống nằm ở file này. Bản bàn giao 06:40 trước khi viết lại: `git diff` file này.

## 0. Bối cảnh tối thiểu

- Repo `odixe06/experiment`: `experiments/<kịch bản>/` (`dagsnet`, `bert`, `tabm`, `sae`, `mlp`, `cnn-bigru`, `mixed-case1..3`), mỗi kịch bản
  14 PP + centralized + **MobiHet L4 (cột mới, method `mobihet_l4`, view `client`)** × seed 41/42/43. Số ghi `results.csv`, README sinh bằng
  `python scripts/experiments_status.py` (`--check` liệt kê ô thiếu). Env: `/home/odixe/Tools/miniconda3/envs/nckh/bin` (luôn `export PATH=…:$PATH`;
  conda.sh hỏng nên `kaggle` không có trong PATH mặc định).
- **Hai worktree, cả hai chưa merge `main`:**
  - `~/project/experiment-wt`, nhánh `fix/central-ddp-compute` — **số liệu sống** (results/compute/README/docs).
    `runs/{central,compute,count-kaggle,regress}-20261120/`, `notebooks/` gitignored.
  - `~/project/experiment-l4`, nhánh `feat/mobihet-l4` — **code L4** (`methods/mobihet*.py`, `scripts/build_l4_fl.py`, `scripts/audit_l4.py`,
    `core/local.py`, `scripts/run_fl.py`); `runs/l4-20261120/` (ledger, pull, `collect.sh`), `runs/l4-port/` (smoke), `notebooks/*-l4-s*`
    gitignored. `experiments/scenarios.json` hai nhánh giống hệt. Số L4 ghi vào `results.csv` của **experiment-wt**.
- Kaggle: luôn `.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py <owner> -- …`; quota `kaggle_quota.py --out <json>`;
  pull `kaggle_as.py <owner> -- python pull_kernel_output.py <owner>/<slug> -p <dir> -j 16 [--pattern REGEX]` (thiếu `kaggle_as` ⇒ 403).
  Log kernel đang chạy: `timeout 40 kaggle_as.py <o> -- kaggle kernels logs -f <o>/<slug> > file` (ghi ra file, không pipe).
  **Quota reset 21/11 00:00 +07.** Mỗi account 2 slot GPU. `dinhan25`: owner cho dùng (20/11); kiểm slot trước (dự án RCD dùng chung).
- Owner 20/11: **bỏ orchestrator MobiHet** (`~/project/mobihet/runs/fbl-r50/orchestrate.py` — không chạy lại); Claude tự điều phối,
  tận dụng mọi slot/quota còn trống.
- Commit: **không** `Co-Authored-By`. Không push GitHub `main` khi owner chưa bảo.

## 1. Kernel Kaggle đang chạy (09:30)

| kernel | việc | khi COMPLETE |
|---|---|---|
| `gaming3/cnnbigru-l4-s41`, `hoangdang20t/cnnbigru-l4-s42` (v2), `bestyorn/cnnbigru-l4-s43` | L4 CNN-BiGRU, push 08:07–08:25, ~350 s/round ⇒ xong ≈ 13:00–13:30 | `~/project/experiment-l4/runs/l4-20261120/collect.sh <owner/slug> cnn-bigru` |
| `abbafan/exp-central-bert-s41/2`, `bestyorn/exp-central-bert-s43/2`, `chiencosieuhang/exp-central-bert-s42/1` | centralized BERT phiên 1, ~9 epoch/phiên ⇒ xong ≈ 13:40, cần ~5 phiên | mục 1a |
| `gaming3/exp-central-cnnbigru-s41/1` | centralized CNN-BiGRU s41 | mục 1a (như s43 đã ghi) |

Nếu L4 CNN-BiGRU dừng trước r50 (`RUN_STATUS.json` `complete: false`): nối phiên
`cd ~/project/experiment-l4 && python scripts/build_l4_fl.py --owner NEW --backbone cnn-bigru-attention --seed S --session 2 --parent <owner>/<slug> --max-hours H`
rồi `kaggle_as.py NEW -- kaggle kernels push -p notebooks/cnnbigru-l4-sS-h2`; collect dùng notebook dir `notebooks/cnnbigru-l4-sS-h2`
(`collect.sh` lấy slug từ ref). Kiểm actual mount của parent (sổ lỗi: kernel source có thể bị từ chối mà push exit 0).

### 1a. Centralized khi COMPLETE

Pull nhẹ vào `runs/central-20261120/pulls/<slug>`:
`--pattern '^(source_snapshot\.json|source_manifest\.json|central-[^/]+/(config\.json|epoch_[0-9]+/metrics\.json))$'`.
Có `epoch_050` + log kernel có `50 EPOCHS AUDITED` + `config.json` đúng seed ⇒ thêm hàng
`centralized,<seed>,global,50,<10 metric>,audited,"kaggle <ref> (in-kernel audit_central_backbones, 50 epoch)"` vào `experiments/<kịch bản>/results.csv`
(như CNN-BiGRU s43). Thiếu ⇒ pull thêm `central-<model>/epoch_<E>/(weights.pt|resume.pt)` rồi nối phiên (cùng seed):

```bash
python scripts/build_central_backbones.py --owner NEW --model bert --slug exp-central-bert-s41-p2 --seed 41 \
  --parent abbafan/exp-central-bert-s41 --parent-artifacts runs/central-20261120/pulls/exp-central-bert-s41 --parent-epoch E --max-hours H
python .agents/skills/kaggle-training-notebook/scripts/kaggle_as.py NEW -- kaggle kernels push -p notebooks/exp-central-bert-s41-p2
```

`--max-hours` = quota account − lề (≤ 11,25). Slot trống lúc tắt máy: `dinhan25` 2 (9,2 h), `goilaaiudi` 2 (18,4 h), `lelovo` 2,
`chiencosieuhang` 1, `abbafan` 1, `hoangdang20t` 1; 6 account nhỏ (anhtruong1909, annhtnt, minhtran0601, nguyenminhtran060113,
thuyanh316, takids1909) quota < 1,5 h tới 00:00. Đọc quota lại trước khi push.

## 2. Đã xong phiên 20/11 sáng (đã ghi results.csv / docs)

- **MobiHet L4 r50: 6/6 chain audit PASS.** DAGSNet .90550 ± .00004 (`~/project/mobihet/runs/fbl-r50/audit.json`), BERT .91260 ± .00043
  (`~/project/fl_veremi_bert/runs/fbl-r50/S4*h2/audit.json`; vượt mọi baseline BERT, CANPerFL .91174); byte ~0,355–0,359 MB/xe/50r, compute
  0,997× FedAvg. Hàng `mobihet_l4` (10 metric từ CM r50) ghi vào `experiments/{dagsnet,bert}/results.csv`. DAGSNet s43: hop orchestrator
  timeout khi upload 9,9 GB ⇒ hop thủ công (`~/project/mobihet/runs/fbl-r50/manual-hop-s43h2.sh`).
- **L4 TabM/SAE/MLP 9/9 audit PASS** (`scripts/audit_l4.py`: round 1–50, hp L4 + seed, source = notebook, method sha = MobiHet, LR = chain
  DAGSNet L4, CM r50 → 10 metric = log, 9.347.679 dòng test): TabM .83272 ± .00367, SAE .71947 ± .00671, MLP .87281 ± .00129 — **dưới
  FedAvg** cùng backbone (.84461 / .75778 / .87759). F1 lớp TB 3 seed: lớp 13 timeDelay SAE .000 / TabM .002 / MLP .099 / DAGSNet .338;
  lớp 11 .238 / .420 / .682 / .842 ⇒ model càng nhỏ, lớp học chậm càng sụp. Chưa so per-class với baseline (thiếu CM baseline, mục 4).
  Ledger 12 kernel: `~/project/experiment-l4/runs/l4-20261120/ledger.json`.
- Centralized: DAGSNet s41/s43 (CM tính lại = log), CNN-BiGRU s43 ⇒ DAGSNet 3/3, CNN-BiGRU 2/3, MLP/SAE/TabM 3/3, BERT 0/3.
- Compute FLOP: 4 notebook đếm 4.096 xong ⇒ `experiments/*/compute.csv` mọi kịch bản + bảng [compute](../research/compute.md).
- Hồi quy code DAGSNet `36ac7ff` vs hiện tại: 12/14 bit-exact, V2X-HtFL lệch có chủ đích, CANPerFL không deterministic trên GPU —
  [audit](../research/dagsnet/audits/2026-11-20-code-regression.md). Checkout cũ `runs/regress-20261120/old` (git worktree detach; xoá bằng
  `git -C ~/project/experiment worktree remove ~/project/experiment-wt/runs/regress-20261120/old` khi không cần).
- Port L4 kiểm: DAGSNet bit-exact với `../mobihet`, FedAvg không đổi, resume TabM exact (`runs/l4-port/`). Lỗi kernel v1
  (`test_sae_fl.py` không nhúng) đã ghi [sổ lỗi](notebook-errors.md).

## 3. Bước tiếp (thứ tự)

1. Bật máy: kiểm status 7 kernel mục 1 (không có orchestrator, không có job local). Collect L4 CNN-BiGRU ×3 → `results.csv`,
   `python scripts/experiments_status.py`.
2. Centralized BERT ×3 COMPLETE ⇒ nối phiên ngay (mục 1a); CNN-BiGRU s41 centralized ⇒ ghi hàng.
3. Báo owner bảng L4 đủ 6 kịch bản đơn model + phân tích lớp 13/11 theo cỡ model; hỏi owner có thử cải tiến cho model nhỏ không
   (không tự tune hp L4 đã khóa).
4. Mixed L4 (owner: làm sau đơn model): cần Q/bootstrap gộp xuyên nhóm model (`methods/mixed.py` mỗi nhóm một Layout) — thiết kế rồi hỏi owner.
5. MobiHet docs: cập nhật `~/project/mobihet/docs/research/{screening-fbl,comparison,tried-methods-index}.md` với kết quả r50 (mục 2).
6. Cuối: merge `feat/mobihet-l4` và `fix/central-ddp-compute` (PR `gh pr create --base main`, body không attribution AI), dời `runs/*`,
   `notebooks/` sang `~/project/experiment/`, `git worktree remove` hai worktree — chỉ khi owner đồng ý và không còn job.

## 4. Việc tay

- **Teammate (dangvanhoang1711):** chuyển sang `main` của `odixe06/experiment`; gửi CM / 10 metric TabM/SAE/MLP/CNN-BiGRU 14 PP × 3 seed
  (repo chỉ có macro-F1 — cần để so per-class với L4); báo khi 5 run mixed xong (case1 V2X-HtFL ×3, case3 pFedES s41/s43, account teammate).
- (tuỳ) `rm -rf ~/project/mobihet/artifacts/fbl-hop/mh-fbl-l4-top-s4{1,2,3}-r50/upload` (bản trùng dataset đã lên Kaggle, ~30 GB).
- (quyết định) push GitHub: nhánh `fix/central-ddp-compute`, `feat/mobihet-l4`; `~/project/mobihet` main; `../fl_veremi_bert` (còn thay đổi
  FB-L chưa commit — hết chain BERT nên đã an toàn để commit).
