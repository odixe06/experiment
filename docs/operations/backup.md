# Git / DVC backup record

## Backup 28/11/2026 (đang chạy)

- Git: `main` push `1cc630f..aa9048f` lên GitHub (rebase lên commit pointer của teammate `1cc630f`, không trailer). Bundle chưa làm lại.
- DVC (remote mặc định `onedrive` = `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/experiment/dvc-storage`):
  7 pointer `artifacts/5k-final-checkpoints/*.dvc`: `dvc status -c` = 1.611 `deleted` (có trên remote, không có cache local), 0 `new`/`missing`.
- Checkpoint R50 của 34 rerun 27–28/11 (85,0 GiB): `runs/ablation-mh/checkpoint-backup-20261125/backup-20261128.py run` chạy nền
  (log `backup-20261128.log`, danh sách `backup-targets-20261128.json`). Mỗi ref: pull hai shard vào ORICO, `dvc add --to-remote`,
  `dvc status -c`, pointer `artifacts/mobihet-checkpoints-2026-11-25/<owner>/<slug>/v1.dvc`. Trước đó phải chạy
  `apply.py file-new` (backup.py chỉ nhận bundle trên ORICO; 28/11 đã dời 43 thư mục).
  10:05: 6/34 DVC_VERIFIED, pointer đã commit (datle211206 cnnbigru-a1-s42, dagsnet-a1-s41, mixmh-a2-case1-s43; datninhz
  cnnbigru-a1-s43, dagsnet-a2-s42; fuocs2k6 tabm-a2-s42). `datninhz/mixmh-abl-a4-case1-s42/1` DVC_BLOCKED (check_space ROOT).
  **Driver đang tạm dừng** (`kill -STOP` pid 101795 và con `backup.py only fuocs2k6/mixmh-abl-a1-case1-s42/1` pid 116368):
  cache rclone `~/.cache/rclone/vfs/onedrive` giữ 54 GiB object chưa upload, ổ gốc còn 16–17 GB, upload ~1 MB/s.
  Chạy tiếp bằng `kill -CONT 116368 101795` khi cache < 15 GiB. Thêm sau: 3 ref PASS sau khi driver nạp danh sách
  (`hunhtnduytn/mixmh-abl-a2-case1-s41-h2/1`, `levanloc1806/mixmh-abl-a4-case1-s41-h2/1`, `nlphk252/dagsnet-mhabl-a4-s41/1`;
  mục 35–37 của danh sách) → chạy `backup-20261128.py size` rồi `run` lần nữa (ref đã VERIFIED được bỏ qua).
  ORICO trống 53 GB: các ref cuối có thể bị `check_space` chặn.
- Khi xong: `git add artifacts/mobihet-checkpoints-2026-11-25/**/*.dvc`, commit, push, tạo lại bundle; FedProto DAGSNet s42 rerun
  (ORICO `dagsnet/fedproto/s42/tuntrnhvminh__fl2-fedproto-s42-checkpoint__v1`, 8,5 GB) cần `dvc add --to-remote` riêng.

## Dữ liệu local theo kịch bản — 27/11/2026

ORICO `experiment-data/experiment/` sắp theo cùng trục với `experiments/<kịch bản>`. BERT và DAGSNet đều là kịch bản của repo này:

```
<kịch bản>/<phương pháp>/s<seed>/<phiên>/        dagsnet bert tabm sae mlp cnn-bigru mixed-case1..3
ablation/<kịch bản>/<arm>/s<seed>/<owner>__<slug>/
bert/_dvc/                                         thư mục DVC nguyên khối: baseline-pulls, t3-pulls, research-evidence-20261116
_shared/                                           round-metrics, archives, git, staging, compute, bản ghi batch (ablation-mh, ablation-l4, central-20261120, bert)
_archive/                                          phương pháp đã loại hoặc thử nghiệm: L4 có LA (ablation-l4-la, l4-la), bert/t3 và artifact fbl/lite2, dev
```

- `<phiên>` giữ nguyên tên thư mục pull, ví dụ `flb-afpha-s41-p2` hoặc `exp-central-bert-s41-p4`. Evidence Kaggle có dạng
  `<owner>__<slug>__v<N>`. MobiHet (A3 cũ) nằm ở `<kịch bản>/mobihet/`, nên không lẫn với ablation.
- Ổ exFAT không có symlink. Vì vậy đường dẫn cũ trên SSD giữ dạng link: `bert/runs/pulls/<run>`, `bert/artifacts/<x>`,
  `runs/ablation-{mh,l4}/pulls/<owner>/<slug>`, `runs/round-metrics`, `runs/central-20261120/pulls/<run>`. Pointer DVC và hash không đổi.
- Mọi đơn vị đều dời bằng `rename` trên cùng ổ. Mỗi đơn vị được đếm số file và byte trước và sau khi dời.
  Journal ghi tại `runs/orico-reorg-20261127/moves.jsonl`, plan tại `plan.tsv`, script là `plan.py`, `apply.py` và `verify.py`.
  `run.json`/`index.json` của round-metrics, `checkpoint-runs.csv` và `checkpoint-files.csv` đã đổi sang đường mới.
  `audit.py` của round-metrics tra evidence qua journal.
- Collector mới vẫn pull vào `runs/ablation-mh/pulls/<owner>/<slug>` trên SSD. Sau audit, chạy
  `python3 runs/orico-reorg-20261127/apply.py file-new` để đưa run vào cây kịch bản rồi để lại link. Làm bước này trước backup checkpoint,
  vì `backup.py` yêu cầu bundle nằm trên ORICO.
- Kết quả: đã dời 389/389 đơn vị (76.795 file, 312.656.653.995 byte), không còn symlink SSD nào gãy. Số file và byte từng đơn vị khớp
  trước và sau. Chênh lệch duy nhất là `_shared/round-metrics`, cùng 1.047 file nhưng +7.217 byte vì đã ghi lại đường dẫn trong `run.json`.
  974/974 file có hash trong round-metrics khớp SHA-256. Đã chạy lại `verify.py bundles` sau khi khởi động máy ngày 27/11: 3.028/3.028 file của 71 bundle khớp SHA-256 và kích thước trong `checkpoint-index.json`, 0 lỗi.
- Các mục cũ bên dưới ghi đường dẫn ORICO trước ngày 27/11. Tra đường mới bằng `moves.jsonl`.
- Backup checkpoint chạy song song từ 27/11. `BACKUP_JOBS` (mặc định 3) đặt số bundle được pull và đẩy cùng lúc, mỗi bundle chạy
  `rclone --transfers 16`. Chỉ luồng chính ghi `backup-results.json`. Chỗ trống staging trên ORICO tính theo kích thước bundle × JOBS.
  Đã thử 2 bundle đã có trên DVC: pointer trùng byte, `rclone check` đạt. Bundle mixed chỉ có 2 file lớn, nên tốc độ tăng
  chủ yếu nhờ chạy nhiều bundle cùng lúc.

## Checkpoint baseline 5k — 27/11/2026

Commit `1cc630f` trên `origin/main` chứa 7 pointer theo backbone. Nhánh Codex cũ đã nằm trong lịch sử
checkout hiện tại. Giữ nguyên pointer và nội dung DVC đã đẩy, ánh xạ `case1..3` thành `mixed-case1..3`
và `cnn-bigru-attention` thành `cnn-bigru` trong registry. Mỗi ô tra theo kịch bản, phương pháp, seed và exact ref.
Mỗi shard có đường dẫn bên trong bundle, kích thước và MD5. [Phủ theo ô](checkpoint-coverage.csv) ·
[Danh mục file](checkpoint-files.csv)

- Kiểm MD5 7 directory manifest, đối chiếu tồn tại/kích thước của toàn bộ 1.813 mục với kho OneDrive và kiểm
  MD5 của 213 file `checkpoint-source.json`. Exact ref, endpoint và kích thước hai shard của mỗi run khớp.
  Backup có tổng 205.765.586.145 byte. Hai shard mẫu MLP AFPHA s41 đọc trực tiếp từ OneDrive và khớp MD5, SHA-256.
  Cả 426 shard của batch nằm trên DVC và không có trong cache DVC local.
- Các bundle giữ cấu trúc nội bộ `<method>/seed<N>/`. Tra pointer và entry từ registry trước khi khôi phục.
  Cả 168 ô FL của bốn backbone đơn và 45 ô mixed đã có đủ shard. CNN-BiGRU tập trung s42 cần checkpoint E50 riêng.
- Archive ORICO có 2 checkpoint gốc R50 đã audit lại của DAGSNet FedAvg và FedProx s42. Giữ bản này tới khi có backup
  đã xác minh. Chỉ dọn payload local của run có backup DVC đủ và đã xác minh. Những file audit nhỏ dùng cho lần đối soát sau.

## Checkpoint MobiHet — 25/11/2026

- Dọn 1.653 file tạm DVC ngày 25/11 (38,668 GiB) vào thùng rác OneDrive. Từng file khớp kích thước và checksum OneDrive với blob ở đường dẫn DVC chuẩn. Hai manifest lịch sử ngoài danh mục checkpoint đã có mục thiếu trước đợt dọn.
- BERT dùng chung remote và cấu hình DVC của repo cha. Đã đối chiếu cả 9.207 blob nguồn và thử fetch từ thư mục con. Git bundle BERT nằm ở `experiment/bert` trên OneDrive. Cache đang dùng là `experiment/.dvc/cache` trên SSD.
- Ngày 26/11, thư mục OneDrive của repo đổi từ `fl_dagsnet_veremi_v2` thành `experiment`. Remote `onedrive` trỏ tới `NCKH/veremi_build/experiment/dvc-storage`.
- Đích DVC `onedrive` theo `.dvc/config`. Checkpoint local nằm dưới `pulls/<owner>/<slug>/checkpoints/vN/`
  trên ORICO, giữ nguyên đường dẫn output Kaggle. DVC pointer theo cùng owner/slug/version ở `artifacts/mobihet-checkpoints-2026-11-25/`.
- Ngày 26/11 đã xác minh **52/52 phiên, 1.854 checkpoint** trên OneDrive, 157.189.865.329 byte gồm metadata.
  Mọi directory manifest khớp MD5 và mọi object khớp kích thước gốc. Phục hồi 5 object 0 byte từ bản gốc
  rồi so MD5 đọc từ cloud hoặc QuickXor do OneDrive trả. Object 0 byte giữ tại `repair-quarantine/2026-11-25/`.
- `checkpoint-index.json` giữ exact ref, byte và SHA-256 từng file. `inventory.json`, `backup-results.json`
  và `cloud-audit-final.json` trong `runs/ablation-mh/checkpoint-backup-20261125/` giữ phạm vi, receipt và kết quả kiểm.
- [Tra checkpoint theo run](checkpoint-runs.csv) · [Danh sách file và MD5](checkpoint-files.csv).
  Ref mới dùng `owner/slug/version`. Danh mục DVC lịch sử DAGSNet/BERT gồm 158 phiên và 6.341 checkpoint.
- Ngoài batch, 8.886 đường dẫn local đã khớp DVC BERT. Đã bổ sung 1.944 đường dẫn thuộc 960 object,
  89.526.542.825 byte (83,4 GiB) trên cùng OneDrive. Rclone báo 960 object khớp, 0 sai lệch.
  MD5 directory manifest `56f074897cac6505d0238598e38b98c2.dir` khớp bytes đọc từ OneDrive.
  [DVC pointer bổ sung](../../artifacts/checkpoints-local-2026-11-26.dvc) · [Bàn giao](continue.md)
- Khôi phục MobiHet/DAGSNet bằng `dvc pull <dvc_pointer>` từ repo gốc. Với `dvc_project=bert`, chạy
  `cd bert` rồi `dvc pull artifacts/<bundle>.dvc`. Cột `checkpoint` là đường dẫn bên trong bundle.
- Bundle bổ sung giữ đường dẫn tương đối dưới `checkout/` và `orico/`. Hai thư mục tương ứng repo và
  `ORICO/experiment-data/` tại lần backup. Biên nhận nằm tại `runs/ablation-mh/checkpoint-backup-20261125/local-supplement-receipt.json`.

### Dọn L4 lịch sử trên ORICO — 26/11/2026

- Đã xóa đúng 12 checkpoint của sáu run L4 có logit adjustment trong `experiment-data/experiment/recovered-checkpoints-20261120/`, thu hồi 15,3 GiB. Giữ log và metadata run, cùng checkpoint của phương án chính thức A3 (`la_mode=off`).
- Đã đối chiếu đường dẫn và MD5 checkpoint với manifest của bundle bổ sung trên OneDrive. Lưu pointer và manifest trên SSD tại `runs/orico-l4-cleanup-20261126/` và trên ORICO tại `experiment-data/experiment/archives/checkpoints-local-2026-11-26/`. Biên nhận xóa là `runs/orico-l4-cleanup-20261126/local-delete.json`.

### Dọn cache BERT và bản staging trùng trên ORICO — 26/11/2026

- Đã xóa cache BERT cũ ở `experiment-data/fl_veremi_bert/dvc-cache/` sau khi kiểm tra lại đủ 9.207 blob trong kho chung theo kích thước và checksum OneDrive. Khôi phục theo [tài liệu BERT](../bert/operations/backup.md).
- Đã xóa 65 file trong `checkpoint-dvc-resume-xktq98dq/objects/` và `checkpoint-dvc-stage-ykg_7ia0/objects/` sau khi đối chiếu kích thước và QuickXor với blob trong kho chung. Giữ 66 file staging còn lại để kiểm tra tiếp.
- Ổ ORICO thu hồi 28,5 GiB. Danh sách file và biên nhận tại `runs/orico-verified-cache-cleanup-20261126/`.

### Gom blob từ kho mobihet trước khi xóa — 26/11/2026

- Owner chỉ giữ DVC cho `experiment` và `experiment_10k_clients`. Kho `veremi_build/mobihet/dvc-storage` và `veremi_build/fl_dagsnet_veremi/dvc-storage` đã xóa. Kho `mobihet-features` chỉ còn v4-scaled*.
- Trước khi xóa, đã quét 133 pointer của repo, gồm cả pointer bị gitignore dưới `runs/`. Hai pointer `runs/recovery-20261120/references/dagsnet-l4/mh-fbl-l4-top-s4{1,2}-r50.dvc` chỉ có object trong kho mobihet. Đã copy 155 object (29 MB) sang `experiment/dvc-storage` và vào `.dvc/cache`. MD5 khớp, `rclone check --download` báo 155/155 khớp.
- 27/11: owner xác nhận L4 không còn là phương pháp chính thức. Đã xóa hai pointer trên cùng 155 object chỉ thuộc chúng, cả trên remote (vào thùng rác OneDrive) lẫn trong `.dvc/cache`. 29 object dùng chung với pointer khác được giữ. Thư mục dữ liệu local `runs/recovery-20261120/references/dagsnet-l4/` (~130 MB, gitignore) vẫn còn.
- Quét lại sau cùng: 131 pointer, 17.046 object, thiếu 0 trên remote `onedrive`. `experiment_10k_clients`: 6 pointer, 245 object, thiếu 0.

### Đổi tên pointer A3 cũ thành MobiHet — 27/11/2026

- A3 cũ (L4 bỏ LA) là MobiHet chính thức nên không còn là ablation. Đã đổi tên 14 thư mục pointer dưới `artifacts/mobihet-checkpoints-2026-11-25/<owner>/`:
  `<model>-abl-a3-<seed>` → `<model>-mh-<seed>` và `mixl4-abl-a3-<case>-<seed>` → `mixmh-<case>-<seed>`. Hash và object không đổi.
- Trong `checkpoint-runs.csv` và `checkpoint-files.csv` chỉ sửa cột `dvc_pointer` (14 run, 428 file). Cột `run` vẫn giữ ref Kaggle gốc (ví dụ `accanniez/cnnbigru-abl-a3-s41/1`), vì slug Kaggle không đổi tên được. Bảng kết quả `experiments/*/results.csv` đã ghi `method=mobihet` cho các run này.
- Biên nhận cũ dưới `runs/` vẫn dùng tên cũ.

### Dọn và cấu trúc lại ORICO — 27/11/2026

- Đã xóa `checkpoint-dvc-resume-xktq98dq/` và `checkpoint-dvc-stage-ykg_7ia0/` (14,5 GB). Có 65/66 object khớp MD5 và kích thước với remote `onedrive`, gồm 34 file `.tmp` đã tính MD5 nội dung. File còn lại là `.tmp` 3.943.985.152 byte, bản upload bị cắt dở và không khớp object nào.
- Đã xóa `local-artifacts/runs/mixed-l4/` và `runs/mixed-l4/` local, tức mixed L4 **có LA** (`la_mode` mặc định `full`), thu hồi khoảng 42 GB checkpoint. Kết quả (config, CM, metrics, log; 1.608 file, 278 MB) đã chuyển sang `../mobihet/archive/l4-la-20261124/experiment/mixed-l4/`, kèm ZIP đã xác minh trên OneDrive. [Archive L4-LA](../../../mobihet/archive/l4-la-20261124/README.md)
- BERT thuộc repo này nên `experiment-data/fl_veremi_bert/{runs,artifacts}` đã đổi tên thành `experiment-data/experiment/bert/{runs,artifacts}` (cùng ổ, không chép lại). 13 symlink dưới `bert/runs/` và `bert/artifacts/` đã trỏ sang vị trí mới, không còn link gãy. Các mục cũ ở trên ghi đường dẫn trước khi đổi tên.
- Giữ nguyên `ablation-l4/` (A3 = MobiHet, 45,4 GB checkpoint, 21 run chỉ có trên ORICO; A1/A2/A4 không có checkpoint), `ablation-mh/`, `bert/` và `transfer_learning_c/` (vrepair). ORICO còn trống 124 GB.

## Ablation audit — 23/11/2026

- [DVC pointer](../../artifacts/ablation-audit-2026-11-23.zip.dvc), remote `onedrive` theo `.dvc/config`.
  ZIP SHA-256 `f09a5e8ef35af365289d0fc715ec5f4c6ce7b2a0e5142840d3941469d2d458f8`.
- 5.428 file của 131 phiên đã audit. Gồm receipt, manifest source, cấu hình, history và CM đến round đã kiểm.
  Cây gốc nằm trên ORICO. Checkpoint giữ tại Kaggle.
- `dvc push artifacts/ablation-audit-2026-11-23.zip.dvc` tải một object. `dvc status -c` báo đồng bộ với `onedrive`.
  Khôi phục bằng `dvc pull artifacts/ablation-audit-2026-11-23.zip.dvc`. ZIP có manifest SHA-256 từng file.


## GitHub centralized CNN-LSTM-feature supplement — 20/11/2026

- Remote: `origin` → `https://github.com/odixe06/experiment.git`; branch:
  `codex/mobihet-v4-relkd`; artifact commit:
  `8b1a3d4a360e5431d2be1a426040ad10c4f4d0fc`.
- Git LFS object: `artifacts/central-cnn-lstm-feature-supplement-2026-11-20.zip`,
  77.224.724 bytes; SHA-256
  `1bca385fd7ed168fc630781d2e6674deb19bf54c07ce40370fdae0e1b29562b5`.
- Scope: all 258 safe non-checkpoint files / 478.011.181 source bytes from exact
  `qwerasdf3004/veremi-central-cnn-lstm-feature-e50` v1 output.
- Checks: config identifies CNN-LSTM-feature seed42/50 epochs; epochs1–50 contiguous;
  every CM is 16×16 with 9.347.679 rows; every prediction reconstructs its CM exactly;
  all ten metrics match each CM; ZIP integrity and credential scan PASS.
- Full backup audit after merge: single-model FL 168/168, mixed-model 45/45, and all
  recorded centralized epochs PASS. No checkpoints included; no cleanup performed.
- Local copy verified on ORICO at
  `experiment-data/experiment/archives/central-cnn-lstm-feature-supplement-2026-11-20.zip`.
  The independent audit checked SHA-256 for all 258 ZIP entries and rebuilt all 50 CMs and their metrics.
  The normalized run is at `runs/round-metrics/legacy/cnn-lstm-feature/centralized/s42/`.

## GitHub mixed-model history supplement — 20/11/2026

- Remote: `origin` → `https://github.com/odixe06/experiment.git`; branch:
  `codex/mobihet-v4-relkd`; artifact commit:
  `7585df08fef18c53bb5eafd8aaf07bd1acca25de`.
- Git LFS object: `artifacts/mixed-history-supplement-2026-11-20.zip`, 1.195.696 bytes;
  SHA-256 `df8032085ba6c0465ae5a85333f0d094ee49ed29b34c4e5c1d38ae676c15e3ba`.
- Scope: complete R1–R50 `rounds.jsonl` for Case2 V2X-HtFL seeds41/42/43 and
  Case3 FedTGP seed41 / TinyProto seeds41/43; CSV contains 300 round rows.
- Checks: exact final kernel/version, non-smoke history, contiguous unique rounds1–50,
  both client/client_proto macro-F1 each round, all ten R50 metrics match final-CM audits,
  ZIP integrity and credential scan PASS; LFS upload 100%.
- No checkpoints included; no local cleanup requested or performed.

## Gộp checkout local và ORICO — 20/11/2026

- Checkout chính `/home/odixe/project/experiment`, nhánh local `main`. FeatureBERT chạy trong `bert/`.
  Git giữ lịch sử BERT tới `49b2d88`, worktree tới `a8cb691` và bổ sung Codex tới `5cf5795`.
- Đã gỡ Git worktree `experiment-wt` sau khi đối chiếu 1.872 file ngoài Git. Bundle BERT, patch local,
  metadata checkout và biên bản lưu dưới `runs/worktree-consolidation/orico-20261120/`.
  Snapshot chứa bản sửa và script T3/Lite2 cũ, theo quyết định loại chúng khỏi code chạy chính.
- Ngày 20/11, đích dữ liệu BERT trên ORICO là `experiment-data/fl_veremi_bert/{runs,artifacts,dvc-cache}`. Cache riêng đã dọn ngày 26/11.
  Dữ liệu checkpoint DAGSNet và Mixed lịch sử chuyển vào `experiment-data/experiment/recovered-checkpoints-20261120`.
  Symlink local trong `bert/` và `runs/` giữ nguyên đường dẫn dùng bởi script. Bốn cây BERT `runs/` đã có trên ORICO được giữ nguyên.
- Tiến độ copy và SHA-256 lưu trong `runs/worktree-consolidation/orico-20261120/`. File `consolidated.json` xác nhận đã đổi symlink
  và thu hồi bản trùng local. Hai symlink cấp checkout `fl_dagsnet_veremi_v2` và `fl_veremi_bert` đã gỡ sau khi chuyển năm script MobiHet local sang `experiment/` và `experiment/bert/`.
- Các checkpoint Mixed L4 case2, L4 port, regression và đầu vào/trace compute chuyển vào
  `experiment-data/experiment/local-artifacts/`. `relocation.json` ghi SHA-256 từng file, các đường dẫn local trỏ tới bản đã xác minh.
- `experiment-data/experiment/round-metrics/` giữ lịch sử metrics, manifest từng run và evidence bổ sung.
  Mở qua `runs/round-metrics/`. [Cấu trúc và trạng thái](continue.md#lịch-sử-từng-round-và-epoch)

## GitHub single-model artifact supplement — 20/11/2026

- Remote: `origin` → `https://github.com/odixe06/experiment.git`; branch:
  `codex/mobihet-v4-relkd`; artifact commit:
  `0af58af785ba24eb2709999d6773880657bb3ff9`.
- Git LFS object: `artifacts/single-fl-final-artifacts-supplement-2026-11-20.zip`,
  88.537.309 bytes; SHA-256
  `46f4ef982fcee7f42ca8d8a8a196396acee252f4534e0f843170336c6c773051`.
- Scope: 390 missing non-checkpoint files / 97.685.945 source bytes from 132 exact final
  runs in the 4-backbone × 14-method × 3-seed matrix.
- Combined with `runs-no-checkpoints-2026-11-20.zip`: 504/504 required files present
  (one final CM, `config.json`, and `rounds.jsonl` for each of 168 runs).
- Checks: 132/132 configs match model/method/seed; round histories contiguous; every CM has
  5.343 clients × 16 × 16, 9.347.679 test rows per view, and all ten final metrics match;
  ZIP integrity PASS; credential/signed-URL scan PASS; LFS upload 100%.
- No checkpoints included; no local cleanup requested or performed.

## GitHub mixed-model results — 20/11/2026

- Remote: `origin` → `https://github.com/odixe06/experiment.git`; branch:
  `codex/mobihet-v4-relkd`.
- Results commit: `5d2648b100e0cb09b1de3ea463553d827acb8663`.
- Scope: mixed128 report, CSV and `scripts/report_mixed_matrix.py`. Local merge uses
  [report](../research/mixed/mixed128-all-results.md) and [CSV](../research/mixed/mixed128-all-results.csv).
- Checks: merged final-CM audits cover 45/45 unique case/method/seed tuples; all headline
  views contain ten metrics; generated report and CSV contain no missing configuration.
- No local cleanup requested or performed.

## GitHub run artifacts — 20/11/2026

- Canonical remote: `origin` → `https://github.com/odixe06/experiment.git` (GitHub redirected the old
  `fl_dagsnet_veremi_v2` URL during push).
- Branch: `codex/mobihet-v4-relkd`; artifact commit:
  `de5904a5facd2ba5f7d9966242808a76ded91016`.
- Git LFS object: `artifacts/runs-no-checkpoints-2026-11-20.zip`, 903.962.154 bytes;
  SHA-256 `9e5d8eb4573785f418ff710c38de0af0b238cdac0bcfd31714a871ee42f2c012`.
- Archive contains 5.176 files / 1.112.281.698 source bytes from `runs/`. Excluded: 1.036
  `.pt`/`.pth`/`.ckpt` checkpoints and 95 files matching signed-URL/credential gates.
- Manifest: `artifacts/runs-no-checkpoints-2026-11-20.manifest.json`; packaging helper:
  `scripts/package_runs_without_checkpoints.py`.
- Checks: ZIP integrity PASS; Git LFS upload 100% (904 MB); no local cleanup requested or performed.
- DVC remains unconfigured; this selected artifact was explicitly stored in GitHub LFS.

## GitHub code + docs — 20/11/2026

- Remote: `origin` → `https://github.com/odixe06/fl_dagsnet_veremi_v2.git`.
- Branch: `codex/mobihet-v4-relkd`.
- Source commit: `c398d5693cbad7df9762748d122740a6d6e59d4f`; `git push` thành công và
  `git -c http.sslBackend=schannel ls-remote` trả đúng SHA.
- Scope: 177 file code, tests, experiment assignment và tài liệu; 41.619 dòng thêm.
- Loại khỏi Git: `docs/paper/`, `tmp/`, `output/`, credentials, signed URLs, checkpoints và run artifacts.
- Checks: secret scan PASS (chỉ token giả trong unit test), `compileall` PASS,
  `git diff --cached --check` PASS, 3 test assignment PASS.
- Giới hạn test local: full suite chưa chạy vì Python host thiếu `torch`/`pytest` và không tìm thấy env `nckh`;
  không cài dependency mới trong phiên backup.
- DVC: không cấu hình, không được yêu cầu; không upload artifact và không cleanup.

## Deployment riêng — 15/11/2026

Owner yêu cầu hourly cloud dispatcher; đã tạo private repo
https://github.com/dangvanhoang1711/kaggle-fl-dispatcher . Source commit triển khai
`cba65ac42971c2fdc62fcf8581269e61ede5881f`,65files (58training source frozen + tooling).
Ledger queue có commit riêng ban đầu; workflow sẽ ghi tiếp trạng thái. Secrets mã hóa ngoài source.
Credential scan so với toàn bộ saved Kaggle tokens PASS;13tests local PASS; dry-run cloud SUCCESS43619921537.
Không backup toàn repo nghiên cứu, không đổi Git HEAD/index gốc, không DVC hoặc cleanup.
Chi tiết vận hành: [cloud dispatcher](cloud-dispatcher.md); current status: [continue](continue.md).

## Backup lịch sử

Updated: 2026-10-31, Asia/Bangkok — pushed MobiHet v4/RelKD source snapshot to new GitHub branch.
Procedure: repository `repo-backup` skill.

## 1. Destinations

- Git: `origin` → `codex/mobihet-v4-relkd`; source commit `9257cec2c0b4b0886cd30e5f7aff591d64543bc9` verified with `git ls-remote`.
- DVC: not configured or requested in this operation.
- Cloud artifacts: not part of this operation; local `runs/`, checkpoints, and generated notebooks remain ignored by Git.

## 2. Tracked artifacts

No DVC manifest touched.

## 3. Checks that passed

| Check | Scope | Result |
|---|---|---|
| Python `compileall` | `core`, `methods`, `scripts`, `tests` | PASS |
| Documentation link check | repository Markdown links | PASS |
| `git diff --cached --check` | source snapshot | PASS |
| credential pattern scan | committed source and docs | no match |
| `git push` + `git ls-remote` | `codex/mobihet-v4-relkd` | source SHA matched |

## 4. Local cleanup

- None requested; no files deleted.

## 5. Next session

- Push later commits with `git -c http.sslBackend=schannel push` because local HTTPS uses Windows certificate store.
- Re-run checks if source or docs change.
- Artifact backup or restore requires separate DVC configuration and owner request.
