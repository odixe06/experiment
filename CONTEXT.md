# CONTEXT — experiment / FL baselines trên VeReMi NextGen v4 (5.343 xe)

Đọc CONTEXT → [LESSONS](LESSONS.md) + [bài học vận hành](docs/operations/lessons.md) → [bàn giao](docs/operations/continue.md).
Bản đồ tài liệu: [docs/README.md](docs/README.md). Trạng thái từng kịch bản: [experiments/README.md](experiments/README.md).

## Bài toán

IDS V2X 16 lớp, 77 feature/bản tin, học cộng tác: **mỗi xe nhận là một client** (5.343 client, dataset v4
`odixe0502/veremi-nextgen-v4-scaled-per-vehicle`). Repo gom **mọi lần chạy baseline** theo kịch bản; phương pháp đề xuất MobiHet
phát triển ban đầu ở `../mobihet`. MobiHet (L4 bỏ LA, 24/11) đã tích hợp vào bảng kịch bản và runner của repo này.

## Quyết định owner

- 29/10: protocol chung ([protocol](docs/research/protocol.md)): 50 round, mọi client mỗi round, 1 epoch local trên fold train + val,
  batch 64, AdamW cosine 1e-3 → 1e-5; headline r50; model toàn cục chấm toàn test, model riêng chấm test của chính xe. Đủ 10 metric.
- 20/11 (gộp repo, phiên này):
  - Repo đổi tên **`experiment`** (GitHub `odixe06/experiment`, thư mục `~/project/experiment`); nhánh codex `mobihet-v4-relkd`
    đã merge; `fl_veremi_bert` nhập bằng subtree vào [bert/](bert/) (giữ lịch sử).
  - **Mỗi thư mục `experiments/<kịch bản>` = một kịch bản**, cấu trúc đồng nhất: `README.md` (sinh), `results.csv`, `compute.csv`.
    Kịch bản: `dagsnet`, `bert`, `tabm`, `sae`, `mlp`, `cnn-bigru`, `mixed-case1..3`.
  - Mỗi kịch bản: **14 phương pháp + centralized, mỗi cái 3 seed (41/42/43)**. 14 = FedAvg, FedProx, AFPHA, FD-IDS, APFed, LwFedNIDS,
    V2X-HtFL, CANPerFL, TinyProto, pFedES, NILM-FL, FedTGP, PerFed-SKD, FedProto. **SCAFFOLD bỏ hẳn.**
  - Mixed case: chỉ 5 phương pháp dị thể (FedProto, FedTGP, TinyProto, V2X-HtFL, pFedES); 9 phương pháp gộp θ + centralized = **N/A**.
  - Centralized: giữ s42 đã có của DAGSNet và CNN-BiGRU; chạy mới s41/s43 cho hai model này, chạy đủ 3 seed cho BERT, TabM, MLP,
    SAE (SAE centralized = **`sae-ce`**, cùng mạng + CE-only như bản FL).
  - Compute: **chỉ FLOP** (đếm op), bỏ GPU-time vì khác phần cứng; job không vừa GPU local thì đếm trên Kaggle.
- Task < 1 h chạy local (RTX 3050 4 GB, env conda `nckh`); task dài chạy Kaggle (account ở `~/.kaggle/accounts`, quota live).

## Trạng thái — 28/11/2026 07:40 (+07)

- Repo phát triển tại `/home/odixe/project/experiment`, nhánh local `main`. FeatureBERT chạy từ `bert/`. [Vị trí dữ liệu ORICO](docs/operations/backup.md)
- **MobiHet = L4 bỏ logit adjustment** (`la_mode=off`). Đủ 27/27 hàng R50; 11 ô rerun 27/11 (checkpoint mất) đã PASS, đủ hai shard,
  dòng cũ đã xóa (owner 28/11: phiên thiếu checkpoint bỏ hẳn). [MobiHet](docs/research/mobihet.md) · [Ablation](docs/research/mobihet-ablation.md) ·
  [Kết quả ablation](experiments/ablation/README.md)
- **Ablation 79/81** ô R50; hai ô mixed-case1 A2/A4 s41 đang chạy phiên nối từ R48. Thêm DAGSNet A2 s42 / A4 s41 đang chạy lại để có checkpoint.
- Centralized BERT P5 đang chạy (từ E27/E24/E25), cần thêm phiên tới E50. Các backbone centralized khác đủ ba seed.
- Mixed128 baseline đủ 45/45 r50; TabM/SAE/MLP/CNN-BiGRU FL baseline đủ 10 metric. Compute và byte đủ cho mọi ô có kết quả.
- Checkpoint coverage, round-metrics index và backup DVC làm sau khi các run trên xong. [Bàn giao](docs/operations/continue.md)
- Repo 10k dùng chung 13 account; MobiHet chính thức (bỏ LA) đang chạy ở đó.

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Đang chạy gì, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Kết quả từng kịch bản | [experiments/](experiments/README.md) (sinh bằng `python scripts/experiments_status.py`) |
| Cấu hình chung, cách chấm | [protocol](docs/research/protocol.md), [methods](docs/research/methods.md) |
| Code: DAGSNet + 4 backbone + mixed | `core/`, `methods/`, `scripts/run_fl.py`, `scripts/run_mixed_fl.py`, `scripts/run_central*.py` |
| Code: FeatureBERT FL | [bert/](bert/README.md) (chạy với cwd `bert/`) |
| Compute | [compute](docs/research/compute.md), `scripts/compute/` |
| Dataset, model, máy | [knowledge/](knowledge/README.md) |
| Kaggle, lỗi kernel, backup | [kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md), [backup](docs/operations/backup.md) |
