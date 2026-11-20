# CONTEXT — experiment / FL baselines trên VeReMi NextGen v4 (5.343 xe)

Đọc CONTEXT → [LESSONS](LESSONS.md) + [bài học vận hành](docs/operations/lessons.md) → [bàn giao](docs/operations/continue.md).
Bản đồ tài liệu: [docs/README.md](docs/README.md). Trạng thái từng kịch bản: [experiments/README.md](experiments/README.md).

## Bài toán

IDS V2X 16 lớp, 77 feature/bản tin, học cộng tác: **mỗi xe nhận là một client** (5.343 client, dataset v4
`odixe0502/veremi-nextgen-v4-scaled-per-vehicle`). Repo gom **mọi lần chạy baseline** theo kịch bản; phương pháp đề xuất MobiHet
phát triển ban đầu ở `../mobihet`. Biến thể L4 đã tích hợp vào bảng kịch bản và runner của repo này.

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

## Trạng thái — 20/11/2026, 20:47 +07

- Repo phát triển tại `/home/odixe/project/experiment`, nhánh local `main`. FeatureBERT chạy từ `bert/`.
  Lịch sử nguồn và bổ sung artifact Codex đã nhập vào main. [Vị trí dữ liệu ORICO](docs/operations/backup.md)
- MobiHet L4 đủ 3 seed r50 cho sáu backbone đơn và mixed case2. Kaggle xác nhận sáu phiên mixed case1/3 vẫn chạy. Collector chờ output cuối. [Kết quả L4](docs/research/mobihet-l4.md)
- Mixed128 baseline đủ 45/45 cấu hình r50. Đã đối chiếu 10 metric mỗi dòng với confusion matrix từ bản sao Kaggle. [Kết quả](docs/research/mixed/mixed128-all-results.md)
- Centralized BERT có ba checkpoint epoch 9 đã audit. Cả ba phiên P2 vẫn chạy, log ghi commit epoch 14. Các backbone centralized khác đủ ba seed.
- Chi phí đủ 9 kịch bản và 114/114 hàng chi tiết. Đã audit thêm 27 trace CNN-BiGRU và mixed case1/3. [Compute](docs/research/compute.md)
- TabM/SAE/MLP/CNN-BiGRU FL baseline đủ 10 metric cuối từ confusion matrix cho cả 168 run. Lịch sử từng round và manifest thống nhất nằm tại `runs/round-metrics/` trên ORICO. [Bàn giao](docs/operations/continue.md#lịch-sử-từng-round-và-epoch) · [Bảng baseline](docs/research/backbones/four-backbone-seed-results-live.md)
- Kernel, kiểm chứng, bản sao artifact và việc tiếp theo nằm ở [bàn giao sống](docs/operations/continue.md).

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
