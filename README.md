# experiment

Federated IDS cho V2X trên VeReMi NextGen v4: **5.343 client = 5.343 xe nhận**, 16 lớp, 77 feature. Repo gom mọi lần chạy
14 phương pháp FL (FedAvg, FedProx, AFPHA, FD-IDS, APFed, LwFedNIDS, V2X-HtFL, CANPerFL, TinyProto, pFedES, NILM-FL, FedTGP,
PerFed-SKD, FedProto) + centralized trên nhiều kiến trúc, **cùng một protocol** ([docs/research/protocol.md](docs/research/protocol.md)).
Trước 20/11/2026 là hai repo `fl_dagsnet_veremi_v2` (DAGSNet, 4 backbone, mixed) và `fl_veremi_bert` (FeatureBERT).

- Kết quả theo kịch bản: [experiments/](experiments/README.md). Tài liệu: [docs/](docs/README.md). Bắt đầu phiên: [CONTEXT.md](CONTEXT.md).
- Dataset: Kaggle [`odixe0502/veremi-nextgen-v4-scaled-per-vehicle`](https://www.kaggle.com/datasets/odixe0502/veremi-nextgen-v4-scaled-per-vehicle) — [mô tả](knowledge/dataset.md).
- Env: conda `nckh` ([environment.yml](environment.yml)).

```
experiments/  một thư mục mỗi kịch bản: README.md (sinh), results.csv (10 metric/seed), compute.csv; mixed-model-v1/ = bảng gán xe → model
core/         DAGSNet + TabM/SAE/MLP/CNN-BiGRU, engine grouped (nhiều client/bước), mixed backbones, eval, metric
methods/      một file mỗi phương pháp (engine DAGSNet, dùng chung cho 4 backbone); methods/mixed.py cho mixed case
scripts/      run_fl.py, run_mixed_fl.py, run_central*.py, builder notebook Kaggle, audit, experiments_status.py, compute/
bert/         FeatureBERT FL (repo fl_veremi_bert cũ, giữ lịch sử): core/, methods/, scripts/, tests/ — chạy với cwd bert/
tests/        test tương đương / đơn vị
knowledge/    dữ kiện bất biến (máy, dataset, DAGSNet; knowledge/bert/ cho FeatureBERT)
papers/       dossier bài báo (PDF không đưa lên Git)
docs/         toàn bộ tài liệu: research/, operations/, methods/, bert/, archive/
```
