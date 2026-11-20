# bert/ — FeatureBERT FL (trước 20/11/2026: repo `fl_veremi_bert`)

Gộp vào repo `experiment` bằng git subtree (giữ lịch sử); tài liệu đã dời sang [docs/bert/](../docs/bert/README.md),
kết quả theo kịch bản ở [experiments/bert/](../experiments/bert/README.md). Chạy lệnh với cwd `bert/`.
Dữ liệu lớn và cách truy cập qua ORICO ở [hồ sơ backup](../docs/operations/backup.md).

Phần này là IDS federated cho V2X trên VeReMi NextGen v4: **5.343 client = 5.343 xe nhận**, 16 lớp. Model client là **FeatureBERT**
(77 token đặc trưng + CLS, 2 lớp Transformer encoder, 77.904 tham số; từ [veremi-bert-77](https://github.com/dangvanhoang1711/veremi-bert-77)).
Repo chạy lại 14 phương pháp FL của [`fl_dagsnet_veremi_v2`](https://github.com/odixe06/fl_dagsnet_veremi_v2) với **cùng protocol**:
FedAvg, FedProx, AFPHA, FD-IDS, APFed, LwFedNIDS, V2X-HtFL, CANPerFL, TinyProto, pFedES, NILM-FL, FedTGP, PerFed-SKD và FedProto.
Owner15/11 bổ sung **MobiHet/T3**, cùng HP/code phương pháp DAGSNet, chỉ đổi model/engine. Protocol: [docs/research/protocol.md](../docs/bert/research/protocol.md).

- Dataset: Kaggle [`odixe0502/veremi-nextgen-v4-scaled-per-vehicle`](https://www.kaggle.com/datasets/odixe0502/veremi-nextgen-v4-scaled-per-vehicle), [mô tả](../knowledge/bert/dataset.md).
- Model: [FeatureBERT](../knowledge/bert/bert.md).
- Điểm bắt đầu: [CONTEXT.md](../docs/bert/context.md). Env: conda `nckh` ([environment.yml](environment.yml)).

```
core/      FeatureBERT tham chiếu + bản grouped (nhiều client/bước), dữ liệu thường trú GPU, AdamW theo client, eval, metric
methods/   một file mỗi phương pháp (chỉ phần cơ chế)
scripts/   run_fl.py (chạy FL), run_batch.py (Kaggle seed41/42/43), audit, report, cost, notebook Kaggle
tests/     test tương đương và test đơn vị
knowledge/ dữ kiện bất biến (máy, dataset, FeatureBERT)
docs/      protocol, danh mục phương pháp, kết quả, vận hành
```

```bash
source ~/Tools/miniconda3/etc/profile.d/conda.sh && conda activate nckh
python -m pytest -q tests
python scripts/run_fl.py --method fedavg --subset 100 --rounds 3 --out runs/smoke --compile --set group_rows=2048   # GPU 4 GiB
```
