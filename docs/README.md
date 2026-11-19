# Bản đồ tài liệu

Mọi tài liệu của repo nằm dưới `docs/` (trừ hub `CONTEXT.md`, `LESSONS.md`, `AGENTS.md` ở gốc; dữ kiện bất biến ở
[knowledge/](../knowledge/README.md); dossier bài báo ở [papers/](../papers/README.md)). Số liệu từng kịch bản: [experiments/](../experiments/README.md).

## Chung mọi kịch bản

| tài liệu | nội dung |
|---|---|
| [research/protocol.md](research/protocol.md) | cấu hình chung: client = xe, 50 round, 1 epoch, batch 64, AdamW cosine, cách chấm, 10 metric |
| [research/methods.md](research/methods.md) | 14 phương pháp FL + centralized, bài gốc, view headline |
| [research/aligned-fl-v2.md](research/aligned-fl-v2.md) | căn chỉnh protocol cho các backbone ngoài DAGSNet |
| [research/compute.md](research/compute.md) | chi phí tính toán (FLOP) mọi kịch bản, cách đo |
| [research/cost.md](research/cost.md) | byte truyền / chi phí đo trên toy (DAGSNet) |
| [methods/](methods/) | ghi chú riêng: [global-baselines](methods/global-baselines.md), [tinyproto](methods/tinyproto.md) |

## Theo kịch bản

| kịch bản | tài liệu |
|---|---|
| DAGSNet | [research/dagsnet/results.md](research/dagsnet/results.md), audit: [research/dagsnet/audits/](research/dagsnet/audits/) |
| FeatureBERT | [bert/](bert/README.md): [results](bert/research/results.md), [comparison](bert/research/comparison.md), [protocol](bert/research/protocol.md), [cost](bert/research/cost.md), [v2-parity](bert/research/v2-parity.md) |
| TabM / SAE / MLP / CNN-BiGRU | [research/backbones/](research/backbones/): [bảng 4 backbone](research/backbones/four-backbone-seed-results-live.md), [audit code](research/backbones/four-backbone-code-alignment-audit-20261116.md), [centralized](research/backbones/central-backbones.md), [TabM](research/backbones/tabm-single-model-baselines.md), [SAE](research/backbones/sae-single-model-baselines.md), [CNN-BiGRU FL](research/backbones/dense-recurrent-fl.md) |
| Mixed case 1–3 | [research/mixed/](research/mixed/): [thiết kế](research/mixed/mixed-model-cases.md), [kết quả](research/mixed/mixed128-all-results.md) |

## Vận hành

| tài liệu | nội dung |
|---|---|
| [operations/continue.md](operations/continue.md) | **bàn giao sống**: job đang chạy, bước tiếp, việc tay |
| [operations/lessons.md](operations/lessons.md) | bài học vận hành (Kaggle, notebook, Git/DVC) |
| [operations/kaggle.md](operations/kaggle.md), [operations/notebook-errors.md](operations/notebook-errors.md) | account, quota, sổ lỗi kernel |
| [operations/backup.md](operations/backup.md) | Git/DVC backup |
| [operations/notion-results.md](operations/notion-results.md) | báo cáo Notion |
| [bert/operations/](bert/operations/README.md) | vận hành riêng phần FeatureBERT (trước khi gộp repo) |

## Lưu trữ

[archive/](archive/): bàn giao/CONTEXT cũ trước khi gộp repo 20/11/2026 và archive của fl_veremi_bert. Không đọc mặc định.
