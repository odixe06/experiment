# Danh mục phương pháp — 14 baseline + centralized (+ MobiHet ở repo ../mobihet)

Owner 20/11/2026: **SCAFFOLD bỏ hẳn** (seed 41/43 NaN r15; code ở lịch sử Git trước commit gộp repo). Bảng dưới là trạng thái DAGSNet 30/10; trạng thái mọi kịch bản: [experiments](../../experiments/README.md).

Nguồn: [bảng owner](https://docs.google.com/spreadsheets/d/1v2tfkEUUfD9wUbOUp4m8ID-H_WK_hgERiC8jq4MSCLU/edit?gid=0) — owner 29/10 chọn
nhóm "Chọn" = 1 và 2 (bỏ FedCAPS, OptiFLIDS, GRU_FedAvg, FedDF). Cấu hình chung: [protocol](protocol.md). Bài báo: [papers/](../../papers/README.md).
Build cũ (dataset Dirichlet 20/50/100 client) chỉ để đối chiếu cơ chế: `odixe06/fl_dagsnet_veremi` (A), `dangvanhoang1711/FL_DAGSNet` (B).

"Model chấm": **toàn cục** = một model, chấm toàn test; **client** = model riêng mỗi xe, chấm test của xe đó (bảng: "đại diện" / "trung bình").

| # | alias | bài | model chấm | trao đổi mỗi round | build cũ | trạng thái v2 |
|---:|---|---|---|---|---|---|
| 0 | centralized | DAGSNet (Khan 2025) | toàn cục | — | A `edl_cmso` | ep50 hoàn tất, F1 0,9593; [results](dagsnet/results.md) |
| 1 | FedAvg | McMahan 2017 | toàn cục | θ | B `fedavg` | r50 hoàn tất; [results](dagsnet/results.md) |
| 2 | FedProx | Li 2020 | toàn cục | θ | B `fedprox` | r50 hoàn tất; [results](dagsnet/results.md) |
| 4 | AFPHA_EDL_CMSO | Khan 2025 (Sci. Rep.) | toàn cục | θ | A `afpha` | r50 hoàn tất; [results](dagsnet/results.md) |
| 5 | FD-IDS | Peng 2025 (Sensors) | toàn cục | θ (+ KD) | A `fd_ids` | r50 hoàn tất; [results](dagsnet/results.md) |
| 6 | APFed_LDwCBN | Sci. Rep. 2025 (DOI 10.1038/s41598-025-17699-3) | client (headline) + global | θ | B `apfed` | r50 metric hoàn tất, bank cuối mất; [results](dagsnet/results.md) |
| 7 | LwFedNIDS | Bouayad 2024 (Access) | toàn cục | θ đã tỉa | A `lwfednids` | r50 hoàn tất; [results](dagsnet/results.md) |
| 8 | V2X-HtFL | OJVT 2025 | client (headline) + client_proto | prototype + feedback sinh | B `v2x_HtFL` | r50 hoàn tất; [results](dagsnet/results.md) |
| 9 | CANPerFL | Hoang 2023 (Appl. Sci.) | client | θ (pretrain FL) → fine-tune local | B `canperfl` | FedAvg r50 + fine-tune 10 epoch; [results](dagsnet/results.md) |
| 10 | TinyProto | Lee 2026 (AAAI) | client_proto (headline) + head | prototype thưa | A `tinyproto` | r3 bất ổn scale; đang chạy r4 μ=1,5e-4; [diagnosis](../methods/tinyproto.md) |
| 11 | pFedES | Yi 2025 (AAAI) | client | extractor proxy nhỏ | A `pfedes` | r50 hoàn tất; [results](dagsnet/results.md) |
| 12 | NILM-FL | Li 2025 (TSG) | client | proxy | A `nilm_fl` | r36 đã audit, session 2 nối tới r50; [continue](../operations/continue.md) |
| 13 | FedTGP | Zhang 2024 (AAAI) | client_proto (headline) + head | prototype ↔ prototype toàn cục huấn luyện được | B `fedtgp` | r50 hoàn tất; [results](dagsnet/results.md) |
| 14 | PerFed-SKD | Singh 2024 (TCE) | client | θ (client được chọn) | A `perfedskd` | r50 hoàn tất; [results](dagsnet/results.md) |
| 15 | FedProto | Tan 2022 (AAAI) | client_proto (headline) + head | prototype | B `fedproto` | r50 hoàn tất; [results](dagsnet/results.md) |
| 16 | **MobiHet** (đề xuất) | repo `mobihet`, `gkd + la + kdpx`, proxy q3_zl | client (private) + proxy | proxy 3-bit + LZMA2 | `mobihet/methods/full-drift` | Cải tiến ở repo ../mobihet; H0–H4 10 round đang chạy; không đưa Notion |

Đặc tả từng phương pháp (công thức theo số phương trình của bài, siêu tham số bài gốc, deviation) nằm ở **docstring đầu file**
`methods/<alias>.py` — một nguồn duy nhất, đọc cùng code. Toàn cục: [global-baselines](../methods/global-baselines.md).
Model chấm headline của FedProto/FedTGP/TinyProto = luật prototype của bài (`client_proto`); head phân loại báo kèm (owner 29/10).
Kết quả tổng: [results](dagsnet/results.md). Báo cáo owner: [Notion](../operations/notion-results.md).
