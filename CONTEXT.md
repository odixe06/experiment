# CONTEXT — fl_dagsnet_veremi_v2 / FL baselines + MobiHet trên VeReMi NextGen v4 (5.343 xe)

Đọc CONTEXT → [LESSONS](LESSONS.md) + [bài học vận hành](docs/operations/lessons.md) → [bàn giao](docs/operations/continue.md).

## Bài toán

IDS V2X 16 lớp, học cộng tác: **mỗi xe nhận là một client** (5.343 client, dataset v4 của repo `mobihet-features`).
Dựng lại **15 phương pháp** trong bảng owner + **MobiHet** (đề xuất của owner: model client khác nhau được, băng thông hạn chế) +
**centralized**, tất cả trên DAGSNet, **cùng một cấu hình** ([protocol](docs/research/protocol.md)). Build cũ (Dirichlet 20/50/100 client,
`odixe06/fl_dagsnet_veremi`, `dangvanhoang1711/FL_DAGSNet`) chưa sát bài gốc và không đồng nhất ⇒ build lại từ đầu, chỉ tham khảo.

## Quyết định owner (29/10/2026)

- Phạm vi: bảng "Chọn" = 1 + 2 ⇒ 15 phương pháp ([danh mục](docs/research/methods.md)), + MobiHet `gkd + la + kdpx` proxy q3_zl, + centralized.
- 50 round; **mọi client tham gia mỗi round, 1 epoch local trên hết dữ liệu train (fold train + val)**; HP chung cố định, headline r50.
- Đánh giá: model toàn cục trên toàn test; model riêng client trên **test của chính xe đó** (gộp = toàn test). Đủ 10 metric.
- DAGSNet được thu nhỏ (giảm kênh, giữ 4 nhánh) nếu centralized macro-F1 val kém full ≤ 0,01; độ rộng chung mọi phương pháp.
- Một seed (42). W&B online. Repo GitHub **public** `odixe06/fl_dagsnet_veremi_v2`; không đẩy data, PDF bài báo, credentials.
- Task < 1 h chạy local; task dài chạy Kaggle (16 account; CLI mới đăng nhập 6, owner bổ sung sau). Env conda `nckh`.
- Tốc độ train phải tối ưu ⇒ `core/grouped.py`: nhiều client một bước (bmm), dữ liệu thường trú GPU.

## Trạng thái — 30/10 ~23:45 (giờ VN)

- W=1, protocol không đổi. **Đủ 16/16 kết quả hoàn tất** (centralized + 15 baseline FL, CANPerFL = FedAvg r50 + fine-tune 10 epoch).
- **NILM-FL**: `accanniez/fl2-nilm-r2s2` nối tiếp s2 thành công đủ 50 round (preflight r1–36 byte-identical), audit PASS_BOUNDED (client F1 0,8846, proxy F1 0,8993).
- **TinyProto-FP**: `anhtruong1909/fl2-tinyproto-r4` (sửa μ=1,5e-4) hoàn tất đủ 50 round, audit PASS_BOUNDED (overlay SHA256 match, target L2 0,2657 ổn định, prototype F1 đạt 0,8534, head F1 0,8693).
- Tất cả CM và 10 metric đã tính lại, audit và cập nhật vào [results](docs/research/results.md). Đã đồng bộ đủ 16/16 phương pháp lên Notion [50 round with new dataset 5000 client](https://app.notion.com/p/409730cee68d81f19259d14bea45ea61) (16 hàng headline + 16 page con có PDF đính kèm).
- Cải tiến MobiHet thuộc **repo `../mobihet`** (H0–H4 đang chạy theo bàn giao này). Owner 31/10 đã cho ghi MobiHet gốc r50 từ branch `codex/mobihet-v4-relkd` lên Notion; đã pull hai session và audit PASS_BOUNDED đủ50round, nhập lịch sử private/proxy r1–50 ([chi tiết](docs/operations/notion-results.md)).
  Quyền chạy / trạng thái chi tiết: [continue](docs/operations/continue.md). Kết quả: [results](docs/research/results.md).

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Đang làm gì, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Kết quả 50 round, audit, lưu ý từng run | [results](docs/research/results.md) |
| Báo cáo Notion, page IDs, bổ sung run hoàn tất | [notion-results](docs/operations/notion-results.md) |
| TinyProto r3 suy giảm và bản sửa r4 | [tinyproto](docs/methods/tinyproto.md) |
| Cấu hình chung, cách chấm, cổng độ rộng | [protocol](docs/research/protocol.md) |
| Danh sách phương pháp, trạng thái | [methods](docs/research/methods.md) |
| Dataset, cột, số liệu, caveat | [knowledge/dataset](knowledge/dataset.md) |
| Dựng DAGSNet cho client | [knowledge/dagsnet](knowledge/dagsnet.md) |
| Máy local, Kaggle, lỗi kernel | [local-env](knowledge/local-env.md), [kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md) |
| Bài báo, cái nào thiếu | [papers/README](papers/README.md) |
| Phương pháp MobiHet: phát triển tiếp, kết quả v1 20/50/100 client | repo [`odixe06/mobihet`](https://github.com/odixe06/mobihet) (v2; v1 ở `history/v1/docs/methods/full-drift/`) |
