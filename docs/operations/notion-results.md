# Báo cáo Notion — kết quả các phương pháp

Owner 30/10 yêu cầu tạo page con dưới NCKH, tiêu đề chính xác **50 round with new dataset 5000 client**.
Sau đó chốt: chỉ ghi phương pháp hoàn tất; phương pháp thiếu tiếp tục training. Owner 31/10 cho đưa MobiHet r50 từ branch lên, chấp nhận chờ artifact để kiểm lại.
Baseline/report thuộc repo này; cải tiến MobiHet thuộc `../mobihet`.

Page đã tạo: [50 round with new dataset 5000 client](https://app.notion.com/p/409730cee68d81f19259d14bea45ea61)
thuộc [NCKH](https://app.notion.com/p/319730cee68d80c3ace9eaaf52bc2cb4).

## Đã xuất và đọc lại kiểm tra

- 16 hàng headline = centralized + 15 baseline FL (bao gồm NILM-FL và TinyProto hoàn tất 50 round).
- 16 page con; tên phương pháp là native mention dẫn tới page đó; mỗi page có PDF upload thật từ `papers/`.
- Tính lại mọi round/view từ CM local; đủ support 5.343 client và 9.347.679 test rows; 10 metric khớp JSONL.
- Đọc lại cả 16 page qua Notion: mọi bảng liên tục 1..50; CANPerFL có thêm hai view fine-tune 1..10; PDF hiện diện.
- Đọc lại bảng chính, đối chiếu 16×10 giá trị với artifact (làm tròn 6 số lẻ). Không điền zero thay dữ kiện thiếu.
- Centralized là 50 **epoch**, CANPerFL là 50 FedAvg + 10 **epoch fine-tune**, ghi rõ trên page và bảng chính.
- APFed mất bank cuối, SCAFFOLD train NaN r15–16, NILM nối phiên r1–36 + r37–50, TinyProto μ=1,5e-4 đều có chú thích. Không tuyên bố đã replay toàn bộ inference.

Map page IDs và bằng chứng đọc lại: [notion-pages-2026-10-30.json](notion-pages-2026-10-30.json).
Payloads, upload IDs, bản local từng page: `runs/notion-report/` (ignored).
Generator local: scripts/build_notion_report.py (`scripts/build_notion_report.py`, gỡ 20/11, xem commit `5b9d0e3`); không tự gọi API hoặc tạo page trùng.

## Trạng thái hoàn thành 16/16 phương pháp

- NILM-FL và TinyProto r4 đã hoàn tất upload, đính kèm PDF gốc, tạo 2 page con và cập nhật vào bảng headline trang root.
- Cả 16 phương pháp baseline của protocol FL VeReMi NextGen v4 đều đã đồng bộ trên Notion. Không đưa các arm MobiHet lên khi chưa có chỉ thị mới của owner.

## Bổ sung MobiHet — 31/10/2026

- Owner cho ghi kết quả r50 hiện có, bổ sung credential sau. Đã thêm private headline và proxy vào bảng chính, tạo [MobiHet v4 — 50 round](https://app.notion.com/p/40a730cee68d81fba3fcfb532fb66a8d).
- Nguồn: branch `codex/mobihet-v4-relkd`, SHA `0de275340506c00588ae054f3c614c97e0315f41`, `docs/research/results.md`: đủ 10 metric r50, làm tròn 4 chữ số; private/proxy macro-F1 0,8953/0,8995.
- Bàn giao branch ghi session2 `minhiw/mobihet-v4-weighted-nowb-s2/1` đủ r1–50 và audit PASS. Phiên này chưa xác minh lại: máy thiếu credential minhiw, account odixe0502 đọc output trả HTTP403. Không coi đây là audit mới.
- Hai page đã đọc lại: đủ private/proxy r50 và chú thích nguồn/giới hạn. Chưa nhập bảng từng round hoặc đính kèm PDF MobiHet. RelKD r10 không đưa vào bảng final.
- Cập nhật sau khi owner gửi hai link: thử lại bằng odixe0502 thành công. Pull version1 session1 34file, session2 104file (54source/config/CM +50snapshot); không thiếu file đã chọn. Không cần owner bổ sung credential cho hai output này.
- [Audit mới PASS_BOUNDED](../research/dagsnet/audits/2026-10-31/mobihet-v4-weighted-nowb-s2.json): đủ50round, HP/W/LR đúng protocol, 50CM×2view khớp support từng xe/lớp, đủ10metric/client quantiles/per-class F1 khớp JSONL, snapshot hữu hạn. Parent25CM khớp byte; source37file SHA256 đúng manifest, hai session giống source. Không pull full bank/state, chưa replay inference hoặc chứng nhận exact resume.
- Đã thêm hai bảng50round lên page MobiHet và nâng bảng tổng lên6chữ số từ CM: private/proxy macro-F1 0,895298/0,899538. Đọc lại kiểm đủ lịch sử và nhãn audit. Payload và kiểm parent/source ở `runs/notion-report/mobihet-verified.json`, `prepare_mobihet_verified.py` (ignored). Map JSON30/10 ở trên là bằng chứng baseline cũ, chưa bao gồm MobiHet. Không còn việc tay bắt buộc.
