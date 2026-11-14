# Continue — bàn giao sống, 30/10/2026 ~23:50 (giờ VN)

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md) → trang này.
Repo này: **thử các phương pháp/baseline và báo cáo**. Cải tiến MobiHet ở `../mobihet`, bàn giao riêng; không chép trạng thái H vào đây.
Snapshot trước phiên: [17:48](continue.md) (khi 2 kernel đang chạy).

## Đang chạy

Không có kernel nào đang chạy trên Kaggle. Toàn bộ 15 baseline FL + Centralized đã hoàn tất 50 round.

## Đã hoàn tất trong phiên

- **NILM-FL session 2 (`accanniez/fl2-nilm-r2s2`)**:
  - Pull 102 file chọn lọc (bỏ `state*.pt`), 0 thiếu, 449 MiB mới.
  - Parent preflight r1–r36 kiểm tra đối chiếu khớp byte-identical với audit session 1.
  - Nối tiếp trơn tru tới round 50: `client/f1_macro` = **0,8846**, `proxy/f1_macro` = **0,8993**.
  - Audit `scripts/audit_run.py`: **PASS_BOUNDED** 50 round complete ([audit json](../research/audits/2026-10-30/fl2-nilm-r2.json)).
- **TinyProto-FP R4 (`anhtruong1909/fl2-tinyproto-r4`)**:
  - Pull 141 file (gồm đầy đủ `provenance/`, bỏ `state*.pt`), 0 thiếu, 1.677 MiB.
  - Overlay `methods/tinyproto.py` khớp SHA256 `1bbf6b6e18f87a47f2ed52831c2709f8a3fbc001a7a2461c13a968ee62895ab4`.
  - Fix scale μ = 1,5e-4 thành công: target L2 norm r50 đạt **0,2657** (không bùng nổ như r3 2.943), loss r50 = **0,0218**.
  - `client_proto/f1_macro` (dense local Eq. 12) = **0,8534** (hồi phục hoàn toàn từ r3 0,2904), `client/f1_macro` (head) = **0,8693**.
  - Audit `scripts/audit_run.py`: **PASS_BOUNDED** 50 round complete ([audit json](../research/audits/2026-10-30/fl2-tinyproto-r4.json)).
- **Cập nhật kết quả**:
  - Đã cập nhật bảng headline [results.md](../research/results.md) đủ **16/16 phương pháp** (Centralized + 15 baseline FL), tính lại trực tiếp từ CM.
  - Chạy `scripts/build_notion_report.py` chuẩn bị toàn bộ JSON payloads cho 16 methods trong `runs/notion-report/`.

- Notion connector đã upload PDF `nilm.pdf` & `tinyproto.pdf`, tạo đủ 2 child page `NILM-FL` & `TinyProto`, cập nhật bảng headline trang root đủ 16/16 phương pháp.
  - Đã cập nhật [notion-pages-2026-10-30.json](notion-pages-2026-10-30.json) và [notion-results.md](notion-results.md).

## Bước tiếp

0. MobiHet gốc branch `codex/mobihet-v4-relkd`: đã pull hai session version1 bằng odixe0502 (quyền đọc hiện hoạt động), audit [PASS_BOUNDED đủ50round](../research/audits/2026-10-31/mobihet-v4-weighted-nowb-s2.json). Source37file khớp manifest; parent25CM khớp byte. Đã nhập hai bảng50round/10metric vào [Notion](https://app.notion.com/p/40a730cee68d81fba3fcfb532fb66a8d), cập nhật bảng tổng: private/proxy F1 0,895298/0,899538. Không cần credential minhiw cho hai output này. Chưa pull full resume banks/replay inference. Chi tiết [notion-results](notion-results.md). Không launch training.
1. Chờ chỉ thị của owner về việc đưa các nhánh cải tiến MobiHet (tại repo `../mobihet`) lên Notion / báo cáo.
2. Kiểm tra tiến độ sàng lọc MobiHet (H0–H4) tại repo `../mobihet` khi được yêu cầu.

## Trạng thái workspace / việc tay

- Git HEAD: `36ac7ffafcb45974ed736b4d838279728d00e884`. Không tự ý commit hay push Git/DVC.
- Toàn bộ artifact đã pull nằm tại `runs/pulls/`.
- Chưa có việc tay bắt buộc.
