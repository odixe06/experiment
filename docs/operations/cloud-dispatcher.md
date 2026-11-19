# Kaggle cloud dispatcher

Repo private: https://github.com/dangvanhoang1711/kaggle-fl-dispatcher

Actions: https://github.com/dangvanhoang1711/kaggle-fl-dispatcher/actions

Được owner cho phép 15/11/2026: check mỗi giờ và push các cấu hình còn thiếu khi máy local shutdown.
Owner xác nhận chưa thêm phương thức thanh toán GitHub. Dùng standard Linux runner, quota Actions
chung tài khoản; không mua compute hoặc thêm phương thức thanh toán.

## Vận hành

- Cron `17 * * * *`, phút17 mỗi giờ; GitHub có thể trễ. Variable `DISPATCH_ENABLED=true` bật lịch.
- `queue.json` remote là ledger hiện hành. Không chạy launch scripts tay song song.
- Scope126 entries: SAE/MLP/CNN-BiGRU-Attention,14methods×3seeds41/42/43. Initial27SAE đã gửi;
  còn99 cấu hình mới. TabM42 đã gửi từ trước, không gửi lại.
- Nine account tokens stored encrypted in `KAGGLE_TOKENS_JSON`; không nằm trong source hoặc notebook.
- Trần2session/account, ít nhất8.25h quota free/session8h; notebook budget7h.
- CANPerFL chỉ dùng FedAvg cùngowner/model/seed đủ50round. Không tự retry ERROR/partial.
- Ghi và commit `reserved` trước API push. Mất reply thì reconcile; chưa rõ thì `review`, không gửi lại mù.
- `submitted` chỉ nghĩa đã được Kaggle nhận. Full scientific audit vẫn thực hiện riêng.
- Tự disable khi không cònpending. Actions → workflow → Disable workflow dừng các lần gửi mới;
  không ảnh hưởng phiên Kaggle đã chạy. Để bật lại sau self-disable, Enable workflow và kiểm variable.

## Source và kiểm chứng

Source tooling local: `scripts/cloud_dispatch/`; deployment package `tmp/kaggle-fl-dispatcher`.
Source training frozen58files trong `source_manifest.json`; không đổi protocol hoặc model.
13 tests local PASS; build/static validation3backbone PASS.
Hai job do cloud gửi đã xác minh published source/private/GPU/version1:
`nguynt298/sae-fl-fedtgp-s41`, `hoangdang1711/sae-fl-fedtgp-s42`.
Scheduler xác nhận source remote sau push; phản hồi SDK đơn lẻ không đủ.
Dry-run cloud success43619921537: nhận27submitted/99pending và1slot khả dụng.
Kết quả tick mới nhất xem [bàn giao](continue.md) và remote queue.

Git main/index của repo nghiên cứu không thay đổi bởi deployment; source được gửi qua GitHub API
vào repo private riêng. Không backup DVC hoặc đưa dataset/PDF/weights lên GitHub.
