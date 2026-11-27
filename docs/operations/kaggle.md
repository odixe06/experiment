# Kaggle — account và quy tắc vận hành

> Kế thừa từ repo `mobihet` / `mobihet-features` (29/10/2026): mục lịch sử là bằng chứng của các run cũ, không phải trạng thái repo này.

Lỗi kernel đã gặp và cách tránh: [sổ lỗi](notebook-errors.md) (đọc trước khi sinh notebook). Kernel đang chạy: [continue](continue.md).
Lịch sử cấp account/quota 15–17/10: [archive](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/kaggle-accounts-history-2026-10-15-to-17.md).

## Kaggle CLI 2.2.5 — cập nhật 27/10/2026

Theo yêu cầu owner từ phiên RCD: nâng `nckh` từ 2.2.4 lên 2.2.5, chỉ package `kaggle`
thay đổi; `kagglesdk` 0.1.38 và runtime ML giữ nguyên. `pip check` PASS.
Pin tại `environment.yml`; metadata tại [local-runtime](../../knowledge/local-runtime.json).
Helper/skill `.agents` và `.claude` cập nhật đồng thời, mỗi bộ test account/pull **29 PASS**.

CLI mới có streaming/resume/timeout cho output, nhưng một lệnh chỉ lấy một trang listing.
Helper giữ all-page traversal/parallel và kiểm file thiếu. Đã sửa helper bỏ mất phiên bản
kernel: `<owner>/<slug>/7` phải truyền `v7` qua mọi trang SDK; regression hai trang xác nhận.
Dùng version cố định khi thu bằng chứng. `kernels push --no-run` không hủy worker;
giữ GUARD và hướng dẫn stop hiện có. Không chạy train/collector/backup/Git/DVC trong tác vụ này.

Nguồn: [release chính thức](https://github.com/Kaggle/kaggle-cli/releases/tag/v2.2.5),
[changelog](https://github.com/Kaggle/kaggle-cli/blob/v2.2.5/CHANGELOG.md).
Số liệu throughput CLI cũ trong skill là lịch sử, chưa benchmark lại 2.2.5.

## Account

Dùng env `nckh` và `kaggle_as.py <owner> -- …` để giữ credential trong process con.
Credential và snapshot nằm dưới `~/.kaggle/accounts/`. Giữ token và email trong kho cục bộ.
Xác minh quyền truy cập cho đúng owner và đúng version trước khi tải output riêng tư.

Ngày 25/11/2026, owner yêu cầu gỡ tài khoản còn `UNVERIFIED` sau kiểm lại quota khỏi cấu hình, danh sách vận hành
và credential local. Dùng exact version để đối chiếu artifact đã audit và xác nhận phiên cần chạy lại.
Xem [bàn giao](continue.md) cho trạng thái batch và bằng chứng.

Ngày 27/11/2026, owner mất quyền `cacaosua`, `preyta`, `nocknnock`, `shippertocdo`. Đã gỡ thông tin đăng nhập cục bộ của bốn tài khoản. Sáu phiên đã nhận trên bốn account này coi như mất. Các notebook gán cho chúng phải đổi owner trước khi push.
Checkpoint chưa backup của chúng cũng mất. [Phủ checkpoint](checkpoint-coverage.csv)

## Quy tắc

- Mọi lệnh chạy theo account bằng `kaggle_as.py`, credential chỉ trong process:
  `conda run -n nckh python .claude/skills/kaggle-training-notebook/scripts/kaggle_as.py <user> -- kaggle kernels status <user>/<kernel>`.
- Chỉ đọc quota khi chuẩn bị một lượt launch đã được giao. Lấy quota, reset và slot từ API ở thời điểm dispatch.
- Dataset source/checkpoint phải PRIVATE, upload lên **account chạy kernel**, verify đủ file/SHA trước khi push.
- Notebook public (`notebooks/`) không chứa key; bản push (`private_push/`, git-ignored, 0600) mang key W&B inline theo quyền owner đã cấp
  (notebook `is_private: true`). Đăng nhập W&B và pin phiên bản: [skill wandb.md](../../.claude/skills/kaggle-training-notebook/references/wandb.md).
- `QUEUED` = chờ GPU, không phải lỗi. `RUNNING`/`COMPLETE`/W&B finished không chứng minh đã train: phải có log/heartbeat và audit artifact.
  Log live: `kaggle kernels logs --follow` (SSE) giới hạn ~35 s; `kernels logs` thường rỗng khi đang chạy.
- Dừng session chỉ làm được trên web (owner bấm **Stop session**); API không hủy được.
- Kaggle native MCP chưa cấu hình; CLI có xác thực là kênh chính.
