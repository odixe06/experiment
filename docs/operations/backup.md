# Backup / restore — nguồn hiện hành

Trạng thái job và quyền thao tác tại [continue](continue.md). Đọc skill `repo-backup` trước push/restore/cleanup.
Lịch sử các lô/cleanup/hash: [snapshot](../../archive/operations-20261116/before/docs/operations/backup.md).

## Đích

- Git `https://github.com/odixe06/fl_veremi_bert.git`, branch `main`.
- DVC default `onedrive`: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/fl_veremi_bert/dvc-storage`.
- Git bundle: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/fl_veremi_bert/fl_veremi_bert.bundle`.
- OneDrive là rclone mount; kiểm mount thật trước ghi. Google Drive không dùng làm upload fallback.

## Artifact và trạng thái đã kiểm

- `artifacts/baseline-pulls.dvc`: baseline session đã backup trước pFedES41p4 cuối.
- `artifacts/t3-pulls.dvc`: T3 BERT, state/provenance.
- `artifacts/research-evidence-20261116.dvc`: audit/report/source receipts đã đóng tại thời điểm backup.
- Output/receipt pFedES41p4 mới đang collect/audit; **chưa xác minh backup cho phần mới**.

Lần xác minh filesystem remote gần nhất:16/11/2026,16:57:46VN, push-j16 + status-c in sync cho các target trong
[receipt reboot](../../archive/operations-20261116/receipts/backup-reboot-20261116.json).
Đó là snapshot, không chứng minh output mới đã backup hoặc cloud upload đã hoàn tất.
VFSdirty có thể còn trênlocal; không xóa cache/artifact hoặc restart mount khỏe để ép sync.

## Restore / cập nhật

```bash
dvc pull PATH/TO/RUN.dvc
```

Kiểmdisk và manifest trước restore. Workspace partial/thiếu model không `dvc add` đè manifest đầy đủ;
đối chiếu `.dir`/entry cũ trước mọi cập nhật. Không dời hoặc sửa byte payload/metadata DVC khi chỉ dọn docs.
DVC cùngrepo chạy nối tiếp; push xong mới status-c. Git stage/push chỉ phạm vi owner giao; không credential/notebook inlinekey,
không `Co-Authored-By:`. Gitbundle tạo `--all`, verify rồi mới thay bản trênmount. Quyền backup không cho phép cleanup.

## Git phiên16/11 cuối

Kết quả pFedES và dọncontext đã push `origin/main`, commit `0eea5d3`; không DVCpush thêm trong yêu cầu hiện tại.
Bản ghi xác nhận cập nhật tiếp trên cùngbranch. Gitbundle allrefs được verify/copy theo commitcuối;
receipt cuối: `runs/maintenance-20261116/final-publish.json`. Output mới vẫn local, không coi snapshot DVC cũ đã chứa p4.
