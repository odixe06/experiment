# Backup / restore — nguồn hiện hành

Trạng thái job và quyền thao tác tại [continue](continue.md). Đọc skill `repo-backup` trước push/restore/cleanup.
Lịch sử các lô/cleanup/hash: [snapshot](../../archive/bert/operations-20261116/before/docs/operations/backup.md).

## Đích

- Git `https://github.com/odixe06/experiment.git`, branch `main`.
- DVC default `onedrive`: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/experiment/dvc-storage`.
- Git bundle BERT lịch sử: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/experiment/bert/fl_veremi_bert.bundle`.
- DVC dùng cấu hình cha của repo `experiment`. Các pointer nằm dưới `experiment/bert/artifacts/`.
- Cache đang dùng là `experiment/.dvc/cache`. Cache BERT cũ trên ORICO đã xóa ngày 26/11 sau khi xác minh đủ 9.207 blob trong kho chung.
- OneDrive là rclone mount; kiểm mount thật trước ghi. Google Drive không dùng làm upload fallback.

## Artifact và trạng thái đã kiểm

- `artifacts/baseline-pulls.dvc`: baseline session đã backup trước pFedES41p4 cuối.
- `artifacts/t3-pulls.dvc`: T3 BERT, state/provenance.
- `artifacts/research-evidence-20261116.dvc`: audit/report/source receipts đã đóng tại thời điểm backup.
- Output/receipt pFedES41p4 mới đang collect/audit; **chưa xác minh backup cho phần mới**.

Ngày 26/11 đã đối chiếu 9.207 blob BERT trong kho chung và thử `dvc fetch` từ `experiment/bert`.
Lần xác minh trước khi gộp ngày 16/11/2026 lúc 16:57:46 +07 dùng `dvc status -c` cho các target trong
[receipt reboot](../../archive/bert/operations-20261116/receipts/backup-reboot-20261116.json).
Đó là snapshot, không chứng minh output mới đã backup hoặc cloud upload đã hoàn tất.
VFSdirty có thể còn trênlocal; không xóa cache/artifact hoặc restart mount khỏe để ép sync.

## Restore / cập nhật

```bash
cd /home/odixe/project/experiment/bert
dvc pull artifacts/t3-pulls.dvc
```

Kiểmdisk và manifest trước restore. Workspace partial/thiếu model không `dvc add` đè manifest đầy đủ;
đối chiếu `.dir`/entry cũ trước mọi cập nhật. Không dời hoặc sửa byte payload/metadata DVC khi chỉ dọn docs.
DVC cùngrepo chạy nối tiếp; push xong mới status-c. Git stage/push chỉ phạm vi owner giao; không credential/notebook inlinekey,
không `Co-Authored-By:`. Gitbundle tạo `--all`, verify rồi mới thay bản trênmount. Quyền backup không cho phép cleanup.

## Git phiên16/11 cuối

Kết quả pFedES và dọncontext đã push `origin/main`, commit `0eea5d3`; không DVCpush thêm trong yêu cầu hiện tại.
Bản ghi xác nhận cập nhật tiếp trên cùngbranch. Gitbundle allrefs được verify/copy theo commitcuối;
receipt cuối: `runs/maintenance-20261116/final-publish.json`. Output mới vẫn local, không coi snapshot DVC cũ đã chứa p4.

## Git khôi phục17/11

Owner yêu cầu commit/push thay đổi trên `main` cảhai repo. Phạm vi: source Lite2/FB, tests có sẵn từ Claude, brief/results/handoff, lỗi mạng Kaggle.
Receipt HEAD/remote/bundle sẽ ghi tại `runs/maintenance-20261117/publish.json` của từngrepo. Không DVCpush hoặc cleanup; artifact Lite2 mới giữ local, chưa backup vào DVCremote.
