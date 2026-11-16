# Git/DVC backup —16/11/2026

Owner16/11 cho phép push toàn bộ thay đổi hợp lệ, không xóa local. Env `nckh`.

## Đích và phạm vi

- Git `https://github.com/odixe06/fl_veremi_bert.git`,branch `main`.
- DVC default `onedrive`: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/fl_veremi_bert/dvc-storage` (rclonemount thật);cachehardlink,copy.
- Bundle dự kiến `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/fl_veremi_bert/fl_veremi_bert.bundle`, verify trước thay.
- `artifacts/baseline-pulls.dvc`: hardlink snapshot cácsession đãdone,source/config/CM/snapshot vàstatecó lưu; sourcegốc `runs/pulls/`.
- `artifacts/t3-pulls.dvc`: outputT3 đãpull, gồmcheckpoint2rank vàprovenance; sourcegốc `runs/t3/pulls/`.
- `artifacts/research-evidence-20261116.dvc`: audit/report/config/pushreceipt/log/testsupport vàendpointCM DAGSNet42 đãrestore/MD5verify.
- Không backupnotebook cóinlinecredential, cachev4npy/file.part, checkpointdatasetupload tạm hoặc bảnpack/copytrùng ởruns/transfer. Dataset gốc dùng Kaggle/DVCnguồn; recoverymodel dùngcheckpointđãgiữtrongpulls. Không xóa cácfiletạm này.

## Trạng thái hiện hành

Baselinepush-j16 **VERIFIED**:108session,17129files/6457073344B,8396objects pushed, status-c in sync; manifestmd5 `c994b3163bee65d450cf08c0cbbcad98.dir`. Controller succeeded;receipt `runs/maintenance-20261116/bert-baseline-backup.json`.
Job: `bert-baseline-backup-20261116-r2`, log `runs/maintenance-20261116/bert-baseline-dvc-push.log`.
T3/evidence **VERIFIED**:811objects pushed;status-c in sync. T3manifest `33ddae696e279eb52c39e2c8f3a45b91.dir`(10989948636B/896files), evidence `235bf5096e0e7fef2dba373833813360.dir`(20056472B/720files).
Job: `bert-t3-backup-20261116-r2`, log `runs/maintenance-20261116/bert-t3-backup-r2.log`.
Không chạyDVCcùngrepo song song. Xác minhhoàn tất riêngtừngtarget bằngstatus-c; chưacoi logquiet làstall.
Cloud upload: kiểm16/11 còn7762VFSdirtyfiles;filesystemmountsync đãPASS. Rclonecache bền trên đĩa, upload tiếp khi máy/mount chạy lại; không gọi cloud đãsync. Receipt `runs/maintenance-20261116/cloud-upload-status.json`. Không cleanup hoặc xóaVFSdirtycache.
Git/bundle cuối sẽghi saupush; trạng tháijob sống tại [continue](continue.md).

## Khôi phục

`dvc pull artifacts/t3-pulls.dvc` hoặc `dvc pull artifacts/baseline-pulls.dvc`; kiểm disk trước.
Snapshotbackup nằm dướiartifacts, không tự chép vềruns. Audit/recoverychỉdùngexactkernel/version/source/HP/roundđãghi.
Manifest lịch sử không đượcaddlại từworkspacepartial.

## Recovery mount16/11 sau reboot

OneDrive đãmount lại tại `/home/odixe/Cloud/OneDrive`, source `onedrive:`,type `fuse.rclone`; servicePID8985active/running, cả2projectpathPASS. Startup trước timeout1h bịsystemdngắt khi reload/uploadVFS. Drop-in `~/.config/systemd/user/rclone-onedrive.service.d/zz-recovery-timeout.conf` đặtTimeoutStartSec6h;daemon-reloadgiữprocess/cache. [Hành vi timeout systemd](https://www.freedesktop.org/software/systemd/man/255/systemd.service.html).
Job `dvc-reboot-push-20261116` fail ởwaitmount60phút, chưa gọiDVC. Jobthaythế `onedrive-remount-dvc-20261116`,PID12313,deliveryqueue, đangpush-j16/status-c hai repo theo target đãkhóa; logrepoBERT `runs/maintenance-20261116/onedrive-remount-dvc.log`. Chưagọi repushVERIFIED trướcreceipt. Lầnđo16:55:225dirtyfile/15.37GB; đó làsnapshotVFSchung, không phải cloudsync hoặc corruption. Không xóa local/cache.
Receiptmount/repair ởrepoBERT `runs/maintenance-20261116/reboot-backup/{mount-restored,mount-timeout-repair,dirty-after-timeout}.json`; jobhiện hành ở [continue](continue.md).
