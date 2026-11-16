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

Baselinepush-j16 đang chạy: `bert-baseline-backup-20261116-r2`, log `runs/maintenance-20261116/bert-baseline-dvc-push.log`.
T3/evidence chờbaseline succeeded rồi mớiadd/push/status: `bert-t3-backup-20261116-r2`, log `runs/maintenance-20261116/bert-t3-backup-r2.log`.
Không chạyDVCcùngrepo song song. Xác minhhoàn tất riêngtừngtarget bằngstatus-c; chưacoi logquiet làstall.
Cloud upload vàfilesystemmountsync phải ghi riêng. Không cleanup hoặc xóaVFSdirtycache.
Git/bundle cuối sẽghi saupush; trạng tháijob sống tại [continue](continue.md).

## Khôi phục

`dvc pull artifacts/t3-pulls.dvc` hoặc `dvc pull artifacts/baseline-pulls.dvc`; kiểm disk trước.
Snapshotbackup nằm dướiartifacts, không tự chép vềruns. Audit/recoverychỉdùngexactkernel/version/source/HP/roundđãghi.
Manifest lịch sử không đượcaddlại từworkspacepartial.
