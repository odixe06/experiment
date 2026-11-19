# Bàn giao sống —16/11/2026, sau reboot

Đọc [CONTEXT](../../../../../../bert/context.md) → [LESSONS](../../../../../../bert/LESSONS.md) → [bài học vận hành](../../../../../../bert/operations/lessons.md).
Snapshot mốc trước reboot: [handoff16/11](../../../../handoffs/2026-11-16-before-shutdown-handoff.md); không dùng snapshot làm trạng thái sống.

## Quyền và công việc đã xong

Owner đã reboot, yêu cầu hoàn tất và push Git+DVC hai repo; không còn yêu cầu dừng/reboot. Được nối pFedES41 r45→50 và cập nhật Notion. Không xóa artifact/cache, không cancel kernel khỏe, không đổi scientific settings.

- Baseline108session:105COMPLETE+3ERROR lịch sử;105source manifest×41file khớp plan từngseed, không overlay:44session base `cb193e6`,61session base `e8246054`; hai commit chỉ khác tài liệu, code41file cùngbyte. 41/42run đủendpoint; pFedES41 cònr45. CANPerFL endpoint10fine-tune sauFedAvg50.
- **MobiHet=T3**, BERT vàDAGSNet đủr50 cả3seed41/42/43. BERT CM/support/10metric+state/source/HP/RNGhai rank/strict rebuildPASS; parentCM36/35/32round byte-identical. Phạm vi PASS_BOUNDED, chưa replay inference toàn mạng.
- BERT exacthead: `minhtran0601/flb-t3-s41-p3/1`, `annhtnt/flb-t3-s42-p3/1`, `trietbackup/flb-t3-s43-p4/1`. Tất cảCOMPLETE; không launchlại.
- [Results](../../../../../../bert/research/results.md), [8bảng hai model](../../../../../../bert/research/comparison.md), [JSON/nguồn](../../../../../../bert/research/comparison.json). NotionProposal2 đã thay bằngT3/MobiHet, ranking/bold vàgiữchildMininote; [receipt](../../../../../../bert/operations/notion-20261116.json). **pFedES tạm ẩn mọi bảng** theo owner, chờr50/audit.
- Tests:35PASS suiteđầu; sửa fixture, rerun9PASS/1SKIP(parquetoffline); T3compute12PASS; notebook8PASS. Linkactive/diffcheckhai repoPASS. Không sửa production train đểqua test.

## Job sống — thread sở hữu phiên này

Owner16/11 yêu cầu phiên tổng hợp brief pull/audit pFedES cuối, cập nhật Notion/tài liệu và push GitHub.
Đã query exact `outlookcheck/flb-pfedes-s41-p4/2`: COMPLETE. Watcher13587 không còn PID sống; không restart.
Collector hiện hành `pfedes41-pull-audit-20261116`, supervisor12879, deliveryqueue;
log `runs/maintenance-20261116/pfedes-final-pull-audit.log`, state `/home/odixe/.codex/long-jobs/jobs/pfedes41-pull-audit-20261116/state.json`.
Pullfull-j16 do job trên sở hữu. Job phụ thuộc `pfedes41-audit-tables-20261116`, supervisor18361, deliveryqueue,
log `runs/maintenance-20261116/pfedes-audit-tables.log`; worker chờ collector exit rồi audit r50/source/HP/parent45/statehai rank, tái sinh comparison hai repo.
Owner thread hiện tại `01a1e452-d34b-7e80-93fa-a0d80183f302`; không dùng thread/watcher cũ ở dưới để restart.
Đã xác nhận running một lần; không mở collector thứ hai.

Ownerthread `01a1e31b-3489-76b3-a36f-f8ce699825eb`, deliveryqueue. Controllerstate chuẩn dưới `/home/odixe/.codex/long-jobs/jobs/<job>/state.json`; sau reboot luôn kiểm PID/startidentity, không tin `running` cũ.

| Job | PID supervisor | Worker / log (từrootrepo) | Công việc |
|---|---|---|---|
| `pfedes41-v2-watch-20261116` |13587| `scripts/watch_kaggle_batch.py`; `runs/maintenance-20261116/pfedes-v2-watch.log` | Watch exact `outlookcheck/flb-pfedes-s41-p4/2` mỗi1200s; terminal đánh thức phiên đểpull/audit |

Theo dõi: `tail -n 200 -F runs/maintenance-20261116/pfedes-v2-watch.log`.
Chỉ còn chờ thì kết thúc turn; worker/completion đánh thức phiên, không modelpoll.

## pFedES41 — version2 đãpush theo yêu cầu owner

- Parent exact `emtrangtri/flb-pfedes-s41-p3/1`,r45; hai state2672+2671client,finite,HP/source/savedmethodgenPASS. Baseline cũ không lưu torchCPU/CUDA RNG riêng; giữ resume semantics gốc, không gọi fullRNGexactresume.
- Private dataset mới **`outlookcheck/flb-pfedes-s41-r45-final`**, ready94file. **94/94file tải thực đãSHA256 khớp parent**; guard import thật nạp đủhai rankr45; mọi cell ASTparsePASS. Receipt `runs/maintenance-20261116/pfedes-reboot/{dataset-file-hashes,offline-preflight}.json`.
- V1push16:27VN rồiownercancel; đãqueryCANCEL_ACKNOWLEDGED/fullpull0file. **V2đãpush**, exact `outlookcheck/flb-pfedes-s41-p4/2`; reply/version/intent/pushed vàSHA notebook/private2T4PASS. Source `e8246054d476f0d3fe96a3171a4609b8dafd9db4` đúngparent,41file khớpcảcb193e6; budget3.8h,seed41/HP/cosine50 giữ nguyên. Receipt hiện hành [launch](../../../receipts/pfedes41-launch-20261116.json); launch/batch/pushreplyV2 ở `runs/maintenance-20261116/pfedes-repush-v2/`. Gate quota19.6296h/0busy. Check `runs/maintenance-20261116/pfedes-repush-v2/launched.json` và `runs/batch-s41/flb-pfedes-s41-p4.{intent,push-reply,pushed}` trước mọi retry.
- r1–r4 lỗi localchecker, không trainingfail: Kaggle tựgiải nénzip; SDK giữrelativepath; thiếuimportos; r4 importmodule khi chạy scripttrực tiếp. Đã sửa đúngstage, giữdataset/checkpoint, offlinepreflightPASS trước r5. r5exit0/PUSH_VERIFIED vàartifactchecksPASS. Không uploadlại hoặc launchtrùng.
- Job `pfedes41-final-20261116`/notifier trước reboot không sống; không chạy lại. Notification lặp jobcũ chỉ là evidence, không yêu cầu rerun.

## Backup

[Backup record](../../../../../../bert/operations/backup.md) giữtarget/hash/remote. Trước reboot: BERT3manifest vàMobiHet10manifest đãpush-j16/status-c in sync; Gitmainhai repo đãpush, bundleverifyPASS. **Push lại sau rebootVERIFIED16:57:46VN**,BERT3/MobiHet10target đềuinsync; [receipt](../../../receipts/backup-reboot-20261116.json).
OneDrive đãmountlạifuse.rclone,servicePID8985active/running; startupTimeout6h qua `~/.config/systemd/user/rclone-onedrive.service.d/zz-recovery-timeout.conf`. JobDVCcũfail do chờmount60phút; systemd đãkill lầnstart trướcdoTimeout1h. RepairgiữPID/cache, mount/pathhai repoPASS. Lầnđo16:55 còn225dirtyfile/15.37GB; không chứngminh cloudsync. Giữ `~/.cache/rclone/` vàlocalartifact; khôngrestart servicekhỏe/xóacache. Không dvcadd lại manifest lịch sử từworkspacepartial.

## Tiếp theo sau completion

1. Watcherterminal: đọcstate/log+terminal.json (ERROR thìattention.json); COMPLETE/ERROR đều pullfulloutput-j16 về `runs/pulls/flb-pfedes-s41-p4`, gồmstatehai rank. Khôngpushlại; queryerror làUNKNOWN. Watcherfail trước terminal chỉ phục hồiwatcher.
2. p4 đủr50: kiểmsourcee8246054/HP/CM/support/10metric/finite2rank vàparent45CMprefix. Cập nhật done/status/reportseed41. Trong comparisonbuilder bỏ cả3chỗlọc pFedES, táisinhhai repo/Notion, giữchildMininote. Chưa đủr50 thì vẫnẩn pFedES.
3. Backup outputp4/receipt mới bằngmanifestmới; khôngghi đè snapshotpartial. Gitpushhai repo, bundleallrefsverify rồi thay trênmountthật. DVCpush/status-c nối tiếp cùngrepo.
4. DVCrepush đãxácminh/receipt16:57:46; không chạy lại do notificationtrùng. Chỉbackup output/receiptmới khi có thêmartifact; táchmountsync/cloudsync.
5. Bàn giao cập nhật tạiđây, hub chỉtrỏ. Chưa có việc tay bắt buộc; nếu quota/access/mount lỗi ngoài recovery thì báo đúngblocker.

Ownercancelp4v1 đãqueryexact **CANCEL_ACKNOWLEDGED**, fullpull-j16 trả0file; khôngcócheckpointmới hơnr45. Đãhủywatcherv1 bằngcontroller (khônghủytrainingkhác), giữreceiptcũ; notificationcancelcũ khôngrerun. RepushV2receipt đãkiểm ởrepoBERT `runs/maintenance-20261116/pfedes-repush-v2/`; batch/watchdir riêng. Joblaunch13421đãsucceeded,đãxửlý; notificationlặp khôngrepush. Watcherv1cancel làchủý, không phục hồi.
