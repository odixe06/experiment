# Bàn giao sống —16/11/2026, sau reboot; tiếp tục hoàn tất công việc

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md).
Snapshot chi tiết phiên: [milestone16/11](../../archive/handoffs/2026-11-16-before-shutdown-handoff.md). Đây là bàn giao hiện hành duy nhất.

## Quyền owner hiện hành

Owner đã reboot và yêu cầu tiếp tục push/hoàn tất; **không còn yêu cầu dừng để reboot**. Giữ quyền Git+DVC hai repo, Notion và nối pFedES41 r45→50. Không xóa artifact/cache, không cancel kernel khỏe hoặc đổi scientific settings.

## Đã hoàn tất / nguồn chính

- Inventory baseline108session:105COMPLETE+3ERROR lịch sử;105source manifest×41file khớp commit khóa `cb193e6fd56fc2f0c6810745cc57d8242c77c366`, không overlay.
- Baseline41/42run đủendpoint (r50; CANPerFL10fine-tune sauFedAvg50). pFedES41p3 chỉr45, auditPASS_BOUNDED/completeFalse; đangnối riêng.
- **MobiHet=T3**: BERT vàDAGSNet đủr50 cả3seed41/42/43. BERT CM/support/10metric+state/source/HP/RNGhai rank/strictrebuildPASS; parentCM36/35/32round byte-identical. Không prediction→CM replay toàn mạng.
- BERT exacthead: `minhtran0601/flb-t3-s41-p3/1`, `annhtnt/flb-t3-s42-p3/1`, `trietbackup/flb-t3-s43-p4/1`, cả3COMPLETE. Không nối/launch lại.
- Kết quả: [results](../research/results.md), [8bảng hai model](../research/comparison.md), [JSON/nguồn](../research/comparison.json); DAGSNetT3 thêm `../mobihet/docs/research/results-t3.md` (từroot repo).
- Macro-F1 BERT .914770238±.000211920(hạng1), DAGSNet .907956319±.000331909(hạng3). Mean±sampleSD3seed, chưa significance/mixed-model proof.
- NotionProposal2 đã thay bằng T3/MobiHet,8bảng/ranking/bold; childMininote giữ nguyên; pFedESẩn. Receipt [notion](notion-20261116.json), page `40d730cee68d80f0a49ad050377772d2`.
- Tests:35PASS ởsuiteđầu; lỗifixture đã sửa, rerun9PASS/1SKIP(parquetoffline); T3compute12PASS. Linkactive vàdiffcheck2repoPASS. Không đổi production train để qua test.

## Backup/Git

[Backup record](backup.md) là nguồn cho target/hash/remote. BERT3manifest đãpush-j16/status-c in sync; MobiHet10manifest tươngtự. Không dvcadd lại manifest lịch sử từworkspacepartial.
Git docs/source/results/manifest đãpushmain cả2repo; bundleallrefs verifyPASS rồi thay trênOneDrive. Finalhandoff commit sẽchốt saureceiptpFedES; `git log -1`/`git ls-remote origin refs/heads/main` là nguồn đỉnh cuối.
**OneDrivecloud còn7762VFSdirtyfiles** khi kiểm; chỉfilesystemmountsync đãPASS. Giữ `~/.cache/rclone/` vàlocalartifact; upload tiếp khi máy/mount chạy lại. Không gọicloudsyncxong hoặc cleanup.

## Việc duy nhất còn đang thực hiện: pFedES41 phiên cuối

- Parent `emtrangtri/flb-pfedes-s41-p3/1`,r45; state2672+2671xe,bankfinite,HP/source vàsavedmethodgenPASS. Baseline cũ không lưu torchCPU/CUDA RNG riêng; giữ nguyên resume semantics, không gọi fullRNGexactresume.
- Owner mới đã chọn **outlookcheck**,quota19.6296h lúcgate; budgetphiên3.8h (5round×~1682s +startup/save), cùngsource/HP/cosine50.
- **Đangstaging private checkpointdataset; chưa gọiPUSHED trướcreceipt.** Target dựkiến `outlookcheck/flb-pfedes-s41-p4/1`. Không launchtrùng.
- Mainjob `pfedes41-final-20261116`,supervisorPID281246;ownerthread `01a1e31b-3489-76b3-a36f-f8ce699825eb`;deliveryqueue. State `/home/odixe/.codex/long-jobs/jobs/pfedes41-final-20261116/state.json`.
- Log `runs/maintenance-20261116/pfedes41-final.log`; workerbảnbền `runs/maintenance-20261116/workers/resume_pfedes41_20261116.py`.
- Mainworker stage/push →watchexact1200s →fullpull-j16 →audit. Notifier `pfedes41-launch-receipt-20261116`,PID287248 chỉchờreceiptlocal, khôngqueryKaggle/launch. Log `runs/maintenance-20261116/pfedes-launch-receipt.log`.
- Receipt khiđãpush: `runs/maintenance-20261116/pfedes41-final/launched.json`, `batch.json`, `watch/latest.json`; ledger `runs/batch-s41/flb-pfedes-s41-p4.{intent,pushed,push-reply}`. Checkpointdataset/ready tại `runs/transfer/flb-pfedes-s41-p3/ready.json`.

## Bật máy / resume — không chạy lại launcher có intent

1. Đọc hub/lessons/bàn giao. Kiểm mountOneDrive vàPID/startidentity; controllerstate running saureboot cóthểstale.
2. Đọc receiptp4; queryexactaccount/slug/version qua helperaccount-scoped. QUERY_ERROR làUNKNOWN, không restart training.
3. RUNNING/QUEUED: chỉ watch1200s bằngworkerlocal/codex-long-jobs. Không modelpoll, không chạy lại `resume_pfedes41_20261116.py` (nó đã sở hữu launch).
4. COMPLETE/ERROR: pulloutputexactp4 về `runs/pulls/flb-pfedes-s41-p4` bằnghelper-j16, auditCM/support/source/HP/round50; kiểm parentCM45round làprefix. ERROR vẫnpull/chẩnđoán trướcretry.
5. Nếu đủr50/auditPASS: cập nhật done/status/reportseed41; bỏ dòng `NAMES.pop('pfedes')` trong `scripts/build_project_comparison.py`, táisinhbảng/Notion, thêm lạipFedES chỉ sauaudit. Backup outputp4/receipt mới, commit/push/bundle theo quyền16/11.
6. Partial: chỉnối checkpoint tươngthích đãaudit trong quyền tới50, giữscientificsettings; khôngstartfresh đểlấykếtquả.

Cácjob collect/server-profile/test/backup trước đã xử lý; notificationlặp không là yêu cầu rerun. Không còn watcher baseline/T3 cũ cần phục hồi. Chưa có thao tác tay bắt buộc ngoài owner tắt/bật máy theo ý muốn sau thông báo bàn giao cuối.

## Recovery sau reboot — kiểm16/11 15:54 VN

Process launcher/watcher cũ không sống; state running cũ là stale. Exact p4 chưa tồn tại trong owned inventory outlookcheck. Dataset cũ `outlookcheck/flb-pfedes-s41` nhận create-reply nhưng không có trong owned listing, status403; không coi đãready hoặc checkpoint đãmount.
- Job **pfedes41-reboot-launch-20261116**, supervisorPID5355, queue/cùng ownerthread. Worker repoBERT `scripts/recover_pfedes41_final.py`; logBERT `runs/maintenance-20261116/pfedes-reboot-launch.log`; state `/home/odixe/.codex/long-jobs/jobs/pfedes41-reboot-launch-20261116/state.json`. Đã kiểm running. Đóng archive parentr45 (2rank/CM/history, bỏ Inductorcache), private dataset mới `outlookcheck/flb-pfedes-s41-r45-final`; verify archiveSHA sau download, rồi generate/testsyntax/private2T4/sourcecb193e6/seed41/50round và pushp4. Không launchtrùng.
- Job **dvc-reboot-push-20261116**, supervisorPID5438, queue/cùng ownerthread. WorkerBERT `scripts/push_after_reboot.py`; logBERT `runs/maintenance-20261116/reboot-dvc.log`; state `/home/odixe/.codex/long-jobs/jobs/dvc-reboot-push-20261116/state.json`. Đã kiểm running. Chờ mountOneDrive thật (serviceđangactivating); sau đó push-j16 BERT3manifest/MobiHet10manifest nối tiếp và status-c. Không ghi vào thưmục bên dưới mount khi chưa mount.
- Script recovery/backup source mới trackedGit; notebook chứa credential vẫn PRIVATE/gitignored. Parent source/model/loss/HP/schedule không đổi; cachemiss chỉ compile lại.
- Completion launch: đọc launched.json/intent/pushed/exactversion, querymộtlần; sau đó watch1200s bằng worker→terminal pull/audit. Đủr50 mới thêm pFedES trở lại comparison/Notion và backup outputp4. Không modelpoll trong lúc chờ.
- Completion DVC: kiểm exit0+status-c in sync hai repo; cập nhật backuprecord. Cloudupload tách riêng mứcmount; giữVFSdirtycache.
