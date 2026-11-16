# Bàn giao sống —16/11/2026, sau reboot

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md).
Snapshot mốc trước reboot: [handoff16/11](../../archive/handoffs/2026-11-16-before-shutdown-handoff.md); không dùng snapshot làm trạng thái sống.

## Quyền và công việc đã xong

Owner đã reboot, yêu cầu hoàn tất và push Git+DVC hai repo; không còn yêu cầu dừng/reboot. Được nối pFedES41 r45→50 và cập nhật Notion. Không xóa artifact/cache, không cancel kernel khỏe, không đổi scientific settings.

- Baseline108session:105COMPLETE+3ERROR lịch sử;105source manifest×41file khớp source khóa `cb193e6fd56fc2f0c6810745cc57d8242c77c366`, không overlay. 41/42run đủendpoint; pFedES41 cònr45. CANPerFL endpoint10fine-tune sauFedAvg50.
- **MobiHet=T3**, BERT vàDAGSNet đủr50 cả3seed41/42/43. BERT CM/support/10metric+state/source/HP/RNGhai rank/strict rebuildPASS; parentCM36/35/32round byte-identical. Phạm vi PASS_BOUNDED, chưa replay inference toàn mạng.
- BERT exacthead: `minhtran0601/flb-t3-s41-p3/1`, `annhtnt/flb-t3-s42-p3/1`, `trietbackup/flb-t3-s43-p4/1`. Tất cảCOMPLETE; không launchlại.
- [Results](../research/results.md), [8bảng hai model](../research/comparison.md), [JSON/nguồn](../research/comparison.json). NotionProposal2 đã thay bằngT3/MobiHet, ranking/bold vàgiữchildMininote; [receipt](notion-20261116.json). **pFedES tạm ẩn mọi bảng** theo owner, chờr50/audit.
- Tests:35PASS suiteđầu; sửa fixture, rerun9PASS/1SKIP(parquetoffline); T3compute12PASS; notebook8PASS. Linkactive/diffcheckhai repoPASS. Không sửa production train đểqua test.

## Job sống — thread sở hữu phiên này

Ownerthread `01a1e31b-3489-76b3-a36f-f8ce699825eb`, deliveryqueue. Controllerstate chuẩn dưới `/home/odixe/.codex/long-jobs/jobs/<job>/state.json`; sau reboot luôn kiểm PID/startidentity, không tin `running` cũ.

| Job | PID supervisor | Worker / log (từrootrepo) | Công việc |
|---|---|---|---|
| `pfedes41-reboot-launch-20261116-r5` |7400| `scripts/recover_pfedes41_final.py`; `runs/maintenance-20261116/pfedes-reboot-launch-r5.log` | Đãkiểm running; checkpoint verified→livequota/slots→push p4; chỉ nhận PUSHED sau exactversion receipt |
| `dvc-reboot-push-20261116` |5438| `scripts/push_after_reboot.py`; `runs/maintenance-20261116/reboot-dvc.log` | Chờ mountOneDrive thật→push-j16 BERT3manifest rồiMobiHet10manifest→status-c; không ghi dưới mountchưa sẵn sàng |

Theo dõi: `tail -n 200 -F runs/maintenance-20261116/pfedes-reboot-launch-r5.log` hoặc `tail -n 200 -F runs/maintenance-20261116/reboot-dvc.log`.
Chỉ còn chờ thì kết thúc turn; worker/completion đánh thức phiên, không modelpoll.

## pFedES41 — checkpoint đã xác minh, chưa kết luận launch thành công

- Parent exact `emtrangtri/flb-pfedes-s41-p3/1`,r45; hai state2672+2671client,finite,HP/source/savedmethodgenPASS. Baseline cũ không lưu torchCPU/CUDA RNG riêng; giữ resume semantics gốc, không gọi fullRNGexactresume.
- Private dataset mới **`outlookcheck/flb-pfedes-s41-r45-final`**, ready94file. **94/94file tải thực đãSHA256 khớp parent**; guard import thật nạp đủhai rankr45; mọi cell ASTparsePASS. Receipt `runs/maintenance-20261116/pfedes-reboot/{dataset-file-hashes,offline-preflight}.json`.
- Target `outlookcheck/flb-pfedes-s41-p4`, budget3.8h, seed41/sourcecb193e6/HP/cosine50 giữ nguyên. Gate gầnnhất quota19.6296h/0busy; launcher đọc lại ngaytrướcpush. Check `runs/maintenance-20261116/pfedes-reboot/launched.json` và `runs/batch-s41/flb-pfedes-s41-p4.{intent,push-reply,pushed}` trước mọi retry.
- r1–r4 lỗi localchecker, không trainingfail: Kaggle tựgiải nénzip; SDK giữrelativepath; thiếuimportos; r4 importmodule khi chạy scripttrực tiếp. Đã sửa đúngstage, giữdataset/checkpoint, offlinepreflightPASS trước r5. Không uploadlại hoặc launchtrùng.
- Job `pfedes41-final-20261116`/notifier trước reboot không sống; không chạy lại. Notification lặp jobcũ chỉ là evidence, không yêu cầu rerun.

## Backup

[Backup record](backup.md) giữtarget/hash/remote. Trước reboot: BERT3manifest vàMobiHet10manifest đãpush-j16/status-c in sync; Gitmainhai repo đãpush, bundleverifyPASS. **Push lại sau reboot chưa kết luận xong** cho tới receipt jobDVC.
OneDrive service đang phục hồiVFS; lầnđo16:20dirty giảm7762→3294file. Đây là snapshot tiếnđộ, không chứngminh cloudsync. Giữ `~/.cache/rclone/` vàlocalartifact; khôngrestart servicekhỏe/xóacache. Không dvcadd lại manifest lịch sử từworkspacepartial.

## Tiếp theo sau completion

1. Launch: kiểmstate/log+launched.json+intent/pushed/exactversion; queryexactmộtlần. RUNNING/QUEUED→watcherworker1200s, khôngpushlại. TerminalCOMPLETE/ERROR đều pullfulloutput-j16 rồiaudit, không lấy exit0 thayartifact.
2. p4 đủr50: kiểmsourcecb193e6/HP/CM/support/10metric/finite2rank vàparent45CMprefix. Cập nhật done/status/reportseed41. Trong comparisonbuilder bỏ cả3chỗlọc pFedES, táisinhhai repo/Notion, giữchildMininote. Chưa đủr50 thì vẫnẩn pFedES.
3. Backup outputp4/receipt mới bằngmanifestmới; khôngghi đè snapshotpartial. Gitpushhai repo, bundleallrefsverify rồi thay trênmountthật. DVCpush/status-c nối tiếp cùngrepo.
4. DVCcompletion: đọc từngpush/statuslog trong `runs/maintenance-20261116/reboot-backup/`; chỉ nhận `ALL_REBOOT_DVC_PUSHES_VERIFIED` cùng status-c insync. Cập nhậtbackuprecord, táchmountsync/cloudsync.
5. Bàn giao cập nhật tạiđây, hub chỉtrỏ. Chưa có việc tay bắt buộc; nếu quota/access/mount lỗi ngoài recovery thì báo đúngblocker.
