# Bàn giao sống — 17/11/2026

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md).
Lịch sử: [snapshot](../../archive/operations-20261116/before/docs/operations/continue.md).

## Trạng thái hiện tại

**Xong (COMPLETE17/11 09:18 +07, pull108file/0thiếu, auditPASS):** pilot [Lite2](../../../mobihet/docs/research/screening-lite2.md) **B + U1** FeatureBERT seed 42,
r1–20 cos50: `annhannh331/flb-lite2-bu1-s42-r20/1`, CAP 8,5 h, base `b93198a` + overlay `methods/mobihet_{compute,lite}.py`
(chưa commit; copy byte-identical từ `../mobihet`, SHA = bản sim đã audit); spec khóa `runs/lite2-pilot/batch.json`,
launcher/collect/audit `scripts/run_lite2_pilot.py`. Watcher detached PID trong `runs/lite2-pilot/watch.pid` (`watch.sh`: poll 1200 s ≤ 32 lượt
→ pull `artifacts/lite2-pulls/` → audit → `DONE`); log `runs/lite2-pilot/watch.log`, theo dõi `tail -f runs/lite2-pilot/watch.log`.
Watcher đã thoát; DONE/audit.json: macro r20 .898173076, Δr20−.004746125, ΔTB−.005172594, bootstrap340663B; verdictPASS.
Bounded auditor chạy lại độc lập trong phiên tổng hợp brief, chưa state2rank local. Không collect lại.
Owner job = phiên Claude Code MobiHet 17/11; phiên mất thì đọc `DONE`/`audit.log`/`audit.json`. Không start watcher/launcher thứ hai.
Luật khóa trước push: audit PASS + Δmacro TB r16–20 và r20 đều ≥ −.010 vs T3 BERT seed 42 (`artifacts/t3-pulls/flb-t3-s42-p3`) + bootstrap ≤ 382 000 B.
**Owner 17/11 hoãn push lại Lite2 BERT r50; không nối r21–50 lúc này, không giữ quota riêng.** Ưu tiên 8 sim FB DAGSNet; FB PASS mới pilot DAGSNet + BERT. Chi tiết bước tiếp: [bàn giao MobiHet](../../../mobihet/docs/operations/continue.md).

Không rerun từ notification cũ.
T3 BERT/DAGSNet đủr50/3seed. Baseline **42/42run** đủendpoint, gồm pFedES41r50.
pFedES exact `outlookcheck/flb-pfedes-s41-p4/2`: pull158file/0thiếu; audit r50/source/HP/parent45/state2rank PASS_BOUNDED.
[Kết quả +10metric](../research/results.md), [audit receipt](../research/pfedes-final-audit.json), [comparison15method](../research/comparison.md).

- `pfedes41-pull-audit-20261116` (supervisor12879) và `pfedes41-audit-tables-20261116` (18361) đã succeeded/exit0, artifact đã kiểm.
  Log `runs/maintenance-20261116/{pfedes-final-pull-audit,pfedes-audit-tables}.log`; state dưới `~/.codex/long-jobs/jobs/<job>/state.json`.
  Ownerthread `01a1e452-d34b-7e80-93fa-a0d80183f302`; completion đã xử lý, không launch/collect lại.
- NotionProposal2 đã thêm pFedES mọi8bảng và10metric, readbackPASS/giữchildMininote; [receipt](notion-20261116.json).
- Dọn context cảhai repo xong:115script nghỉ; [manifest repo này](../../archive/operations-20261116/migration.json) ghi13script đã archive.
  Core/methods/tests/DVC giữ byte; link/AST/importPASS;12test notebook/batchPASS;T3real-loss-gradientPASS.
  Checker parity13/11 đãarchive vì inventoryT3 mới ngoài scope; kết quả20PASS lịch sử vẫn giữ tại [parity](../research/v2-parity.md).

## Tiếp theo

1. GitHub BERT đã nhận kết quả pFedES + dọn context (`0eea5d3`, xác nhận `ac72528`).
   Owner yêu cầu push tiếp mọi thay đổi hiện tại cảhai repo trên `main`, gồm brief/context/archive MobiHet.
   Gitbundle theo [backup](backup.md); receipt xác minh cuối ở `runs/maintenance-20261116/final-publish.json`.
2. Owner chốt sau cùng17/11: compute chấp nhận≤1,01×FedAvg và top5; riêng compute bằng FedAvg là đạt, không cần thấp hơn; performance top5+gap≤.015, communicationtop5.
   Lite2≤1,37× chỉ là gate cũ, không chứng nhận đạt mục tiêu mới. Brief §13–15 đã bổ sung auditpilot2model, seed42 và thiếu evidence.
   Task ưu tiên theo [task_brief](../../../mobihet/task_brief.md) §1 và [FB](../../../mobihet/docs/research/screening-budget.md):
   owner duyệt8sim FB DAGSNet; candidate PASS → pilot DAGSNet + BERT. Lite2 BERT r50 hoãn; không giữ quota riêng.
   DAGSNet Lite2 r50 đã auditPASS gate cũ (.905877; .354116MB; estimator1,364×), chưa đạt đíchFB1,01×.
   MobiHet đã xác nhận đủ8simFB DAGSNet version1; watcher duy nhất ở [bàn giao MobiHet](../../../mobihet/docs/operations/continue.md). Chưa launch pilotFB BERT; chỉ sau simcandidatePASS.
3. Output pFedES cuối giữ local; DVC snapshot trước16:57 không bao gồm output mới. Chưa có lệnh DVCpush mới trong yêu cầu hiện tại.

Không xóa artifact/cache, không đổi protocol/HP/source frozen. Chưa cần thao tác tay.
