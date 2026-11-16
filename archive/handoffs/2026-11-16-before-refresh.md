> Snapshot16/11/2026; trạng thái sống tại docs/operations/continue.md của repo.

# Continue: bàn giao sống, 13/11/2026 10:00 VN

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [bài học vận hành](lessons.md) → trang này.
Artifact, state, exact session/version ưu tiên hơn tóm tắt.

## Đã xác minh

- Code public GitHub `odixe06/fl_veremi_bert`; source train batch khóa **cb193e6fd56fc2f0c6810745cc57d8242c77c366**.
  [Parity v2](../research/v2-parity.md): 20 kiểm tra PASS, chỉ đổi model/phụ thuộc BERT; CANPerFL đúng code v2.
- Local smoke 13 method PASS (100 xe, compile, group_rows 2.048; 2–3 round); LwFedNIDS sửa SDPA head-dim 5.
  `python -m pytest -q tests`: **37 PASS**, log `runs/pytest_partial_batch.log`.
- Controller **flb-batch-s42-20261113-r3** succeeded 09:59:14 VN; marker `BATCHW_FIRST_ROUND`.
  Log `runs/batch-watch/controller-r3.log`; state `/home/odixe/.codex/long-jobs/jobs/flb-batch-s42-20261113-r3/state.json`.
  Completion đã xử lý. Hai controller trước failed ở inventory/plan, không có training launch từ chúng.
- **13 kernel v1 đã push** (reply + `.intent` + `.pushed`); exact status 09:59:08 VN: cả 13 RUNNING.
  W&B đã có 13 run: config world2/client5343/compiled/HP đúng protocol; mới AFPHA có r1 lúc kiểm.
  Chưa pull/audit output, chưa hoàn tất 50 round.
- AFPHA r1 T4: train **573,432 s**, eval **56,981 s**. So v2 seed42 r1: **1,693× train**, **1,355× train+eval**.
  Đã ghi [LESSONS](../../LESSONS.md); bằng chứng `runs/batch-watch/first_round.json`, `initial-health.json`.

## Batch đang chạy

Plan bất biến `runs/batch/plan.json`; không replan, không push lại slug đã có intent. Deferred r3 = 0.
Owner thread **01a1d288-3551-7de3-9e0d-3b0b4f8b3401**; mọi controller delivery queue.

| method | exact kernel | budget phiên đầu (h) |
|---|---|---:|
| nilm | outlookcheck/flb-nilm/1 | 11.50 |
| apfed | outlookcheck/flb-apfed/1 | 11.50 |
| pfedes | anhtruong1909/flb-pfedes/1 | 10.49 |
| perfedskd | thuyanh316/flb-perfedskd/1 | 8.60 |
| v2x_htfl | chuchimnon/flb-v2x-htfl/1 | 8.53 |
| fedtgp | minhtriethihi/flb-fedtgp/1 | 8.31 |
| tinyproto | accanniez/flb-tinyproto/1 | 6.69 |
| fd_ids | heykewtiie/flb-fd-ids/1 | 6.60 |
| afpha | nguyetquedo/flb-afpha/1 | 5.83 |
| fedproto | khanhmay0304/flb-fedproto/1 | 5.44 |
| fedprox | minhtrit06/flb-fedprox/1 | 5.05 |
| fedavg | nguyenminhtran060113/flb-fedavg/1 | 4.73 |
| lwfednids | odixeuit/flb-lwfednids/1 | 2.45 |

**CANPerFL chưa launch**: collector tự dùng output FedAvg r50, fine-tune 10 round, cùng account nguyenminhtran060113.
Phần chưa đủ 50 round tự nối `-pN` khi quota >= 3h; reset thứ Bảy **14/11 00:00 UTC = 07:00 VN**.
Không đổi loss/optimizer/participation/HP; group_rows production 8.192 như v2.

## Watcher / bước tiếp

> **Lỗi thời (13/11):** watcher Codex dưới đây đã dừng. Hiện hành: mục *Trạng thái 14/11 chiều* cuối file.

Watcher cũ đã kết thúc có chủ ý khi thấy r1. **Job hiện hành flb-batch-s42-20261113-watch1 RUNNING**,
supervisor PID **126921**; chỉ chạy `scripts/watch_batch.py`, không plan/launch lại.
Controller log `runs/batch-watch/controller-watch1.log`; durable state
`/home/odixe/.codex/long-jobs/jobs/flb-batch-s42-20261113-watch1/state.json`.
Delivery **queue** (`auto-queue`) về owner thread ghi trên.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/batch-watch/controller-watch1.log`.
Nó giữ `runs/batch-watch/state.json` với `first_round` đã có, nên không đánh thức r1 lần nữa.
Log watcher `runs/batch-watch/watch.log`; poll **1.200 s**. Chỉ worker polling, không model polling.

1. Completion watcher → đọc state + log + artifact thực. ERROR: collect/chẩn đoán/sửa đúng bước, source/version mới;
   giữ kernel khỏe, không restart từ đầu nếu có checkpoint tương thích đã xác minh.
2. ALL_TERMINAL → audit → [results](../research/results.md): 10 metric từ CM, so v2; nêu giới hạn PASS_BOUNDED.
3. Cập nhật handoff hiện hành, kiểm link; không chạy lại plan/launch batch.

## Quyền / việc tay

Owner cho phép 14 baseline seed42, cấu hình v2; không SCAFFOLD/MobiHet/centralized.
Chừa **novahihi, keocaosu, emtrangtri, minhtran0601**; kiểm artifact xác nhận chưa dùng 4 account này, <=2 kernel/account.
Public code GitHub; không data/PDF/credential, không Co-Authored-By. Notebook PRIVATE có W&B key inline theo quyền đã cấp;
key ở version history private vĩnh viễn. Không đổi global credential, không hủy job khỏe, không xóa artifact. Chưa có việc tay.


## Kiểm trực tiếp 13/11 19:15 VN (Codex)

- Watcher PID126921/126926 không còn sống; state `running` cũ là stale. Collect một lượt đã pull10 phiên COMPLETE.
- APFed/NILM/pFedES còn RUNNING. Bảng round và quota chính xác: `runs/batch/status.json`.
- 10 phiên dừng budget r6–32, cần nối; account còn0.7–3.2h, chưa đủ session tối thiểu3h sau margin1.15. Collector chưa push-p2; chờ reset14/11 07:00VN. CANPerFL tiếp tục chờ FedAvg r50.
- Audit partial đang chạy; complete=False không phải utility/training FAIL. Evidence `runs/batch/audit/`; report10metric `runs/batch/report.md` khi audit xong.

- Controller phục hồi: `flb-watch-20261113-evening`, supervisor PID10705, trạng thái `running`; owner thread `01a1d4cf-a312-7411-a6dc-df58a2c8d9f5`, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/flb-watch-20261113-evening/state.json`; log `runs/batch-watch/evening.log`. Poll1200s. Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/batch-watch/evening.log`.

- Audit partial hoàn tất10/10 PASS_BOUNDED; tất cả complete=False. `run_batch.py audit` cố ý trả nonzero ở completion gate do thiếu r50/3run sống/CANPerFL chưa chạy; không phải10artifact FAIL. Kết quả từng method ở `runs/batch/audit/`, 10metric ở `last_metrics`.


## Owner đổi chính sách13/11 19:25 VN

- Đã hủy watcher `flb-watch-20261113-evening` qua controller; durable state cancelled, SIGTERM effective19:25:19. Không tạo watcher BERT mới.
- Owner cho phép nối phiên ngắn: `MIN_SESSION=.9h`, margin1.15 và reserve.4h kết thúc giữ nguyên. Đang chạy collect một lượt để push-p2 trên account còn quota; không đổi scientific settings/source train.
- Generator mới lưu `inductor-cache.tar` sau subprocess, restore trước train; cache lỗi/miss không thay method. NPY build ở working, phiên sau đọc trực tiếp parent cache nếu có. Parent v1 chưa lưu cache: p2 vẫn phải cold build/compile; chưa có bằng chứng speedup T4.

- Owner bổ sung19:28: tận dụng từng round/quota còn lại. Từ lượt kế: min session.55h, margin1.03, reserve kết thúc.15h (9phút), thay ngưỡng.9h/margin1.15/reserve.4h ở trên. Lượt collect đang chạy đã nạp constants cũ; chạy thêm lượt sau để nhận phần quota ngắn còn lại. Account đang chạy giữ reservation, không cấp quota hai lần.

- Collect ngắn13/11 19:25–19:34: pFedES v1 COMPLETE r23, pull + audit PASS_BOUNDED. Đã push-p2 cho AFPHA/FD-IDS/FedAvg/FedProto/FedProx/LwFedNIDS/PerFedSKD/pFedES/TinyProto (9phiên, tổng11.93h); FedTGP/V2X-HtFL đang push với min.55h/margin1.03. Không watcher BERT. Evidence `runs/batch/*-p2.intent`, `.push-reply`, `.pushed`.

- Push ngắn hoàn tất: **11 phiên p2 v1**, tổng13.47h session budget. FedTGP.70h, V2X-HtFL.84h bổ sung. Mỗi phiên mount đúng private parent và kiểm state2rank trước decode. Không replan/relaunch. Kiểm status một lượt lưu `runs/batch/short-session-status.json`.


## Kiểm trực tiếp 14/11 01:50 VN

- Cả 13 phiên mới (APFed v1, NILM v1 và 11 phiên p2) đều COMPLETE sạch sẽ do budget stop.
- Đã pull toàn bộ 13 session output về `runs/pulls/` (100% files ok, không thiếu file).
- Đã kiểm provenance source manifest cho cả 13 run: base SHA `cb193e6fd56fc2f0c6810745cc57d8242c77c366`, 0 overlay, 41 files trùng khớp SHA256 với repo.
- Audit hoàn tất: **13/13 PASS_BOUNDED** (complete=False vì chưa đạt r50; không phải lỗi model/utility).
  - Rounds hiện tại: afpha r37, v2x_htfl r34, perfedskd r33, fedtgp r33, apfed r31, fd_ids r31, fedprox r31, fedproto r29, nilm r29, tinyproto r29, fedavg r27, pfedes r24, lwfednids r14.
  - CANPerFL tiếp tục chờ FedAvg đạt r50.
- Bảng 10 metric cập nhật tại `runs/batch/report.md`; chi tiết từng run ở `runs/batch/audit/*.json`. Trạng thái lưu tại `runs/batch/status.json`.
- Kaggle GPU hiện rảnh; chờ quota reset thứ Bảy 00:00 UTC (07:00 VN) để tiếp tục nối phiên p3 tới r50.

## Nối phiên 14/11 06:30 VN (Claude Code)

Ưu tiên phương pháp 1 model chung (state ~0,3 MB/rank). Chuyển chéo account: `scripts/transfer_global.py <method>:<account>`
→ pull `<run>/` từ owner cũ, kiểm state 2 rank cùng round, dataset private `<account>/<run>` (slug = tên run, file ở gốc để glob
`**/<run>/state.pt` khớp), kiểm size state qua `scripts/list_dataset.py`, push `-pN` với `--resume-dataset` (generator mới).
Chuỗi sau đó sống trên account mới. Kaggle tự giải nén `inductor-cache.tar` → phiên chuyển compile lạnh (không đổi khoa học).
Receipt `runs/transfer/<slug>/`, log `runs/transfer/transfer.log`. Dataset thừa `takids1909/flb-xfer-fedavg-r27` (bản thử, không dùng).

| run | phiên mới | account | budget | từ round |
|---|---|---|---:|---:|
| fedavg | flb-fedavg-p3 | takids1909 (chuyển) | 2,23 h | 27 |
| fedprox | flb-fedprox-p3 | catbaochau (chuyển) | 2,15 h | 31 |
| fd_ids | flb-fd-ids-p3 | odixe0502 (chuyển) | 1,30 h | 31 |
| lwfednids | flb-lwfednids-p3 | odixeuit | 0,80 h | 14 |
| apfed | flb-apfed-p2 | outlookcheck | 6,16 h | 31 |
| pfedes | flb-pfedes-p3 | anhtruong1909 | 0,57 h | 24 |

Cả 6 RUNNING 06:30. AFPHA (global, r37) chưa nối: hết account rảnh trước reset. `collect` tự push APFed lên outlookcheck — account
đang chạy mobihet T143 → 2 slot đầy. EXCLUDE thêm dinhan25/annhtnt/trietbackup/annhannh331 (mobihet B21).
Sau reset 07:00: `python scripts/run_batch.py collect` nối các run còn lại cùng owner. Chưa có watcher.

## Trạng thái 14/11 chiều (hiện hành)

- **Watcher:** `/home/odixe/project/mobihet/scripts/watch_chains.sh` (setsid, sống qua đóng phiên), tick 1200 s, mỗi tick
  `FLB_SEED={42,41,43} run_batch.py collect` (seed chỉ chạy khi có `runs/batch-s<seed>/plan.json`). Log
  `/home/odixe/project/mobihet/runs/chain-watch/watch.log`, `bert-<tick>.log`. Dừng `pkill -f watch_chains.sh`.
- **Seed 42 audit:** AFPHA, APFed, FD-IDS, FedAvg, FedProx r50 `PASS_BOUNDED` complete (`runs/batch/audit-summary.json`,
  `runs/batch/report.md`). CANPerFL RUNNING. 8 run còn lại đang nối phiên. Audit lại: `python scripts/run_batch.py audit`
  (assert cuối chỉ PASS khi đủ 14 run).
- **Seed 41:** 13 phiên p1 đã push (plan `runs/batch-s41/plan.json`); CANPerFL s41 tự đẩy khi FedAvg s41 r50.
- **Seed 43:** NILM/APFed trên chuchimnon; 11 còn `deferred`.
- **Sửa 14/11:** `launch_deferred` trước chỉ nhận account 0 kernel bận → bỏ phí slot trống thứ hai; giờ nhận account bận 1.
  EXCLUDE bỏ 3 account B20 (novahihi/keocaosu/emtrangtri, B20 xong); giữ minhtran0601 + 4 account B21.

## 15/11 hiện hành — owner mở T3 BERT và nối baseline 3 seed

- Owner: pull/nối baseline seed42/41/43 khi cần; thêm T3 FeatureBERT seed41/42/43. T3 cùng code/cấu hình DAGSNet, chỉ khác model/engine. Không thay scientific settings baseline.
- Collect r1 không có PATH conda nên status UNKNOWN rỗng, không chứng minh kernel lỗi. Đã sửa PATH worker và start job `flb-three-collect-20261115-r2`, supervisor PID191137, delivery queue; log `runs/three-collect-20261115-r2.log`; state `/home/odixe/.codex/long-jobs/jobs/flb-three-collect-20261115-r2/state.json`. Đây là collect một lượt + nối bằng launcher cũ, chưa là completion toàn bộ 3 seed.
- T3 source `runs/t3/port.json`; launcher `scripts/run_t3_batch.py`; reserved owners41 dinhan25 /42 annhtnt /43 trietbackup (đã EXCLUDE khỏi baseline). Cap11.5h/run; thiếu r50 phải nối, không restart. Baseline còn sống giữ nguyên.
- Real-row tier loss/gradient PASS: `python scripts/check_t3_port.py` (FeatureBERT77904 params, B=0; so từng xe unpadded 64/32/16 dòng).
- Smoke compile 100 client/2round cosine50 đang chạy: controller `flb-t3-smoke-20261115`, supervisor PID192294, log `runs/t3/smoke.log`, state `/home/odixe/.codex/long-jobs/jobs/flb-t3-smoke-20261115/state.json`, delivery queue. Group_rows1024 chỉ cho bộ nhớ GPU local; production4096 giống T3 DAGSNet. Chưa launch BERT T3; completion phải kiểm finite/applied/CM/state, resume rồi launch trong quyền owner.
- Không sửa source T3 trong khi smoke sở hữu. Không baseline watcher trùng; collector một lượt trên có thể chạy dài. Khi kết thúc: kiểm UNKNOWN/partial, tiếp tục deterministic watcher1200s nếu còn kernel.

- Smoke đầu dừng trước decode do parquet local đã offline (`label_mapping.json` không còn ở data root). Cached100 X/Y/off/ids và train_sizes vẫn có. Smoke r2 dùng `V4_ROOT=.../fl_veremi_bert/knowledge` làm metadata root, đọc dữ liệu thật từ cache100 (không tạo/train dữ liệu giả); production giữ dataset Kaggle. Controller `flb-t3-smoke-20261115-r2`, supervisor PID192588, log `runs/t3/smoke-r2.log`, state `/home/odixe/.codex/long-jobs/jobs/flb-t3-smoke-20261115-r2/state.json`, queue; owner thread `01a1db5a-19af-7920-a785-66374f2ab856`. Job r1 failed, không còn sở hữu source.

## T3 smoke completion 15/11 01:52 VN

- `flb-t3-smoke-20261115-r2` succeeded; artifact `runs/t3/smoke/`: rounds1–2, CM đủ100xe, 10metric recompute, finite, skip≤1%, mọi xe applied, strict FeatureBERT77904 rebuild và RNG/state PASS (`audit.json`). Allocator báo retry OOM khi chọn workspace local4GB nhưng run hoàn tất, audit finite PASS; không xóa guard hoặc đổi production group4096.
- Controller kế `flb-t3-launch-watch-20261115`, supervisor PID194998, owner thread `01a1db5a-19af-7920-a785-66374f2ab856`, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-launch-watch-20261115/state.json`; log `runs/t3/launch-watch.log`.
- Worker `scripts/t3_launch_after_resume.sh`: audit smoke2 → clone artifact sang smoke-resume → resume đúng r2→r3 (cosine50, giữ state/RNG) → audit3 → `run_t3_batch.py launch` → watch exact3 kernels1200s. Nếu bất kỳ stage lỗi, dừng trước stage kế. Không launch tay trùng khi worker sống.
- Chưa chứng minh T4/2rank hoặc full r50. Completion terminal phải pull + audit, thiếu round thì nối checkpoint hai rank; không restart fresh.
- Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/t3/launch-watch.log`.

## Phục hồi launcher T3 15/11 01:55 VN — hiện hành

- Resume smoke r2→r3 thành công, `runs/t3/smoke-resume/audit.json` PASS (CM/10metric/state/RNG/rebuild). Không cần train lại smoke.
- Worker `flb-t3-launch-watch-20261115` failed **trước push**: random nbformat cell IDs thay notebook hash. Repro xác nhận khác đúng `/cells/*/id`, overlay source không đổi. Fix cố định cell IDs, rebuild PASS byte-identical. Giữ batch cũ `runs/t3/batch-before-cell-id-fix.json`, receipt `cell-id-fix.json`; không có intent/pushed trước sửa.
- Worker thay thế **flb-t3-launch-watch-20261115-r2**, supervisor PID196485, queue về thread `01a1db5a-19af-7920-a785-66374f2ab856`; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-launch-watch-20261115-r2/state.json`; log `runs/t3/launch-watch-r2.log`. Command `scripts/t3_launch_verified.sh`: audit3 → live quota/slots/identity → push seed41/42/43 → exact-status watch1200s. Không chạy lại worker cũ hoặc smoke-resume clone.
- Source scientific không đổi sau smoke; không đổi model/HP để sửa lỗi launcher. Mọi completion cần kiểm marker/artifact thực trước kết luận launched/completed.

## Phục hồi inventory T3 15/11 01:58 VN — hiện hành

- Worker r2 lỗi **trước intent/push** vì inventory dinhan25 gặp ref lịch sử `sgs-echohack2-c4` HTTP404. Reprobe trực tiếp xác nhận; không bỏ UNKNOWN hoặc suy thiếu credential. Không lỗi training.
- Seed41 chuyển sang **minhtran0601** (DAGSNet T341 cùng owner dùng1slot); phải live-gate quota còn≥13.225h và slot<2 trước push. Seed42 annhtnt /43 trietbackup giữ nguyên. Không source/HP đổi. Receipt `runs/t3/account41-replan.json`, batch trước giữ `batch-before-account41-replan.json`.
- Controller **flb-t3-launch-watch-20261115-r3**, supervisor PID198101, delivery queue về thread `01a1db5a-19af-7920-a785-66374f2ab856`; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-launch-watch-20261115-r3/state.json`; log `runs/t3/launch-watch-r3.log`. Chỉ launch/watch, không train lại smoke. Gate stdout/stderr giờ lưu `<slug>.inventory.log`, unknown vẫn chặn push.
- Không launch trùng khi controller sống. Kernel launched chỉ khi `<slug>.pushed` tồn tại cùng exact version reply.

## 15/11 02:00 VN — T3 đã push; baseline watcher hiện hành

- T3 3/3 **đã push version1**: seed41 `minhtran0601/flb-t3-s41/1`, seed42 `annhtnt/flb-t3-s42/1`, seed43 `trietbackup/flb-t3-s43/1`; intent + pushed có đủ. Watch lần đầu01:58:30: 1RUNNING/2QUEUED. Chưa có artifact r50. Controller T3 `flb-t3-launch-watch-20261115-r3` tiếp tục watch1200s, không launch lại.
- Baseline collector r2 pull seed42/41, nối7run seed41 p2 rồi lỗi trước push PerFedSKD vì quota odixe0502 thay đổi. Run đã push giữ nguyên; seed43 chưa tới trong lượt đó.
- Sửa riêng collector: quota drift trước intent → WAIT, tiếp tục run khác (như slots busy). Regression thực qua collect2run RED→GREEN; `tests/test_batch.py`4PASS. Không thay source train baseline.
- **Watcher baseline mới `flb-three-watch-20261115`**, supervisor PID200707, queue về thread `01a1db5a-19af-7920-a785-66374f2ab856`; state `/home/odixe/.codex/long-jobs/jobs/flb-three-watch-20261115/state.json`; log `runs/three-watch-controller.log`.
- Worker `scripts/watch_three_seeds.py`: collect+nối42/41/43 tuần tự, seed đủ terminal→audit; mọi lỗi thật ghi per-seed log và xử lý các seed khác trước báo lỗi. Chỉ success khi3seed audit đủ; còn round thì worker chờ1200s. Lock `runs/three-watch/worker.lock`; trạng thái/log mỗi seed `runs/three-watch/`.
- Không watcher baseline cũ còn sống; không khởi watcher trùng. Gate thiếu quota phải chờ, không restart run hoặc nới ngân sách.

## Kiểm trực tiếp 15/11 13:17 VN

- T341 DAGSNet RUNNING; BERT T3 seed41 COMPLETE, seed42/43 RUNNING (exact v1 query). COMPLETE chưa chứng minh đủ r50.
- Baseline BERT seed42: 14/14 audit PASS_BOUNDED complete=True. Seed41/43 mỗi seed còn APFed/NILM/LwFedNIDS/pFedES; APFed r42, NILM r28 đang WAIT quota (0.3–0.4h); watcher hiện hữu tiếp tục collect/nối, không tạo trùng.
- Pull BERT T3 seed41: controller `flb-t3-s41-pull-20261115`, supervisor PID359562, delivery queue, state `/home/odixe/.codex/long-jobs/jobs/flb-t3-s41-pull-20261115/state.json`, log `/home/odixe/project/fl_veremi_bert/runs/t3/pull-s41.log`; owner thread theo state controller. Output `runs/t3/pulls/flb-t3-s41` trong repo BERT. Completion phải audit artifact/round/RNG/config; thiếu r50 thì nối đúng checkpoint. Chưa audit T3 seed41.
- Các watcher cũ T341/T3 BERT/baseline vẫn sống; không sửa file chúng sở hữu. Không Git/DVC push, không xóa artifact.

## Owner 15/11 — cho phép nối chéo account

- Owner cho phép chuyển run thiếu quota sang account khác, giữ checkpoint/source/HP/RNG và tổng round.
- Quota sống: novahihi17.42h, thuyanh31616.06h, accanniez15.35h, minhtriethihi14.61h. Controller `flb-cross-resume-20261115` chuyển tuần tự APFed41→novahihi, NILM41→thuyanh316, APFed43→accanniez, NILM43→minhtriethihi qua `scripts/transfer_global.py`; kiểm inventory/slot/quota trước push. Chưa gọi launched trước pushed receipt.
- Log `/home/odixe/project/fl_veremi_bert/runs/cross-resume-20261115.log`; state `/home/odixe/.codex/long-jobs/jobs/flb-cross-resume-20261115/state.json`; delivery queue. Watcher baseline hiện hữu sẽ nhận head mới sau receipt. Completion cần audit parent/artifact, xử lý stage lỗi, không restart từ r1.

## T3 BERT seed41 pull/audit 15/11 13:40 VN

- Pull job succeeded exit0: 147/147 files, 0missing, 3902MiB. Artifact `runs/t3/pulls/flb-t3-s41` trong repo BERT.
- CM/support/10metric audit PASS_BOUNDED r1–36, complete=False (`runs/t3/audit-s41.json`). State audit PASS r36 cả2rank, 2672+2671xe, finite private banks, strict FeatureBERT rebuild, source overlay/base và HP khớp batch, RNG/gen hai rank (`runs/t3/state-audit-s41.json`). Không stored predictions nên không claim replay inference.
- Nối r36→50 cùng owner minhtran0601: job `flb-t3-s41-resume-20261115`, PID363969, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-s41-resume-20261115/state.json`, log `/home/odixe/project/fl_veremi_bert/runs/t3/resume-s41.log`. Live gate quota≥6.18h/slot<2 trước push `flb-t3-s41-p2`; chưa claim launched trước pushed marker. Giữ source/HP/50round schedule, import parent trước decode. Worker watch1200s sau push; completion pull/audit tiếp.

## T3 BERT seed41 resume launcher phục hồi PATH

- Job `flb-t3-s41-resume-20261115` failed trước gửi push: subprocess không tìm thấy `kaggle` trong PATH. Inventory parent hợp lệ, state audit vẫn PASS r36; không training failure. Intent cũ giữ `flb-t3-s41-p2.intent-before-path-fix`; reply lỗi giữ nguyên tới retry. CLI nckh2.2.5 verified.
- Sửa worker prepend thư mục sys.executable vào PATH. Job thay thế `flb-t3-s41-resume-20261115-r2`, supervisor PID364383, delivery queue về thread hiện hành; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-s41-resume-20261115-r2/state.json`; log `/home/odixe/project/fl_veremi_bert/runs/t3/resume-s41-r2.log`. Kiểm inventory/quota/slot lại trước push; không restart r1. Chưa claim launched trước pushed receipt.

## T3 BERT seed41 p2 ERROR — sửa import guard

- Resume r2 push thành công `minhtran0601/flb-t3-s41-p2/1`, sau đó ERROR14:02; pull listing `.log` rỗng, chưa có traceback remote. Notebook frozen có lỗi xác định: generator thiếu --method nên import guard đòi method=fedavg, checkpoint=mobihet_compute. Replay guard checkpoint thật r36 RED→GREEN PASS khi sửa method; không coi đây là traceback remote.
- Giữ p2 intent/pushed/reply và artifact parent. Sửa launcher truyền --method mobihet_compute; thêm assert ô import sinh đúng method. Slug mới p3, parent vẫn phiên gốc r36, không dùng ERRORp2 làm parent. Source train/HP không đổi.
- Controller `flb-t3-s41-resume-20261115-r3`, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-s41-resume-20261115-r3/state.json`; log `/home/odixe/project/fl_veremi_bert/runs/t3/resume-s41-r3.log`. Kiểm quota/slot lại trước push, watch1200s. Chưa kết luận launched/r50 trước receipt/audit.

## T3 BERT seed41 p3 đã push — phục hồi watcher

- Job resume-r3 failed ở watcher, không ở push: receipt xác nhận `minhtran0601/flb-t3-s41-p3/1` đã push. Watcher dùng lại resume-watch của p2, batch SHA mismatch nên dừng trước query. Không launch lại training.
- Sửa launcher dùng thư mục watch theo slug. Watch-only job `flb-t3-s41-p3-watch-20261115`, supervisor PID368898, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-s41-p3-watch-20261115/state.json`; log `/home/odixe/project/fl_veremi_bert/runs/t3/p3-watch.log`; poll1200s, out `runs/t3/flb-t3-s41-p3-watch`. Giữ state p2 làm evidence. Completion cần collect/audit round thực.

## Nối chéo hoàn tất staging/push 15/11 14:55 VN

- Controller cross-resume succeeded exit0; đủ4 ready dataset + parent round + resume_dataset + intent/pushed/version1 verified. APFed41 r42→novahihi/flb-apfed-s41-p3 RUNNING; NILM41 r28→thuyanh316/flb-nilm-s41-p2 RUNNING; APFed43 r42→accanniez/flb-apfed-s43-p3 RUNNING; NILM43 r28→minhtriethihi/flb-nilm-s43-p2 QUEUED (query14:54:53–54). Chưa audit r50.
- Baseline watcher PID200710 sống, đã nhận head mới seed41; seed43 mới push sẽ nhận qua sessions/intent ở lượt kế. Không watcher trùng.
- pFedES41 vẫn r27 WAITquota0.4h. Khởi transfer bổ sung `flb-pfedes41-cross-20261115` tới emtrangtri, live quota/slot gate trước push; log `/home/odixe/project/fl_veremi_bert/runs/pfedes41-cross-20261115.log`; state `/home/odixe/.codex/long-jobs/jobs/flb-pfedes41-cross-20261115/state.json`, delivery queue. Không claim launched trước receipt. LwFedNIDS41 đang nối p3 từr47; LwFedNIDS43/pFedES43 head trước còn running theo collector.

## pFedES41 chuyển account xong 15/11 15:15 VN

- Job `flb-pfedes41-cross-20261115` succeeded exit0; pull59/59file,0missing. Dataset `emtrangtri/flb-pfedes-s41` ready, state2rank sizes832642276/832329888 bytes; checkpoint r27 từ `odixe0502/flb-pfedes-s41-p2/1`.
- Receipt intent/pushed + metadata private mount đúng dataset verified; không mount kernel parent trùng. Kernel `emtrangtri/flb-pfedes-s41-p3/1` đã PUSHED cap8.95h, query15:15:34 QUEUED. Chưa có bằng chứng round28/r50.
- Watcher baseline PID200710 vẫn sống, sessions() nhận head mới từ intent/pushed; tiếp tục collect/nối/audit3seed1200s. Không tạo watcher trùng. Không cần thao tác tay.


## Kiểm 2026-11-15T21:21:01.161052+07:00 — hiện hành, thay trạng thái watcher cũ

- Process inventory: không watcher/collector T3 hoặc baseline cũ còn sống (máy đã restart); không dựa state running cũ. Có backup project khác đang chạy, không can thiệp.
- Exact query: `minhtran0601/flb-t3-s41-p3/1`, `trietbackup/flb-t3-s43/1`, `minhtran0601/mh-k-b21-t341-r50/1` COMPLETE. Local BERT41 r50, BERT43 r32; DAGSNet41 r50. Pull local trước phiên thiếu rank1, chưa đủ chứng minh exact-resume; không dùng state-audit-s41-r50 cũ thay audit hai rank.
- Owner yêu cầu xử lý cả DAGSNet nếu chưa xong. DAGSNet42/43 đã audit r50 theo mobihet hub; DAGSNet41 cần hoàn tất audit so CBB41, không launch lại r1.
- Job **t3-collect-all-20261115-evening**, supervisor PID29693, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/t3-collect-all-20261115-evening/state.json`; log `/home/odixe/project/fl_veremi_bert/runs/t3/collect-all-evening.log`. Worker `scripts/collect_t3_completed.sh`: pull BERT41p3/42/43, audit CM + state/source/HP/RNG hai rank tuần tự, rồi pull DAGSNet41. Không model polling.
- Completion: đọc state/log; thiếu file/audit lỗi sửa đúng bước. Nếu BERT43 vẫn r32: quota/slot sống → tạo slug mới resume đúng parent r32, source/HP/RNG/schedule50 giữ nguyên; không chạy `resume_t3_s41.py` cho seed43. DAGSNet41 audit với control `../mobihet/artifacts/cbb-s41/mh-k-cbb-s41-r50`, không dùng CBB42. Sau đó cập nhật results/report10metric.
- Script migration: [index](../../scripts/README.md). Watcher + bounded CLI chép nguyên byte; launcher BERT dùng helper và HP reference local. Shared originals mobihet giữ vì DAGSNet dùng. Compile/import/status-parser + shell syntax PASS; link active PASS. Không sửa source train đóng băng; không Git/DVC push.
- Baseline watcher cũ chết: sau T3, kiểm head exact trước phục hồi watcher; không chạy launcher batch fresh. Chưa cần thao tác tay.


## Completion xử lý 2026-11-15T21:58:19.503742+07:00 — hiện hành

- Job `t3-collect-all-20261115-evening` succeeded/exit0, marker thật; pull BERT41p3 164/164, BERT42 145/145, BERT43 139/139, DAGSNet41 188/188; 0missing.
- BERT CM/10metric PASS_BOUNDED + full state/source/HP/RNG hai rank PASS: seed41 r50 macro .914839434; seed42 r35 .911838427; seed43 r32 .910273506. Report `runs/t3/report.md`, CM audits `runs/t3/audit-<slug>.json`, state `runs/t3/state-audit-<slug>.json`. Không replay inference từ predictions.
- DAGSNet41 CM audit + state hai rank PASS r50, so đúng CBB41; utility PASS. Evidence `../mobihet/runs/t3-s41/audit.json`, `state-audit.json`, `report.md`. DAGSNet T3 cả3seed hoàn tất; không nối nữa.
- BERT42/43 **đã PUSHED v1** `annhtnt/flb-t3-s42-p2/1` từr35 cap6.5h; `trietbackup/flb-t3-s43-p2/1` từr32 cap7.5h. Live quota trước push15.75h/10.52h, busy0 cả2; audit parent hai rank+generated guard/syntax/metadata PASS, overlay hash giữ nguyên. First exact query21:57:35 cả2QUEUED, chưa chứng minh round mới/health. W&B giữ OFF theo frozen T3; không dùng run baseline W&B làm bằng chứng T3.
- Controller **flb-t3-resume-42-43-20261115**, supervisor PID36825, owner thread `01a1df90-f93b-7872-810f-f8437450fa9d`, delivery queue; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-resume-42-43-20261115/state.json`; log `/home/odixe/project/fl_veremi_bert/runs/t3/resume-42-43.log`. Worker `scripts/resume_t3_remaining.py`: push receipts có đủ rồi watch exact2kernel1200s, không model polling.
- Completion: inspect state/log/terminal; pull exact p2 vào thư mục mới, audit CM+hai rank, kiểm overlap CM round cũ byte-identical; nếu partial nối từ checkpoint mới, không rerun launcher có intent. Nếu ERROR vẫn pull, chẩn đoán trước retry. Cập nhật report10metric/results. Không launch lại seed41/DAGSNet.
- Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/t3/resume-42-43.log`. Chưa cần thao tác tay. Không Git/DVC push hoặc xóa artifact.


## Owner cancel queue + retry 2026-11-16T00:31:37.017251+07:00 — hiện hành

- Owner đã cancel hai T3 p2, yêu cầu push lại và collect/audit kernel hoàn tất (xác nhận pFedES41). Exact query cả `annhtnt/flb-t3-s42-p2/1`, `trietbackup/flb-t3-s43-p2/1` = CANCEL_ACKNOWLEDGED. `emtrangtri/flb-pfedes-s41-p3/1` = COMPLETE.
- Job **flb-t3-retry-p3-pfedes41-20261116**, supervisor PID67647, delivery queue về thread hiện hành; state `/home/odixe/.codex/long-jobs/jobs/flb-t3-retry-p3-pfedes41-20261116/state.json`; log `/home/odixe/project/fl_veremi_bert/runs/t3/retry-p3-pfedes41.log`. Worker `scripts/retry_cancelled_t3.py`.
- Worker query exact cancellation → pull output p2 (seed42 0file; không giả thiếu checkpoint thành fresh run) → parent audit + notebook hash → live quota/slot → push slug p3 version1 → pull pFedES41 đầy đủ + audit CM10metric → watch hai p3 interval1200s. Nếu cancelled p2 có state mới, dừng để audit trước chọn parent. Không sửa worker khi sống.
- Retry notebook giữ nguyên byte code của p2, chỉ metadata id/title/code_file đổi slug; parent seed42 r35/seed43 r32, source/HP/RNG/schedule50 giữ nguyên. Receipts `runs/t3/flb-t3-s4[23]-p3.intent/.push-reply/.pushed`, manifest `retry-p3-batch.json`; chưa claim launched trước receipt.
- Watcher p2 cũ chỉ read status, sẽ terminal/error khi thấy cancellation. Completion p2 đó đã được xử lý ở đây: không retry p2 hoặc launch trùng p3. Watcher mới dùng thư mục `runs/t3/retry-p3-watch`.
- Completion hiện hành: đọc log/receipt/audit pFedES41, verify provenance SHA cb193e6 + source bytes và round50; cập nhật baseline status/report, nếu partial nối checkpoint đã audit. T3 p3 terminal phải pull/audit hai rank; lỗi stage nào sửa stage đó. Không Git/DVC push/xóa artifact. Chưa cần thao tác tay.
- Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/t3/retry-p3-pfedes41.log`.


## Completion watcher p2 đã xử lý — 16/11 00:37 VN

- `flb-t3-resume-42-43-20261115` failed/exit1 vì child watcher exit2 thấy cả2 CANCEL_ACKNOWLEDGED, đúng owner đã hủy queue. Không lỗi train/source, không retry watcher p2. Pull cả2 cancelled p2 =0file, không checkpoint mới.
- Retry p3 cả2 đã PUSHED v1, receipt/reply + notebook SHA + parent metadata verified: annhtnt/flb-t3-s42-p3 từr35, trietbackup/flb-t3-s43-p3 từr32. Không push lại.
- Job hiện hành `flb-t3-retry-p3-pfedes41-20261116` vẫn running, đang pull pFedES41p3 (listing4588file); chưa có audit cuối. Sau pull/audit worker watch p3 interval1200s. State/log giữ như mục trước.

## Kiểm và push lại T3 BERT seed43 — 2026-11-16 02:13 VN

- Owner hủy `trietbackup/flb-t3-s43-p3/1`; exact Kaggle status `CANCEL_ACKNOWLEDGED` lúc 02:11; pull 0 file, không có checkpoint mới. Không nối từ p3.
- Parent không đổi: `trietbackup/flb-t3-s43/1`, r32, state audit `source_hp`, `RNG_two_rank`, `strict_rebuild` PASS. Notebook p4 byte-identical p3, chỉ thay metadata ID/title/code_file; seed43, source, HP, lịch 50 round giữ nguyên.
- Live inventory trước push: trietbackup GPU quota 10.5248747h, busy0, slug p4 chưa tồn tại. `trietbackup/flb-t3-s43-p4/1` **PUSHED**; receipt `runs/t3/flb-t3-s43-p4.intent`, `.push-reply`, `.pushed`, `.inventory.json`; notebook `notebooks/flb-t3-s43-p4/`.
- Exact Kaggle status **RUNNING** lúc 02:13:51 VN, chưa audit output hay round mới; không launch trùng. Chưa tạo watcher mới trong lượt push này. Job `flb-t3-retry-p3-pfedes41-20261116` có state cũ `running` nhưng PID supervisor/worker/child không còn tồn tại lúc kiểm 02:10; không coi job đang sống, xử lý phần pFedES riêng nếu cần.
