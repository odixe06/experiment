# Bàn giao sống — 16/11/2026

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [bài học vận hành](lessons.md).
Artifact/source/config/round thực ưu tiên bản tóm tắt. Lịch sử: [snapshot trước cập nhật](../../archive/handoffs/2026-11-16-before-refresh.md).

## Quyền owner phiên 16/11

Ưu tiên kiểm toàn bộ kernel hoàn thành, pull đa luồng và audit; cập nhật tài liệu hai repo BERT/mobihet; gọn CONTEXT;
cập nhật Notion Proposal 2 chỉ dùng T3 tên MobiHet bên cạnh 14 baseline; commit/push Git và DVC cả hai repo.
Ranking performance theo macro-F1 giảm dần, communication tổng50round **gồm bootstrap** tăng dần (cột riêng không bootstrap), compute tăng dần.
Không xóa artifact, không khởi training mới trong bước collect này; không đổi source/HP frozen.

## Job đang sở hữu collect/audit

- `repo-collect-audit-20261116`, supervisor PID269238; owner thread `01a1e31b-3489-76b3-a36f-f8ce699825eb`; delivery queue.
- State `/home/odixe/.codex/long-jobs/jobs/repo-collect-audit-20261116/state.json`.
- Log `runs/maintenance-20261116/collect.log`; worker `/tmp/collect_repo_20261116.py`.
- Status toàn bộ baseline42/41/43 (4 query đồng thời), pull phiên terminal chưa done bằng16luồng, audit từng seed;
  T3 head41p3/42p3/43p4 pull đầy đủ + CM/state/source/HP/RNG hai rank khi terminal. Không launch/nối training.
- Đã kiểm running lúc14:05 VN. Process watcher/collector cũ không sống lúc đầu phiên. Không mở collector trùng.
- Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/maintenance-20261116/collect.log`.

## Trước khi có completion

T3 BERT41 r50 audit CM+state PASS;42 r35/43r32 là kết quả partial cũ, head mới cần audit.
DAGSNet T3 cả3seed đã r50; evidence `../mobihet/runs/t3-s41/` và `../mobihet/runs/b21-r50/`.
Không dùng seed42/43 partial vào headline r50. Baseline seed42 trước phiên14/14 complete; seed41/43 cần lượt collect hiện hành.

## Sau completion — việc đã được ủy quyền

1. Đọc log/state/inventory; kiểm provenance baseline base `cb193e6`, CM/support/10metric/round50 (CANPerFL10), giới hạn PASS_BOUNDED.
2. T3: audit hai rank, overlap CM với parent, source/config và rounds; xác định đủ r50 hay partial. Không restart run để sửa collector.
3. Tính bảng performance/bytes/client-compute/server-compute hai model; sửa method/proposal/results + mọi thông tin hiện hành sai.
4. Notion `40d730cee68d80f0a49ad050377772d2`: thay mô tả + bảng cũ bằng T3/MobiHet; tổng8bảng (2performance+2communication+2client+2server). Giữ child pages nếu có.
5. Kiểm link active hai repo, test thay đổi code có sẵn; DVC repo BERT chưa init/remote, dùng OneDrive sibling đã cấu hình ở mobihet; kiểm mount/disk và manifest đầy đủ trước add.
6. DVC push nhiều luồng, status-cloud xác minh; commit/push toàn bộ thay đổi hợp lệ hai repo, không credential/notebook có key/PDF. Bundle verify. Không xóa local.

## Việc tay

Chưa phát hiện blocker cần owner.

## Job phân tích độc lập cùng phiên

- `repo-server-cost-20261116`, PID272088: profile server FedTGP/V2X-HtFL bằng FlopCounterMode trên số prototype thật; log `runs/maintenance-20261116/server-cost.log`.
- `mobihet-inventory-20261116`, PID272971: query B20/B21/T341 đã push; log `/home/odixe/project/mobihet/runs/maintenance-20261116/inventory.log`.
- `dagsnet42-evidence-20261116`, PID273507: restore chọn lọc config/rounds/CM endpoint14baseline DAGSNet42 từ OneDrive DVC, kiểm MD5; log `runs/maintenance-20261116/dagsnet42-restore.log`. Không sửa manifest/workspace partial repo DAGSNet.
- Cả3 delivery queue, owner thread như collector; state `/home/odixe/.codex/long-jobs/jobs/<job>/state.json`. Kết quả mới phải inspect khi completion.
- Đã cập nhật method T3 đúng Laplace local/global LA, loss synthetic r1 hard/r2+Q, dose/floor; protocol BERT sửa scope3seed+T3; hub gọn. Chưa cập nhật bảng Notion/Git/DVC: chờ audit ưu tiên owner.

## Phân tích đã chuẩn bị, chưa publish

`scripts/build_project_comparison.py` sinh8bảng + JSON evidence; yêu cầu mọi baseline/T3 đủ endpoint và audit.
DAGSNet42 đã restore chọn lọc từ14manifest DVC, MD5 PASS (`runs/maintenance-20261116/dagsnet42/restore.json`);
không thay manifest cũ. Mobihet inventory24/24 COMPLETE; artifact/audit B20/B21 trước đó phải đối chiếu marker.
DAGSNet CANPerFL41/43 audit PASS_BOUNDED rounds10 nhưng `complete=False` do auditor cũ đòi50; script nhận đúng endpoint10 **chỉ CANPerFL**, không nới cho phương pháp khác.
Server profiler lần đầu lỗi API `env_at(w)` ở DAGSNet và sum nhầm chiều count; kết quả lần đầu không hợp lệ.
Job thay thế `repo-server-cost-20261116-r2`,PID274259,queue; log `runs/maintenance-20261116/server-cost-r2.log`;
script `/tmp/profile_server_20261116.py` sửa sum fold axis1 và env_at(1.0), ghi đè output profile sai. Kiểm Np~hàng chục nghìn, không32.
Sau collect xong chạy build_project_comparison.py, xử lý assert thiếu bằng chứng; tuyệt đối không bỏ completeness gate để publish partial.

## Completion server-cost xử lý16/11

Job `repo-server-cost-20261116` failed đã xử lý: lỗi env_at thiếu width; output32prototype sai đã bị thay, không sử dụng. Bảnr2 succeeded, Np69993/batch3500 đã kiểm từ artifact; BERT FedTGP30.242GFLOP/V2X33.389GFLOP; DAGSNet139.847/190.187GFLOP mỗi round theo phạm vi matmul. Kết quả tại `runs/maintenance-20261116/{fl_veremi_bert,fl_dagsnet_veremi_v2}-server.json`.
Inventory mobihet24/24COMPLETE; restore DAGSNet42 MD5 PASS. Baseline105COMPLETE+3ERROR lịch sử, chưa đồng nghĩa108run khoa học. Collector đang audit; không mở trùng.
Job kiểm tra `repo-verify-20261116`, supervisorPID275566, đã kiểm running, delivery queue cùng ownerthread; log `runs/maintenance-20261116/tests.log`, state `/home/odixe/.codex/long-jobs/jobs/repo-verify-20261116/state.json`. Kiểm baseline tests + link hai repo + T3 compute regression; không training production.

## Completion mobihet-inventory xử lý16/11

Inventory24/24exact kernel COMPLETE. Đã kiểm từng slug có artifact/config local và đối chiếu audit B20/B21/T341; mọi artifact check PASS, riêng T4P utility FAIL vẫn giữ verdict. Không pull trùng24phiên đã lưu. T3DAGSNet3seed r50 macro .907956319 ± .000331909(sampleSD), bảng12metric tại `docs/research/results-t3.md` trong repo mobihet. Collector BERT vẫn sở hữu phần baseline/T3BERT; không khởi trùng.

## Backup MobiHet đang chạy16/11

`mobihet-dvc-backup-20261116`, supervisorPID277319, deliveryqueue về ownerthread `01a1e31b-3489-76b3-a36f-f8ce699825eb`; đã kiểm running. State `/home/odixe/.codex/long-jobs/jobs/mobihet-dvc-backup-20261116/state.json`; log `/home/odixe/project/mobihet/runs/maintenance-20261116/backup.log`. Worker `/tmp/backup_mobihet_20261116.py`: chỉ dvcadd run mới chưa có manifest trong b21-chain/b21-transfer/t3-s41 + sanitized receipts/inventory; push-j16 rồi status-c. Không đè manifest partial cũ, không xóa. Target/log chi tiết trong mobihet/runs/maintenance-20261116/. Sau completion kiểm manifest/status thực, cập nhật backup record; Git/bundle và BERT DVC vẫn chờ.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/mobihet/runs/maintenance-20261116/backup.log`.

Completion DAGSNet42 restore đã xử lý:14method MD5+CMsupport5343×16 PASS, receipt `runs/maintenance-20261116/dagsnet42-verified.json`. Baseline BERT105session COMPLETE đã kiểm executedsource41file/session với frozen GitSHA+manifest, PASS (`baseline-provenance.json`, tái kiểm `scripts/verify_batch_provenance.py`). BaERROR lịch sử không được gọi scientificPASS; head kế tiếp đánh giá riêng bởi collector. Không cần pull lại các phiên này.

## Owner16/11 bổ sung pFedES41 phiên cuối

pFedES41p3 COMPLETE nhưng audit chỉr45, không đủ50. Owner yêu cầu push bổ sungr45→50 và **tạm bỏ pFedES khỏi mọi bảng**. Script comparison đã loại method này (không dùng partial/2seed thay3seed).
Job `pfedes41-final-20261116`,supervisorPID281246,deliveryqueue cùng ownerthread,đã kiểm running; state `/home/odixe/.codex/long-jobs/jobs/pfedes41-final-20261116/state.json`;log `/home/odixe/project/fl_veremi_bert/runs/maintenance-20261116/pfedes41-final.log`. Worker `/tmp/resume_pfedes41_20261116.py` audit2rankr45/finite/gen → chọn owner quota≥4.2h+slot → private checkpointdataset → pushp4cap3.8h (5round×~1682s +startup/save) → watch1200s → fullpull/audit. Emtrangtri quota0, phải chuyểnaccount; chưa claim launched trước receipt.
Historical baseline state không lưu riêng torchCPU/CUDARNG; giữ saved methodgen và resume semantics gốc, không gọi exactfullRNGresume. Source/HP/50roundschedule/privatebanks giữ nguyên; không chạy r1. Không sửa source tốc độ trong phiên cuối. Saucompletion kiểm artifact+round50, rồi mới thêm lại pFedES vào bảng.
Theo dõi: `tail -n 200 -F -- /home/odixe/project/fl_veremi_bert/runs/maintenance-20261116/pfedes41-final.log`.

## Owner yêu cầu chuẩn bị tắt máy

Hoàn tất các task đang dở rồi **dừng phiên**, cập nhật toàn bộ bàn giao để bật máy tiếp tục. pFedES có thể tiếp tục trên Kaggle khi local tắt; không cancel remote. Không tạo nghiên cứu/training mới ngoài phiên cuối đã ủy quyền. Trước dừng: hoàn tất pull/audit/T3/Notion, Git+DVC/bundle; ghi trạng thái cloudVFS riêng và exact run còn sống. Copy worker `/tmp/*.py` dùng trong phiên vào `runs/maintenance-20261116/workers/` (repoBERT) để không mất sau reboot.

MobiHet DVC10manifest/438objects VERIFIED mứcmount (push-j16 exit0/status-c in sync); cloud upload chưa xác minh riêng. RepoBERT baselinebackup đang chạy: `bert-baseline-backup-20261116`,supervisorPID282599,queue;state `/home/odixe/.codex/long-jobs/jobs/bert-baseline-backup-20261116/state.json`;log `/home/odixe/project/fl_veremi_bert/runs/maintenance-20261116/bert-baseline-backup.log`. Khởi tạo DVCOneDrive sibling `fl_veremi_bert/dvc-storage`,hardlink snapshot phiên baseline đãdone, không ghi vào pull collector. T3/receipts backup saucollect.

BERT baselinebackup r1failed trước DVCadd vì phát hiện `v4npy/train_X.npy.part`5.49GB còn từ pull full trước. Đây là cache dataset tạm, không phải model/CM đã audit; không xóa hoặc backup file dở. Worker r2 loại toàn v4npy (dataset cache tái tạo được), tái dùng hardlink đã staging đúnginode; chỉ phần cache này được phép .part, runartifact .part khác vẫn chặnbackup. Job `bert-baseline-backup-20261116-r2`,PID283142,queue;log `runs/maintenance-20261116/bert-baseline-backup-r2.log`,state theo tênjob. DVCinit/remote đã xong; chưa xác nhậnpush trước completion.

## Verify r1 lỗi fixture, sửa test16/11

Suite r1=35PASS/4FAIL:2notebooktest namespace thiếu resume_dataset; NPYintegration thấy root offline nhưng không cònparquet; AFPHAtoy thiếu clientsmetadata. Sửa chỉtest: bổ sung field+case datasetresume; skip integration nếu thiếu parquet train/test; AFPHA toy ghi clients.parquet6xe/tmproot. Không sửa production method/source frozen. Job `repo-verify-20261116-r2`,PID283468,queue,state theo tênjob;log `runs/maintenance-20261116/tests-r2.log`; rerun đúng3testfile nhánh lỗi +link2repo+T3compute;35testđãPASS giữbằngchứng r1.

Verifyr2 completion xử lý:9PASS/1SKIP(offlineparquet)/14deselected cho notebook/NPY/AFPHA; T3compute12PASS; linkactive hai repoPASS.35testkhác đãPASS ởr1, không sửa productioncode để làmtestpass. `git diff --check` hai repoPASS, credential-patternscan candidateGit không phát hiệntokenliteral. Log `runs/maintenance-20261116/tests-r2.log`.

## Collector/Notion hoàn tất —16/11

Collector succeeded nhưng baseline41 vẫnincomplete đúngpFedES45; không đọc exit0 thànhallbaseline50. Cả3T3BERT r50, fullpull164/164file cho42/43,0missing; CM/state/source/HP/RNG2rankPASS. PrefixCM so parent36/35/32round byte-identical. Kết quả/8bảng đã ghi `docs/research/results.md`, `comparison.md`, `comparison.json` hai repo. MacroBERT .914770238±.000211920(hạng1),DAGSNet .907956319±.000331909(hạng3).
NotionProposal2 replace/readbackPASS:8heading/table, childmininote giữnguyên, bảngcũđãxóa,T3/MobiHet đượcinđậm; pFedESẩn. Receipt `docs/operations/notion-20261116.json`, page40d730cee68d80f0a49ad050377772d2.
T3backup r1failed DVC lock do baselinepushđangsống, không cólỗiartifact. Worker thaythế `bert-t3-backup-20261116-r2`,PID286160,queue;log `runs/maintenance-20261116/bert-t3-backup-r2.log`,state theo tênjob; chờbaselinebackup succeeded bằngworkerlocal10s rồi mới dvcadd/push/status. Khôngmodelpoll hoặc haiDVCprocesscùngrepo. Workerđãlưu ở `runs/maintenance-20261116/workers/`.
Còn:Gitcommit/push/bundle2repo;DVCBERTcompletion+backuprecord;pfedescheckpointdataset/pushreceipt;gọnbàn giao cuối vàdừngphiên theoowner.

Git source/docsđãpush16/11: BERT `72a5b024ae78f6f3b5b3d999f5c81612911c7518`, MobiHet `5dfae538428b287b649b40fee36b309e8ce39dc8`, main; còncommitbàn giao/manifestcuối/bundle saubackup. Notion8bảng đãreplace/readbackPASS,childpagegiữnguyên; source kếtquảhiện hành `docs/research/comparison.md`/`.json`.
Notifier `pfedes41-launch-receipt-20261116`,PID287248,queue;logBERT `runs/maintenance-20261116/pfedes-launch-receipt.log`; chỉchờreceiptlocal,cấmqueryKaggle hoặclaunchthêm. MainpFedESworker vẫnowns staging/push/watch; notifierbáo sớm khiđãpush đểchốtbàn giao/tắtmáy, khôngchờtraining3.8h. State theo tênjob ởlong-jobs/jobs;workerđãcopydurable trongruns/maintenance/workers.

Baselinebackup-r2 completion VERIFIED:108session,17129files,6.457GB logical;8396objects pushed;status-c in sync,manifestc994b3163bee65d450cf08c0cbbcad98.dir. T3/evidence jobđãqua baselinegate. Kiểm foldcounts axis1 train/val:32685166+4676107=37361273, khớp train_sizes từngxe; Np69993 hợp lệ, khônglẫntest.

BERTT3/evidencebackup-r2 completion VERIFIED:811objects pushed/status-c in sync;T3logical10.990GB/896files,evidence20.06MB/720files. BackupBERT3manifest đãxongmứcmount. OneDrivecloud còn7762VFSdirtyfiles lúc kiểm; không xóaVFS/cache hoặc gọicloudsyncxong. Localartifact cònnguyên; rcloneupload tiếp khi máy/mount chạy lại. pFedES đangstagingdatasetoutlookcheck (quota19.63hđãgate), chưaclaimpush trướcreceipt.
