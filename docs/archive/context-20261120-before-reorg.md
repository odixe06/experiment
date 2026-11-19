# CONTEXT — fl_dagsnet_veremi_v2 / FL baselines + MobiHet trên VeReMi NextGen v4 (5.343 xe)

Đọc CONTEXT → [LESSONS](LESSONS.md) + [bài học vận hành](docs/operations/lessons.md) → [bàn giao](docs/operations/continue.md).

## Bài toán

IDS V2X 16 lớp, học cộng tác: **mỗi xe nhận là một client** (5.343 client, dataset v4 của repo `mobihet-features`).
Dựng lại **15 phương pháp** trong bảng owner + **MobiHet** (đề xuất của owner: model client khác nhau được, băng thông hạn chế) +
**centralized**, tất cả trên DAGSNet, **cùng một cấu hình** ([protocol](docs/research/protocol.md)). Build cũ (Dirichlet 20/50/100 client,
`odixe06/fl_dagsnet_veremi`, `dangvanhoang1711/FL_DAGSNet`) chưa sát bài gốc và không đồng nhất ⇒ build lại từ đầu, chỉ tham khảo.

## Quyết định owner (29/10/2026)

- Phạm vi: bảng "Chọn" = 1 + 2 ⇒ 15 phương pháp ([danh mục](docs/research/methods.md)), + MobiHet `gkd + la + kdpx` proxy q3_zl, + centralized.
- 50 round; **mọi client tham gia mỗi round, 1 epoch local trên hết dữ liệu train (fold train + val)**; HP chung cố định, headline r50.
- Đánh giá: model toàn cục trên toàn test; model riêng client trên **test của chính xe đó** (gộp = toàn test). Đủ 10 metric.
- DAGSNet được thu nhỏ (giảm kênh, giữ 4 nhánh) nếu centralized macro-F1 val kém full ≤ 0,01; độ rộng chung mọi phương pháp.
- Một seed (42). W&B online. Repo GitHub **public** `odixe06/fl_dagsnet_veremi_v2`; không đẩy data, PDF bài báo, credentials.
- Override owner 30/10 cho lần chạy MobiHet v4 mới: **không dùng W&B**, dùng log/artifact Kaggle; dataset `odixe0502/veremi-nextgen-v4-scaled-per-vehicle`. Chi tiết phiên mới ở bàn giao.
- Task < 1 h chạy local; task dài chạy Kaggle (16 account; CLI mới đăng nhập 6, owner bổ sung sau). Env conda `nckh`.
- Tốc độ train phải tối ưu ⇒ `core/grouped.py`: nhiều client một bước (bmm), dữ liệu thường trú GPU.

## Trạng thái — 01/11 (giờ VN)

- 19/11 mới nhất: audit mixed tăng 40/45 PASS. Case3 V2X đủ3seed đã audit; Case1 V2X s42
  dừng budget tại r42 và đã resume r43→50 trên `minhgia2026`. Cả5 cấu hình còn lại đang RUNNING;
  không còn cấu hình chưa push. `nguynlahhqun` vẫn loại vì worker0 GPU.
- 19/11 11:51ICT:32/45 mixed runs final-CM audit PASS; updated [results](docs/research/mixed/mixed128-all-results.md) and CSV. 4 V2X long runs active,9 other incomplete tuples await continuation/repair.
- 19/11 11:33ICT: 4 V2X long resumes RUNNING on new accounts nguyntngquckhnh/minhgia2026, 2 each. Total23 accounts; current ledger in handoff.
- 19/11 10:25ICT mixed128:30/45 final-CM PASS; all15 incomplete tuples submitted as checkpoint continuations using measured round durations. [Results](docs/research/mixed/mixed128-all-results.md); ledger runs/mixed128-launch-20261119-b.json.
- 17/11 22:02ICT mixed128:case2 đủ15/15 COMPLETE50/final-CM audit PASS.
  Đãpush đủ30run case1/3; toàn bộ45cấu hình đãsubmitted, còn chờtrain/resume.
  [Thiết kế](docs/research/mixed/mixed-model-cases.md), [bàn giao](docs/operations/continue.md).

- 17/11 mixed-model: đã khóa3 bảng xe→model seed2026, giữ nguyên partition từng xe;
  dùng chung mọi method/seed41/42/43. Coverage5343/count/hash/shard tests PASS.
  Mới xong assignment, chưa mixed runner/BERT-FL hoặc push. Xem [3case](docs/research/mixed/mixed-model-cases.md).

- 17/11: bảng4backbone×14method×3seed đã đủ168/168 ô. CNN NILM43 alternate huygia1611
  COMPLETE50, F1=0,82286; marker/manifest đã kéo, history/source audit thêm tạm lỗi mạng.
  Xem [bảng live](docs/research/backbones/four-backbone-seed-results-live.md) và [bàn giao](docs/operations/continue.md).

- 16/11 22:26VN: theo owner yêu cầu push ngay, NILM43 alternate trên huygia1611 đã RUNNING v1;
  bản voxuandatapply vẫn QUEUED, chưa hủy. Ưu tiên collect alternate; tránh lấy duplicate thành seed mới.
  Ref và manual cancel bản cũ tại [bàn giao](docs/operations/continue.md).

- 16/11 20:55VN: đã push bản cuối CNN NILM43 resume29→50 lên `voxuandatapply` (v1 QUEUED).
  Account parent chỉ còn3,71h; chuyển checkpoint trực tiếp bằng URL output tạm thời, giữ source/state gốc.
  Theo dõi [bàn giao](docs/operations/continue.md); bảng vẫn167/168 tới khi bản này hoàn tất.

- 16/11 19:36VN: MLP và TabM V2X a4 đều COMPLETE50 đủ3seed, smoke/history telemetry PASS.
  CNN9 resume đã COMPLETE50. Bảng live167/168, chỉ thiếu CNN NILM seed43 chưa resume do quota lần trước.
  Đây là completion/telemetry gate; chưa audit CM/weights độc lập cho các bản mới.

- 16/11 17:04VN: MLP V2X a4 seed41/42/43 đã push projection FP32 chung; local regression PASS,
  còn chờ GPU smoke. Ref/owner tại [bàn giao](docs/operations/continue.md); không dùng MLP V2X a2 lỗi.

- 16/11 kiểm tiếp: CNN pFedES43 COMPLETE50, bảng live158/168; còn4 CNN RUNNING,
  TabM V2X41/43 QUEUED. Log live rỗng nên chưa có round/ETA mới; xem bàn giao.

- 16/11 16:42VN: kiểm đủ21 account; [bảng live](docs/research/backbones/four-backbone-seed-results-live.md)
  đã điền157/168 ô. Mới: TabM V2X seed42 fp32proj-a4 COMPLETE50 F1=0,84772,
  history đủ50round, max skip0,16524% <1%; CNN pFedES seed42 COMPLETE50 F1=0,74134.
  TabM V2X seed41/43 còn QUEUED. Đây là kết quả marker/history, chưa audit CM độc lập cho bản mới.

- 16/11 audit code 4 backbone×14 baseline so DAGSNet: 13/14 PASS hoặc PASS-adapter; V2X-HtFL FAIL
  GPU health trên TabM/MLP dù protocol/HP khớp. Xem [audit alignment](docs/research/backbones/four-backbone-code-alignment-audit-20261116.md).

- 16/11 13:40VN: refresh bảng live 4 backbone×14method×3seed =153/168 ô có số. TabM/SAE đủ42/42.
  Đã submitted đủ84/84 cấu hình MLP/CNN; CNN có2 CAN +8 resume đang RUNNING, còn2 resume chờ slot/quota.
  MLP V2X cả3seed ERROR lặp lại do smoke non-finite skip16–41% > gate1%, không retry mù.
  Xem [continue](docs/operations/continue.md) và [bảng live](docs/research/backbones/four-backbone-seed-results-live.md).

- 15/11 23:xxVN: đã xuất [bảng live Macro-F1 4 model×14method×3seed](docs/research/backbones/four-backbone-seed-results-live.md):
  128/168 ô có kết quả,40 trống; TabM42, SAE40, MLP34, CNN-BiGRU12. CNN APFed3 run
  chưa đủ50round nên để trống. MLP/CNN chưa audit final CM độc lập.

- 15/11 22:48VN: direct MLP/CNN-BiGRU đủ78/78 submitted. CAN đãpush2/6 sau parent artifact gate
  (CNN s41, MLP s42), tổng80/84; còn MLP CAN41/43 + CNN CAN42/43 chờ slot đúng parent owner.
  Cloud vẫn tắt. Xem [continue](docs/operations/continue.md).

- 15/11 21:53VN: sau phone verification, probe `iamabighotdog` COMPLETE đúng2×T4 capability7.5.
  Đã mở khóa và retry version2 hai job: NILM41 RUNNING, LwFedNIDS43 QUEUED. Ledger73/84,
  còn11; production startup marker chưa xuất ở lần check cuối; cloud vẫn tắt.

- 15/11 21:28VN: roster20, push wave20+4. `iamabighotdog` nhận0 GPU ở cả4 kernel, ERROR tại
  gate2×T4; đã blacklist launcher và đánh dấu ledger failed_zero_gpu. Còn19 account GPU-capable đều
  đã lấp slot an toàn. MLP/CNN-BiGRU71/84 job submitted không-ERROR, còn13; cloud vẫn tắt.

- 15/11 19:50VN: account thứ17 `anduyvo06` identity/quota PASS; push2 LwFedNIDS-a2
  (CNN-BiGRU s41, MLP s42), 1 QUEUED+1 RUNNING. Ledger51/84, còn33; cloud vẫn tắt.

- 15/11 19:02VN: roster16 sau khi thêm `huygia1611`,`hunhthanhhong`; identity PASS. Push10/10
  aligned-v2 không lỗi, ledger MLP/CNN-BiGRU49/84, còn35. Post-refresh mọi slot/quota an toàn
  đã dùng; cloud vẫn tắt. Xem [continue](docs/operations/continue.md).

- 15/11 18:xxVN: check39 MLP/CNN-BiGRU-a2:12 COMPLETE gate50round+smoke PASS,22 RUNNING,
  5 QUEUED,0 ERROR. MLP đủ3seed: APFed F1 tốt nhất sơ bộ0,88179±0,00345; FedAvg0,87759;
  FedProx0,87154; FedProto0,84714. Chưa audit final CM/source/config; CNN-BiGRU chưa COMPLETE.

- 15/11 18:15VN: roster14 account sau khi thêm `buidinh0303`,`huynhkyy`; identity/quota PASS.
  Đã lấp16 slot mới, push16/16 aligned-v2 không lỗi. Post-refresh mọi slot/quota an toàn đã dùng;
  ledger MLP/CNN-BiGRU39/84 submitted, còn45. Cloud vẫn tắt. Xem [continue](docs/operations/continue.md).

- 15/11 17:01VN: account thứ12 `voxuandatapply` identity/quota PASS; đã push2 job FedProto-a2
  (CNN-BiGRU s42, MLP s43), 1 RUNNING+1 QUEUED. Tổng ledger MLP/CNN-BiGRU23 submitted;
  cloud vẫn tắt. Xem [continue](docs/operations/continue.md).

- 15/11 16:13VN: roster Kaggle đủ11 account. Đã push21 MLP/CNN-BiGRU aligned-v2 version1,
  gồm đủ6 FedAvg để mở khóa CAN; 10 RUNNING +11 QUEUED, toàn bộ slot/quota khả dụng đã dùng,
  không lỗi. Còn63/84; cloud vẫn tắt. Ledger/launcher/inventory xem [continue](docs/operations/continue.md).

- 15/11 15:04VN: corrective scope đãpush đủ SAE42/42 + TabM-TinyProto3/3.
  CAN đúngparent-a2; V2X3 đãpush. Chờ smoke/round/audit; cloud giữtắt.

- 15/11 14:52VN: aligned-v2 đãpush34;29 COMPLETE đủ50round/smokePASS,5 RUNNING,
  0ERROR; final skip max0. Chưa audit CM/metric/weights. Cloud giữtắt.

- 15/11 14:22VN: wave1-a2 đủ17×50round/smokePASS theo marker; đãpush tiếp17SAE-a2.
  TổngSAE31/42đãpush. Ledger wave2 trongruns; cloud vẫn tắt. Chưa audit CM mới.

- 15/11 13:46VN: đãpush17 aligned-v2 version1 (TabM TinyProto3 + SAE14), source remote khớp;
  chờ GPU smoke/round. Ledger `runs/aligned-v2-launch-wave1.json`. Cloud giữtắt.

- 15/11 owner chốt [aligned-v2](docs/research/aligned-fl-v2.md): SAE-FL CE-only/prototype93,
  TinyProto đúngμ DAGSNet, telemetry chung. Notebook-a2 riêng, chưapush; cloud vẫn tắt.

- 15/11: đã collect84/84 TabM/SAE (14method × 3seed × 2model), kiểm CM cuối/source/config/history PASS.
  [Bảng so sánh](docs/research/backbones/tabm-sae-dagsnet-comparison-20261115.md) dùng DAGSNet seed42 theo owner;
  TinyProto khác μ giữa hai nhánh, chưa phải so sánh chỉ đổi backbone. Cloud vẫn tắt.

- 15/11: owner yêu cầu DỪNG hourly cloud dispatcher tại private repo `dangvanhoang1711/kaggle-fl-dispatcher`.
  Đã xác minh workflow `disabled_manually`, `DISPATCH_ENABLED=false`; không có tick đang chạy/chờ.
  Không tự bật lại; các job đã gửi Kaggle vẫn tiếp tục.
  Không push tay song song; kiểm `queue.json`/Actions trước. Dry-run cloud PASS99pending;
  tick11:02VN SUCCESS, vừa gửi17MLP; TabM42/SAE42/MLP38/BiGRU0 đãpush, còn46pending.
  Xem [bàn giao](docs/operations/continue.md).

- 14/11 mới: TabM đãpush42/42 cấu hình; SAE đãpush20/42 qua2waves. Batch mới9SAE nhậnRUNNING/QUEUED,
  chưa xác nhận smoke GPU. CNN-LSTM feature báoCOMPLETE, chưa audit sốepoch cuối. Xem continue cho roster.

- 14/11: MLP và CNN-BiGRU-Attention: **84 notebook local** (mỗi model14 method×3seed41/42/43).
  CPU reference/gradient/BN và28 method smoke PASS;26 direct-method resume CM checks PASS.
  Chưa push, chưa kiểm2T4. [Chi tiết](docs/research/backbones/dense-recurrent-fl.md).

- 14/11: đã build local ma trận mới **single-backbone SAE × 14 baseline × seed 41/42/43**: 42/42 notebook
  (39 direct + 3 CANPerFL), static audit PASS, chưa push. SAE giữ objective reconstruction + KL sparsity + CE;
  prototype dùng bottleneck16. Host không có PyTorch nên dynamic smoke là hard gate đầu notebook Kaggle.
  CANPerFL phụ thuộc FedAvg cùng seed round50. Chi tiết: [SAE baseline matrix](docs/research/backbones/sae-single-model-baselines.md).

- 14/11: mở ma trận mới **single-backbone TabM k=4** cho 14 baseline (bỏ SCAFFOLD), seed 41/42/43.
  Đây là phạm vi mới, không thay hoặc diễn giải lại kết quả DAGSNet seed42 cũ. Protocol/adapters và trạng thái launch:
  [TabM baseline matrix](docs/research/backbones/tabm-single-model-baselines.md), [continue](docs/operations/continue.md).
  16 trajectory đã COMPLETE và local audit đủ50 round; 5 method đã đủ3 seed: FedAvg, FedProx, AFPHA, FD-IDS,
  APFed. Đã lấp lại16/16 slot:15 TabM QUEUED + CNN-LSTM RUNNING. CANPerFL chờ parent FedAvg đã audit rồi build/push
  khi có slot. Chi tiết version, kết quả và phạm vi audit ở [continue](docs/operations/continue.md).

- 19/19 phương pháp code + test. **W = 1 khóa** (DAGSNet 395.408 tham số).
- 14 run FL 50 round + centralized + CANPerFL **xong, audit PASS** ⇒ [results](docs/research/dagsnet/results.md) (centralized 0,9593;
  FL tốt nhất APFed client 0,9119; FedAvg 0,9042). Còn chạy: NILM-FL (cần session 2), TinyProto r3 (sửa luật prototype).
- Chi phí FLOPs/băng thông mọi phương pháp: [cost](docs/research/cost.md). MobiHet: repo riêng (bên dưới).
- MobiHet v4 r50 audit PASS: private/proxy macro-F1 0,8953/0,8995, class13 ~0,116. P0 local-prior FAIL;
  P1a 77-feature oracle PASS (F1 lớp13 0,5423),
  mở method branch; RelKD và pairwise 0↔13 r10 audit PASS nhưng gate FAIL; private-only global statistics bỏ proxy
  train→val r10 audit/gate PASS (macro +0,01398, class13 +0,27724, train −73%, round bytes 0). Final train+val→test r50
  COMPLETE `hoangdang1711/mobihet-v4-stats-final-r50/1`: CM/metric audit PASS, macro/client mean 0,887913/0,846076,
  class13 0,339987; weights chưa audit. Arm mới `hoangdang1711/mobihet-v4-stats-realbn-val-r10/1`: BN chỉ từ real,
  paired train/val R10 đã COMPLETE, gate FAIL. Arm adaptive Gaussian theo class train count đã push
  `hoangdang1711/mobihet-v4-stats-adaptive-val-r10/1` COMPLETE, CM audit PASS nhưng gate FAIL (macro +0,001025).
  `dangthiet/mobihet-v4-stats-massnorm-val-r10/1` COMPLETE: source/CM metric audit PASS, gate FAIL
  (macro −0,005279; lớp11/15 không hồi phục). Đóng nhánh massnorm.
  Diagnostic tiếp `dangthiet/mobihet-v4-stats-nogauss-val-r10/1`: paired bỏ Gaussian, 2 T4, 14 test PASS;
  COMPLETE R10, source/20 CM metric audit PASS; gate FAIL (macro −0,008963; client p10 −0,075048).
  Train nhanh hơn26,35% nhưng chất lượng giảm; không mở R50. Chi tiết trong design.
  Arm `dangthiet/mobihet-v4-stats-prior-val-r10/1` COMPLETE02/11, source/config/20 CM metrics audit PASS,
  gate FAIL: macro +0,003950 nhưng mean −0,012551, p10 −0,044708, class13 −0,051158. Không mở R50.
  CPU diagnostic `dangthiet/mobihet-v4-stats-client-diagnostic/1` COMPLETE, pull/audit count-CM identity PASS.
  Gaussian cứu lớp thiếu nhưng lợi ích khác nhau theo lớp; chi tiết train-support trong design.
  Gaussian fidelity CPU `dangthiet/mobihet-v4-gaussian-fidelity/1` COMPLETE; pull/audit predictions/CM/metrics PASS.
  Domain AUC1,0 cả4 lớp kể cả bỏ6 flags; synthetic→real binary macro kém0,0922/0,0643 ở0↔13/11↔15.
  Bằng chứng phân phối Gaussian lệch real, chưa cô lập nguyên nhân hoặc mở arm FL mới; chi tiết trong design.
  Tiếp02/11 đã push `dangthiet/mobihet-v4-stats-hist-val-r10/1`: paired histogram marginal transport vs stats gốc.
  COMPLETE03/11, source/config/20 CM metrics audit PASS; gate FAIL sát ngưỡng: macro+0,004766<+0,005.
  Mean/p10/class13 đều tăng; train+0,255%. Không mở R50, không nới gate. Không W&B.
  03/11 đã push `dangthiet/mobihet-v4-stats-histmask-val-r10/1`: sửa6 cặp flag=1⇒value=0,
  COMPLETE03/11,3 arms/30 CM metric audit PASS; histmask FAIL cả2 gates: macro+0,004756 vs Stats,
  macro−0,000251/p10−0,006338 vs histogram. Đóng repair, không R50. Hist control lại sát ngưỡng+0,005007.
  03/11 đã push `dangthiet/mobihet-v4-stats-hist-s43-r10/1`: paired seed43 Stats/histogram,
  COMPLETE04/11, seed43 R10 macro +0,004496, p10 −0,006306. Gate FAIL, đóng nhánh histogram, không lên R50.
  Chưa cần dataset v5/raw. Không W&B.

## Cần biết → mở

04/11 MobiHet Core proposal: phiên `dangthiet/mobihet-core-copula-q-r10/2` đã nhận 2 T4 và pass 3 tests. Đang chờ smoke/R10. Xem bàn giao cho thiết kế 5 nhánh.

| câu hỏi | tài liệu |
|---|---|
| Đang làm gì, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Tự check/push Kaggle khi máy tắt? | [cloud dispatcher](docs/operations/cloud-dispatcher.md) |
| Kết quả 50 round, audit, lưu ý từng run | [results](docs/research/dagsnet/results.md) |
| Báo cáo Notion, page IDs, bổ sung run hoàn tất | [notion-results](docs/operations/notion-results.md) |
| TinyProto r3 suy giảm và bản sửa r4 | [tinyproto](docs/methods/tinyproto.md) |
| Cấu hình chung, cách chấm, cổng độ rộng | [protocol](docs/research/protocol.md) |
| Danh sách phương pháp, trạng thái | [methods](docs/research/methods.md) |
| Dataset, cột, số liệu, caveat | [knowledge/dataset](knowledge/dataset.md) |
| Dựng DAGSNet cho client | [knowledge/dagsnet](knowledge/dagsnet.md) |
| Máy local, Kaggle, lỗi kernel | [local-env](knowledge/local-env.md), [kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md) |
| Git backup gần nhất | [backup](docs/operations/backup.md) |
| Bài báo, cái nào thiếu | [papers/README](papers/README.md) |
| Phương pháp MobiHet: phát triển tiếp, kết quả v1 20/50/100 client | repo [`odixe06/mobihet`](https://github.com/odixe06/mobihet) (v2; v1 ở `history/v1/docs/methods/full-drift/`) |
