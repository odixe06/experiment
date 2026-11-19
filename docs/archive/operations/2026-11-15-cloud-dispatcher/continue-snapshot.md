# Snapshot 15/11/2026 — live handoff: [continue](../../../operations/continue.md)

# Continue — bàn giao sống, 01/11/2026 (giờ VN)

## 15/11 00:35 — GitHub Actions cloud dispatcher ACTIVE, tick SUCCESS

Owner yêu cầu tự check/push mỗi giờ kể cả shutdown máy. Đã cho phép repo private + GitHub Secrets;
owner xác nhận chưa thêm phương thức thanh toán GitHub. Repo mới:
https://github.com/dangvanhoang1711/kaggle-fl-dispatcher . Source triển khai từ
`scripts/cloud_dispatch/`, package local `tmp/kaggle-fl-dispatcher`.
Đã upload 58 file source frozen và encrypted secret 9 account; không đổi global Kaggle login.
Workflow minute17 mỗi giờ, `DISPATCH_ENABLED=true`; chạy không cần máy local.
Dry-run cloud SUCCESS: https://github.com/dangvanhoang1711/kaggle-fl-dispatcher/actions/runs/43619921537 .
Tick mới nhất SUCCESS: https://github.com/dangvanhoang1711/kaggle-fl-dispatcher/actions/runs/43621866191 .
Ledger29submitted/97pending/0review; thêm2SAE do cloud gửi:
nguynt298/sae-fl-fedtgp-s41/version1 (QUEUED lúc xác minh),
hoangdang1711/sae-fl-fedtgp-s42/version1 (RUNNING lúc xác minh).
Cả2 đã so source remote/private/GPU/version với frozen build PASS; chưa kiểm GPU smoke/round mới.
Tổng ma trận: TabM42/42 đãpush,SAE29/42,MLP0/42,BiGRU0/42.
Lịch phút17 mỗi giờ; lần kế tiếp dự kiến01:17VN15/11 (GitHub có thể trễ).
Không chạy launch scripts tay song song khi scheduler được bật. Source/code nghiên cứu không đổi.
Scheduler không tự retry ERROR/partial run; CANPerFL phải cùng owner/model/seed FedAvg đủ50round.
13 unit/integration tests local PASS; source hashes và3backbone notebook builds PASS.
Giới hạn: status/submitted không phải audit đủ50round; không tự nối session ERROR/partial.
Scheduler tự disable khi hếtpending; inventory/API lỗi thì fail closed, không push trùng.
Source deployment commit cba65ac42971c2fdc62fcf8581269e61ede5881f; ledger có commit riêng.
Bằng chứng snapshot: docs/archive/operations/2026-11-15-cloud-dispatcher/deployment.json.
Docs link check scoped docs/operations PASS; whole-repo checker vấp junction tmp/central-sheet-input/node_modules
không tồn tại (không sửa thư mục ngoài phạm vi).

## 14/11 — yêu cầu đưa tất cả vào queue, bị Kaggle giới hạn

Owner cho phép push toàn bộ remaining99 (SAE15,MLP42,BiGRU42). Live9account có18RUNNING/QUEUED.
Thử push ttalgies/sae-fl-fedtgp-s41 bị từ chối: Maximum batch GPU session count of 2 reached.
CLI exit0 nhưng không có successful version; KHÔNG tính là đãpush. Tổng vẫnTabM42,SAE27,MLP0,BiGRU0.
Không thể gửi hết lên remote queue khi đầy slot; cần nộp theo wave khi backend nhảslot.
CANPerFL cần parent FedAvg cùngbackbone/seed đủ50round trước khi launch.

## 14/11 — account mới nguynt298

Owner yêu cầu dùng nguynt298. Saved token introspection/quota PASS30h; notebook cũ duy nhất
mobihet-dagsnet-20c-relaxed CANCEL_ACKNOWLEDGED. Đã push version1 nguynt298/sae-fl-nilm-s42
RUNNING và nguynt298/sae-fl-nilm-s43 QUEUED; limit8h,train budget7h,private2T4.
Chưa xác nhận startup/smokeGPU. Script launch_sae_nguynt298.py; checker đã thêm roster.
SAE đãpush27/42; hiện9account được dùng. Không đổi global login.

## 14/11 — SAE wave3 thêm5run, kiểm sau push

Đã push version1: TinyProto43 dangthiet; pFedES41 hoangdang1711,42 hoangdangnhi,43 qwerasdf3004;
NILM41 qwerasdf3004. Limit8h,train budget7h; qwer còn16.07h trước2push.
Kiểm sau push: pFedES41 RUNNING,4run mới cònQUEUED. TổngSAE25/42 đãpush:
10COMPLETE theo API (chưa audit đủround),1RUNNING,14QUEUED. TabM CAN41 QUEUED;
15SAE active/queued +1TabM =16slot. Không cóslottrống tại thời điểm kiểm.
Wave3 script launch_sae_wave3.py; check_current_wave.py đã cập nhật roster.

## 14/11 — SAE wave2 thêm9run

Kiểm live:9slot trống; CNN-LSTM feature COMPLETE (chưa kiểm đủ50epoch), TabM CAN42/43 vàV2X42 COMPLETE.
TabM CAN41 cònQUEUED; SAE FedAvg41,FedProx41/42/43,AFPHA42,FD-IDS41 cònQUEUED.
Đã push version1 thành công9SAE qua scripts/launch_sae_wave2.py:
FD-IDS43 dangthiet; APFed41/42 hoangdang1711,43 hoangdangnhi;
LwFedNIDS41 minhiw,42 phamquang0606,43 ttalgies; TinyProto41/42 qwerasdf3004.
Sau push:RUNNING FD-IDS43/APFed41/APFed43/TinyProto41/42;QUEUED APFed42/LwFedNIDS41/42/43.
Chưa xác nhận smoke/production mới. Limit8h/session,max-hours7; qwer quota16.86h trước2push.
TổngSAE đãpush20/42; chưa launchMLP/BiGRU. check_current_wave.py đã bao gồm wave2.

## 14/11 — final TabM + SAE wave1 launch

Live read: mọi TabM đã push trước đều COMPLETE, CNN-LSTM feature còn RUNNING. Quota27–29h trên7account,
qwer18.71h. Đã push version1 batch15 qua scripts/launch_tabm_final_sae_wave.py:
TabM CANPerFL41/42 dangthiet,43 hoangdang1711; V2X42 ttalgies.
Parent CAN41 chính xác dangthiet/tabm-fl-fedavg-s41-probe,42/43 slug bình thường; parent COMPLETE đã kiểm.
SAE FedAvg41/42 hoangdangnhi,43 hoangdang1711; FedProx41/42 hoangdv1711,43 minhiw;
AFPHA41 minhiw,42/43 phamquang0606; FD-IDS41 ttalgies,42 qwerasdf3004.
Kaggle nhận các run RUNNING/QUEUED, chưa xác nhận smoke2GPU/production logs. Tổng16slot kểCNN-LSTM.
SAE CAN41/42 local metadata cũ phải rebuild theo parent owner hoangdangnhi; không dùng dangthiet.
Không push MLP/BiGRU trong batch này. Các run TabM vừa COMPLETE chưa được collect/audit mới.

## 14/11 — đã lấp7slot TabM mới

Owner yêu cầu account rảnh thì push. Quota5account chọn28–29h trước launch.
Version1 push thành công và status RUNNING cả7: hoangdv1711/perfedskd41/42,
phamquang0606/perfedskd43 và fedproto42, hoangdangnhi/fedproto41,
minhiw/fedproto43, ttalgies/v2x-htfl41 (slug đều tabm-fl-<method>-s<seed>).
Builder source57files; V2X mới có gen_s resume fix. Chưa xác nhận startup/2T4/smoke logs.
Các8TabM wave trước đã RUNNING, CNN-LSTM feature RUNNING:16slot theo roster.
Còn chưa push TabM V2X42 và CANPerFL41/42/43. Không launch SAE/MLP/BiGRU trong wave này.

## 14/11 — kiểm wave TabM sau build84

Live status:7 mới COMPLETE (TinyProto42,pFedES41/42/43,NILM42/43,FedTGP41),8 QUEUED
(LwFedNIDS41/42/43,TinyProto41/43,NILM41,FedTGP42,V2X43); CNN-LSTM feature RUNNING.
Đã selective-pull RUN_STATUS,manifest,config,rounds vào runs/pulls/*-status-check, xác nhận7run đủ1–50.
Tổng23/42 trajectory có50round:16audit trước+7 mới chỉ kiểm history/status, chưa audit CM/weights.
MacroF1: TinyProto42 head0.65101277/proto0.28755920 (cần điều tra chênh lệch, chưa kết luận lỗi);
pFedES41/42/43=0.82226393/0.82527947/0.81947116;
NILM42/43 client=0.82318158/0.81878390; FedTGP41 head0.82464771/proto0.75674923.
Chưa push thêm trong lượt status này. 84MLP/BiGRU và42SAE vẫn local-only.

## 14/11 — MLP/CNN-BiGRU-Attention build84

Đã hoàn thiện backend grouped, pruning dependency groups, APFed classifier và V2X projection.
Builder `scripts/build_dense_recurrent_matrix.py`:2 model×14method×3seed41/42/43=84 notebook private2T4;
CANPerFL parent cùng model/seed,50FedAvg+10fine-tune. Chưa push.
CPU test `tests/test_dense_recurrent_fl.py` PASS forward eval/train, mask, gradients, BN,
28 one-round smokes và26 round-boundary resume CM checks. V2X thiếu gen_s đã sửa lưu/restore.
Runtime test local riêng `tmp/fl-test-runtime` (torch2.15CPU,pyarrow); Python311 với PYTHONPATH thêm thư mục này và repo.
Không diễn giải CPU PASS thành GPU PASS. Notebook chạy mandatory tests trước2GPU smoke/resume/production.
Thiết kế, adaptation và giới hạn: [dense-recurrent-fl](../../../research/dense-recurrent-fl.md).

## 14/11/2026 — SAE × 14 baseline × 3 seed (build local)

Owner yêu cầu build sẵn 14 baseline đơn model SAE, seed41/42/43. Đã thêm backend `--backbone sae` và adapter cho
FedAvg/FedProx/AFPHA/FD-IDS/APFed/LwFedNIDS/CANPerFL/TinyProto/pFedES/NILM/FedTGP/PerFed-SKD/FedProto/V2X-HtFL.
SAE 7.613 tham số giữ objective centralized: reconstruction +1,2 KL sparsity +1,5 CE; prototype dùng code16.
42/42 notebook local đã tạo dưới `notebooks/sae-fl-*`:39 direct +3 CANPerFL. `scripts/audit_sae_fl.py` PASS nbformat,
AST mọi cell, private2×T4/dataset/owner/seed, embedded source/manifest và CAN parent wiring. Python local hiện không có
PyTorch nên chưa chạy dynamic `tests/test_sae_fl.py`; test này là hard gate remote trước smoke2 round+resume và production.
Chưa push theo đúng yêu cầu chỉ build. CANPerFL chỉ push sau FedAvg cùng seed COMPLETE có state round50.
Thiết kế khóa: [SAE baseline matrix](../../../research/sae-single-model-baselines.md). Builder:
`scripts/build_sae_fl.py`; toàn ma trận: `scripts/build_sae_matrix.py`.

## 14/11/2026 — single-model TabM × 14 baseline × 3 seed

Owner yêu cầu chạy 14 baseline (bỏ SCAFFOLD, centralized, proposal) với duy nhất TabM k=4, seed 41/42/43,
50 round và metric test từng round. Đã khóa adapter/protocol tại
[TabM baseline matrix](../../../research/tabm-single-model-baselines.md). Backend thêm `--backbone tabm`; TabM grouped khớp
reference tuyệt đối trên CPU, 6.548 tham số. CPU smoke đủ 14/14 PASS, gồm pruning/KD/prototype/V2X/CAN parent.
Builder private 2×T4: `scripts/build_tabm_fl.py`. Probe v1 xác nhận đúng 2 Tesla T4 rồi ERROR trước test do package
`tests` của image che namespace local khi gọi `python -m`; không train. Đã sửa gọi file trực tiếp với `PYTHONPATH`, push
`dangthiet/tabm-fl-fedavg-s41-probe/2`, SHA256 notebook
`c3768d378ae7bb8ee8ff8e52bc9d726056ccdcc1d1e2c259f378b28d3b499faa`; chờ smoke/round thật.
Sau đó CPU smoke đủ14 method PASS trên Kaggle image ở `minhiw/tabm-fl-fd-ids-s41/1`; đã mở wave1 tối đa
15 session/queue: FedAvg s41/42/43, FedProx s41/42/43, AFPHA s41/42/43, FD-IDS s41/42/43, APFed s41/42/43.
Mỗi account dùng tối đa2 slot; qwerasdf3004 có CNN-LSTM cũ + APFed nên không mở thêm. 5 run đầu đã RUNNING, phần còn
QUEUED/đang nhận worker. GPU smoke v1 phát hiện `compile_gate` gọi `torch.cat([])` vì TabM không có BN; mọi v1 dừng
trước production. Sửa buffer rỗng `(G,0)`, `bn_maxdiff=0`, local smoke PASS; repush v2/v3 theo slot được giải phóng.
Ba trajectory bản sửa đã COMPLETE/audit đủ50 round: `phamquang0606/tabm-fl-fd-ids-s42/2` macro-F1 r50
=0,805610; `hoangdangnhi/tabm-fl-fedprox-s42/3`=0,833102; `qwerasdf3004/tabm-fl-apfed-s43/2`
client/global=0,857185/0,845119. `scripts/audit_tabm_fl.py` kiểm source hash, config,50 CM×9.347.679 dòng,
10 metric+phân phối client+F1 lớp, finite/skipped và50 checkpoint; executed notebook có19/19/21 output. Local chưa
deserialize tensor checkpoint vì Windows Python không có torch; remote smoke đã strict-load/resume.
`phamquang0606/tabm-fl-fd-ids-s43/2` RUNNING; `hoangdangnhi/tabm-fl-fedprox-s43/3` và
`qwerasdf3004/tabm-fl-fedtgp-s43/1` QUEUED. V1 lỗi/queued cũ còn giữ slot; repush bản sửa khi backend nhả.
Lần tiếp 14/11: backend đã nhả; repush thành công FedAvg s41 v3/s42 v2/s43 v2, FedProx s41 v2,
FD-IDS s41 v2, APFed s41 v2. Lấp slot còn lại bằng TinyProto s42; pFedES s41/s42/s43; NILM s42/s43;
FedTGP s41. Tổng16/16 batch session trên8 account: 7 TabM + CNN-LSTM feature đang RUNNING,8 TabM QUEUED.
Các run mới vừa provisioning nên lần đọc đầu chưa có log health; cần xác nhận2T4, smoke/resume và compile gate trước
khi gọi production khỏe. Không repush khi RUNNING/QUEUED.
Wave2 notebook đã build (LwFedNIDS/TinyProto/pFedES/NILM/FedTGP×3); wave3 build
(PerFed-SKD/FedProto/V2X-HtFL×3), chưa push vì ưu tiên thay đủ wave1 v1. Hai slug FD-IDS đầu được Kaggle
canonicalize `_`→`-`; builder đã sửa default slug dùng dấu `-` cho notebook sinh sau.
Quota live sau các run khoảng234,82 GPU-h;7 account còn29,4–30h, qwerasdf3004 còn26,03h.
`qwerasdf3004/veremi-central-cnn-lstm-feature-e50` đang RUNNING. Lần kiểm cuối mọi account đủ2 slot active/queued
hoặc còn terminal session chưa được backend thu hồi, nên chưa nhận thêm push.
Account tool đã sửa khóa Windows (`fcntl`→`msvcrt` fallback), không đổi credential. CANPerFL phải chờ FedAvg cùng seed
r50 rồi mount đúng `.../<parent-slug>/state.pt`, chạy 10 epoch fine-tune; builder đã bỏ glob `**/state.pt` mơ hồ.

Tiếp 14/11: tổng16 trajectory TabM COMPLETE và local audit PASS. Ngoài3 run audit trước, đã kéo/audit đủ
FedAvg s41v3/s42v2/s43v2, FedProx s41v2/s43v3, AFPHA s41/s42/s43v2, FD-IDS s41/s43v2,
APFed s41/s42v2 và FedTGP s43v1. Mỗi pull117file,0 missing; `LOCAL_AUDIT.json` kiểm source/config,50 round,
CM đủ9.347.679 dòng/view/round, tái tính metric, finite/skipped và50 checkpoint. Executed notebooks đã lưu.
Kết quả mean±sample-std 3 seed: FedAvg0,844610±0,002501; FedProx0,829823±0,003643;
AFPHA0,825933±0,003591; FD-IDS0,805425±0,000541; APFed client0,858280±0,003044.
Đã push lấp lại16/16 slot: LwFedNIDS s41/s42/s43, TinyProto s41/s42/s43, pFedES s41/s42/s43,
NILM s41/s42/s43, FedTGP s41/s42, V2X-HtFL s43 và CNN-LSTM centralized đang RUNNING. Lần kiểm ngay sau push:
15 TabM QUEUED, CNN-LSTM RUNNING; chưa có health log mới. Không Git/DVC push.

## 14/11/2026 — CNN–LSTM feature196k đã push

Owner yêu cầu sửa rồi push CNN–LSTM dùng một bản tin77feature; Conv256/128 +2LSTM96,196.240tham số, alias `cnn-lstm-feature`. Đã push private `qwerasdf3004/veremi-central-cnn-lstm-feature-e50/1`, cap12h, train mới từ epoch1 tới mục tiêu50. Quota trước29,650h/reserved0; owner-scoped listing không có slug này. Notebook SHA256 `74b7782d9d6e34559a5e45348d01d762a2ef0987b78d3d732245b1ed24fb7ab8`;16source/hash/schema/AST PASS. Thêm CUDA-fp16 gradient test và probe-budget guard (epoch×1,5 phải vừa session). Local14tests:12PASS+2CUDA skipped; Worker đã nhận đúng 2 Tesla T4; 14 tests chạy 15,569s: 13 PASS, 1 skip (Mamba extension), gồm CUDA-fp16 CNN–LSTM PASS. Probe PASS: 970,900s/epoch train+eval (~16,18phút), 13,48472h/50epoch, peak8,427GiB, skipped0. Production đã vào epoch1, step500/9122 cả2rank, grad3,585 finite, skipped0. Budget phiên còn11,166h lúc bắt đầu; dự kiến cần nối session để đủ50. Kiểm14/11 11:38VN: live log COMMITTED epoch1 macro-F1=0,910649; chưa kéo/audit đủ10metric. Epoch2 tới3500/9122, grad0,3751 finite, AMP skipped2; statusRUNNING. [Chi tiết](../../../research/central-backbones.md). Phiên này sở hữu run; không repush khi còn chạy. Không thay DAGSNet/CNN-BiGRU/SAE sang các kích thước mới đề xuất; không khởi chạy FL. MambaCNN vẫn đóng.

## 14/11/2026 — chuẩn bị FL trộn5backbone

Owner xác nhận **trộn model giữa client trong cùng thí nghiệm**, không phải5run đồng kiến trúc. Chọn DAGSNet25%, BERT-style77feature25%, TabM/SAE/CNN-BiGRU-Attention chia đều50% còn lại. Làm tròn5.343client:1.336/1.336/891/890/890. [Đánh giá tương thích và việc cần chuẩn bị](../../../research/heterogeneous-backbones.md). Static audit: FL backend hiện chỉ DAGSNet; không gộp tham số khác kiến trúc bằng các baseline FedAvg-like cũ. Proxy/prototype/statistics có hướng thích nghi nhưng cần adapter/loss/layout và thử nghiệm. Chưa chọn biến thể cho các phương pháp không hỗ trợ heterogeneity, chưa sinh manifest, chưa sửa runner/train hoặc launch. BERT centralized hiện chỉ kiểm tới epoch18. Không dùng checkpoint centralized làm init FL mặc định. MambaCNN vẫn đóng.

## 14/11/2026 — MambaCNN centralized đã push

**ĐÃ ĐÓNG theo owner14/11: bỏ Mamba, dùng các model cũ. Không tiếp tục sửa/push/resume MambaCNN hoặc tự chạy lại model khác.** Version3 đã xác nhận ERROR trước production; cả3version không có epoch production. Code/log giữ để tham khảo. Bản chỉnh gate key-bias mới chỉ ở local, chưa push version4, chưa kiểm GPU. [Bằng chứng cuối](../../../../papers/feng-2026-mambacnn/rebuild.md). Các đoạn dưới là lịch sử, không còn yêu cầu tiếp tục chạy.

Owner yêu cầu push/chạy, sau đó xác nhận sửa tốc độ. Version1 **ERROR trước production** tại budget gate: PyTorch scan dự kiến21,79001h/epoch, chưa có epoch/checkpoint train. Đã kéo4JSON+log, zero missing vào `runs/pulls/central-mambacnn-v1`. Đổi sang CUDA scan chính thức, giữ129.328tham số và protocol. Pinned wheel/hash, kiểm scan output/gradient và20paired real-data AMP steps; sweep micro32/128/256/512, logicalbatch4096. Không dùng lại trajectory cũ vì chưa có production. Local10PASS+1CUDA skipped;17source hashes/AST/schema PASS. Đã push private `qwerasdf3004/veremi-central-mambacnn-e50/2`, statusRUNNING; quota trước29,888h/reserved0, version1 terminalERROR đã xác nhận. SHA256 `7a8cc7dd33d144880bf2f6ee69bfe740fde0fa24b141eb3859ef5d8f6197e646`. Cap12h, budget11,25h trừ setup/probe; mục tiêu50epoch, nối checkpoint nếu cần. [Chi tiết và gates](../../../../papers/feng-2026-mambacnn/rebuild.md). Phiên này sở hữu run. Tiếp: kiểm CUDA install/tests/equivalence/probe→production; không repush trùng khi RUNNING. Không Git push/W&B.

## 13/11/2026 — thêm CNN-BiGRU-Attention và SR-LiCNN

Owner đã cho phép build và push cả hai model centralized lên Kaggle, chọn account nhiều quota. Kế thừa mục tiêu 50 epoch, dataset v4 và protocol của các backbone trước. Đã tải PDF CNN-BiGRU, đọc bảng và công thức, render trang kiến trúc. [Nguồn và điểm chưa rõ](../../../../papers/mnkash-2026-cnn-bigru/paper.md), [phương án chuyển thể](../../../../papers/mnkash-2026-cnn-bigru/rebuild.md).

Owner đã đồng ý dùng Table 2 và GRU chuẩn PyTorch. Đã build cấu hình 77×1, Conv64/128, BiGRU64, Attention128, head16, đếm đúng 118.288 tham số trainable. 8 test CPU PASS, gồm strict weights, gradient, attention, prediction/CM/metrics và continuous-vs-resume bit-exact cả BN buffer. BN buffer rank0 broadcast trước sharded eval để dùng cùng trạng thái checkpoint. Đã push private `qwerasdf3004/veremi-central-cnn-bigru-attention-e50/1`, status ban đầu QUEUED. Notebook SHA256 `902133e9937dde0e905d66fdcad868fb947c317203e83b7f5077bca742e56ca6`. Notebook kiểm 2×T4, test CPU, cache v4, probe real data, rồi khởi tạo mới train tối đa 50 epoch; nếu hết budget, nối từ epoch đã commit bằng builder và source snapshot của parent. Phiên này sở hữu run mới, tiếp theo kiểm startup/probe/production.

Kiểm 14/11 khoảng09:25VN: CNN-BiGRU **COMPLETE50/50**, Kaggle verifier PASS. Kéo161file/19MiB về `runs/pulls/central-cnn-bigru-attention-v1`. Local audit source snapshot/manifest/notebook SHA/config fingerprint/data identity với v4 trước đó; đủ50CM→10metric/CSV, final predictions→CM và strict weights/BN/AdamW state đều PASS. Phạm vi cụ thể ở `LOCAL_AUDIT.json`; chưa kéo prediction/weights các epoch1–49 hoặc chạy lại inference từ weights trên full-data local. Final accuracy0,9499481101, macro-F1 0,9490269133. Best test macro-F1 tại epoch44:0,9495313240 (quan sát hậu nghiệm). Train+eval6,05600giờ, chưa gồm setup và một phần I/O/audit. Tổng177AMP skips/456100step, tối đa6/epoch; không vượt guard. [Đủ50epoch×10metric và từng lớp](../../../../papers/mnkash-2026-cnn-bigru/results.md). Không cần nối tiếp run này.

Owner đã bỏ PDF MambaCNN và đồng ý bản student supervised100k–150k thay SR-LiCNN. Đã đọc23trang, đối chiếu hình3/5, ghi rõ thiếu cấu hình student và lỗi bảng/công thức ở [paper](../../../../papers/feng-2026-mambacnn/paper.md). Build `mambacnn` **129.328 tham số**, đầy đủ77feature/16lớp, 3nhánh×2Mamba + CNN + self/cross-attention. [Cấu hình, deviation và tests](../../../../papers/feng-2026-mambacnn/rebuild.md). **10 test local PASS** gồm scan outputs/gradients so với recurrence độc lập, strict weights, AMP CPU và exact epoch resume. Đã tích hợp factory/runner/builder, sinh notebook private2T4 draft `hoangdv1711/veremi-central-mambacnn-e50`, SHA256 `288154b895080b89223e8f430cc5ffa611c3b93d355a673087cbb59c8d3524a5`; **chưa push**, chưa có GPU probe hoặc metric. Bước tiếp khi launch: recheck quota/access, probe thật trước production50epoch; không suy tốc độ từ số tham số. Không dùng tên KD cho bản supervised. [Nguồn SR-LiCNN lịch sử](../../../../papers/luo-2025-sr-licnn/paper.md).

Quota live qua SDK ngày 13/11, xác thực identity 8 account, reserved đều 0. qwerasdf3004 30h, hoangdv1711 29,447h, ttalgies 29,133h, hoangdangnhi 28,988h, phamquang0606 28,334h, dangthiet 4,427h; hoangdang1711/minhiw đã hết. Recheck trước push. Host Windows dùng Python313 (torch 2.10 CPU, numpy/pyarrow/nbformat) cho test, Python311 (kaggle/requests/pypdf/fitz) cho API và PDF. Python mặc định trỏ MSYS, không có các package này. Shell network cần escalation; quota đã kiểm thành công ngoài sandbox. Global Kaggle login được giữ nguyên.

## GRU + Attention centralized — Kaggle version 1

Phiên private [ttalgies/veremi-central-gru-attention-e50/1](https://www.kaggle.com/code/ttalgies/veremi-central-gru-attention-e50/versions/1) COMPLETE, Kaggle audit 50/50 epoch. Local kéo 108 file vào `runs/pulls/central-gru-attention-v1`, kiểm source snapshot/manifest với notebook đã push, 50 CM → 10 metric và dự đoán epoch 50 → CM. SHA256 notebook `69557cae9717caf079d60e2e789ef1880a61a94389f83b8ffe79e471d77942ad`. [Kết quả và protocol](../../../research/central-backbones.md).

## 10/11/2026 — CNN–LSTM centralized, session 2

Phiên private `hoangdang1711/veremi-central-cnn-lstm-e50-s2/1` COMPLETE. Tiếp từ epoch 46 của `hoangdang1711/veremi-central-cnn-lstm-e50/1`. Khôi phục weights, AdamW và AMP scaler, giữ source, seed 42, causal windows và cosine horizon 50. Notebook SHA256 `20911f257f6f2d91fa7bc0217929d52eb69b89a31844b14b6fb567cfd3b0d33a`. Kaggle audit đủ 50 epoch. Đã kéo 72 file vào `runs/pulls/central-status-cnn-lstm-s2-v1`, xác minh predictions → CM → metrics cho cả 4 epoch mới. [Kết quả CNN–LSTM](../../../research/central-backbones.md#cnnlstm-trên-veremi-v4-10112026).

## 10/11/2026 — SAE centralized thay GRANDE

Private `phamquang0606/veremi-central-sae-e50/1` COMPLETE. Kaggle audit đủ 50 epoch. Local đã kéo metrics 1–50, dựng lại CM và 10 metric epoch 50 từ dự đoán hai rank, strict-load weights cuối. Artifact ở `runs/pulls/central-status-sae-v1`. [Kết quả SAE](../../../research/central-backbones.md). Notebook SHA256 `00ab19f17c26543d1a381b5373ca2ad3d5a4ec71b27ec1f1c4d0b3292bd76b78`.

Log production xác nhận PyTorch 2.10.0+cu128, 2 Tesla T4, 6 test PASS. Hai rank bắt đầu epoch 1/50, tổng 9.122 step. Ở step 250 grad norm 0,299, skipped 0.

## 10/11/2026 — BERT centralized, session 2

Owner yêu cầu nối BERT. Parent `dangthiet/veremi-central-bert-e50/1` audit đủ epoch 1–9,
macro-F1 epoch 9 = 0,947931. Source snapshot/manifest, config, metrics, weights và resume
ở `runs/pulls/central-status-bert-v1`; local strict weights, AdamW và AMP scaler PASS.
Đã push `dangthiet/veremi-central-bert-e50-s2/1`, cùng seed 42, global batch 4096,
micro 32/GPU, cosine horizon 50. Builder dùng `--parent-artifacts` và `--parent-epoch 9`
để giữ nguyên source phiên 1. Notebook SHA256 `3efde1eb6545441e5aabb5c9526b253fa3eb117ae79258240cc064b429eb5101`.
Worker RUNNING trên 2 Tesla T4. Log xác nhận import epoch 9 và production epoch 10 step 1/9122
ở cả hai rank, grad 0,2129, skipped 0. Quota trước push 19,44 giờ; session cap 12 giờ.

## 10/11/2026 — GRANDE thay TCNN

Đã push private `phamquang0606/veremi-central-grande-e50/1`, status RUNNING. Notebook SHA256
`f63e0c5e8086d40209101673fc9d59ec3e6a2c459f83a7df6bcb372d3dac218e`. Quota trước push 30h, reserved 0.
Phiên đã COMPLETE, Kaggle audit đủ 50 epoch. Local tính lại từ dự đoán trên 9.347.679 dòng test, khớp 10 metric epoch 50. Accuracy 0,860158, macro-F1 0,726454. Đã kéo metrics 1–50, nhãn và dự đoán cuối, cùng CM/weights một số epoch vào `runs/pulls/central-status-grande-v1`. [Phân tích lớp và ngưỡng chia](../../../research/central-backbones.md#grande-trên-veremi-v4-10112026).
TCNN lịch sử giữ checkpoint e1 (macro-F1 0,905491), ngừng kế hoạch nối tiếp.
Lần nối bất cứ run cũ nào phải dùng đúng source snapshot đã push cho run đó.

## 09/11/2026 — centralized 5 backbone

Owner chốt BERT-style supervised trên 77 feature, TCNN kích thước paper, TabM k=4,
CNN–LSTM cửa sổ causal 10 cùng cặp nhận–gửi và đệm đầu; MLP 64/32.
Đã triển khai [thiết kế, cấu hình và runner](../../../research/central-backbones.md).
Test CPU PyTorch 2.10 trên fixture đạt cho đủ 5 model, gồm epoch cuối lẻ, resume, strict weights,
prediction→CM→metrics và ghép thời gian trên parquet nhiều file. Owner yêu cầu push trực tiếp cả 5 run production.
Xác minh 8 tài khoản, mỗi tài khoản còn 30 giờ GPU, reserved=0 trước push.
Kiểm label_mapping/scaler từ dataset Kaggle khớp repo. Cả 5 notebook private được Kaggle nhận version 1,
mỗi phiên yêu cầu 2×T4, mục tiêu 50 epoch. Trạng thái quan sát sau push:

| Model | Phiên | Trạng thái |
|---|---|---|
| TCNN | `minhiw/veremi-central-tcnn-e50/1` | RUNNING, log đầu chưa xác nhận GPU/epoch |
| BERT-style | `dangthiet/veremi-central-bert-e50/1` | QUEUED |
| CNN–LSTM | `hoangdang1711/veremi-central-cnn-lstm-e50/1` | QUEUED |
| TabM | `hoangdangnhi/veremi-central-tabm-e50/1` | QUEUED |
| MLP | `hoangdv1711/veremi-central-mlp-e50/1` | QUEUED |

Lần kiểm tiếp: xem worker log và epoch đã commit; nếu dừng budget trước epoch 50, audit phần đã commit rồi resume đúng parent.

## Cập nhật MobiHet Core — 05/11/2026

- Run hiện tại của chat này: `dangthiet/mobihet-core-mixed-q-blend-r10/1`, push và status RUNNING đã xác nhận.
  Owner chấp thuận target synthetic 50% one-hot + 50% Q. Q cập nhật R2/4/6/8/10,
  floor16, widths [1,0.5], seed42, train→val R10/cosine50, không W&B.
  Builder `scripts/build_mobihet_core_val.py --blend-q`. Quota trước push 5.7577 giờ GPU,
  ước tính khoảng 2 giờ. Test target 50/50 và smoke resume 2→3 chạy trước full train.
- Evidence: `dangthiet/mobihet-core-copula-q-r10/2` và
  `dangthiet/mobihet-core-mixed-q-every2-r10/1` COMPLETE, outputs dưới `runs/pulls/`
  cùng slug + suffix -v2/-v1. Đã kiểm CM ids/counts và macro R10.
  Mixed CE/Q/Q2 macro 0.874586/0.877830/0.877934.
  Q2 vs CE macro +0.0033479, mean +0.006239, p10 +0.007544, class13 +0.003334,
  train R3–10 +12.35%; vs Q train -6.13%. Gate macro +0.005 FAIL.
  Q2 payload recurring 5440 bytes/client R10, chưa bootstrap/framing.
- Blend là giả thuyết exploratory mới: pooled Q có thể làm yếu tín hiệu đúng lớp.
  Trộn nhãn giữ class identity. Cần kiểm F1 thực nghiệm, không mặc định sẽ tăng.
  So archived mixed CE và Q2. Gate vs CE macro >=+.005, mean/p10 >=-.005,
  class13 >=-.01, train overhead <=20%. Không tự mở R50.
- Bước tiếp: kiểm log test/smoke/full train; pull non-PT output đúng version khi COMPLETE,
  audit CM/metrics/bytes và nhóm 979 xe dưới256 dòng train. GPU tests bản mới chưa xác nhận tại handoff.

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../../../CONTEXT.md) → [LESSONS](../../../../LESSONS.md) →
[bài học vận hành](../../../operations/lessons.md) → trang này. Sửa tại chỗ. Lịch sử 29–30/10: [snapshot](../../continue-2026-10-30.md).

## Đang chạy (Kaggle; W&B `21522798-uit/fl-dagsnet-veremi-v2`)

### MobiHet v4 mới — owner yêu cầu bỏ W&B, 30/10/2026

- Phiên này sở hữu `minhiw/mobihet-v4-weighted-nowb/1`; push thành công, status **RUNNING**.
  [Notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-weighted-nowb).
- Log trực tiếp xác nhận torch 2.10.0+cu128, Python 3.12.13, **2 Tesla T4**;
  `test_method_runs[mobihet]` + `test_next_round_resume`: **2 passed**, 15 deselected.
  Lần kiểm mới nhất: **RUNNING, SMOKE PASS**; NPY hoàn tất sau 1.433s. Smoke r1 train774s, r2 train856s;
  đã restart/resume thành công. Smoke compile gate FAIL → eager; **production compile gate cả hai rank PASS**
  (gradient finite, grad_rel ~0,043). Production bắt đầu 30/10 13:30:50 UTC (20:30:50 VN), budget còn8,68h;
  Kiểm 31/10: **COMPLETE ở production r25/50**, log `budget stop before round 26`.
  Phiên dừng bình thường khoảng04:54 giờ VN ngày31/10, chưa hoàn tất mục tiêu50round.
  R25 private/proxy macro-F1:0,883182/0,887442; accuracy:0,914857/0,915196;
  client F1 mean:0,824924/0,849356. Applied586396, skipped9 (0,001535%).
  Cần kiểm artifact/checkpoint hai rank và resume r26–50; chưa khởi chạy session2.
  Các số dưới đây là log các round đầu:
  Macro-F1 private/proxy r1:0,8011/0,4371; r2:0,7095/0,7888; r3:0,8183/0,8357; r4:0,8355/0,8478.
  Client F1 mean r4:0,6481/0,8137. R1 train1341s/eval402s; r3–4 train1143s/eval36s (~19,7 phút/round).
  Nếu tốc độ giữ nguyên, 50 round ~16,5h train/eval; phiên này budget8,68h sẽ cần nối tiếp, chưa đủ r50.
  Smoke 32 xe: r1 private/proxy macro-F1 0,7920/0,6065; r2 0,7546/0,8036,
  client F1 mean r2 0,5975/0,7443. Không phải kết quả full dataset.
- Dataset owner chọn: `odixe0502/veremi-nextgen-v4-scaled-per-vehicle`. Không chuẩn hóa lại.
  Recipe MobiHet hiện hữu, aggregation sample-weighted; W=1, batch64, seed42, cosine50.
- Notebook chạy test CPU → decode/audit 5.343 client và số dòng → smoke 32 client r1,
  khởi động process mới resume r2 → kiểm applied/skipped/finite → train đủ client tới r50 hoặc budget.
  Smoke không dùng điểm test để chọn HP. Session timeout 11h, budget nội bộ 10h kể cả setup, chừa 0,4h finalize.
- Builder: `scripts/build_mobihet_v4.py --owner minhiw`; notebook tại `notebooks/mobihet-v4-weighted-nowb/`.
  Snapshot nhúng 37 file local; base SHA `36ac7ffafcb45974ed736b4d838279728d00e884` **không chứa patch local**.
  SHA256 notebook: `c760a77f38b9756535a93e7ffe9a9f265f2726a94e65cb6d614de027e642fef8`.
  Output có `source_snapshot.json`, `source_manifest.json`, `SMOKE_PASS.json` khi smoke thành công,
  `smoke/` và `mobihet-v4-weighted-nowb/` (metrics/CM/models/state theo rank).
- Quota trước launch: minhiw 20,9296h free, reserved0; hoangdang1711 13,8727h; dangthiet 15,6100h.
  Host Windows này không có torch; kiểm động thực hiện trên Kaggle. CLI2.2.4, Python311 có SDK/nbformat.
- Tiếp: đọc log account-scoped `kaggle kernels logs minhiw/mobihet-v4-weighted-nowb --follow`;
  status/output dùng `/1`. Log stream có `/1` trả404; stream không version hoạt động.
  Khi xong: pull/audit đủ round, CM, config, checkpoint; budget stop thì dựng session resume cùng source snapshot,
  không dùng generator mặc định clone base SHA (thiếu patch). Chưa có bằng chứng hiệu năng cải thiện.

- **Session 2 COMPLETE, r50 audit PASS:** `minhiw/mobihet-v4-weighted-nowb-s2/1`, push 31/10;
  [notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-weighted-nowb-s2).
  Worker xác nhận torch2.10.0+cu128, Python3.12.13, 2 Tesla T4. Đã restore byte-hash đúng 37 file source;
  resume preflight PASS: checkpoint round25 đủ cả hai rank, tensor finite, RNG MobiHet có mặt, metric liên tục r1–25.
  R50 private/proxy macro-F1 0,89530/0,89954; client mean 0,84678/0,86605; class13 0,11591/0,11727.
  Pull 104 file bỏ state lớn exit0; audit 50 CM ×2 view, 9.347.679 dòng/view/round, metric/source/finite/skipped PASS.
  W&B tắt. Quota minhiw đầu session2:30h free.
  Builder `scripts/build_mobihet_v4_resume.py`; notebook SHA256
  `b9a49cee2bd3ae6d5e53c1dd4520445bdc658d41fcbd9ba3da1ea431f3a929cd`.

- **P0 eval-only r25 kết thúc ERROR sau khi đã ghi artifact; audit khoa học PASS, gate FAIL:** `minhiw/mobihet-v4-p0-local-prior-r25/1`;
  [notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-p0-local-prior-r25). Không train lại: nạp checkpoint hai rank r25,
  kiểm `private_raw` khớp artifact cũ, rồi chấm `private = z + log π_k − log π_g`. Gate khóa trước launch:
  local-prior làm macro-F1 0,88319→0,88122, client mean 0,82493→0,81169, p10 0,67724→0,64526.
  ERROR do raw eager-vs-compiled lệch nhỏ vượt assertion 1e-12, không phải eval crash; không rerun. Builder
  `scripts/build_mobihet_v4_p0_eval.py`; notebook SHA256
  `3d1c64fba5d623293a813d84c0582b0a8ad2a6d2b28b20bd9cfd3a60fd8fff50`; W&B tắt.

- **P1a identifiability COMPLETE, audit PASS:** `minhiw/mobihet-v4-p1-identifiability/1`;
  [notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-p1-identifiability). Chỉ đọc `train/`, HGB + MLP 3×256
  fit fold train và chấm full fold val cho binary benign/timeDelay trên 77 feature; thêm exact/rounded collision audit.
  Gate khóa: best val F1 lớp13 ≥0,50 ⇒ method branch; <0,30 ⇒ raw dataset P1b; giữa hai mức ⇒ P1b.
  HGB/MLP val F1 lớp13 =0,54138/0,54233 ⇒ gate `REPRESENTATION_SUFFICIENT_METHOD_BRANCH` PASS.
  Predictions đủ1.389.784 dòng val; confusion, counts, source hash và footer fingerprint audit PASS. Notebook SHA256
  `0e60027104ccb19459fd8fb599fc66f7c52e8282679ac49cf0eb25dfc044f730`; W&B tắt.

- **RelKD train→val paired arm v2 COMPLETE, audit PASS, gate FAIL:** `minhiw/mobihet-v4-relkd-val-r10/2`,
  [notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-relkd-val-r10); local notebook `notebooks/mobihet-v4-relkd-val-r10/`, builder
  `scripts/build_mobihet_relkd_val.py`, SHA256 `c095997e9c1c4e75b21fd59aa2007db63f992648b7b0efba9f090e5d5c3018e9`.
  Chỉ decode train/, chạy matched `mobihet` và `mobihet_rel` tới r10 trên horizon50; unit/resume + smoke2-round là hard gate
  trong notebook. V1 train+resume smoke r1–2 khỏe nhưng checker đòi metadata `method` ở rank1 ⇒ ERROR sau smoke;
  sửa assertion rank-aware, không đổi khoa học. V2 startup 2×T4; unit/method/exact-resume RelKD 3 PASS.
  Audit source/hash, 20 CM, 5.343 client, 4.676.107 dòng/view/round, metric/finite/skipped PASS. So control r10:
  class13 +0,20996, macro +0,01499, nhưng client mean −0,01331 và p10 −0,05480 ⇒ gate FAIL. Không kéo dài/tune,
  không chạy official test. Nhánh tiếp `mobihet_pair`: binary CE raw logits 0↔13, builder
  `scripts/build_mobihet_pair_val.py`; đã push arm bên dưới. Gate khoa học ghi ở [MobiHet improvement](../../../methods/mobihet-improvement.md).
  W&B tắt.

- **Pairwise 0↔13 train→val paired arm COMPLETE, audit PASS, gate FAIL:** `minhiw/mobihet-v4-pair-val-r10/1`, push 31/10;
  [notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-pair-val-r10). Account `minhiw` còn 13,19 GPU-h trước push;
  notebook SHA256 `fe0950aa5cd9ee3a4174b7178dfd20417d196b8780538ac99ed0e3b30d402659`.
  Worker xác nhận torch 2.10.0+cu128, Python 3.12.13, 2 Tesla T4; unit/method/exact-resume filter 3 PASS.
  Train-only NPY xong sau 995s; smoke resume PASS. Audit source/hash, 20 CM, 5.343 client,
  4.676.107 dòng/view/round, metric/finite/skipped PASS. So control r10: class13 +0,21690, benign +0,00303,
  macro +0,01320, client mean +0,00473, p10 +0,00009 ⇒ gate FAIL. Không nới gate/kéo dài/tune; không official test.
  Nhánh tiếp `mobihet_stats`: bỏ proxy, one-time global class statistics, builder `scripts/build_mobihet_stats_val.py`.

- **Proxy-free statistics paired arm COMPLETE, audit PASS, gate PASS:** `hoangdang1711/mobihet-v4-stats-val-r10/1`, push 01/11;
  [notebook](https://www.kaggle.com/code/hoangdang1711/mobihet-v4-stats-val-r10). Account `hoangdang1711` còn 30,00 GPU-h trước push;
  notebook SHA256 `0fa5186816d0ea58a5099a3066dce526b38859ed872ab87e8aa738d10b458431`.
  Worker xác nhận torch 2.10.0+cu128, Python 3.12.13, 2 Tesla T4; method/resume filter 3 PASS; train-only cache 145s;
  `STATS SMOKE PASS` sau resume r1→r2. Pull 54 file/761 MiB exit 0. Audit source/hash, 20 CM, 5.343 client,
  4.676.107 dòng/view/round, metric/finite/skipped PASS. So control r10: macro +0,01398, client mean +0,00710,
  p10 +0,01133, class13 +0,27724; train r3-r10 nhanh hơn 73,02%; payload mỗi round 0 ⇒ gate khóa trước PASS.
  Không đổi HP theo val. Không W&B.

- **Proxy-free statistics final r50 RUNNING:** `hoangdang1711/mobihet-v4-stats-final-r50/1`, push 01/11;
  [notebook](https://www.kaggle.com/code/hoangdang1711/mobihet-v4-stats-final-r50). Account còn 25,72 GPU-h trước push;
  notebook SHA256 `cc1cfefbc9a2409dbb587e0f247b57f384d34e0e5088627862f747bcab0dd3d2`. Train train+val,
  eval official full test mỗi round, 50 round; worker xác nhận 2 Tesla T4 và method/resume filter 3 PASS. Cache train+test
  xong sau 1.296s; `STATS FINAL SMOKE PASS` sau resume r1→r2. Production compile gate PASS; đã commit ít nhất r36/50.
  Private macro/client mean tăng r1 0,8005/0,4838 → r10 0,8703/0,7755 → r25 0,8837/0,8275 →
  r36 0,8870/0,8415; loss 1,2620→0,2591; r2-r36 train/eval ổn định 291/33s. Không NaN/OOM. Không W&B.

### Các phiên kế thừa từ bàn giao trước (chưa xác minh trên host này)

| kernel | account | SHA | việc | ghi chú |
|---|---|---|---|---|
| `fl2-nilm-r2` | accanniez | 903ac42 | NILM-FL 50 round | ~1.017 s/round ⇒ budget 11,5 h dừng ~r38 (state theo rank) |
| `fl2-tinyproto-r3` | anhtruong1909 | 55d13cc | TinyProto chạy lại (view prototype theo Eq. 12) | push 30/10 ~10:50, ~4,5 h |

Kiểm: `python .claude/skills/kaggle-training-notebook/scripts/kaggle_as.py <acct> -- kaggle kernels status <acct>/<slug>`
(API Kaggle chập chờn: status rỗng ⇒ thử lại). W&B: `wandb.Api().runs("21522798-uit/fl-dagsnet-veremi-v2")`.

## Đã xong (30/10) — kết quả: [results](../../../research/results.md), chi phí: [cost](../../../research/cost.md)

- 14 run FL 50 round + centralized 50 epoch + CANPerFL đã pull về `runs/pulls/<slug>/` (git-ignored, bỏ `state*.pt`) và audit
  PASS (đủ round, CMs, không NaN, HP đúng protocol). Bảng 10 metric tính lại từ CMs (`scripts/report.py`).
- Lỗi đã xử lý: APFed ERROR sau r50 khi ghi state (RAM host; metric đủ; sửa `bda3c07`, không chạy lại); TinyProto r2 dùng sai
  prototype suy luận (sửa `55d13cc`, chạy r3); SCAFFOLD không ổn định r1–r17 (NaN r15–16) rồi hồi phục — ghi chú trong results.
- Kernel cũ không lưu model mỗi round đều đã dừng (`dinhan25/fl2-v2x-htfl` cũ tự hết; bỏ qua).

## Cập nhật mới nhất 01/11 — stats final và real-BN

Stats final R50 COMPLETE; pulled 60 non-.pt artifacts tại `runs/pulls/mobihet-v4-stats-final-r50-v1`.
CM/metrics audit đủ 50 round, 2 views, 5.343 ids, 9.347.679 rows/view. Weights/resume chưa audit.
Macro 0,887913; client mean 0,846076; p10 0,725657; class13 0,339987.
Đã push `hoangdang1711/mobihet-v4-stats-realbn-val-r10/1`: paired control stats và candidate real-only BN.
2 Tesla T4, 5 method/resume tests PASS; chờ cache/smoke. Quota trước push 20,35h.
Gate và SHA ở [design](../../../methods/mobihet-improvement.md). Ưu tiên theo dõi arm này; mục theo dõi final bên dưới đã hoàn tất.

## Adaptive arm — cập nhật mới nhất

04/11 Core theo proposal đang chạy `dangthiet/mobihet-core-copula-q-r10/2` trên 2 T4.
V1 dừng ở source hash trước train do thay chuỗi quá rộng. V2 hash đúng, 3 tests floor/Q và resume PASS.
SHA notebook 305115f4f62c3fc1c1e929bfa6025e836eca789c533962451ce85e88b0f6f098.
5 arms seed42 train→val R10: ce/q W1, mixed_ce/mixed_q W1 và W0,5, q_no_floor W1.
Q từ snapshot đầu round, hard labels R1. Payload mô phỏng 576 B upload/512 B download từ R2, bootstrap riêng.
Chờ smoke và production. Khi collect kiểm CORE_RESULTS, CORE_GROUPS, config, Q13, counts, thời gian và CM từng arm.

04/11 seed43 COMPLETE. Pull 32 file tại `runs/pulls/mobihet-v4-stats-hist-s43-v1`. Source snapshot khớp seed42, cả hai config seed43, 20 CM × 2 views đủ 5.343 xe và 4.676.107 dòng/view, metrics khớp. Gate FAIL macro/p10. Dừng histogram theo kế hoạch.

03/11 mới nhất: owner yêu cầu seed43 replication, đã push `dangthiet/mobihet-v4-stats-hist-s43-r10/1`.
Paired Stats/histogram cũ R10, horizon50; dùng exact source snapshot46files từ hist seed42 audited.
Smoke vàproduction dùng `--set seed=43`; hard assertions config seed43; không thay phương pháp hoặc gate.
Builder `scripts/build_mobihet_stats_hist_seed43.py`; notebook SHA
`526d7e298044c77d3974bce7be7f81a7af1b24518ecbfc9af8c007b077722eb8`.
Local frozen-source identity/CLI seed/syntax/metadata PASS. Quota15,98h, dự kiến2–3h.
RUNNING, đúng2 Tesla T4,torch2.10.0+cu128/Python3.12.13;22 frozen-source tests PASS,21 deselected.
Chờ cache/smoke seed43/R10; run do phiên sở hữu. Báo cùng2 lần seed42; không chọn lần tốt hay tự launchR50.
Không W&B/Git push; dataset giữ nguyên. Các run histmask/hist cũ đều COMPLETE.

03/11 mới nhất: histmask `dangthiet/mobihet-v4-stats-histmask-val-r10/1` COMPLETE đủ3 armsR10.
Pull45 non-.pt files/21MiB exit0 tại `runs/pulls/mobihet-v4-stats-histmask-dangthiet-v1`.
Source/config/30 CM×2views/5343ids/4676107rows/metrics/finite/bytes/gate audit PASS; weights chưa audit.
Stats/hist/histmask macro0,875623/0,880629/0,880379; mean0,782102/0,787265/0,786506;
p10 0,569974/0,580995/0,574657; class13 0,293487/0,300418/0,299931.
Histmask FAIL original Stats gate (+0,004756<0,005) và repair gate (macro âm,p10−0,006338).
Đóng repair, không R50 hoặc launch mới. Histogram control lần này macro+0,005007 vs Stats, sát ngưỡng;
lần trước+0,004766. Không chọn lần tốt rồi gọi xác nhận độc lập: cùngseed, backend có dao động.
Mọi dòng histmask RUNNING bên dưới đã cũ. Kết quảR9 nhỉnh hơn không giữ được ởR10.

03/11 histmask kiểm mới nhất: RUNNING, candidate sửa đã tớiR9/10, macro0,8781/mean0,7767,
train265s/eval19s, compile gate PASS2rank. Histogram chưa sửa cùngR9 macro0,8777/mean0,7750.
Hai control đã COMPLETE R10: Stats0,8756/0,7821; hist0,8806/0,7873 (log làm tròn, chưa audit).
Chờ candidateR10/finalize rồi pull45 artifacts dự kiến và audit hai gate; không kết luận từ số làm tròn.

03/11 histmask kiểm tiếp: RUNNING. Stats control COMPLETE R10 macro0,8756/mean0,7821.
Nhánh histogram chưa sửa tớiR4/10 macro0,8583/mean0,6787, train265–266s/eval19s, compile gate PASS2rank.
Chưa thấy histmask production bắt đầu. Chưa có kết quả candidate/gate; không so điểm khác round.

03/11 histmask kiểm tiến độ: RUNNING; cache352s,25 tests PASS, candidate smoke2 round+resume PASS.
Smoke compile gate FAIL→eager; production Stats control compile PASS cả2 GPU.
Log mới nhất controlR4/10 macro0,8537/client mean0,6744, train265s/eval19s;
chưa tới histogram/histmask production. Chưa có kết quả so sánh R10 hoặc gate. Không repush.

03/11 mới nhất: owner yêu cầu triển khai missingness repair. Đã push
`dangthiet/mobihet-v4-stats-histmask-val-r10/1`,3 arms Stats/hist/histmask R10; gate chính vs Stats giữ nguyên,
thêm gate cải thiện macro>0 vs hist và guard mean/p10/class13/time. Không tự mởR50.
Builder `scripts/build_mobihet_stats_histmask_val.py`; source47 files; notebook SHA
`a1e60c66e2f2d12bfe2a139121660182f5fda7e47dd257306012e37de3507543`.
Local AST/3arms/metadata/smoke wiring PASS; remote hard tests trước train. Quota18,93h; dự kiến3–4h.
Probe trên sampled hist xác nhận flag/value inconsistent, scope ghi design. Payload bằng hist, không W&B.
Worker xác nhận2 T4,torch2.10.0+cu128/Python3.12.13;25 tests PASS,21 deselected.
Chờ cache/smoke/production. Run này do phiên sở hữu; hist cũ đã COMPLETE.

03/11 mới nhất: hist `dangthiet/mobihet-v4-stats-hist-val-r10/1` COMPLETE cả2 arm R10.
Pull32 non-.pt files exit0 tại `runs/pulls/mobihet-v4-stats-hist-dangthiet-v1`.
Source/config/20 CM×2 views,5343 ids,4676107 rows/view/round, metrics/finite/skipped/bytes audit PASS;
weights chưa kiểm. `CM_AUDIT.json` ghi phạm vi. Gate FAIL duy nhất macro+0,004766<+0,005.
Control→hist: macro0,875697→0,880463; mean0,781884→0,785900; p10 0,568208→0,577876;
class13 0,290622→0,300157. Train268,30→268,98s(+0,255%); bootstrap tăng như design, round bytes0.
Không nới gate/tune bins/kéo R50. Chưa launch mới. Các dòng RUNNING hist bên dưới đã cũ.

02/11 hist kiểm tiếp: RUNNING, control đã xongR10 (macro0,8757/client mean0,7819).
Candidate histogram production compile gate PASS cả2 rank, đã commitR1: macro0,8037/mean0,4738,
loss1,2557, train274s/eval88s. Control cùngR1: macro0,7988/mean0,4714.
Đây là log làm tròn, chưa artifact audit; chưa kết luận gate hay overhead từR1 vì warm-up/compile khác nhau.
Chờ candidateR10, không repush. Dòng chưa thấy candidate bên dưới đã cũ.

02/11 kiểm hist mới nhất: `dangthiet/mobihet-v4-stats-hist-val-r10/1` RUNNING.
Cache907s; candidate smoke32 client r1→resume r2 PASS (macro0,8106→0,8303).
Smoke compile gate FAIL→eager; production control compile gate PASS cả2 rank.
Log đã tới control stats r9/10: macro0,8727, client mean0,7719; train268–269s/eval20s ổn định.
Chưa thấy production candidate histogram trong log; notebook chạy control trước rồi candidate.
Chưa có paired gate/kết luận cải thiện; tiếp chờ control10 và candidate10, không repush.

02/11 mới nhất: owner yêu cầu chỉnh và chạy. Đã push `dangthiet/mobihet-v4-stats-hist-val-r10/1`.
Candidate histogram marginal transport256 bins theo train class/feature, giữ Gaussian latent correlation;
loss/BN/HP/inference của stats gốc. Không giữ prior correction thất bại; không đổi dataset.
Bootstrap/client tổng5.243.456 bytes up/2.720.320 down; round0bytes. Gate R10 giữ nguyên cả time overhead≤20%.
Builder `scripts/build_mobihet_stats_hist_val.py`; SHA notebook
`23209043c147c803768f3ab0fba02acbc39a5e3118c7c97c6517bd01e17d08cd`.
Local AST/embedded modules/paired arms/metadata PASS; host không có torch, dynamic tests là hard gate remote.
Quota dangthiet21,24h trước push, dự kiến2–3h cộng bootstrap. RUNNING, torch2.10.0+cu128/Python3.12.13,
đúng2 Tesla T4;22 tests PASS,21 deselected (sampler/accounting/control/no-proxy/CPU exact resume).
Chờ cache/smoke2 GPU/production. CPU resume PASS chưa chứng minh bit-exact CUDA resume.
Theo dõi histogram arm tiếp; fidelity/prior đều đã COMPLETE. Run do phiên này sở hữu; không Git push.

02/11 kiểm mới nhất: Gaussian fidelity `dangthiet/mobihet-v4-gaussian-fidelity/1` COMPLETE.
Pull8 files/32MiB exit0 về `runs/pulls/mobihet-v4-gaussian-fidelity-v1`; `LOCAL_AUDIT.json` ghi phạm vi:
source hashes/manifest totals/12 predictions→CM/10 metrics/AUC/sample IDs/shapes/flags PASS.
Chưa độc lập rescan raw data hoặc recompute full moments local. Domain AUC1,0 cả4 lớp, cả khi bỏ6 flags.
Real→real vs Gaussian→real macro binary:0↔13 0,673399/0,581192;11↔15 0,966949/0,902665.
Không phải FL score. Dấu hiệu fidelity hypothesis đạt nhưng chưa chứng minh multimodality hay nhân quả với FL.
Tiếp: audit miền/quan hệ feature, thử sampler tôn trọng biến rời rạc và phụ thuộc có căn cứ;
chưa launch mới trong lần kiểm, không R50. Các dòng chờ fidelity bên dưới đã hoàn tất.

02/11 mới nhất: owner đồng ý diagnostic trước sửa sampler. Đã push CPU-only
`dangthiet/mobihet-v4-gaussian-fidelity/1`; startup log scan1/410 thành công. Không W&B/official test.
Builder `scripts/build_gaussian_diagnostic.py`, runner `scripts/diagnose_gaussian.py`.
Full train moments4 lớp, sampled domain/transfer, protocol và dấu hiệu bác bỏ ở design.
SHA notebook `576c2f5aeef1cf1fe37ed2b0f71de0f4807c7da0a61725947a2aa0f010c60c4d`.
Local AST/nbformat và priority reservoir row-identity/chunking PASS. Dynamic full-data validation thực hiện remote.
Account dangthiet quotaGPU21,24h trước push; diagnostic không sử dụng GPU.
Tiếp: theo dõi scan→domain/transfer; COMPLETE thì pull, audit source/counts/samples/predictions→CM/metrics,
rồi mới chọn sửa sampler. Đây là run hiện tại do phiên này sở hữu; prior/nogauss/client diagnostic đều đã COMPLETE.

02/11 kiểm mới nhất: prior `dangthiet/mobihet-v4-stats-prior-val-r10/1` COMPLETE cả hai arm R10.
Pull32 non-.pt files exit0 tại `runs/pulls/mobihet-v4-stats-prior-dangthiet-v1`. Source/config/20 CM ×2 views,
5.343 ids/4.676.107 rows/view/round và metric audit PASS; weights chưa kiểm. `CM_AUDIT.json` ghi phạm vi.
Gate FAIL: macro+0,003950 (<+0,005), mean−0,012551, p10−0,044708, class13−0,051158.
Train262,29→262,38s, round bytes0. Đóng arm prior, không mở R50; mọi dòng prior RUNNING bên dưới đã cũ.
Không launch mới trong lần kiểm này. Bước tiếp: khảo sát chất lượng Gaussian theo lớp trước đề xuất cơ chế mới;
giữ stats uniform làm control, không tune prior strength theo val.

02/11 diagnostic CPU COMPLETE: pull9 files exit0 tại `runs/pulls/mobihet-v4-stats-client-diagnostic-v1`.
Local audit source hashes/completion/count totals/CM per-client-class identity PASS (32.685.166 train,4.676.107 val).
Phân tích ghi trong design: bỏ Gaussian giảm mean mạnh nhất nhóm256–4095 train rows; các lớp11/13/15 vắng
ở train có F1=0 khi bỏ Gaussian. Prior arm vẫn là thí nghiệm train đang chạy; không mở arm mới từ cutoff val.
Các dòng chờ diagnostic bên dưới đã hoàn tất.

02/11: owner yêu cầu tranh thủ phân tích và push. Prior vẫn RUNNING khi kiểm; không repush/đổi run đó.
Đã push CPU-only `dangthiet/mobihet-v4-stats-client-diagnostic/1`, builder `scripts/build_stats_client_diagnostic.py`.
Script `scripts/analyze_stats_clients.py` đọc3 cột client_id/label/fold từ train parquet, không đọc test;
đối chiếu chính xác5343×16 val counts với embedded audited nogauss/control R10 CMs.
Xuất analysis.json (10metrics, train-size/missing-class groups, support theo lớp, nhầm11/13/15),
clients.csv, train_val_class_counts.npz, ANALYSIS_PASS.json. Chờ scan; chưa kết luận theo train-count.
SHA notebook `f7b09790ede6875b7e84d09e1691834d50378d28064128357241dea204bbb1ce`.
Lần push đầu400 và status404; bỏ view private_local dư thừa khỏi payload, push tiếp version1 thành công.
Đây là diagnostic CPU chạy song song, không phải arm train mới. Tiếp: collect diagnostic rồi ghép với prior R10 khi xong.

02/11 mới nhất: owner yêu cầu build/push tiếp. `dangthiet/mobihet-v4-stats-prior-val-r10/1` RUNNING,
đã xác nhận torch2.10.0+cu128/Python3.12.13/2 Tesla T4;17 tests PASS,21 deselected; chờ cache/smoke và paired R10.
Candidate synthetic CE(z−log_pg), giữ real CE, uniform sampling, BN và optimizer. Prior từ bootstrap train;
không thêm proxy/payload. Control stats; gate giữ nguyên, không tự mở R50. Quota trước23,54h, ETA2–3h.
Builder `scripts/build_mobihet_stats_prior_val.py`; source45 file. SHA notebook
`fc8805c21c47187b04073233e313a204fb970c173a98f6259a18040bb05fdf5d`.
Local kiểm AST mọi embedded module/cell, metadata, paired arms PASS; host không có torch nên dynamic tests là hard gate trên Kaggle.
Theo dõi run prior này tiếp; nogauss bên dưới đã đóng. Không W&B; không Git push.

02/11 kiểm mới nhất: `dangthiet/mobihet-v4-stats-nogauss-val-r10/1` COMPLETE đủ hai arm R10.
Pull32 file non-.pt exit0 tại `runs/pulls/mobihet-v4-stats-nogauss-dangthiet-v1`; source/20 CM/metrics PASS,
5.343 ids, 4.676.107 rows/view/round; weights chưa audit. `CM_AUDIT.json` lưu phạm vi kiểm.
Gate FAIL: macro −0,008963, mean −0,033406, p10 −0,075048, class13 +0,024601; train nhanh26,35%.
Không kéo dài nogauss R50. Giữ stats làm control; bước tiếp phân tích hỗ trợ theo lớp/client trước đề xuất arm mới.
Các dòng chờ nogauss phía dưới là lịch sử, đã hoàn tất.

02/11 mới nhất: đã push `dangthiet/mobihet-v4-stats-nogauss-val-r10/1`, paired diagnostic bỏ toàn Gaussian.
Đúng2 T4; 14 tests PASS (bao gồm không sample Gaussian và exact resume). Chờ cache/smoke/production.
Quota trước25,64h; SHA/gate/diễn giải trong design. Theo dõi run này tiếp; không tự mở final từ diagnostic.

02/11 mới nhất: massnorm dangthiet version1 COMPLETE; pull 32 non-.pt artifacts exit0 về
`runs/pulls/mobihet-v4-stats-massnorm-dangthiet-v1`. Source hash + 20 CM/metrics PASS; weights chưa kiểm.
Gate FAIL: macro 0,875736→0,870457; mean 0,781558→0,779740; class11/15 không hồi phục. Đóng massnorm.
Không launch R50 hoặc arm mới trong lần kiểm này. Các trạng thái chờ massnorm bên dưới đã cũ.

02/11 tiếp: owner yêu cầu điều chỉnh dựa hoangdang1711. Đã build/push
`dangthiet/mobihet-v4-stats-massnorm-val-r10/1`: normalize adaptive class weights về mean1 mỗi client;
control uniform stats. Nhận đúng 2 T4; 11 test method/weight/resume PASS. Chờ cache/smoke/production.
Quota trước push 27,73h; gate/SHA và cơ chế ghi trong design. Không kết luận cải thiện trước R10 audit.

02/11: `hoangdang1711/mobihet-v4-stats-adaptive-val-r10/1` COMPLETE, 32 non-.pt files pulled exit 0
về `runs/pulls/mobihet-v4-stats-adaptive-hoangdang1711-v1`. Source hashes + 20 CM/metrics PASS,
đủ 5.343 ids và 4.676.107 dòng/view/round; weights chưa audit. Gate FAIL: macro +0,001025 < +0,005.
Control→adaptive: macro 0,875763→0,876788; mean 0,782237→0,785772; p10 0,572870→0,576334;
class13 0,294855→0,332395. Train 265,37→264,44s; round bytes 0. Không nâng R50 dựa trên gate này.
Thông tin yêu cầu dừng phiên hoangdang1711 bên dưới đã hết hiệu lực vì phiên đã COMPLETE.

Owner xác nhận chuyển account sang dangthiet. Đã push `dangthiet/mobihet-v4-stats-adaptive-val-r10/1`,
giữ nguyên notebook SHA, chỉ đổi metadata owner. Account còn 30h trước push.
Phiên hoangdang1711 đã RUNNING trước chuyển; CHƯA xác nhận dừng. Cần owner Cancel run trong UI để tránh trùng.
CLI 2.2.4 không hỗ trợ `--no-run`; lệnh đó thất bại trước upload, sau đó push thường thành công.

Real-BN COMPLETE, 20 CM audit PASS, quality gate FAIL; không kéo dài arm BN.
Đã push `hoangdang1711/mobihet-v4-stats-adaptive-val-r10/1`, trạng thái mới nhất QUEUED.
Candidate class/client Gaussian weight batch/(batch+train count), paired control stats, train→val R10.
Quota trước push 18,14h; SHA/gate ở design. Chưa có GPU/test/smoke confirmation cho kernel mới.
Bước tiếp: xác nhận 2 T4, test/resume filter và smoke PASS; khi COMPLETE kéo CM, kiểm gate trước quyết định.

## Bước tiếp

1. Theo dõi `hoangdang1711/mobihet-v4-stats-final-r50/1`; xác nhận `STATS FINAL SMOKE PASS`, rồi chờ r50, pull/audit
   source/hash + 50 CM + 10 metric + class13 + finite/skipped/zero-byte. Không chọn best test round, không tune theo test.
2. `fl2-nilm-r2` dừng budget (~r38) ⇒ session 2, cùng `--run`:
   `python scripts/gen_notebook.py --owner accanniez --slug fl2-nilm-r2s2 --method nilm --run fl2-nilm-r2 --resume-kernel accanniez/fl2-nilm-r2 --max-hours 11.5`
   rồi `embed_wandb_key.py` + `chmod 600` + `ast.parse` mọi cell + push `-t 43200`.
3. `fl2-tinyproto-r3` xong ⇒ pull (`pull_kernel_output.py ... --pattern '^(?!.*state(_rank\d+)?\.pt$)'`), audit, thêm vào results
   (headline `client_proto`); kiểm view prototype ≈ view head (smoke 100 xe r2: 0,819 vs 0,825).
4. Viết `docs/methods/` cho APFed, V2X-HtFL, CANPerFL; sửa bảng "còn thiếu" trong [papers/README](../../../../papers/README.md).
5. MobiHet tiếp ở repo [`odixe06/mobihet`](https://github.com/odixe06/mobihet) (bàn giao riêng `docs/operations/continue.md` của repo đó);
   arm cuối 50 round so với bảng results ở đây.

## Lưu ý vận hành

- `--set` qua `json.loads` ⇒ chuỗi cần dấu `"` JSON (xem [lessons](../../../operations/lessons.md) 30/10). Notebook push phải nhúng key W&B
  (`embed_wandb_key.py`; quên ⇒ `SystemExit: W&B credential not embedded`, bản 1 fl2-canperfl).
- Commit không có trailer `Co-Authored-By` (owner 30/10).

## Việc tay (owner)

- Không có việc gấp. minhtrit06 hết GPU quota tới 31/10. dinhan25 chỉ có token (chạy được qua `kaggle_as.py`).
