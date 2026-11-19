# MobiHet v4 — chẩn đoán round 25 và hướng cải tiến

Trạng thái 01/11/2026: MobiHet v4 r50 audit **PASS**; private/proxy macro-F1 0,8953/0,8995. P0 r25 **FAIL**;
P1a identifiability **PASS**; RelKD và pairwise 0↔13 train→val r10 **FAIL**. Không dùng official test để chọn feature, loss, round hay ngưỡng.

## Bằng chứng hiện có

Nguồn: `rounds.jsonl` của `minhiw/mobihet-v4-weighted-nowb/1`, đủ r1–25; số dưới đây chưa được audit lại từ confusion matrix.

| view r25 | macro-F1 | accuracy | client-F1 mean | p10 | F1 lớp 13 `timeDelayAttack` | macro-F1 bỏ lớp 13 |
|---|---:|---:|---:|---:|---:|---:|
| private | 0,8832 | 0,9149 | 0,8249 | 0,6767 | **0,0559** | **0,9383** |
| proxy | 0,8874 | 0,9152 | 0,8494 | 0,7266 | **0,0368** | **0,9441** |

So FedAvg r50 0,9042, private thiếu 0,0210 và proxy thiếu 0,0168. Nếu chỉ lớp 13 thay đổi, private cần F1 lớp 13
0,392 và proxy cần 0,305 để bằng FedAvg. Các lớp yếu kế tiếp là benign (0), positionMirroring (7) và
suddenConstantSpeed (11); 12/16 lớp private đã ≥ 0,919.

Ba dấu hiệu không ủng hộ việc ưu tiên tăng bit hoặc tăng lượng Gaussian:

- Q3 residual tương đối chỉ 0,00364 ở r25; chưa chứng minh lượng tử là nút thắt.
- Proxy hơn private cả pooled macro-F1 (+0,0043) lẫn client-F1 mean (+0,0244); cơ chế tri thức hiện tại chưa tạo lợi cá nhân hóa.
- Lớp 13 có 1.924.480 dòng train, nên lỗi không giải thích bằng thiếu support toàn cục. Dataset hiện ghi nhận
  benign ↔ timeDelay là cặp nhầm chính vì attack dịch cả thời gian gửi và nhận.

## P0 — kiểm đúng luật quyết định, không train lại

**Vấn đề.** Private train bằng `CE(z + log π_k − log π_g, y)` nhưng `evaluate()` chấm logits thô `z`.
Logits thô là view đã hiệu chỉnh về prior toàn cục; headline lại chấm model k trên test của chính client k.

**Can thiệp.** Từ checkpoint r50, báo ba view trên cùng predictions:

1. `private_raw = z` (view hiện tại);
2. `private_local = z + log π_k − log π_g` (đúng logits đi vào CE local);
3. hỗn hợp logits private/proxy với hệ số khóa trên fold `val`, không fit trên test.

**Dấu hiệu bác bỏ.** `private_local` không tăng client-F1 mean ít nhất 0,005 hoặc làm pooled macro-F1 giảm quá 0,005;
hỗn hợp không tăng ít nhất 0,005. Khi đó không đổi luật headline.

**Kết quả r25 — FAIL.** Artifact CM đủ 5.343 client/9.347.679 dòng và metric tính lại khớp JSON. So raw,
`private_local` làm macro-F1 0,88319 → 0,88122, client mean 0,82493 → 0,81169 và p10 0,67724 → 0,64526.
F1 lớp 13 tăng 0,05587 → 0,07312 nhưng không bù thiệt hại các lớp khác. Giữ logits raw làm headline; không rerun P0.

## P1 — cổng định danh lớp 13 trước khi sửa thuật toán

**Vấn đề.** F1 lớp 13 gần 0 ở cả private lẫn proxy, trong khi 15 lớp còn lại đạt 0,94. Có thể 77 feature hiện tại không chứa
đủ tín hiệu quan sát để tách timeDelay khỏi benign.

**Đo trên train/val, không official test:**

- audit exact/near duplicate feature giữa nhãn 0 và 13, theo client và sender;
- binary oracle 0-vs-13 bằng HGB và MLP trên 77 feature, chấm fold `val` theo thời gian/client;
- lặp oracle trên raw timing/provenance hợp lệ: receive time độc lập, send time, message sequence và cửa sổ causal theo sender;
- cấm đưa `source_run`, world/attack archive hoặc đường dẫn file vào model vì chúng tiết lộ loại tấn công.

**Dấu hiệu quyết định.** Nếu oracle raw đạt F1 lớp 13 ≥ 0,50 còn oracle 77 feature < 0,30, lỗi nằm ở dataset representation.
Nếu cả raw oracle < 0,30, lớp 13 không định danh được dưới observation model hiện tại: không tune loss để tạo điểm giả;
hoặc bổ sung sensor/timestamp độc lập, hoặc công bố task 15 lớp sau khi gộp observationally-equivalent class và chạy lại mọi baseline.

**P1a kết quả — PASS method branch.** Fit fold train, chấm full fold val; official test không đọc. HGB F1 lớp13 0,54138
(recall 0,7864), MLP 3×256 đạt 0,54233 (recall 0,8506); best vượt gate 0,50. Exact collision trên mẫu cân bằng
200k/lớp = 0; rounded 0,1 chỉ 1/200k. Artifact predictions 1.389.784 dòng, confusion, source hash và dataset footer
`16d3475e...` audit PASS. Kết luận giới hạn: 77 feature có tín hiệu cho binary 0↔13; chưa chứng minh specialist sẽ nâng
multiclass MobiHet. Không mở dataset v5/P1b lúc này; chuyển P3, ưu tiên reliability-gated KD rồi specialist.

## P2 — dataset v5 nếu raw còn tín hiệu

Chỉ mở khi P1 chứng minh raw có tín hiệu. Dựng feature causal, fit bằng fold train:

- giữ `rel_delay`, `ses_dt`, `ses_dt_send`, `ses_dt_skew` ở cả physical/log scale và normal-score; tránh mất ngưỡng vật lý do clipping/tie;
- thêm chuỗi 8–16 message theo `(client, sender)`: inter-arrival, inter-send, clock-skew drift, message-age residual,
  position prediction tại thời điểm nhận và neighbor-consensus residual;
- thêm nhánh temporal 1D nhỏ hoặc specialist 0-vs-13; classifier 16 lớp chỉ gọi specialist khi top candidates thuộc {0,13};
- scaler, rolling state và feature selection chỉ fit trên train. Fold val dùng chọn thiết kế; sau khi khóa mới retrain train+val một lần.

**Cổng val:** F1 lớp 13 ≥ 0,35; benign F1 giảm ≤ 0,005; macro-F1 tăng ≥ 0,015; client p10 không giảm quá 0,005.
Nếu dataset đổi, phải chạy lại tối thiểu centralized, FedAvg, APFed và MobiHet trên cùng v5; số v4 không còn là đối chứng công bằng.

## P3 — sửa phương pháp nếu representation đã đủ

Ưu tiên theo thứ tự, mỗi arm đổi một yếu tố:

1. **KD có reliability theo lớp/client.** Trên calibration từ train/val, client chỉ nhận KD từ proxy ở lớp/cặp lỗi mà teacher
   tốt hơn private; tắt KD cho timeDelay khi proxy đang kém. Không thêm byte. Falsifier: private class13/client mean không tăng ≥ 0,005.
2. **Tách head local và head trao đổi.** Head local chỉ CE trên dòng thật và dùng cho own-client; proxy/global head nhận KD và Gaussian.
   Backbone có thể chung, head local không rời client nên vẫn hỗ trợ kiến trúc dị thể. Falsifier: private không vượt proxy client mean
   ít nhất 0,005 hoặc p10 giảm.
3. **Control proxy CE-only q3.** Bỏ `KL(private→proxy)` nhưng giữ Q3/error feedback để tách tác động KD khỏi nén.
   Chạy thêm dense/q3 matched control chỉ khi CE-only q3 vẫn hụt. Residual Q3 hiện nhỏ nên chưa ưu tiên tăng bit.
4. **Specialist benign–timeDelay** từ P2, loss pairwise/focal chỉ trên cặp 0/13. Nhận khi đạt cổng P2; không dùng class weight
   toàn cục vì lớp 13 không hiếm và có thể chỉ đổi recall lấy false positive benign.

### Arm P3.1 đã build — `mobihet_rel`

Chỉ teacher→private KD đổi: với từng client/batch/class có mặt, tính mean true-label NLL của proxy snapshot và private;
KD real và Gaussian của lớp đó chỉ bật khi teacher NLL thấp hơn private. Lớp vắng mặt tắt gate. Proxy CE/KD, Gaussian,
Q3/error feedback, aggregation và byte không đổi. Không thêm HP tune (`margin = 0`). Baseline module `mobihet` giữ gate tắt.

Notebook paired `minhiw/mobihet-v4-relkd-val-r10/2` COMPLETE. V1 chỉ lỗi checker metadata rank1 sau smoke r2;
v2 sửa assertion, không đổi source phương pháp/gate. Cả control và candidate fit fold train, chấm full fold val,
dùng horizon cosine 50 và dừng r10; official test không decode. Audit source/hash, 20 CM, 5.343 client,
4.676.107 dòng/view/round, metric/finite/skipped PASS. RelKD tăng F1 lớp13 0,01298→0,22294 (+0,20996) và
macro-F1 0,86133→0,87632 (+0,01499), nhưng client mean 0,77463→0,76132 (−0,01331), p10
0,55991→0,50511 (−0,05480) ⇒ gate **FAIL**. Không kéo dài/tune RelKD; không chạy official test/final.

### Arm P3.2 đã build — `mobihet_pair`

Giữ nguyên MobiHet baseline; chỉ cộng binary CE hệ số 1 trên hai raw private logits `[0, 13]` cho dòng thật có nhãn
benign/timeDelay. Không thêm head/model/byte, không đổi proxy/KD/Gaussian/Q3. Cổng train→val r10 khóa trước launch:
F1 lớp13 ≥0,35; benign giảm ≤0,005; macro-F1 +0,015; client mean +0,005; p10 giảm ≤0,005.
FAIL thì dừng nhánh pairwise; không tune hệ số trên val.

**Kết quả — FAIL.** Kernel `minhiw/mobihet-v4-pair-val-r10/1` COMPLETE; audit source/hash, 20 CM,
5.343 client, 4.676.107 dòng/view/round, metric/finite/skipped PASS. So control r10: class13
0,01299→0,22989 (+0,21690), macro +0,01320, client mean +0,00473, p10 +0,00009, benign +0,00303.
Không đạt ba ngưỡng class13/macro/mean; không nới gate, kéo dài hoặc tune hệ số.

### Arm P3.3 PASS — `mobihet_stats`

Bỏ trainable proxy hoàn toàn. Mỗi client chỉ giữ private model; một lần bootstrap gửi count/Σx/Σxxᵀ lên server và nhận
mean/covariance/prior đủ 16 lớp. Mỗi round private học CE local + CE trên Gaussian pooled; không teacher KD, Q3 hay payload
mỗi round. Đây là cost/utility arm, không gọi là FL lặp đầy đủ sau bootstrap. Gate train→val r10 khóa trước launch:
macro/client mean/p10 giảm không quá 0,005; class13 không giảm; train time r3–r10 giảm ≥30%; bytes mỗi round bằng 0.

**Kết quả — PASS.** Kernel `hoangdang1711/mobihet-v4-stats-val-r10/1` COMPLETE; audit source/hash, 20 CM,
5.343 client, 4.676.107 dòng/view/round, metric/finite/skipped PASS. So control r10: macro +0,01398,
client mean +0,00710, p10 +0,01133, class13 +0,27724 (0,01338→0,29061). Train r3–r10 giảm 73,02%
(1.004,35→271,00 s/round); payload mỗi round bằng 0. Theo gate khóa trước, dựng final train+val→official-test r50,
không đổi HP theo val.

Không ưu tiên: tăng Gaussian batch, copula/GMM khác, tăng nhiệt độ, cap KD đơn giản, tăng Q3→Q4. Các build MobiHet cũ đã cho
thấy những hướng này thường chỉ tăng tốc đầu hoặc cải thiện proxy mà không chuyển ổn định sang private; v4 chỉ mở lại khi control mới
chỉ ra đúng cơ chế.

### Final stats R50 và real-BN ablation (01/11)

Final `hoangdang1711/mobihet-v4-stats-final-r50/1` COMPLETE. Đã kéo 60 artifact không gồm .pt;
audit 50 round × 2 view: 5.343 ids, 9.347.679 dòng/view, metric khớp CM. Chưa audit weights/resume.
Private macro 0,887913; accuracy 0,903361; client mean 0,846076; p10 0,725657; class13 0,339987.
Macro thấp hơn MobiHet cũ R50 ~0,0074; không kết luận thắng toàn diện. Round payload 0, bootstrap vẫn có chi phí.

Giả thuyết mới: Gaussian pooled đang tham gia BN cùng local real, có thể lệch normalization của xe.
`mobihet_stats_realbn` chỉ đổi mask BN để lấy moment từ dòng thật; Gaussian vẫn có CE và gradient.
Khóa train→val paired r10 với control stats: macro +0,005; mean/p10 >=−0,005; class13 >=−0,01;
train overhead <=20%; round bytes 0. FAIL thì đóng nhánh, không tune hệ số để vượt gate.
Đây là exploratory development sau khi đã xem test; cần independent holdout xác nhận các phương pháp tiếp theo.
Kernel `hoangdang1711/mobihet-v4-stats-realbn-val-r10/1` đã push; quota trước push 20,35h.
SHA notebook `38a9e61df0253605dd40ad48d859d68b4b7f16de4c0ef232f86324b14de02afe`.

### Adaptive Gaussian CE — exploratory paired R10

Real-BN R10 COMPLETE, CM/metrics PASS, gate FAIL: macro −0,000062; mean −0,001725; p10 −0,001072;
class13 +0,003001. Đóng nhánh BN.
Stats vs original control val R10: chia theo quartile số dòng val (54/172/717,5), client-F1 delta trung bình
0,01303/0,00355/0,00719/0,00457. Đây là proxy kích thước val, không phải train-count; chưa chứng minh xe lớn bị hại.
Giả thuyết: Gaussian hỗ trợ lớp thiếu nhưng regularize quá mạnh lớp đủ dữ liệu. Thử weight theo client/class
`batch/(batch+n_kc)` với count lấy duy nhất train; lớp thiếu giữ 1. Không thay real CE, BN hoặc LR.
Kỳ vọng: macro +0,005; mean/p10 >=−0,005; class13 >=−0,01; train overhead <=20%; bytes/round 0.
FAIL đóng arm; không tune theo test. Đây là exploratory sau khi đã xem test, cần independent confirmation.
Builder `scripts/build_mobihet_stats_adaptive_val.py`; account hoangdang1711 quota 18,14h; budget dự kiến 2–3h.
Notebook SHA `d30be75971de3c0067407dab30084514f0c5c7cb9c1cbdbd52caa9b5f7fdf29e`.

Adaptive result 02/11 (`hoangdang1711`, version 1): source/20 CM audit PASS, weights chưa kiểm.
Gate FAIL: macro +0,001025 (yêu cầu +0,005), mean +0,003535, p10 +0,003464, class13 +0,037539;
train overhead không tăng, bytes/round 0. Cải thiện nhỏ, chưa đạt tiêu chí mở final R50.

### Mass-normalized adaptive weights — 02/11, exploratory

Adaptive val R10 class13 +0,03754 nhưng class11 −0,03301, class15 −0,01316. Công thức cũ đồng thời
đổi allocation theo lớp và giảm tổng trọng số synthetic. Chưa chứng minh yếu tố nào gây đánh đổi.
Arm mới giữ allocation rồi chia weight cho mean từng client: tổng weight 16, cùng tổng mass như uniform control.
Không tune hệ số. Kỳ vọng khôi phục lớp 11/15; dấu hiệu bác bỏ: các lớp này vẫn giảm hoặc primary gate FAIL.
Control stats uniform, paired train→val R10, horizon50 seed42. Gate giữ: macro +0,005, mean/p10 >=−0,005,
class13 >=−0,01, overhead <=20%, bytes/round 0. Theo dõi lớp11/15 như diagnostic, không dùng đổi gate.
Builder `scripts/build_mobihet_stats_massnorm_val.py`; dangthiet quota 27,73h trước push, dự kiến 2–3h.
Notebook SHA `c31e993edfb6ebb4e94d58dd0f13bbd0bda53f26d340e45a4bf374991b52e3c8`.

Massnorm result 02/11: `dangthiet/mobihet-v4-stats-massnorm-val-r10/1` COMPLETE. Source hash + 20 CM/metrics
audit PASS (5.343 ids, 4.676.107 rows/view/round); weights chưa audit. Gate FAIL: macro −0,005279,
mean −0,001818, p10 +0,002072, class13 +0,014520. Control→candidate class11 0,714526→0,667520,
class15 0,940677→0,928591: không hồi phục như giả thuyết. Train 272,48→272,55s, round bytes 0.
Đóng massnorm, không mở R50. Bằng chứng không ủng hộ việc chỉ khôi phục tổng synthetic weight để cứu lớp11/15.

### No-Gaussian diagnostic — 02/11

Sau real-BN/adaptive/massnorm không đạt gate, chưa có phép đo loại bỏ Gaussian toàn bộ. Arm
`mobihet_stats_nogauss` bỏ sinh mẫu/forward rows/CE Gaussian; giữ real logit-adjusted CE và bootstrap cũ
(vẫn tính/ghi chi phí moments/prior). BN chỉ thấy real; dropout RNG đổi theo shape, nên đo tác động của toàn pipeline
augmentation, không riêng synthetic CE. Không gọi là local-only hoàn toàn vì prior toàn cục vẫn dùng.
Paired stats uniform vs nogauss train→val R10 horizon50 seed42. Dấu hiệu: giảm macro/class13 => Gaussian có lợi
cho các chỉ số đó; tăng thì cần kiểm chi tiết class/client trước kết luận. Báo mọi lớp, đặc biệt 11/13/15.
Gate cải thiện giữ nguyên +0,005 macro, mean/p10 >=−0,005, class13 >=−0,01, overhead <=20%, round bytes0;
đây là diagnostic, không tự động mở R50. Candidate không đọc official test.
Builder `scripts/build_mobihet_stats_nogauss_val.py`, dangthiet quota25,64h, dự kiến2–3h.
SHA `2850d811b1d9292ef22f6c8faa3325849ba12aaa678958428ec0ec63b2bad8b4`.

Kết quả02/11: COMPLETE, pull32 file non-.pt exit0; source/20 CM ×2 views/metrics audit PASS,
5.343 ids và4.676.107 rows/view/round. Weights chưa kiểm. Control→nogauss R10:
macro0,875855→0,866892; client mean0,781763→0,748357; p10 0,569234→0,494186;
class11 0,708145→0,665608; class13 0,298322→0,322922; class15 0,941006→0,915944.
Train r3–10:277,29→204,22s (−26,35%); round bytes0. Gate FAIL; không mở R50.
Gaussian pipeline hỗ trợ macro và phân bố client trong phép đo này, nhưng không giúp lớp13 so với bỏ Gaussian.
Không suy ra tác động riêng CE hoặc thiếu train-data vì BN/dropout cũng đổi và chưa phân tầng train-count.
Artifacts: `runs/pulls/mobihet-v4-stats-nogauss-dangthiet-v1`, audit `CM_AUDIT.json`.

## Trình tự thí nghiệm

### Prior-consistent Gaussian CE — 02/11, exploratory

No-Gaussian R10 làm client F1 giảm ở65,86% client. Chia quartile số dòng val54/172/717,5,
delta mean lần lượt −0,01465/−0,05831/−0,04605/−0,01420. Đây không phải phân tầng train-count.
Giữ Gaussian; kiểm một cơ chế khác với thay trọng số: real CE(z+log_pk−log_pg) nhắm raw logits
theo global prior, trong khi Gaussian lấy đều16 lớp dùng CE(z) nhắm uniform prior.
Candidate dùng synthetic CE(z−log_pg): log uniform là hằng số nên bỏ được. Nếu class-conditionals
khớp, hai objective cùng nhắm global prior. Gaussian xấp xỉ có thể không khớp real; đây là giả thuyết,
không phải lỗi đã chứng minh hoặc bảo đảm tăng điểm. Không đổi BN, số mẫu/weight, LR, dataset, inference.
log_pg lấy từ bootstrap train hiện có, không thêm collective/payload; tái dựng khi resume.
Kỳ vọng macro +0,005 mà không giảm mean/p10 quá0,005 hoặc class13 quá0,01;
train overhead<=20%, round bytes0. FAIL đóng arm, không tune prior strength hay tự mở R50.
Paired stats control vs `mobihet_stats_prior`, seed42, full5343 client, train→val R10/horizon50.
Builder `scripts/build_mobihet_stats_prior_val.py`; dangthiet quota23,54h; dự kiến2–3h.
Tests: posterior algebra/sign, no-proxy/round-bytes và exact next-round resume; remote smoke2 GPU trước production.
Push `dangthiet/mobihet-v4-stats-prior-val-r10/1` thành công; worker2 T4 RUNNING.
Notebook SHA `fc8805c21c47187b04073233e313a204fb970c173a98f6259a18040bb05fdf5d`.

Kết quả02/11: COMPLETE, pull32 non-.pt artifacts exit0, source/config/20 CM ×2 views/metrics audit PASS;
weights chưa audit. Control→prior R10: macro0,875491→0,879441; accuracy0,898386→0,905399;
client mean0,781403→0,768852; p10 0,564370→0,519661; class13 0,292364→0,241206.
Class11 0,710595→0,755062; class15 0,940354→0,948023. Train r3–10 262,29→262,38s; bytes/round0.
Gate FAIL: macro dưới+0,005, mean/p10/class13 đều vi phạm giới hạn. Đóng prior, không kéo R50/tune strength.
Sự nhất quán target prior về công thức không đủ bảo đảm personalized utility; chưa chứng minh nguyên nhân
là Gaussian approximation hay prior toàn cục lệch local. Artifacts `runs/pulls/mobihet-v4-stats-prior-dangthiet-v1`.

### Histogram seed43 replication — 03/11

04/11 kết quả R10 Stats→histogram: macro 0,877049→0,881545, mean 0,783131→0,785793, p10 0,570782→0,564476, lớp13 0,287839→0,293837. Train 266,35→266,61s/round. Macro gain hai lần seed42 +0,004766/+0,005007, seed43 +0,004496. Seed43 không đạt macro và p10. Đóng nhánh.

Owner đồng ý kiểm độ ổn định: paired Stats/histogram seed43, R10/horizon50 trên cùng train/val.
Source snapshot giữ nguyên byte-text từ audited `dangthiet/mobihet-v4-stats-hist-val-r10/1`
(46files, gồm tests), chỉ thêm CLI `--set seed=43` cho smoke và2 production arms; assertions config seed43.
Không histmask, không đổi bins/HP/gate. Primary macro≥+0,005, mean/p10≥−0,005, class13≥−0,01,
train overhead≤20%, round bytes0. Báo cùng cả2 lần seed42, không chọn lần tốt. Seed mới không tạo holdout mới.
Gate PASS chỉ mở cân nhắc R50, không auto launch; FAIL thì dừng nhánh, không tune theo validation.
Builder `scripts/build_mobihet_stats_hist_seed43.py`; dangthiet quota15,98h, dự kiến2–3h.
Notebook SHA `526d7e298044c77d3974bce7be7f81a7af1b24518ecbfc9af8c007b077722eb8`.

### Histogram missingness consistency — 03/11, pre-run plan

Probe lưu `runs/pulls/mobihet-v4-gaussian-fidelity-v1/HIST_FLAG_PROBE.json`: histogram từ20k train samples/lớp,
full-train L đã lưu, numpy reproduction. Trên0/11/13/15, real flag=1 luôn value=0; generated flag=1 vi phạm
92–100% tùy feature/lớp. Đây là sampled histogram, chưa đo full production histogram.
Candidate `mobihet_stats_histmask` áp quy tắc preprocessing6 cặp: flag==1 ⇒ value=0 sau histogram transport.
Giữ nguyên flag, các tọa độ khác, RNG, bins, loss/BN/model/optimizer. Không tuyên bố marginal giữ nguyên sau sửa;
đây là projection onto missingness constraint, có thể tăng khối xác suất0 và vẫn không sửa mọi joint relation.
Không đổi bootstrap/round bytes so histogram. Giả thuyết: bỏ mẫu inconsistent tăng classifier utility.
Three-arm matched R10 full train→val/horizon50 seed42: stats gốc, histogram, histogram sửa. Gate stats giữ nguyên
macro≥+0,005; mean/p10≥−0,005; class13≥−0,01; time overhead≤20%; round bytes0.
Gate bổ sung vs histogram: macro>0; mean/p10≥−0,005; class13≥−0,01; overhead≤20%. Cả2 gate cần PASS.
FAIL thì đóng arm, không tune projection theo val, không tự mởR50. Kiểm implication/idempotence/no mutation,
unchanged flags/payload/RNG và exact next-round resume. Builder `scripts/build_mobihet_stats_histmask_val.py`.
Dangthiet quota18,93h trước push, dự kiến3–4h gồm smoke/cache/3arms; không W&B.

Kết quả03/11 COMPLETE, pull45 non-.pt files exit0; source/config/30 CM×2views/metrics/gates audit PASS,
weights chưa kiểm. Stats/hist/histmask macro0,875623/0,880629/0,880379;
mean0,782102/0,787265/0,786506; p10 0,569974/0,580995/0,574657;
class13 0,293487/0,300418/0,299931. Histmask train265,05s vs Stats264,69s (+0,137%),
vs hist +0,0244%. Round bytes0. Original gate FAIL macro+0,004756; repair gate FAIL macro−0,000251
và p10−0,006338. Đóng repair, không kéoR50/tune projection.
Hist control lần này macro+0,005007, lần trước+0,004766; same-seed rerun không là independent replication.
Sát ngưỡng và backend dao động: báo cả2 lần, không chọn lần PASS bỏ lần FAIL.
Artifacts `runs/pulls/mobihet-v4-stats-histmask-dangthiet-v1/CM_AUDIT.json`.

### Histogram marginal transport — 02/11, pre-run plan

Evidence: real/Gaussian domain AUC1,0 kể cả bỏ NaN flags; binary transfer gap0,0922/0,0643.
Sample80k train rows4 lớp cho thấy ses_first chỉ2 giá trị, trf_snd_1s3, nbr_n_close7,
ses_dt ~94,45% cùng giá trị. Đây là scan mẫu, chưa chứng minh support toàn dataset.
Giả thuyết: Gaussian smooth phá các khối xác suất và miền marginal của feature rời rạc/skewed.
Candidate `mobihet_stats_hist`: bootstrap thêm histogram256 bins cố định[-4,4], counts/sums theo lớp/feature
từ toàn train; cộng giữa rank; inverse CDF trả occupied-bin mean. Latent z vẫn qua L cũ, chuẩn hóa mỗi tọa độ
theo marginal sigma rồi Φ trước inverse CDF. Giữ latent Gaussian correlation, không phải Gaussian copula fit
normal-scores; chỉ xấp xỉ marginal, chưa phục hồi joint constraints/multimodality. Không interpolation qua atoms.
Cờ0/1 giữ binary và xác suất theo histogram; có thể gộp các giá trị liên tục/discrete gần nhau trong bin rộng0,03125.
Không thay loss/BN/số synthetic/LR/seed/width/data/inference. Giữ original stats control, không dùng prior arm thất bại.
Bootstrap/client tăng5.046.272 bytes up +2.523.136 down; tổng5.243.456/2.720.320 bytes.
Đây là cost tăng thật, không gọi miễn phí; round payload0. Không thêm raw-row sharing hoặc proxy.
Gate khóa paired train→val full5343 R10 horizon50: macro≥+0,005; mean/p10≥−0,005; class13≥−0,01;
train overhead r3–10≤20%; round bytes0. Không tune bin count theo val; FAIL đóng arm. Không auto R50.
Đây là kiểm giả thuyết mới sampler, không đảm bảo khôi phục fidelity hoặc tăng FL score.
Tests: binary mass/constant support, histogram additivity, byte accounting, unchanged control sampler,
no-proxy + exact resume. Derived hist/CDF tái dựng từ cùng train khi resume; RNG Gaussian vẫn lưu.
Builder `scripts/build_mobihet_stats_hist_val.py`; dangthiet quota21,24h trước push, dự kiến2–3h cộng bootstrap.
Đã push `dangthiet/mobihet-v4-stats-hist-val-r10/1`; SHA notebook
`23209043c147c803768f3ab0fba02acbc39a5e3118c7c97c6517bd01e17d08cd`.

Kết quả03/11: COMPLETE,32 non-.pt files pulled exit0; source/config/20 CM×2 views/metrics audit PASS,
weights chưa audit. Control→hist macro0,875697→0,880463(+0,004766); mean0,781884→0,785900;
p10 0,568208→0,577876; class13 0,290622→0,300157. Class11 0,711113→0,718423,
class15 0,940446→0,939603. Train r3–10 268,30→268,98s(+0,255%); round bytes0.
Gate FAIL: primary macro cần+0,005, thiếu0,000234; mọi gate phụ đạt. Không làm tròn thành PASS,
không nới gate/tune bins hoặc kéo R50. Dấu hiệu tích cực cho marginal transport, chưa đạt tiêu chí nâng cấp.
Artifacts `runs/pulls/mobihet-v4-stats-hist-dangthiet-v1/CM_AUDIT.json`.

### Gaussian fidelity diagnostic — 02/11, pre-run plan

Prior arm FAIL; Gaussian hỗ trợ lớp thiếu nhưng hại lớp13 khi có train. Cần kiểm chất lượng phân phối synthetic,
không tiếp tục tune trọng số. CPU `mobihet-v4-gaussian-fidelity`: đọc train/ 410 parquet, moments full fold train
cho0/11/13/15, quantize input fp16 như loader, covariance jitter1e-4, Cholesky, clamp±4.
Numpy RNG là distributional reproduction, không bit-exact torch. Reservoir uniform theo row:20k train/10k val mỗi lớp,
seed42; lưu row offsets theo manifest file cùng samples/moments để audit, không đọc official test.
Ba phép đo khóa trước: flags NaN binary violations; HGB real-vs-Gaussian domain AUC (all77/loại6 flags);
HGB train real vs Gaussian cùng40k mẫu cân bằng/cặp, đánh giá cùng20k val trên0/13 và11/15.
HGB100 iterations/15 leaves/L2=1, không early stop/tune. Domain fit20k/eval20k rows tách nhau từ train sample;
moments đã fit toàn train, nên không gọi đây là generative held-out test hoặc client-held-out generalization.
Ghi predictions/CM/10 metrics/AUC; binary balanced sample không phải điểm full16-class FL.
Dấu hiệu hỗ trợ hypothesis: domain AUC≥0,8 cùng synthetic→real macro-F1 kém real→real≥0,05.
Nếu domain chỉ lệch ở flags: khảo sát sampler cho biến rời rạc; nếu vẫn lệch khi bỏ flags: khảo sát multimodality.
Nếu transfer gần real: chưa đủ bằng chứng sửa sampler, quay lại cơ chế local training. Không tự chọn class-specific
cutoff hoặc auto mở R50. Kết quả chỉ định hướng, cải tiến tiếp vẫn phải paired FL gate đã khóa.
Push thành công `dangthiet/mobihet-v4-gaussian-fidelity/1`, CPU startup scan train PASS.
Notebook SHA `576c2f5aeef1cf1fe37ed2b0f71de0f4807c7da0a61725947a2aa0f010c60c4d`.

Kết quả02/11: COMPLETE, pull8 files32MiB exit0 tại `runs/pulls/mobihet-v4-gaussian-fidelity-v1`.
Local audit source hashes, manifest totals37.361.273,12 prediction→CM/10metrics/AUC, sample shapes/unique IDs,
train/val ID disjointness và flags PASS. Không độc lập recompute full moments/rescan raw dataset local.
HGB domain AUC=1,0 ở0/11/13/15 cho cả77 features và bỏ6 NaN flags. Gaussian flags không binary
99,955–100%;31,205–47,625% giá trị flag ngoài[0,1], real flags binary100% trong mẫu.
Transfer binary macro real→real vs Gaussian→real:0↔13 0,673399→0,581192(−0,092207);
11↔15 0,966949→0,902665(−0,064283). AUC tương ứng0,756674→0,618385 và0,995155→0,955380.
Đạt dấu hiệu đã khóa domain≥0,8 + gap≥0,05 cả2 cặp. Kết luận Gaussian lệch real và transfer kém trong
probe HGB cân bằng này; chưa chứng minh đây là nguyên nhân duy nhất của FL, cũng không chứng minh multimodality.
Bỏ6 flags khỏi discriminator vẫn AUC1,0 nên sửa flags đơn lẻ chưa đủ bằng chứng khôi phục fidelity;
feature còn lại có thể có miền rời rạc/quan hệ xác định. Tiếp audit constraints trước chọn sampler mới.

### Train-support diagnostic song song — 02/11

Audited nogauss/control R10: class11 recall0,75081→0,58198; 11→15 errors4713→6834.
Class13 recall0,22749→0,23793; benign→13 false positives67570→49616. Không suy ra Gaussian
gây hại mọi lớp hiếm: cần phân tầng train support. CPU kernel `dangthiet/mobihet-v4-stats-client-diagnostic/1`
scan train/val labels và kiểm val CM identity; không train hoặc xem official test.
Bins train rows [0,256,1024,4096,infinity), missing classes [0,1,5,9,17);
per-class support absent/1–63/≥64, chỉ chấm client có lớp đó ở val. Khóa trước khi scan.
Xuất đủ10 pooled metrics mỗi nhóm, client delta và nhầm11/13/15; mô tả liên hệ, không khẳng định nhân quả.
Notebook SHA `f7b09790ede6875b7e84d09e1691834d50378d28064128357241dea204bbb1ce`.

Log CPU diagnostic đã scan410 file, vượt assertion train/val counts và CM identity, đang finalize.
Delta client mean nogauss−stats theo train rows: <256 −0,01123 (979 xe),256–1023 −0,04858(1491),
1024–4095 −0,05167(1418),≥4096 −0,01498(1455). Nhóm thiếu≥9 lớp tăng mean0,00756(553 xe),
nhưng chưa đủ cơ sở chọn cutoff/gating từ val. Lớp11/13/15 vắng ở train nhưng có ở val:
control F1 lần lượt0,37066/0,06116/0,75882; nogauss đều0 (support268/258/196 dòng).
Gaussian hỗ trợ lớp chưa thấy ở local, không chỉ regularization; lợi ích vẫn phụ thuộc lớp và client.
Lớp13 đã có train lại cải thiện khi bỏ Gaussian, khác lớp11/15: không áp dụng một quy tắc giảm Gaussian theo count cho mọi lớp.
Diagnostic COMPLETE; pull9 artifacts exit0 tại `runs/pulls/mobihet-v4-stats-client-diagnostic-v1`.
Local source-hash/completion/count-total/CM-identity audit PASS. Prior arm vẫn đang chạy,
chưa mở thêm adaptation dựa cutoff post-hoc.

1. Hoàn tất và audit r50 hiện tại; không thay run đang chạy.
2. P0 hậu kỳ trên checkpoint r50.
3. P1 identifiability trên train/val.
4. Chọn đúng một nhánh: P2 nếu raw có tín hiệu; P3 nếu 77 feature đã đủ.
5. Smoke 2 round → arm 10 round seed42 với cổng khóa → arm 50 round; arm PASS mới lặp seed43.
