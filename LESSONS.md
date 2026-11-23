# LESSONS — bài học kỹ thuật (dữ liệu, model, train, đánh giá, code)

- 21/11 ablation L4 ([thiết kế](docs/research/mobihet-l4-ablation.md)). Q gộp từ trung bình lớp fp32 có tổng hàng < 1. Lý do: xe có lớp c nhưng round đó không train dòng nào của c (ngân sách tầng) nên đóng góp 0. Decoder 8-bit và fp16 của L4 đều chuẩn hóa lại hàng, nên bản raw fp32 phải giữ bước này. Bỏ nén không được bỏ phần decode. Test bắt lỗi bằng tổng hàng Q. L4 full-data 2×T4 không bit-exact giữa hai phiên cùng notebook (case2 r2 macro .676158 vs .676146, r1 lệch 668 dòng): atomics CUDA. Đối chứng Kaggle so theo dải nhiễu; bit-exact chỉ khẳng định được ở test local. `build_l4_fl.py` được `run_mixed_l4.py`/`audit_l4.py` import như module, nên import phụ thuộc thư mục `scripts/` phải đặt trong `main()`.

- 21/11 communication budget ([audit](artifacts/communication-budget/audit_report.md)). `payload_bytes()` là byte/client/round theo quy ước dense, không phải tổng của run. Uplink prototype chỉ gồm lớp có dữ liệu: N_p = 69.993 cặp, bằng 81,9 % dense. PerFed-SKD chỉ tính xe được chọn (`selected_frac`). Hàng MobiHet cũ (Q3 proxy-delta) trong `cost.md` đã thay bằng L4. Mask TinyProto sinh từ seed nên tính 0 B. Tổng L4 = K·Σ`bytes_*` mỗi round, cộng bootstrap 340,7 KB/xe một lần (94–96 % tổng). Run canonical thiếu config thì không lấy K hoặc payload từ panel compute-trace cùng tên (K = 200).

- 20/11 profiling CUDA. Luồng autograd native nhận `TorchDispatchMode` nhưng không nhận nhãn pha Python `ContextVar`. Bridge nhãn qua lệnh backward đã sửa thiếu 111,78 GFLOPs backward của MLP FedAvg r2. Lockstep sao chép context riêng khi sinh luồng. Regression GEMM CPU/CUDA kiểm cả dòng padding và phép cộng trace chi tiết.
- 20/11 mixed L4 regression. `row_score[-1]` là ô sink khi ghi nhiều padding index trùng. Bỏ đúng ô sink khi so trạng thái, kiểm tất cả điểm dòng thật, tensor còn lại và CM.
- 20/11 centralized profiling. `inference_mode` có thể đưa linear, Conv1d và GRU lên dispatcher ở dạng op tổng hợp. Dùng `no_grad` và SDPA math trong probe FLOP để quan sát phép nhân ma trận ở eval.

- 20/11 mixed L4 (Lockstep thread mỗi nhóm backbone): torch 2.13 giữ **config dynamo/inductor theo thread** — main đặt
  `recompile_limit=64`, thread mới đọc 8 ⇒ case3 (5 backbone) chạm giới hạn và dynamo **lặng lẽ chạy eager** frame đó.
  `TorchDispatchMode` (op counter compute) cũng thread-local ⇒ op trong thread **đếm 0**. Sửa: chép config khác mặc định +
  enter dispatch mode của thread gọi vào mỗi thread nhóm (`methods/mobihet_mixed_l4.py`, test toy kiểm cả hai).
  Bootstrap gộp qua nhóm: `cop_*` exact, KM bằng nhau **sau hoán vị cluster** — tổng float64 khác thứ tự lật dấu vector
  riêng ⇒ tâm `mu ∓ d`; so KM phải theo hoán vị, không theo chỉ số.

- 20/11 compute L4 (panel 200, FLOP): chi phí cố định không phụ thuộc model (lấy mẫu copula/KM + KL synthetic ≈ 0,16–0,19
  Tops/round op không-matmul, bootstrap 0,064 Tops) ⇒ DAGSNet 0,920× / BERT 0,917× FedAvg nhưng TabM 1,310×, MLP 1,952×,
  SAE 2,445×. Mixed panel 200: compact lấy độ rộng **max trong nhóm grouped**; mỗi backbone chỉ 53–86 xe nên nhóm xe lớn
  lẫn xe tầng 0 ⇒ 64 thật + 64 synthetic mọi bước ⇒ executed 2× FedProto (case2 212 vs 96 Tops). Không suy ra full-scale.

- 18/11 BERT/V2X regression: loss thật, batch64, scale4096, seed41/42/43 trên CPU fp16 tái hiện gradient nonfinite ở attention in_proj_bias/out_proj.bias block0. FP32 cùng fixture/loss hữu hạn cả3seed. Sửa precision policy cho BERT+SAE riêng mixed V2X; không đổi kiến trúc/loss/LR/gate1%. tests/test_mixed_bert_precision.py; GPU T4 regression cũng PASS cả3seed (runs/mixed128-bert-precision-gpu.json); còn cần full-data gate, fixture không chứng minh đủ50round.

- 18/11 mixed128: cả 6 V2X case1/3 seed41/42/43 fail full round đầu vì BERT AMP skip 2,62998–3,68301%, vượt gate1%. SAE-only FP32 fix không bao phủ BERT. Không nối/chạy lại nguyên mã lỗi; cần replay/precision proof riêng trước khi sửa và chạy lại.

- 17/11 mixed128 V2X case2 seed42: pilot2round PASS nhưng full-data tới r18
  SAE skip1,15867% vượt gate1%; seed41/43 complete50. Pilot không chứng minh
  health dài hạn. Gate riêng backbone phát hiện lỗi dù tỷ lệ skip gộp r17 chỉ0,31531%.
  Replay checkpoint15 trên T4 bắt batch thật r16 xe1976/1961: AMP scale4096 làm
  encoder SAE gradient nonfinite; projection-only/loss-only FP32 vẫn lỗi. Scale1
  hoặc toàn bộ phép tính FP32 đều hữu hạn. a2 cho SAE local training của mixed V2X
  dùngFP32, giữ AMP của backbone/phương pháp khác và ngưỡng1%. Chạy lại cả3seed
  cùngpolicy, không ghép a1/a2. Evidence `runs/mixed128-precision-evidence.json`.

- 17/11 mixed-model: kiểm tổng số dòng bằng integer/float64;37.361.273 vượt miền
  integer chính xác của float32. Dùng env.n.double().sum() cho identity gate, không
  thay quy tắc trọng số loss của baseline. Health mixed phải kiểm riêng backbone;
  tỷ lệ skip gộp có thể che nhóm nhỏ bị lỗi. Bảng gán phải tra globalclient_id,
  không tra vị trí local sau chiaGPU/sortsize; hash cố định qua method/seed/resume.

- 16/11 MLP V2X: notebook a2 còn nhúng projection cũ dù workspace đã sửa FP32 cho TabM.
  Test gọi loss thật của Method.round, batch64/scale4096, tái hiện seed42 gradient proj_b nonfinite
  với projection lấy từ notebook a2; projection hiện tại PASS cả41/42/43. Không đổi loss/HP/epsilon.
  Regression được nhúng trong a4 trước cache/smoke GPU. CPU fixture chưa chứng minh full-data GPU;
  phải build/push snapshot mới, sửa workspace không sửa notebook đã gửi.

- 16/11 V2X precision follow-up: CPU autocast-fp16 synthetic fixture tái hiện gradient `proj_b`
  nonfinite ở seed41/42 khi scale4096; FP32 hoặc scale1 hữu hạn. `project()` dùng baddbmm trong
  autocast ép bias về half; gradient cộng qua batch overflow. Tắt autocast riêng projection,
  giữ FP32 và khôi phục eps1e-12: fixture cả3seed PASS scale4096; reference TabM PASS.
  Chưa suy fixture thành full-data proof. a4 có GPU regression + smoke2round trước production.

- 16/11 TabM V2X-HtFL seed42 không suy giảm muộn: F1 đã 0,01888 ở r1 và train loss gần0.
  Adapter đã bỏ `applied/skipped` của `local_epoch`, nên bước AMP non-finite bị skip có thể hiện thành loss0
  nhưng marker vẫn COMPLETE. Telemetry mới chứng minh rerun: seed42 skip100% ở r1/r2; seed43 skip100%/98,3%.
  Normalize fp32 `eps=1e-2` **không sửa được** và giả thuyết gradient gần vector0 bị bác bỏ như nguyên nhân đủ.
  Không push production tiếp; cần diagnostic eager-FP32/eager-AMP/compiled-AMP trên cùng batch, đồng thời audit
  DAGSNet V2X với telemetry mới trước khi gọi hai backbone đồng nhất về health.

- 15/11 owner đổi SAE-FL sang CE-only, prototype pre-classifier93; đây là biến thể mới,
  không còn bắt buộc reconstruction/KL như quyết định14/11 bên dưới. Centralized SAE cũ
  vẫn giữ objective gốc. Decoder trong layout CE-v2 có gradient0; phân biệt tổngparams
  vàparams ảnh hưởng prediction. Không resume checkpoint khác objective/representation.

- 15/11: metric history thiếu skipped_frac phải là unknown, không phải zero. V2X-HtFL bỏ
  st.skipped/applied khi trả round info; CM/history đủround không chứng minh mọi bước optimizer
  được áp dụng. SAE classifier dùng skip77+code16 nhưng prototype chỉcode16: phải kiểm
  representation trước quy điểm prototype thấp cho toàn bộ classifier.

- 15/11: khi so backbone giữa các nhánh repo, kiểm cả HP override của run gốc. TinyProto DAGSNet r4
  dùng μ=0.00015 nhưng TabM/SAE dùng heuristic mặc định; chênh lệch không cô lập tác động backbone.
  Audit parameter phải tính sau pruning (LwFedNIDS) hoặc thêm projection (V2X), không ép mọi method
  bằng sốparams backbone gốc. [Bằng chứng](docs/research/backbones/tabm-sae-dagsnet-comparison-20261115.md).

Đọc sau [CONTEXT](CONTEXT.md), cùng [bài học vận hành](docs/operations/lessons.md). Bài học kế thừa:
`../mobihet-features/LESSONS.md` (dataset v4), `../mobihet/LESSONS.md` (grep theo chủ đề; FL/KD/đánh giá trên DAGSNet).

- 14/11 SAE trong FL không được thay objective bằng CE thuần: decoder chỉ nối với reconstruction nên sẽ không nhận
  gradient. Mọi method có loss riêng phải giữ `reconstruction + KL sparsity + classification CE`, rồi cộng KD/prototype/
  proximal term của method. Khóa bằng grouped-reference regularizer test và one-round smoke mọi method.

## Kế thừa — áp dụng thẳng cho repo này

- 14/11: V2X-HtFL server prototype training có generator shuffle riêng gen_s; checkpoint thiếu RNG này
  làm resume round2 khác confusion matrix uninterrupted. Đã bổ sung state/load và CPU26 adapter resume CM test PASS.
  Không suy CM equality fixture thành bit-exact weights hoặc CUDA equivalence.

- 14/11 TabM trong FL: member axis không phải client/model federation axis. CE phải trung bình trên 4 member;
  prediction/KD phải trung bình **xác suất**; prototype dùng trung bình 4 hidden vector. Nếu đưa tensor `(G,B,K,C)`
  thẳng vào code DAGSNet `(G,B,C)`, chương trình có thể vẫn chạy nhưng objective/eval sai nghĩa. Khóa bằng grouped-reference
  test và smoke đủ từng nhóm phương pháp trước launch. [Thiết kế](docs/research/backbones/tabm-single-model-baselines.md).
- 14/11 backbone không có BatchNorm cần đường compile/eval trả buffer phẳng `(G,0)` và diff bằng 0; không được
  `torch.cat([])`. Method smoke eager PASS không cover compile gate. Probe GPU phải chạy gate trước production.

- Model client chỉ học nhãn của chính nó đạt đỉnh trên test có prior toàn cục sau 1–2 epoch rồi giảm (mobihet 13/09). Ở đây model client
  chấm trên test của chính xe ⇒ prior khớp hơn, nhưng Sybil test 17,7 % vs train 8,0 % vẫn lệch theo thời gian.
- Luôn báo F1 train cạnh F1 test; sụp đột ngột khi đổi cấu hình là dấu hiệu tối ưu hỏng (HGB L2 = 0, mobihet-features 28/10).
- Val "xe chưa thấy" ≈ val cùng xe khi train gộp nhiều xe; không lộ được học thuộc theo xe (mobihet-features 29/10).

## Dữ liệu

- GRANDE v4 (10/11), với đầu vào chặn ở ±3,72, đếm ngưỡng của **cột được chọn** ngoài miền để phát hiện nút luôn rẽ một phía. Epoch 50 có 153/480 nút vượt ±3,73. Đếm lá khả thi phải xét cả điều kiện từ nút cha. [Checkpoint và kết quả từng lớp](docs/research/backbones/central-backbones.md#grande-trên-veremi-v4-10112026).

- 03/11 histmask: sửa đúng implication flag=1⇒value=0 chưa tăng utility; vs hist macro−0,000251,
  p10−0,006338, cả2 gates FAIL. Không suy chất lượng miền mẫu cải thiện ⇒ FL metric chắc chắn tăng.
  Histogram same-seed hai lần macro gain0,004766/0,005007 nằm hai phía gate0,005; cần báo dao động,
  không chọn lần vượt ngưỡng làm xác nhận độc lập.

- 03/11 histogram marginal transport R10 tăng macro0,004766, mean0,004016, p10 0,009668,
  class13 0,009535 với train overhead0,255%; nhưng primary gate+0,005 vẫn FAIL.
  Không làm tròn/nới gate cho kết quả sát ngưỡng; tính cả chi phí bootstrap tăng khi đánh giá utility/cost.

- 02/11 Gaussian fidelity: full-train moment Gaussian trên0/11/13/15 dễ tách khỏi real (HGB domain AUC1,0,
  kể cả bỏ6 NaN flags). Flags synthetic gần100% không binary,31–48% ngoài[0,1]. Binary transfer macro
  kém real-trained0,0922 và0,0643 ở0↔13/11↔15. Mean/cov đúng không bảo đảm mẫu có miền/quan hệ feature hợp lệ.
  Đây là sampled HGB diagnostic, không phải FL score hoặc bằng chứng riêng cho multimodality.

- 02/11 Gaussian prior-consistency R10: synthetic CE(z−log_pg) tăng pooled macro0,00395
  và lớp11/15 nhưng giảm client mean0,01255, p10 0,04471, lớp13 0,05116. Công thức target-prior
  nhất quán không bảo đảm utility khi Gaussian chỉ xấp xỉ và các client khác nhau. Gate FAIL, không tune strength.

- 02/11 train-support diagnostic trên nogauss paired R10: Gaussian cứu các lớp11/13/15 vắng ở train
  nhưng xuất hiện ở val (bỏ Gaussian thì F1 đều0; support nhỏ268/258/196). Đồng thời lớp13 có train
  lại tốt hơn khi bỏ Gaussian, trong khi lớp11/15 kém đi. Không suy từ count thấp ra cùng nhu cầu
  synthetic cho mọi lớp; không chọn cutoff/gating từ kết quả val rồi coi là xác nhận độc lập.

- 02/11 no-Gaussian paired val R10: bỏ toàn Gaussian nhanh26,35% nhưng macro −0,008963,
  client mean −0,033406 và p10 −0,075048; class13 lại +0,024601. Không loại bỏ augmentation
  chỉ dựa trên compute hoặc một lớp. Đây là tác động cả pipeline (BN/dropout đổi), chưa cô lập synthetic CE.

- 01/11 pairwise 0↔13 train→val r10: F1 lớp13 0,0130 → 0,2299, macro +0,0132, client mean +0,00473,
  p10 +0,00009, nhưng không đạt gate khóa (class13 ≥0,35; macro +0,015; mean +0,005). Objective đúng hướng nhưng
  chưa đủ mạnh; không nới gate hoặc tune hệ số sau khi xem val. Chuyển câu hỏi cost: bỏ proxy, giữ one-time global statistics.
- 31/10 RelKD train→val r10: reliability-gated teacher→private KD tăng F1 lớp13 0,0130 → 0,2229 và macro-F1
  0,8613 → 0,8763, nhưng client mean 0,7746 → 0,7613 và p10 0,5599 → 0,5051. Cải thiện pooled không đủ cho
  personalized FL: phải khóa cổng cả client mean/p10. Không kéo dài/tune arm; chuyển objective chỉ tác động cặp 0↔13.
- 31/10 MobiHet P0 r25: thêm `log π_k − log π_g` lúc own-client inference tăng F1 timeDelay 0,0559 → 0,0731 nhưng
  giảm pooled macro-F1 0,8832 → 0,8812, client mean 0,8249 → 0,8117 và p10 0,6772 → 0,6453. Giữ logits raw;
  prior correction lúc train không suy ra rằng phải cộng lại lúc inference.
- 31/10 P1a train→val: trên đúng 77 feature, binary HGB/MLP tách benign–timeDelay đạt F1 lớp13 0,541/0,542;
  exact collision mẫu 200k/lớp bằng0. Representation có tín hiệu, nên không đổ lỗi chính cho dataset hoặc mở v5 trước;
  lỗi còn lại là chuyển tín hiệu binary vào multiclass/private qua objective/KD. Đây là oracle binary, không phải điểm MobiHet dự kiến.

- 30/10 MobiHet v4: resume cần RNG riêng của Gaussian (`gen_g`), stochastic quantization (`gen_q`),
  lịch mẫu (`env.gen`) và torch CPU/CUDA theo rank. Chỉ lưu weights/version không tái hiện round kế tiếp.
  Test CPU train r1 → serialize → r2 so với restore → r2 đã PASS bit-exact trên Kaggle;
  chưa chứng minh bit-exact CUDA/compile. Smoke hai GPU kiểm riêng đường restart.
- 30/10: tính loss bằng `loss * ok` vẫn cho NaN khi loss NaN và ok=0; dùng `where`, chia số bước thực sự áp dụng.
  Thay đổi thống kê loss, không thay gradient/update. Theo dõi applied/skipped để tránh model đứng yên nhưng loss giả thấp.

- `pyarrow.dataset(<v4>/train)` lỗi "Unable to merge: Field scenario has incompatible types: large_string vs string" (cột trong file là
  `large_string`, partition hive suy `string`) ⇒ đọc từng file bằng `pq.read_table` (`core/data.py`).
- 77 đầu vào = 11 × 7 ⇒ DAGSNet giữ lưới 11 vị trí (`view(B, 11, 7).transpose(1, 2)`).

## Compile

- 13/11 MambaCNN: tổng tham số trong paper không xác định duy nhất kiến trúc student. Cần phân biệt số báo cáo128,11K sau feature selection/KD với bản77feature supervised đã build129.328. Các dimension thiếu phải ghi là lựa chọn chuyển thể; không gọi bản không teacher là KD. Prefix scan kiểm cả output lẫn gradient với recurrence độc lập, kể cả L=1 (transition không ảnh hưởng khi trạng thái đầu0). Đối chiếu (`papers/feng-2026-mambacnn/rebuild.md`, gỡ 20/11, xem commit `5b9d0e3`).

- **`torch.compile` + AMP fp16 làm gradient NaN ở nhánh AlexNet grouped** (29/10): forward khớp eager (Δlogit 2e-3) nhưng
  backward của `amax` (maxpool 2, ceil) cho NaN ở mọi bước ⇒ optimizer bỏ mọi bước, model đứng ở init, FedAvg smoke acc 0,047 /
  loss "0,028" (chỉ là tổng các bước được áp = 0). fp32 compile và eager không bị. Sửa: `torch.maximum(x[..., 0::2], x[..., 1::2])`.
  Chặn tái diễn: `local.compile_gate` (một bước thật, so gradient compile vs eager, FAIL ⇒ eager) + cảnh báo `skipped_frac` mỗi round.
  Test tương đương fp64 eager không bắt được lỗi chỉ có ở đường compile + fp16.
- Shape tĩnh cho train (nhóm đệm tới đủ G client rỗng), shape động cho eval; mỗi shape mới = một lần compile ~2 phút.

## Tốc độ (đo 29/10, RTX 3050 4 GiB local)

- 14/11 MambaCNN129.328tham số: prefix scan PyTorch trên2T4/micro32 dự kiến21,79h/epoch (full VeReMi v4), không thể suy nhẹ từ số tham số. Scan materialize B×L×H×N và nhiều phép toán trong6block. Probe-budget gate đã chặn production; bản sửa CUDA phải qua kiểm output/gradient và timing thật trước train. Bằng chứng và gate (`papers/feng-2026-mambacnn/rebuild.md`, gỡ 20/11, xem commit `5b9d0e3`).

- DAGSNet đơn, AMP, batch 8.192: full 46k mẫu/s eager, 59k compile; W 0,375: 78k / 113k.
- Grouped (G model/bước, bmm channels-last) compile: ~45k slot/s full width ≈ 77 % model đơn; eager chậm hơn ~2,5×.
  Lần compile đầu ~2 phút. Hiệu suất thực = slot thật / slot (padding): 0,85 ở batch 64 / 8.192 dòng mỗi bước grouped trên phân bố
  client v4 (nhóm theo kích thước); phần mất chủ yếu do client lớn (95k dòng = 1.487 bước tuần tự).
- **Stride của tensor đầu vào cũng là guard của `torch.compile` tĩnh** (29/10): `mask = valid[:, s*b:(s+1)*b]` có stride phụ thuộc số bước
  của nhóm ⇒ mỗi nhóm một lần recompile, quá 8 lần dynamo **im lặng chạy eager** (probe Kaggle FedAvg W 1: r1 1.243 s thay vì vài phút).
  Sửa: `.contiguous()` mỗi bước, `recompile_limit = 64`; kiểm bằng `TORCH_LOGS=recompiles` (còn 2 lần hợp lệ: dropout 0 của gate, no_grad eval).
- Nạp parquet trên Kaggle: mỗi rank đọc cả 410 file từ `/kaggle/input` mất 10–30 phút ⇒ một lần `data.build_npy` trong notebook rồi
  `V4_NPY` memmap theo rank.
- 30/10: luật suy luận prototype phải theo đúng bài từng phương pháp: FedProto/FedTGP dùng prototype toàn cục, **TinyProto Eq. 12 dùng
  prototype cục bộ dense c̄^L_ij**. Target thưa × μ của TinyProto chỉ là đích regularizer (0 ngoài mask) ⇒ dùng làm prototype suy luận
  cho F1 0,24 vs head 0,79. Audit: view prototype thấp xa view head là dấu hiệu sai luật suy luận.
- 30/10 (bổ sung, r3): gap prototype/head **không chỉ** do luật suy luận. TinyProto đã dùng dense local Eq.12 nhưng r2 F1 0,8165 → r50
  0,2904; norm target 35,5→2.943, loss r5 43,2. μ=1/mean(n_ij) là heuristic bản dựng; trên v4 làm gain benign 3,43× mỗi gộp nếu
  feature cùng hướng. Đã tái hiện khuếch đại qua `Method.round`; sửa μ=1,5e-4 theo cấu hình mẫu tác giả, chưa chứng minh hội tụ full-K/r50.
  Không quy kết phương pháp yếu trước khi audit scale + deviation. Chi tiết: [TinyProto](docs/methods/tinyproto.md).
- 30/10: audit CM/metric và tensor snapshot hữu hạn chỉ là **PASS_BOUNDED**. Harness lọc snapshot >10M phần tử nên bank private và
  prototype local của 2.672 xe có thể không còn trong snapshot từng round; thiếu prediction thì không replay CM từ suy luận được.
  Ghi giới hạn cạnh kết quả, không gọi đầy đủ "audit model PASS" chỉ từ các metric đúng.
- 30/10: `run_fl.py` cũ lưu `env.gen` (shuffle), nhưng dropout grouped dùng global Torch RNG; checkpoint không lưu CPU/CUDA RNG toàn cục.
  Nối NILM đã phục hồi bank/round/shuffle state, **không chứng nhận bit-exact** với run không gián đoạn. Giữ source chuỗi đang chạy;
  bổ sung contract/test global RNG cho harness tương lai, không gọi state round-trip/eval là test exact resume.
- 01/11: với MobiHet v4, trainable proxy không cần để cứu lớp hiếm. Bootstrap một lần bằng sufficient statistics theo lớp
  (count/Σx/Σxxᵀ), rồi private model học local real CE + pooled Gaussian CE, tăng đồng thời macro/client mean/p10/class13 trên val,
  giảm 73% train time và đưa payload mỗi round về 0. Kết luận chỉ áp dụng cho arm val seed42; cần final official-test r50.

## Compute (20/11/2026)

- Harness `scripts/compute/total_compute.py` móc `importlib.import_module("methods.<m>")`; runner import thẳng (`from methods.mixed
  import Method` trong `run_mixed_fl.py`) không qua hook ⇒ phải wrap trước khi chạy script, không thì không có pha bootstrap/round/eval.
- Bỏ time-mode thì count phải chạy đúng `group_rows` production (4096 đơn model, 8192 mixed): report chỉ quy đổi được count ở
  group nhỏ (BERT eager OOM) về dòng executed của engine khi có time run cùng cấu hình. OOM local ⇒ đếm trên Kaggle T4, không hạ group.
- CANPerFL = FedAvg r1–50 + 10 epoch fine-tune; constructor cần state FedAvg r50. FLOP/epoch không phụ thuộc trọng số ⇒ state FedAvg
  1 round cùng panel/cfg gắn nhãn r50 là đủ để đếm pha fine-tune; tổng = FedAvg 50r + bootstrap + 10 × (train + ngoài train).
