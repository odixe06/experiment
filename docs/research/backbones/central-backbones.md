# Centralized VeReMi v4 — năm backbone, 50 epoch

> 20/11/2026 (owner): code CNN–LSTM (cả `cnn-lstm-feature`), TCNN, GRANDE, GRU-Attention, MambaCNN và predecessor windows
> đã gỡ khỏi repo; nhật ký dưới đây là lịch sử. Khôi phục: `git show 5b9d0e3:<path>`. Centralized còn dùng: mlp, tabm, bert,
> sae / sae-ce, cnn-bigru-attention (`scripts/run_central_backbones.py`); DAGSNet dùng `scripts/run_central.py`.

## CNN–LSTM một bản tin — đã push14/11/2026

Owner đồng ý thống nhất thông tin đầu vào: một dòng77feature, không lịch sử10bản tin. Alias mới **`cnn-lstm-feature`**, [model](../../../core/central_backbones.py): input ngoài(B,77), nội bộ77vị trí×1giá trị → Conv1d1→256→128(kernel1,Tanh) → hai LSTM hidden96 → hidden vị trí cuối → Linear96→16. **196.240tham số train/inference**, gồm Conv768+32.896, LSTM86.784+74.496, head1.552. Đây là bản chuyển thể theo ngân sách~200k; quan hệ tuần tự là thứ tự feature, không phải thời gian. Conv kernel1 không trộn vị trí; LSTM đảm nhiệm phần đó.

Factory/runner/builder nhận alias mới, không nạp hoặc tạo predecessor windows. Alias `cnn-lstm` cũ giữ nguyên503.056tham số/10bản tin để đọc checkpoint lịch sử. Bản mới phải train từ đầu; strict-load weights cũ không tương thích, không dùng lại kết quả cũ làm baseline mới. Bốn cấu hình DAGSNet~310k/CNN-BiGRU~140k/BERT~80k/SAE~10k được thảo luận là đề xuất; lần sửa này chỉ đổi CNN–LSTM bằng alias mới.

Local Python3.13/PyTorch2.10 CPU: suite13tests,12PASS+1CUDA skip,10.744s. Model mới kiểm196.240parameters, output(B,16), independence theo dòng/reordering, reject temporal input, gradients, strict weights và fixture train/stop/resume2epoch bit-exact; prediction→CM→10metrics PASS. Chưa kiểm CUDA, tốc độ, bộ nhớ hoặc chất lượng trên full dataset.

Notebook draft local `notebooks/veremi-central-cnn-lstm-feature-e50/veremi-central-cnn-lstm-feature-e50.ipynb`, SHA256 `476095e547bc802b70c236a3e1f37b5b94bc80d9b99c6c73855d4dd9f945592f`; private2T4, account metadata qwerasdf3004 chỉ là draft, chưa kiểm quota/chưa push. Giữ protocol centralized50epoch/batch4096/seed42, micro128 tạm thời, probe real data trước production. Cần xác nhận resource budget trước launch. Checkpoint/source cũ không sửa.

Owner sau đó yêu cầu push chạy lại. Đã thêm test CUDA-fp16 forward/backward cho model mới và gate probe epoch×1,5 phải vừa session. Local14tests:12PASS+2CUDA skips,17.613s. Regenerate16source files/hash/AST/schema/private2T4/no-windows/no-resume PASS; SHA256 mới `74b7782d9d6e34559a5e45348d01d762a2ef0987b78d3d732245b1ed24fb7ab8`. Quota qwerasdf3004 trước push29,650h/reserved0; listing owner không có slug mới, input access đã được chứng minh qua các run cũ trên cùng account. **Đã push [version1](https://www.kaggle.com/code/qwerasdf3004/veremi-central-cnn-lstm-feature-e50/versions/1), statusRUNNING**. Cap12h, budget11,25h trừ setup/probe; train mới horizon50. Kiểm startup14/11 khoảng11:21VN: đúng 2 Tesla T4, 14 tests trong15,569s:13PASS+1Mamba-extension skip; CUDA-fp16 CNN–LSTM PASS. Status version1 RUNNING. Probe real-data PASS: train874,834s + eval96,066s ≈16,18phút/epoch; ước tính13,48472h/50epoch, peak8,427GiB, skipped0. Budget guard PASS; production bắt đầu với11,166h còn lại và đã tới epoch1 step500/9122 cả2rank, grad3,585 finite/skipped0. Dự kiến cần nối session để đủ50; kiểm14/11 11:38VN: live log COMMITTED epoch1 macro-F1=0,910649; chưa kéo/audit đủ10metric. Epoch2 tới3500/9122, grad0,3751 finite, AMP skipped2; statusRUNNING. Log tại `tmp/veremi-central-cnn-lstm-feature-e50-v1-live.json`; không dùng điểm bản10bản tin cho bản này. Không W&B/Git push.

**MambaCNN — đã bỏ theo owner14/11/2026.** Giữ các model cũ; không tiếp tục tối ưu/push/resume. Model (`core/mambacnn.py`, gỡ 20/11, xem commit `5b9d0e3`) chuyển thể supervised129.328tham số, không teacher/KD; đối chiếu paper (`papers/feng-2026-mambacnn/paper.md`, gỡ 20/11, xem commit `5b9d0e3`), cấu hình và chẩn đoán (`papers/feng-2026-mambacnn/rebuild.md`, gỡ 20/11, xem commit `5b9d0e3`). Cả3phiên kết thúc trước production. CUDA scan/micro128 nhanh hơn bản đầu khoảng17,21× nhưng vẫn ước tính63,29giờ/50epoch. Gate trọng số phát hiện lệch ở key bias attention; bản điều chỉnh kiểm tra mới chỉ ở local, chưa xác minh/push. Không có kết quả production hoặc checkpoint để resume.

**CNN-BiGRU-Attention, 13/11/2026.** Owner chốt dùng Table 2 và native PyTorch GRU. [Model](../../../core/cnn_bigru_attention.py) nhận 77×1, Conv64 k3 valid → BN/ReLU → Pool2 → Conv128 k3 valid/ReLU → BiGRU64 mỗi chiều → additive Attention128 → Dropout0,3 → head16. Đếm được 118.288 tham số trainable. [Nguồn](../../../papers/mnkash-2026-cnn-bigru/paper.md) và [chuyển thể, kiểm tra](../../../papers/mnkash-2026-cnn-bigru/rebuild.md). Kế thừa protocol centralized 50 epoch, microbatch 256/GPU. SR-LiCNN đã được thay bằng MambaCNN theo yêu cầu owner.

**GRU + Attention.** [Saran & Kesswani, JJCIT 2025](https://doi.org/10.5455/jjcit.71-1725609265), Figure 3 (ICU), Algorithm 1 và Eq. 5–7. Mỗi bản tin là một mẫu. PCA lấy 5 thành phần từ 77 cột v4 đã scale, fit trên train+val. Năm điểm PCA theo thứ tự phương sai tạo input 5×1 cho GRU 96. Attention cộng dùng trạng thái cuối làm query, gộp 5 trạng thái thành vector 96, rồi phân 16 lớp. Model (`core/gru_attention.py`, gỡ 20/11, xem commit `5b9d0e3`) có 48.688 tham số trainable. Checkpoint chứa cả tâm và cơ sở PCA. Chọn nhánh ICU nhẹ hơn nhánh NF-TON-IoT rộng 160. Dùng protocol centralized v4 (phân bố mẫu gốc, seed 42, split cố định, AdamW, 50 epoch) thay cho SMOTE, tuning và early stopping của nghiên cứu IoMT.

**Cập nhật 10/11/2026.** Bộ hiện tại gồm BERT-style, CNN–LSTM, MLP, TabM và **SAE**. GRANDE đã chạy đủ 50 epoch. TCNN dừng ở checkpoint epoch 1.

**SAE.** [TAS-VANET](../paper/SAE_2025.pdf), Scientific Reports 2026 (DOI 10.1038/s41598-026-75542-9), mục Phase 2b, công thức 9–12 và Bảng 2. Detector gốc dùng 13 feature min–max, encoder ReLU 64 → sigmoid 16, decoder đối xứng với đầu ra sigmoid, head nhị phân nhận phép nối input với bottleneck. WOA chọn LR 0,0015, mục tiêu thưa 0,05, hệ số KL 1,2. Hệ số classification 1,5. Trust, clustering và cập nhật online thuộc phần hệ thống của bài.

**Bản centralized v4.** [SAE PyTorch](../../../core/sae.py) dùng encoder 77→32→16, decoder 16→32→77, head 93→16. Có 7.613 tham số lúc train, 4.528 lúc inference. Ánh xạ cố định 71 cột z bằng (z+3,72)/7,44, giữ nguyên 6 cờ để decoder tái tạo đúng miền. Loss bằng tổng sai số bình phương theo cột (trung bình theo mẫu) + 1,2 × tổng KL Bernoulli trên 16 unit + 1,5 × CE 16 lớp. Mean activation của KL tính trên toàn logical batch, kể cả batch cuối lẻ và khi tách hai GPU. Regularization theo AdamW weight decay 1e-4 của protocol thay cho L2 cộng vào loss của Eq. 12. Giữ lịch LR chung 1e-3→1e-5, seed 42. Dùng hệ số đã chọn trong paper, không chạy lại WOA trên v4. Đánh giá cùng official test tại epoch 50, theo dõi thêm ba lớp GRANDE hụt nhiều nhất. Chất lượng và tốc độ cần đo từ run mới.

**GRANDE.** [PDF của owner](../paper/GRANDE_2024.pdf), Marton et al., ICLR 2024, Eq. 2–7; đối chiếu [release TensorFlow 0.1.6](https://pypi.org/project/GRANDE/0.1.6/). Mỗi nút học vector ngưỡng và điểm chọn cột. Entmax-1.5 và hardmax straight-through chọn một cột; theo code tác giả, nhánh trái lấy round((1 + softsign(ngưỡng − giá trị))/2). Nhân các quyết định trên đường đi ra chỉ báo lá. Softmax theo cây trên trọng số của lá được chọn rồi gộp logits riêng của các cây cho từng dòng.

**Bản v4.** PyTorch model (`core/grande.py`, gỡ 20/11, xem commit `5b9d0e3`) có 32 cây đầy đủ sâu 4, 77 feature đã scale, 16 logits/lá; 32 × [15 × 77 × 2 + 16 × (16 + 1)] = 82.624 tham số. Ma trận chọn/định tuyến tính fp32 dưới outer AMP để tránh đổi nhánh gần ngưỡng. Dùng đủ feature và batch cho mỗi cây, tree dropout 0. Bài gốc đánh giá 19 bộ nhị phân, batch 64 và tune optimizer/regularization; run này dùng CE 16 lớp, AdamW/cosine chung và global batch 4.096 như các baseline. CPU kiểm routing bằng traversal độc lập (cả tie), entmax gradcheck và resume. Notebook chạy probe DAGSNet W1 và GRANDE trên cùng cache và global batch, 3 step warmup + 6 step đo train, eval mẫu riêng. Sau probe khởi tạo mới cho 50 epoch.

Quyết định owner 09/11/2026. Mỗi run train riêng trên fold train + val của cả 5.343 xe, chấm toàn bộ test mỗi epoch. Tỷ lệ phân xe của thí nghiệm FL không áp dụng. Headline là epoch 50. DAGSNet có kết quả từ [baseline](../dagsnet/results.md).

| Backbone | Đối chiếu paper gốc | Cấu hình VeReMi đã chọn | Tham số |
|---|---|---|---:|
| BERT-style | [Fu et al., IEEE TVT](../paper/BERT_2025.pdf), phần III–IV. IoV-BERT-IDS biến payload CAN/packet thành byte sentences, BPE, token/segment/position embedding. Pretrain MBWM và NBSP trên cặp câu, rồi fine-tune classifier qua CLS. | Mỗi feature thành token bằng phép nhân vô hướng với vector học riêng và cộng embedding nhận diện cột. Thêm CLS, 2 block bidirectional post-LN, 4 head, d=64, FFN=128, GELU, dropout 0,1. Supervised CE từ đầu. Pha byte pretraining và trọng số paper nằm ngoài thí nghiệm đã chọn. | 77.904 |
| TCNN | [Khan et al., Computer Networks](../paper/TCNN_2025.pdf), §3.6, Tables 3–4. MHA (32 head × 64) → LN → residual; Conv1D 256 k5 same + ReLU → cộng đầu ra MHA → pool/LN. Năm block, bốn max-pool, flatten, 13 hidden dense. SHAP dùng sau train. | 77×1 → 38 → 19 → 9 → 4 vị trí. Giữ head/channel và dãy dense 128/512/128/256/128/256/64/32/64/128/512/256/128. Dense đầu đọc 1.024 phần tử. Lớp cuối 16 logits. Table 3 của paper ghi đầu ra 20 nhưng 258 tham số ở lớp cuối không khớp 128→20; build dùng 128→16 có bias. | 10.369.587 |
| CNN–LSTM | [Kamal et al., Scientific Reports](../paper/CNN_LSTM_2026.pdf), Tables 5–6. Nhánh 2 CNN–2 LSTM: input 10×21, 2 Conv1D kernel 1, 512/256 filter, tanh; 2 LSTM 128; hidden cuối → classifier. Adamax, LR 0,0003, batch 100, 50 epoch. Paper bỏ chuỗi ngắn. | 10×77. Mỗi dòng là đích, lấy tối đa 9 dòng quá khứ cùng client, sender và fold, chỉ lấy thời gian nhỏ hơn đích. Sắp xếp các dòng quá khứ trùng timestamp bằng message_id, rồi thứ tự file ổn định. Đệm zero bên trái. ID, thời gian, nhãn không được thêm vào 77 feature. Chấm đủ từng dòng. | 503.056 |
| MLP | [Gorishniy et al., NeurIPS](../paper/MLP_2021.pdf), Eq. 1. Mỗi block Linear → ReLU → Dropout, linear classifier ở cuối. Paper tune độ rộng/độ sâu theo dataset. | Hidden 64, 32; dropout 0,1. Đây là kích thước nhỏ đã chọn cho v4. | 7.600 |
| TabM | [Gorishniy et al., ICLR](../paper/TabM_2025.pdf), §3.3–3.4, Fig. 1–2. MLP dùng shared weight, adapter nhân đầu vào/ra và bias riêng từng member; adapter nhân khởi tạo 1 trừ adapter đầu; head riêng. Paper dùng k=32 và khảo sát thêm biến thể embedding. | Bản raw numeric, hai block rộng 32, k=4, dropout 0,1; random signs ở adapter đầu. Bốn member dùng chung minibatch, train trung bình CE từng member; eval trung bình xác suất trước argmax. | 6.548 |

## Protocol và điểm khác runner DAGSNet

Theo [protocol](../protocol.md) và [runner DAGSNet](../../../scripts/run_central.py): global batch 4.096, seed 42, AdamW, weight decay 1e-4, cosine từ 1e-3 đến 1e-5, grad clip 1, AMP fp16 trên T4. Train 37.361.273 dòng, test 9.347.679 dòng. Dùng 77 cột đã scale, 16 nhãn. Các HP optimizer/split lấy từ thí nghiệm DAGSNet của repo.

Runner mới giữ batch cuối (1.657 dòng với toàn bộ train + val) và tích lũy gradient theo đúng số mẫu thực từng rank. Runner DAGSNet cũ dùng floor(N / batch); chênh này và stream shuffle/dropout riêng phải được ghi khi so sánh. Không chọn epoch hoặc chỉnh kích thước qua official test. Lưu dự đoán theo thứ tự cache từng rank, nhãn tại epoch 1, ma trận nhầm 16×16, 10 metric, weights và resume state tại mỗi epoch. Thư mục epoch được đổi tên sau khi ghi đủ. LR/trạng thái AdamW/scaler tiếp nối qua session; RNG shuffle/dropout xác định theo epoch. Các phiên bản backend CUDA có thể tạo sai số số học.

## CNN-BiGRU-Attention trên VeReMi v4, 14/11/2026

`qwerasdf3004/veremi-central-cnn-bigru-attention-e50/1` COMPLETE50/50. Final accuracy0,94995; precision macro/micro/weighted0,96112/0,94995/0,94698; recall0,94216/0,94995/0,94995; F1 0,94903/0,94995/0,94592. Best test macro-F1 0,94953 tại epoch44, quan sát hậu nghiệm; headline giữ epoch50. Train+eval6,05600giờ. timeDelayAttack F1 thấp nhất0,52397, benign0,90722. Local source/config/data identity +50CM→10metrics PASS; final predictions→CM và strict weights/BN/AdamW state PASS. Earlier predictions/weights chưa kéo local; verifier Kaggle báo đủ50epoch. [Bảng đầy đủ và phạm vi kiểm](../../../papers/mnkash-2026-cnn-bigru/results.md).

## GRU + Attention trên VeReMi v4

Epoch 50 trên 9.347.679 dòng test đạt accuracy 0,71156. Precision macro/micro/weighted 0,57329 / 0,71156 / 0,69169. Recall 0,46930 / 0,71156 / 0,71156. F1 0,46403 / 0,71156 / 0,66602. Macro-F1 test cao nhất là 0,46519 ở epoch 36, accuracy cao nhất là 0,71243 ở epoch 15. Epoch 50, F1 feignedBraking 0,00099 và suddenConstantSpeed 0,00338. Source và metric từ ma trận nhầm đủ 50 epoch đã kiểm local, ma trận epoch cuối khớp dự đoán trên toàn bộ test.

## CNN–LSTM trên VeReMi v4, 10/11/2026

Phiên nối từ epoch 46 đã hoàn thành epoch 47–50. Macro-F1 tương ứng 0,990416 / 0,990419 / 0,990393 / 0,990398. Epoch 50 accuracy 0,991800. Precision macro/micro/weighted 0,991855 / 0,991800 / 0,991800. Recall 0,988979 / 0,991800 / 0,991800. F1 0,990398 / 0,991800 / 0,991783. Kaggle audit toàn bộ 50 epoch. Local khớp source và config với parent, tính lại CM, 10 metric và F1 từng lớp từ dự đoán trên 9.347.679 dòng test cho từng epoch mới, strict-load weights hữu hạn cả bốn epoch.

## SAE trên VeReMi v4, 10/11/2026

Epoch 50, seed 42, 9.347.679 dòng test. Accuracy 0,911541. Precision macro/micro/weighted 0,893940 / 0,911541 / 0,881726. Recall 0,868417 / 0,911541 / 0,911541. F1 0,874345 / 0,911541 / 0,889464. F1 lớp accelerationMultiplication 0,91855, feignedBraking 0,93804 và suddenConstantSpeed 0,77934. Epoch cuối train 50,95 giây, bỏ 3 bước AMP. Log Kaggle audit 50 epoch. Local dựng lại CM cuối từ cả hai rank, khớp 10 metric và F1 từng lớp, nạp strict weights cuối và kiểm hữu hạn.

## GRANDE trên VeReMi v4, 10/11/2026

Phiên `phamquang0606/veremi-central-grande-e50/1` đã audit 50 epoch. Local dựng lại CM epoch 50 từ 9.347.679 cặp nhãn–dự đoán của hai rank, khớp CM, 10 metric và F1 từng lớp. Artifact local ở `runs/pulls/central-status-grande-v1/central-grande`.

| Epoch | Accuracy | Precision macro | Precision micro | Precision weighted | Recall macro | Recall micro | Recall weighted | F1 macro | F1 micro | F1 weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0,860158 | 0,815319 | 0,860158 | 0,861314 | 0,719671 | 0,860158 | 0,860158 | 0,726454 | 0,860158 | 0,832080 |

Macro-F1 test đạt 0,824179 ở epoch 2, còn 0,709440 ở epoch 10 và 0,726454 ở epoch 50. Train CE trung bình trong epoch tương ứng là 0,4475, 0,5547 và 0,5391. Epoch 50, ba lớp accelerationMultiplication, feignedBraking và suddenConstantSpeed giải thích 67,2% chênh lệch macro-F1 với MLP cùng epoch. F1 GRANDE lần lượt 0,0239, 0,4004 và 0,4053. F1 timeDelayAttack là 0,0003, MLP cũng thấp ở lớp này (0,1316).

Checkpoint cho thấy 57/480 ngưỡng của cột đang được chọn vượt ±3,73 ở epoch 1, 123 ở epoch 2 và 153 ở epoch 50. Các nút này chỉ có thể rẽ một phía với đầu vào v4. Lần theo mọi đường root→leaf, giao khoảng hợp lệ của từng feature với điều kiện từng nút, còn tối đa 331/512 lá khả thi ở epoch 1, 201 ở epoch 2 và 170 ở epoch 50. Đây là cận trên theo miền giá trị, chưa đo tần suất vào lá trên dữ liệu thực. Một run chưa tách được ảnh hưởng của learning rate, số cây và phân bố lớp. Phép kiểm tiếp theo có thể dùng fold train/val tách riêng, giữ 32 cây sâu 4, chỉ giảm LR nhóm ngưỡng và đo tỉ lệ nút một nhánh cùng macro-F1 val.

## Chạy và kiểm

- [Runner](../../../scripts/run_central_backbones.py), [backbone](../../../core/central_backbones.py), temporal index (`core/central_windows.py`, gỡ 20/11, xem commit `5b9d0e3`), [builder](../../../scripts/build_central_backbones.py), [tests](../../../tests/test_central_backbones.py).
- Notebook đóng băng source local, hash source/data, đính dataset v4 public, chạy 2 T4. Dữ liệu cache ở temp. Model và output ở working. Probe đo trên global batch thật, sau warmup, cả train và eval, để lập ngân sách. File probe không chứa kết quả chất lượng của 50 epoch.
- Tạo notebook bằng `python scripts/build_central_backbones.py --owner ACCOUNT --slug SLUG --model MODEL`. MODEL là `probe`, `bert`, `cnn-lstm`, `mlp`, `tabm`, `sae` hoặc `grande`. Nối tiếp bằng slug mới và `--parent ACCOUNT/PARENT_SLUG` cùng MODEL, cùng thư mục artifact và epoch đã commit. Notebook kiểm source của parent rồi mới train.
- Kết quả run cần đối chiếu epoch 1…50, hash source/data, dự đoán → CM → 10 metric. Kiểm local CPU dùng fixture để kiểm forward/backward, đệm, biên split, resume và metric. Chi phí/độ chính xác thực nghiệm chỉ xác định khi chạy trên Kaggle.
