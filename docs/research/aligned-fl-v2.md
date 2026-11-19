# FL aligned-v2 — 15/11/2026

Owner yêu cầu đồng nhất baseline với DAGSNet; chốt riêng SAE: CE thuần, representation93.
Chuẩn đối chiếu: odixe06/fl_dagsnet_veremi_v2 commit
68a01b29cb3dd47d221d48a16289a8c9f1dfa155, source lưu runs/comparison-20261115/dagsnet/source.

## Những gì đồng nhất

- 14/14 method có cùng hp.get defaults với source DAGSNet (AST audit,
  runs/aligned-v2-hp-audit.json). TinyProto khôi phục mu=0.00015 và log APS/target
  trực tiếp từ source gốc; notebook truyền rõ override này.
- Cùng core local optimizer/reset mỗi epoch, batch64, cosine1e-3→1e-5, decay1e-4,
  clip1, AMP;50round, seeds41/42/43, cùng dataset/client partition và evaluation views.
- CANPerFL10fine-tune sau FedAvg50 cùng model/seed; notebook a2 chỉ trỏ parent a2.
- Mỗi local_epoch ghi applied/skipped; runner tổng hợp tất cả rank/method mỗi round.
  Smoke bắt buộc có telemetry, applied>0 và skipped<1%. Không xem field thiếu là zero.
  Telemetry đếm local_epoch client optimizer, không bao gồm server/proxy optimizer riêng.
- Kiến trúc backbone, dimension, dropout/BN và khởi tạo riêng vẫn khác theo model.
  Không thêm warm-start hoặc chọn best test view/round. LwFedNIDS giữ pruning70%.

## SAE-FL mới (khác centralized SAE cũ)

- core.sae.fl_config: CE hệ số1; bỏ reconstruction/KL khỏi loss FL.
- Encoder77→32→16 như cũ. Classifier93→16; representation trước classifier là
  concat(unit-scaled input77, sigmoid code16). FedProto/FedTGP/TinyProto dùng93chiều.
- Decoder giữ trong layout để tránh phá checkpoint/model code centralized; gradient bằng0
  trong CE variant, không đóng góp prediction. Tổng layout7613params, phần ảnh hưởng
  prediction4528params (decoder3085). Không mô tả toàn bộ7613 là được tối ưu bởi CE.
- Pruning SAE vẫn cắt encoder32→10, code16→5, decoder32→10; representation82,
  head vẫn nhận đủ77input+5code. APFed tách đúng classifier cuối.
- V2X projection93→93, server attention3heads vì93 không chia hết4. DAGSNet/TabM
  giữ4heads. Đây là adaptation kích thước được công khai, không phải HP giống tuyệt đối.
  V2X SAE layout16448params (7613+93²+2×93).
- SAE centralized cũ dùng config() và sae_loss() không đổi; kết quả cũ không được gắn nhãn CE-v2.

## Build và giới hạn

Builder scripts/build_aligned_matrix.py tạo4model×14method×3seed=168notebook riêng,
hậu tố-a2; không sửa frozen notebook cũ hoặc cloud ledger. Roster owner chỉ là gán local,
chưa kiểm quota/quyền mount khi launch. Build ledger runs/aligned-v2-build.json.

CPU tests: reference CE/93D/zero decoder gradients/TinyProto/step counters,
smoke14method SAE mới; regression TabM và SAE gốc. Static notebook/source hash/AST kiểm
trong builder. Kiểm GPU2T4/AMP/compile và performance chưa thực hiện; không khẳng định
đã chữa V2X seed42 collapse hoặc chất lượng sẽ tăng.

TinyProto cũ cần chạy lại để so cùngHP. Toàn bộ SAE cần fresh run vì objective/representation
đã đổi; không resume checkpoint cũ. Các TabM method khác chưa có thay đổi objective;
V2X cần diagnostic GPU có telemetry trước khi chạy dài. Cloud vẫn tắt, chưa push.
