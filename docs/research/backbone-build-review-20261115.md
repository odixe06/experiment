# Kiểm tra build TabM/SAE — 15/11/2026

Phạm vi: source đã chạy trong artifacts comparison-20261115, config/history/final CM;
đối chiếu reference centralized và grouped FL. Không sửa training hoặc launch lại.

## Kết quả xác nhận

- Source snapshot kiểm tra khớp local cho core/sae.py, tabm.py, grouped.py, local.py,
  evaluate.py và methods/fedtgp.py, v2x_htfl.py, tinyproto.py.
- Chạy trực tiếp tests/test_sae_fl.py và tests/test_tabm_fl.py với runtime CPU
  tmp/fl-test-runtime: cả hai PASS (reference forward, objective SAE, smoke các method,
  CAN parent). Đây không phải kiểm gradient AMP/compile trên GPU production.
- TabM giữ CE trung bình các member, inference trung bình xác suất; SAE giữ skip77,
  reconstruction + KL sparsity + CE như centralized. Chưa thấy lỗi nối tensor ở các đường này.
- FL optimizer được tạo lại mỗi local epoch; centralized giữ optimizer xuyên epoch.
  Đây là protocol FL hiện tại, không thể coi centralized50epoch tương đương FL50round.

## Những điểm phải xử lý trước kết luận về backbone

1. TinyProto thiếu override mu=0.00015 của DAGSNet r4. TabM/SAE dùng heuristic.
   Chắc chắn cấu hình khác; chưa cô lập mức ảnh hưởng tới kết quả.
2. SAE classifier nhận concat(input77,code16); prototype chỉ dùng code16 sigmoid,
   còn KL kéo activation về rho=0.05. Prototype không có skip77 mà classifier có.
   Seed42 FedTGP: classifier0.65844, prototype0.06875; FedProto:0.65774/0.16768.
   Bằng chứng đường prototype kém; giả thuyết code ít phân biệt lớp cần kiểm checkpoint,
   độ phân tán và khoảng cách class trên train/val. Chưa đo collapse của code.
3. TabM V2X-HtFL seed41/42/43 headline0.67054/0.01951/0.74679.
   Seed42 đã0.01888 ởround1 và gần đứng yên. Chưa xác định nguyên nhân tối ưu/AMP.
   methods/v2x_htfl.py lấy st['loss'] nhưng bỏ st['skipped']/st['applied']; history thiếu
   skipped_frac. scripts/run_fl.py và audit dùng get(...,0), nên không được diễn giải
   giá trị thiếu là bằng chứng không skip. Kiểm CM PASS không phải health PASS.
4. LwFedNIDS prune70%: TabM6548→2082, SAE7613→3070. Đây là adapter có chủ đích,
   không phải lỗi đếm tham số; chất lượng giảm cần phân biệt với model chưa prune.

## Bước kế tiếp đề xuất

- Bổ sung telemetry applied/skipped/nonfinite cho các method đang thiếu; đánh dấu unknown
  trong audit khi field không có. Chạy diagnostic ngắn trên đúng2T4/source/seed42 V2X,
  đối chiếu eager-fp32, eager-AMP và compiled-AMP trên cùng batch, trước khi chọn sửa.
- Khôi phục override TinyProto từ baseline gốc; khóa config trước chạy kiểm chứng mới.
- SAE: quyết định representation prototype có bao gồm skip77 hay không. Đây là thay đổi
  adapter/phương pháp, cần chốt và kiểm validation; không tự thay bằng view có test cao hơn.
- Không tăng round hoặc tune theo official test để chữa số thấp. Giữ kết quả cũ và caveat.

[Bảng số liệu và nguồn](tabm-sae-dagsnet-comparison-20261115.md).
