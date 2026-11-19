# Năm backbone cho FL dị kiến trúc — lựa chọn owner14/11/2026

> 17/11: owner replaced the allocation below with [three fixed mixed-model cases](../mixed/mixed-model-cases.md).
> Assignment seed2026, unchanged vehicle data, shared mapping across training seeds41/42/43 and methods.
> The five single-backbone FL choices now exist; mixed-model runner and FL BERT remain pending.
> SAE-FL uses CE-only/representation93 per aligned-v2, overriding the historical objective below.
> Remaining sections preserve the earlier design discussion, not current launch readiness.

**Thảo luận tiếp14/11:** nhóm trưởng muốn5mức~310k/200k/140k/80k/10k. Đã đề xuất DAGSNet thu nhỏ/CNN–LSTM/CNN-BiGRU/BERT/SAE. Owner đã yêu cầu sửa riêng CNN–LSTM một bản tin77feature thành196.240tham số (`cnn-lstm-feature`), code+CPUtests xong, chưa train. Các kích thước còn lại chưa triển khai; bảng chọn TabM dưới đây ghi quyết định trước đó, cần cập nhật bộ5cuối cùng trước sinh manifest/FL. [Bản CNN–LSTM mới](central-backbones.md).

Owner chọn trộn DAGSNet, BERT-style77feature, TabM(k=4), SAE(hidden32) và CNN-BiGRU-Attention giữa client trong cùng thí nghiệm. Tỷ lệ25%/25%/50% chia đều ba model cuối. MambaCNN đã bỏ. Đây là đánh giá và chuẩn bị thiết kế; chưa sửa FL runner hoặc launch thí nghiệm mới.

## Phân bổ

| Model | Tỷ lệ yêu cầu | Client trong5.343xe | Tham số train hiện tại |
|---|---:|---:|---:|
| DAGSNet W1 | 25% | 1.336 | 395.408 |
| BERT-style77feature | 25% | 1.336 | 77.904 |
| TabM k4 | 50%/3 | 891 | 6.548 |
| SAE hidden32 | 50%/3 | 890 | 7.613 |
| CNN-BiGRU-Attention | 50%/3 | 890 | 118.288 |

Làm tròn theo phần dư lớn nhất; tie theo thứ tự model trong bảng. Đây là số client, không phải tỷ lệ bản tin. Khi triển khai: cố định một manifest client→model, đề xuất shuffle ID bằng seed42 không dùng nhãn/test để chọn kiến trúc; mọi phương pháp dùng cùng manifest. Báo thêm số dòng train/test theo kiến trúc, không đánh đồng tỷ lệ client với tỷ lệ dữ liệu. Manifest chưa sinh.

## Kết luận kiểm code hiện tại

Bộ model khả thi về giao diện dữ liệu: cùng77feature,16lớp, mỗi mẫu là một dòng; CNN-BiGRU chạy chuỗi theo feature, không cần cửa sổ10bản tin của CNN-LSTM. Chưa có benchmark FL cho bốn backbone mới; tham số nhỏ không chứng minh train nhanh.

FL hiện vẫn gắn với DAGSNet: [runner](../../../scripts/run_fl.py) dùng `width`, [Env](../../../core/fl.py) có một `cfg/Layout`, [grouped](../../../core/grouped.py) triển khai riêng DAGSNet. [Local loss](../../../core/local.py) nhận logits(G,B,C). Nên chưa thể chỉ đổi tên model và chạy các phương pháp cũ.

| Nhóm cơ chế trong code cũ | Khả năng dùng5kiến trúc |
|---|---|
| FedAvg, FedProx, SCAFFOLD, AFPHA, FD-IDS, APFed, PerFed-SKD; pha FedAvg của CANPerFL | Không thể gộp trực tiếp tham số khác kiến trúc. Nếu gộp riêng theo kiến trúc phải ghi rõ baseline5nhóm và đánh giá theo model của client; không gọi đó là một global model dùng chung5kiến trúc. |
| LwFedNIDS | Còn có pruning plan phụ thuộc tên/kênh DAGSNet; cần thiết kế pruning hợp lệ riêng cho mỗi backbone trước khi xét gộp. |
| FedProto, FedTGP, TinyProto, V2X-HtFL | Có hướng chuyển thể bằng biểu diễn prototype chung; phải chốt feature/projection, dimension và cách học để các model trao đổi được. Không chỉ sửa kích thước tensor. |
| pFedES, NILM-FL, MobiHet bản proxy | Có hướng giữ private model khác nhau và proxy chung. Code hiện cùng layout cho private/proxy hoặc grouped DAGSNet; cần tách. Kiến trúc/chi phí proxy phải ghi riêng, không tính chỉ private parameters. |
| MobiHet Stats/Core trong repo | Trao đổi thống kê77feature hoặc thông tin lớp không buộc private model đồng kiến trúc; vẫn phải thay backend DAGSNet và xử lý loss/eval riêng. Cần nêu đúng biến thể khi chọn phương pháp chạy. |

## Những hợp đồng phải giữ khi triển khai

- TabM:4member thuộc **một client**; local CE trung bình từng member, inference trung bình xác suất. KD/prototype cần định nghĩa riêng, không đưa tensor4member vào CE/KL cũ như16logit thông thường.
- SAE: giữ reconstruction + sparsity KL +1,5CE đã chốt. Mean activation KL tính trên batch **của từng client**; không dùng `all_reduce` của helper centralized để trộn client khác nhau. Decoder thuộc train/checkpoint, inference dùng4.528tham số.
- CNN-BiGRU và DAGSNet: BN thống kê riêng client; padding không tham gia thống kê. Cơ chế trao đổi buffer theo phương pháp, không tự áp global BN cho mọi phương pháp.
- Hidden hiện khác nhau: DAGSNet256, BERT64, CNN-BiGRU128; SAE bottleneck16 nhưng classifier input93; TabM4×32. Chưa chốt cách tạo prototype chung. Projection thêm phải tính vào tham số/byte.
- FL theo protocol khởi tạo mới seed42, cùng khởi tạo trong mỗi kiến trúc. Không tự warm-start bằng checkpoint centralized đã thấy dữ liệu toàn bộ client. BERT centralized mới có kết quả1–18; hoàn thành50 cần cho mốc centralized đồng đều, không phải điều kiện kỹ thuật để khởi tạo FL.
- Giữ các phần độc lập kiến trúc của [protocol](../protocol.md): partition theo5.343xe,50round,1epoch local/batch64, optimizer/lịch LR, headline cuối,10metric và test riêng xe cho personalized models. Mục kiến trúc và các deviation mới phải chốt trước production; chưa sửa protocol cũ để tránh diễn giải lại run đã công bố.

## Trước khi chạy dài

Kiểm mỗi backbone so reference (logits/loss/gradient), client nhỏ/tail/padding, strict checkpoint và resume; kiểm mỗi cơ chế aggregation/KD/prototype. Sau đó smoke vài round và benchmark trên dữ liệu thật/T4 để dự trù quota, bộ nhớ bank client/proxy, thời gian và bytes. Chỉ dùng train/val cho lựa chọn thiết kế; không chọn projection hoặc tỷ lệ từ official-test score.
