import json

old_table1 = open('/tmp/old_table1.txt').read()
new_table1 = open('/tmp/new_table1.txt').read()

p1_sub1_old = 'Đủ 50 round, audit PASS từ 50 CM (F1 đạt 0.907399, Recall 0.909542 — cao nhất toàn mạng, Client P10 0.776663, tổng lưu lượng 50 round chỉ 1,16 MB/xe gồm 1,15 MB bootstrap unicast và 13,3 KB cho 50 round, giảm 99,27% so với FedAvg).'
p1_sub1_new = 'Đủ 50 round, audit PASS từ 50 CM (F1 trung bình 3 seed đạt 0.9078 ± 0.0004, Recall 0.9097 ± 0.0002 — cao nhất toàn mạng, Client P10 0.7748 ± 0.0023, tổng lưu lượng 50 round chỉ 1,16 MB/xe gồm 1,15 MB bootstrap unicast và 13,3 KB cho 50 round, giảm 99,27% so với FedAvg).'

p1_sub2_old = 'Dataset có benign/timeDelay khó phân biệt, shortcut theo world và Sybil lệch prior theo thời gian. Một seed; headline là điểm cuối, không chọn round tốt nhất trên test.'
p1_sub2_new = 'Dataset có benign/timeDelay khó phân biệt, shortcut theo world và Sybil lệch prior theo thời gian. Báo cáo trung bình 3 seed (41, 42, 43) kèm độ lệch chuẩn (ddof=1) cho các phương pháp hoàn tất; NILM-FL đang chạy Wave 2; SCAFFOLD giữ seed 42 (seed 41/43 phân kỳ NaN r15); headline là điểm cuối, không chọn round tốt nhất trên test.'

args = {
    "page_id": "409730cee68d81f19259d14bea45ea61",
    "command": "update_content",
    "content_updates": [
        {"old_str": old_table1, "new_str": new_table1},
        {"old_str": p1_sub1_old, "new_str": p1_sub1_new},
        {"old_str": p1_sub2_old, "new_str": p1_sub2_new}
    ],
    "allow_async": False
}

with open('/tmp/page1_update_args.json', 'w') as f:
    json.dump(args, f)

print('Wrote /tmp/page1_update_args.json')
