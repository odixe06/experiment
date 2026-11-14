import json

# Page 1
with open('/home/odixe/.gemini/antigravity-cli/brain/23911446-f552-4d01-b120-52d83befda76/.system_generated/steps/554/output.txt') as f:
    p1 = json.loads(f.read())['text']

old_t1 = open('/tmp/old_table1.txt').read()
new_t1 = open('/tmp/new_table1.txt').read()

old_rows1 = [r for r in old_t1.split('<tr>\n') if r.strip()]
new_rows1 = [r for r in new_t1.split('<tr>\n') if r.strip()]

assert len(old_rows1) == len(new_rows1)

p1_updates = []
for old_r, new_r in zip(old_rows1, new_rows1):
    if old_r == new_r:
        continue
    old_full = '<tr>\n' + old_r
    new_full = '<tr>\n' + new_r
    # Ensure exact match in p1
    assert p1.count(old_full) == 1, f"old_full count not 1: {p1.count(old_full)}"
    p1_updates.append({"old_str": old_full, "new_str": new_full})

# Add text updates
p1_sub1_old = 'Đủ 50 round, audit PASS từ 50 CM (F1 đạt 0.907399, Recall 0.909542 — cao nhất toàn mạng, Client P10 0.776663, tổng lưu lượng 50 round chỉ 1,16 MB/xe gồm 1,15 MB bootstrap unicast và 13,3 KB cho 50 round, giảm 99,27% so với FedAvg).'
p1_sub1_new = 'Đủ 50 round, audit PASS từ 50 CM (F1 trung bình 3 seed đạt 0.9078 ± 0.0004, Recall 0.9097 ± 0.0002 — cao nhất toàn mạng, Client P10 0.7748 ± 0.0023, tổng lưu lượng 50 round chỉ 1,16 MB/xe gồm 1,15 MB bootstrap unicast và 13,3 KB cho 50 round, giảm 99,27% so với FedAvg).'
assert p1.count(p1_sub1_old) == 1
p1_updates.append({"old_str": p1_sub1_old, "new_str": p1_sub1_new})

p1_sub2_old = 'Dataset có benign/timeDelay khó phân biệt, shortcut theo world và Sybil lệch prior theo thời gian. Một seed; headline là điểm cuối, không chọn round tốt nhất trên test.'
p1_sub2_new = 'Dataset có benign/timeDelay khó phân biệt, shortcut theo world và Sybil lệch prior theo thời gian. Báo cáo trung bình 3 seed (41, 42, 43) kèm độ lệch chuẩn (ddof=1) cho các phương pháp hoàn tất; NILM-FL đang chạy Wave 2; SCAFFOLD giữ seed 42 (seed 41/43 phân kỳ NaN r15); headline là điểm cuối, không chọn round tốt nhất trên test.'
assert p1.count(p1_sub2_old) == 1
p1_updates.append({"old_str": p1_sub2_old, "new_str": p1_sub2_new})

print(f"Page 1 has {len(p1_updates)} granular updates, all unique!")

# Page 2
with open('/home/odixe/.gemini/antigravity-cli/brain/23911446-f552-4d01-b120-52d83befda76/.system_generated/steps/558/output.txt') as f:
    p2 = json.loads(f.read())['text']

old_t2 = open('/tmp/old_table2.txt').read()
new_t2 = open('/tmp/new_table2.txt').read()

old_rows2 = [r for r in old_t2.split('<tr>\n') if r.strip()]
new_rows2 = [r for r in new_t2.split('<tr>\n') if r.strip()]

assert len(old_rows2) == len(new_rows2)

p2_updates = []
for old_r, new_r in zip(old_rows2, new_rows2):
    if old_r == new_r:
        continue
    old_full = '<tr>\n' + old_r
    new_full = '<tr>\n' + new_r
    assert p2.count(old_full) == 1, f"old_full count not 1: {p2.count(old_full)}"
    p2_updates.append({"old_str": old_full, "new_str": new_full})

p2_sub1_old = 'MobiHet (KM16+Codec) đạt macro F1 0,9074 (trung bình 2 seed 0,9078), macro recall 0,9095 (cao nhất toàn mạng) và giữ client P10 ở mức 0,7767 (cao nhất trong các phương pháp FL).'
p2_sub1_new = 'MobiHet (KM16+Codec) đạt macro F1 0.9078 ± 0.0004 (trung bình 3 seed), macro recall 0.9097 ± 0.0002 (cao nhất toàn mạng) và giữ client P10 ở mức 0.7748 ± 0.0023 (cao nhất trong các phương pháp FL).'
assert p2.count(p2_sub1_old) == 1
p2_updates.append({"old_str": p2_sub1_old, "new_str": p2_sub1_new})

p2_sub2_old = 'MobiHet (KM16+Codec) vượt trội FedProx 0,8998, AFPHA 0,8990, FD-IDS 0,8992 và FedAvg 0,9042 để lọt Top 3 toàn hệ thống (đứng ngay sau APFed 0,9119 và CANPerFL 0,9088)'
p2_sub2_new = 'MobiHet (KM16+Codec) vượt trội FedProx 0.9005 ± 0.0017, AFPHA 0.8993 ± 0.0012, FD-IDS 0.8990 ± 0.0016 và FedAvg 0.9048 ± 0.0009 để lọt Top 3 toàn hệ thống (đứng ngay sau APFed 0.9130 ± 0.0010 và CANPerFL 0.9097 ± 0.0009)'
assert p2.count(p2_sub2_old) == 1
p2_updates.append({"old_str": p2_sub2_old, "new_str": p2_sub2_new})

p2_sub3_old = 'đạt Macro F1 0,9074 (Top 3 toàn mạng, cao nhất nhóm model heterogeneity), Macro Recall 0,9095 (cao nhất toàn mạng) và Client P10 0,7767 với communication budget chỉ 1,16 MB qua 50 round'
p2_sub3_new = 'đạt Macro F1 0.9078 ± 0.0004 (Top 3 toàn mạng, cao nhất nhóm model heterogeneity), Macro Recall 0.9097 ± 0.0002 (cao nhất toàn mạng) và Client P10 0.7748 ± 0.0023 với communication budget chỉ 1,16 MB qua 50 round'
assert p2.count(p2_sub3_old) == 1
p2_updates.append({"old_str": p2_sub3_old, "new_str": p2_sub3_new})

p2_sub4_old = '<li>\nChạy 3 seed 41 42 43 - cộng trừ sai số (ưu tiên 1)\n</li>'
p2_sub4_new = '<li>\nChạy 3 seed 41 42 43 - cộng trừ sai số (đã cập nhật 3 seed cho 14 phương pháp; NILM đang chạy Wave 2)\n</li>'
assert p2.count(p2_sub4_old) == 1
p2_updates.append({"old_str": p2_sub4_old, "new_str": p2_sub4_new})

print(f"Page 2 has {len(p2_updates)} granular updates, all unique!")

json.dump({"page_id": "409730cee68d81f19259d14bea45ea61", "command": "update_content", "content_updates": p1_updates, "allow_async": False}, open('/tmp/p1_granular.json', 'w'))
json.dump({"page_id": "40d730cee68d80f0a49ad050377772d2", "command": "update_content", "content_updates": p2_updates, "allow_async": False}, open('/tmp/p2_granular.json', 'w'))
