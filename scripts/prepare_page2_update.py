import json

old_table2 = open('/tmp/old_table2.txt').read()
new_table2 = open('/tmp/new_table2.txt').read()

p2_sub1_old = 'MobiHet (KM16+Codec) đạt macro F1 0,9074 (trung bình 2 seed 0,9078), macro recall 0,9095 (cao nhất toàn mạng) và giữ client P10 ở mức 0,7767 (cao nhất trong các phương pháp FL).'
p2_sub1_new = 'MobiHet (KM16+Codec) đạt macro F1 0.9078 ± 0.0004 (trung bình 3 seed), macro recall 0.9097 ± 0.0002 (cao nhất toàn mạng) và giữ client P10 ở mức 0.7748 ± 0.0023 (cao nhất trong các phương pháp FL).'

p2_sub2_old = 'MobiHet (KM16+Codec) vượt trội FedProx 0,8998, AFPHA 0,8990, FD-IDS 0,8992 và FedAvg 0,9042 để lọt Top 3 toàn hệ thống (đứng ngay sau APFed 0,9119 và CANPerFL 0,9088)'
p2_sub2_new = 'MobiHet (KM16+Codec) vượt trội FedProx 0.9005 ± 0.0017, AFPHA 0.8993 ± 0.0012, FD-IDS 0.8990 ± 0.0016 và FedAvg 0.9048 ± 0.0009 để lọt Top 3 toàn hệ thống (đứng ngay sau APFed 0.9130 ± 0.0010 và CANPerFL 0.9097 ± 0.0009)'

p2_sub3_old = 'đạt Macro F1 0,9074 (Top 3 toàn mạng, cao nhất nhóm model heterogeneity), Macro Recall 0,9095 (cao nhất toàn mạng) và Client P10 0,7767 với communication budget chỉ 1,16 MB qua 50 round'
p2_sub3_new = 'đạt Macro F1 0.9078 ± 0.0004 (Top 3 toàn mạng, cao nhất nhóm model heterogeneity), Macro Recall 0.9097 ± 0.0002 (cao nhất toàn mạng) và Client P10 0.7748 ± 0.0023 với communication budget chỉ 1,16 MB qua 50 round'

p2_sub4_old = '<li>\nChạy 3 seed 41 42 43 - cộng trừ sai số (ưu tiên 1)\n</li>'
p2_sub4_new = '<li>\nChạy 3 seed 41 42 43 - cộng trừ sai số (đã cập nhật 3 seed cho 14 phương pháp; NILM đang chạy Wave 2)\n</li>'

args = {
    "page_id": "40d730cee68d80f0a49ad050377772d2",
    "command": "update_content",
    "content_updates": [
        {"old_str": old_table2, "new_str": new_table2},
        {"old_str": p2_sub1_old, "new_str": p2_sub1_new},
        {"old_str": p2_sub2_old, "new_str": p2_sub2_new},
        {"old_str": p2_sub3_old, "new_str": p2_sub3_new},
        {"old_str": p2_sub4_old, "new_str": p2_sub4_new}
    ],
    "allow_async": False
}

with open('/tmp/page2_update_args.json', 'w') as f:
    json.dump(args, f)

print('Wrote /tmp/page2_update_args.json')
