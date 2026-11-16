# FeatureBERT FL — điểm vào dự án

Đọc CONTEXT → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) → [bàn giao sống](docs/operations/continue.md).
Không đọc `archive/` mặc định.

IDS V2X16lớp,77input,5.343xe/client; FeatureBERT77.904tham số.
Repo14baseline FL + MobiHet/T3; DAGSNet đối chiếu tại `../fl_dagsnet_veremi_v2`, phương pháp tại `../mobihet`.

## Trạng thái — 16/11/2026

T3 BERT/DAGSNet đủr50 cả3seed41/42/43, audit CM/state có phạm vi PASS_BOUNDED.
Kết quả hiện hành: [results](docs/research/results.md), [comparison](docs/research/comparison.md).
pFedES đủr50/3seed, đã bổ sung bảng/Notion; GitHub và bước tiếp: [continue duy nhất](docs/operations/continue.md).

Owner đang lập [brief xây lại MobiHet](../mobihet/task_brief.md): seed42, đơnmodel DAGSNet/BERT,
sim→pilot→full; performance/communication xếp riêng top5, Macro-F1 cách bestbaseline≤.015, compute giảm tối đa.
Chưa launch phương pháp mới; mixed-model full-scale chưa chạy.

## Quy tắc đang dùng

- [Protocol](docs/research/protocol.md):50round,1epoch thật/round,batch64,AdamWcosine; chỉ ngoại lệ method gốc được ghi rõ.
- MobiHet=T3 giữ HP/code phương pháp DAGSNet, chỉ đổi model/engine. Không sửa source/config frozen của run.
- COMPLETE không thay audit; CM PASS_BOUNDED không chứng minh prediction replay/full private bank/exact resume.
- Giữ artifact/cache và credential process-scoped; không `Co-Authored-By:`. Job đang sở hữu file thì không sửa/dời.

| Cần biết | Mở |
|---|---|
| Job/owner/quyền/bước tiếp/việc tay | [Continue](docs/operations/continue.md) |
| Protocol/baseline/parity | [Protocol](docs/research/protocol.md), [methods](docs/research/methods.md), [parity](docs/research/v2-parity.md) |
| Kết quả/byte/FLOPs | [Comparison](docs/research/comparison.md), [cost](docs/research/cost.md) |
| Dataset/model/máy | [Knowledge](knowledge/README.md) |
| Kaggle/audit/backup | [Operations](docs/operations/README.md), [scripts](scripts/README.md) |

Kết quả→docs/research; job→continue; dữ kiện bất biến→knowledge; bài học→hai lessons.
