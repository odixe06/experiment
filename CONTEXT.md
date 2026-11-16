# FeatureBERT FL trên VeReMi NextGen v4

Đọc CONTEXT → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) → [bàn giao sống](docs/operations/continue.md).
Không đọc `archive/` mặc định.

IDS V2X16lớp,77đầu vào;5343xe nhận là5343client. Repo so sánh14baseline FL và **MobiHet/T3** bằng FeatureBERT77904tham số,
đối chiếu DAGSNet ở `../fl_dagsnet_veremi_v2` và phương pháp ở `../mobihet`.

## Trạng thái kiểm16/11/2026

Sau reboot16/11: owner yêu cầu hoàn tất, không cần dừng/reboot; DVCpush lại và recovery pFedES41 đang thực hiện.
Ưu tiên phiên này: kiểm kernel hoàn thành → pull đa luồng → audit → cập nhật tài liệu/Notion → Git+DVC push hai repo.
Collector hiện hành và mọi exact head ở [continue](docs/operations/continue.md); không dùng trạng thái watcher cũ sau reboot.
T3 BERT đủr50 cả3seed, auditCM+statehai rankPASS. DAGSNet T3 đủr50 cả3seed. Không trộn partial vào headline50.

## Quyết định đang áp dụng

- [Protocol](docs/research/protocol.md):50round,1epoch thật/round, batch64, AdamW cosine, mọi client; ngoại lệ lựa chọn client theo method gốc.
- Seed41/42/43;14baseline,bỏSCAFFOLD/centralized. T3 bổ sung15/11, giữ cùng HP/code DAGSNet, chỉ đổi model/engine.
- **MobiHet=T3**: synthetic S64→64/M64→32/L48→16 từr26;floor16. Không đổi source/config frozen của run.
- `COMPLETE` không thay audit. CM audit **PASS_BOUNDED** không chứng minh replay inference/full private bank.
- Owner16/11 cho phép Git+DVC push hai repo; không xóa artifact, credential process-scope, không Co-Authored-By.

## Cần biết → mở

| Câu hỏi | Nguồn |
|---|---|
| Job nào sống, ai sở hữu, bước kế/việc tay? | [continue duy nhất](docs/operations/continue.md) |
| Cấu hình chung và luật đánh giá? | [protocol](docs/research/protocol.md) |
| Baseline và thay đổi cho BERT? | [methods](docs/research/methods.md), [parity](docs/research/v2-parity.md) |
| Kết quả và giới hạn audit? | [results](docs/research/results.md) |
| Byte/FLOPs đo thế nào? | [cost](docs/research/cost.md) |
| T3 và cơ chế đang chọn? | `../mobihet/docs/research/method.md`, `../mobihet/docs/research/proposal.md` |
| Dataset/model/máy? | [knowledge](knowledge/README.md) |
| Kaggle, script, lỗi, backup? | [kaggle](docs/operations/kaggle.md), [scripts](scripts/README.md), [sổ lỗi](docs/operations/notebook-errors.md) |

Kết quả/phương pháp→docs/research;job sống→continue;dữ kiện bất biến→knowledge;bài học→hai lessons.
