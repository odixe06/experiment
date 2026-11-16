# Công cụ fl_veremi_bert

Chạy Python trong env `nckh`, từ root repo.

| Script | Công việc |
|---|---|
| `run_batch.py`, `watch_three_seeds.py` | Baseline 3 seed: collect, nối checkpoint, audit |
| `run_t3.py`, `run_t3_batch.py` | Driver T3 BERT và build/launch batch khóa |
| `resume_t3_remaining.py` | Nối seed42/43 từ checkpoint đã audit, live quota/slot gate + watcher |
| `resume_t3_s41.py` | Receipt/launcher lịch sử seed41 p3; không chạy lại khi intent tồn tại |
| `watch_kaggle_batch.py` | Watch exact owner/slug/version, interval 1200s; không launch |
| `kaggle_cli_bounded.py` | Kaggle CLI timeout hữu hạn, dùng qua kaggle_as.py |
| `audit_run.py`, `audit_t3_state.py` | CM/10metric và state/source/HP/RNG hai rank |
| `collect_t3_completed.sh` | Pull T3 BERT 3 seed + audit, rồi pull DAGSNet41 về repo mobihet |
| `build_project_comparison.py` | Tái sinh8bảng performance/communication/client/server cho2model từaudit; pFedES tạm loại theoowner16/11 |
| `verify_batch_provenance.py` | Kiểm executedsource baseline với commit/manifest đãkhóa |
| `run_local_guarded.py` | Guard RAM cho tác vụ local |

## Tách công cụ khỏi mobihet — 15/11/2026

`watch_kaggle_batch.py`, `kaggle_cli_bounded.py` được chép nguyên byte từ `../mobihet/scripts/`.
T3 launcher, resume và smoke dùng helper local. HP tham chiếu DAGSNet được giữ tại
[knowledge/t3-dagsnet-reference.json](../knowledge/t3-dagsnet-reference.json), nguyên byte từ
`../mobihet/runs/t3-s41/batch.json`. Các method T3 và driver đã port trước phiên này.
Helper shared vẫn giữ trong mobihet vì các pipeline DAGSNet còn dùng; script nghiên cứu DAGSNet khác thuộc repo đó.
Không sửa overlay/source train đã khóa hoặc notebook đã push khi chuyển helper.
