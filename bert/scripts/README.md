# Công cụ còn dùng — FeatureBERT

Từ repo root, env `nckh`; quyền/job tại [continue](../../docs/bert/operations/continue.md).
Launch/resume/reboot/watcher riêng các đợt đã đóng ở [archive16/11](../../docs/archive/bert/operations-20261116/README.md).

| Tác vụ | Entry point |
|---|---|
| Driver baseline / T3 | `run_fl.py`, `run_t3.py` |
| Pilot Lite2 B + U1 (build/launch/collect/audit; control T3 seed 42) | `run_lite2_pilot.py` |
| Ledger baseline: audit/collect theo quyền | `run_batch.py` (collect có thể launch continuation; không dùng chỉ để đọc status) |
| Sinh notebook / import dataset checkpoint | `gen_notebook.py`, `verify_checkpoint_dataset.py`, `transfer_global.py` |
| Watch exact kernel / inventory / CLI bounded | `watch_kaggle_batch.py`, `kaggle_inventory.py`, `kaggle_cli_bounded.py` |
| Audit CM/10metric / T3 state | `audit_run.py`, `audit_t3_state.py`, `audit_t3_smoke.py` |
| Tái sinh bảng hai model | `build_project_comparison.py`; require đủ endpoint/audit mọi hàng |
| Source / parity / port | `verify_batch_provenance.py`, `check_t3_port.py` |
| Đo cost / báo cáo | `cost.py`, `report.py` |
| Local guard/fixture/benchmark/data listing | `run_local_guarded.py`, `cache_subset.py`, `bench_grouped.py`, `list_dataset.py` |
| Link active | `check_docs.py` |

`run_p21_recovery.py` tên cũ nhưng notebook generator còn import helper; giữ nguyên byte/đường dẫn,
không phải hướng dẫn launch P21. T3 reference: [t3-dagsnet-reference](../knowledge/t3-dagsnet-reference.json).
Worker pFedES hiện tại và script one-off của phiên nằm trong `runs/maintenance-20261116/`; owner tại continue,
không di chuyển trong khi job sống. [Manifest dọn context](../../docs/archive/bert/operations-20261116/migration.json).
