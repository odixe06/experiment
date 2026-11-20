"""Build current four-backbone, per-seed final Macro-F1 tables; unverified cells stay blank."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/research/backbones/four-backbone-seed-results-live.md"
METHODS = ["fedavg", "fedprox", "afpha", "fd_ids", "apfed", "lwfednids",
           "canperfl", "tinyproto", "pfedes", "nilm", "fedtgp", "perfedskd",
           "fedproto", "v2x_htfl"]
NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "afpha": "AFPHA",
         "fd_ids": "FD-IDS", "apfed": "APFed", "lwfednids": "LwFedNIDS",
         "canperfl": "CANPerFL", "tinyproto": "TinyProto", "pfedes": "pFedES",
         "nilm": "NILM-FL", "fedtgp": "FedTGP", "perfedskd": "PerFed-SKD",
         "fedproto": "FedProto", "v2x_htfl": "V2X-HtFL"}
MODEL_INFO = {'TabM': '**Model:** 77 → 32 → 32 → 16, ensemble k=4 dùng chung trọng số; **6.548 tham số**, dropout 0,1. Representation 32 chiều; dự đoán bằng trung bình xác suất của 4 thành viên.', 'SAE': '**Model:** encoder 77 → 32 → 16; nối input77 với code16 → classifier 93 → 16. **7.613 tham số lưu**, trong đó 4.528 ảnh hưởng dự đoán; decoder 3.085 tham số không dùng trong loss FL. Bản FL CE-only, representation 93 chiều, dropout 0.', 'MLP': '**Model:** 77 → 64 → 32 → 16, ReLU; **7.600 tham số**, dropout 0,1. Representation 32 chiều; nhận trực tiếp vector feature của một bản tin.', 'CNN-BiGRU-Attention': '**Model:** input 77×1 → Conv64 + BN + MaxPool → Conv128 → BiGRU64 mỗi chiều → Attention128 → 16 lớp. **118.288 tham số**, representation 128 chiều, dropout 0,3. Chuỗi chạy trên 77 feature của một bản tin, không phải cửa sổ nhiều bản tin.'}
COMMON_CONFIG = '**Cấu hình chung:** 77 feature/bản tin, 16 lớp; 5.343 client theo xe; 14 phương pháp, seed 41/42/43. Train 50 round, 1 epoch local/round, batch 64; AdamW, LR cosine 1e-3 → 1e-5. CANPerFL fine-tune thêm 10 round sau FedAvg50. Số tham số dưới đây là backbone gốc, chưa tính module riêng của phương pháp hoặc pruning.'
values = {}


def method_view(method):
    return ('global' if method in ('fedavg', 'fedprox', 'afpha', 'fd_ids', 'lwfednids')
            else 'client_proto' if method in ('tinyproto', 'fedtgp', 'fedproto') else 'client')


def cached_final(folder, view):
    marker = folder / "RUN_STATUS.json"
    if not marker.exists() or not (folder / "SMOKE_PASS.json").exists():
        return None
    run = json.loads(marker.read_text(encoding="utf-8"))
    final = run.get("final", {})
    if not run.get("complete") or run.get("last_round") != run.get("target_round"):
        return None
    if final.get("skipped_frac") is None or final['skipped_frac'] >= 0.01:
        return None
    return final.get(f"{view}/f1_macro")


def aligned_identity(row):
    slug = row.get("slug", "")
    model = "TabM" if slug.startswith("tabm-") else "SAE"
    method = row.get("method")
    if method is None:
        method = next((name for name in METHODS if f"-fl-{name.replace('_', '-')}-" in slug), None)
    seed = row.get("seed")
    if seed is None:
        seed = next((value for value in (41, 42, 43) if f"-s{value}" in slug), None)
    return model, method, seed

# Full TabM matrix was final-CM audited; corrected aligned-v2 TinyProto overrides its old run.
comparison = json.loads((ROOT / "runs/comparison-20261115/audit_summary.json").read_text(encoding="utf-8"))
for row in comparison["audited"]:
    if row["backbone"] == "tabm" and row["method"] != "v2x_htfl":
        values[("TabM", row["method"], row["seed"])] = row["metrics"]["f1_macro"]

aligned = json.loads((ROOT / "runs/aligned-v2-live.json").read_text(encoding="utf-8"))["rows"]
for row in aligned:
    if (row.get("status") == "COMPLETE" and row.get("smoke") and row.get("complete")
            and row.get("last_round") == row.get("target_round") and row.get("f1_macro") is not None):
        model = "TabM" if row.get("backbone") == "TabM" else "SAE"
        if model == "SAE" or row.get("method") == "tinyproto":
            values[(model, row["method"], row["seed"])] = row["f1_macro"]
    elif row.get("slug"):
        model, method, seed = aligned_identity(row)
        value = cached_final(ROOT / "runs/aligned-v2-live" / row["slug"], method_view(method))
        if value is not None and method is not None and seed is not None and (model == "SAE" or method == "tinyproto"):
            values[(model, method, seed)] = value

backbones = json.loads((ROOT / "runs/aligned-v2-backbones-live.json").read_text(encoding="utf-8"))["rows"]
for row in backbones:
    if (row.get("status") == "COMPLETE" and row.get("smoke") and row.get("complete")
            and row.get("last_round") == row.get("target_round") and row.get("f1_macro") is not None):
        model = "MLP" if row["backbone"] == "mlp" else "CNN-BiGRU-Attention"
        values[(model, row["method"], row["seed"])] = row["f1_macro"]
    elif row.get("slug"):
        model = "MLP" if row["backbone"] == "mlp" else "CNN-BiGRU-Attention"
        value = cached_final(ROOT / "runs/aligned-v2-backbones-live" / row["slug"], method_view(row['method']))
        if value is not None:
            values[(model, row["method"], row["seed"])] = value

resume_path = ROOT / "runs/aligned-v2-resumes-live.json"
if resume_path.exists():
    resumes = json.loads(resume_path.read_text(encoding="utf-8"))["rows"]
    for row in resumes:
        if (row.get("status") == "COMPLETE" and row.get("smoke") and row.get("complete")
                and row.get("last_round") == row.get("target_round") == 50
                and row.get("skipped_frac", 1) < 0.01 and row.get("f1_macro") is not None):
            values[("CNN-BiGRU-Attention", row["method"], row["seed"])] = row["f1_macro"]

        elif row.get('status') == 'UNVERIFIED' and row.get('slug'):
            folder = ROOT / 'runs/aligned-v2-resumes-live' / row['slug']
            value = cached_final(folder, method_view(row['method']))
            if value is not None:
                manifest = json.loads((folder / 'experiment_manifest.json').read_text(encoding='utf-8'))
                if manifest.get('method') == row['method'] and manifest.get('seed') == row['seed']:
                    values[("CNN-BiGRU-Attention", row['method'], row['seed'])] = value

v2x_path = ROOT / 'runs/v2x-a4-live.json'
if v2x_path.exists():
    for row in json.loads(v2x_path.read_text(encoding='utf-8'))['rows']:
        if (row.get('status') == 'COMPLETE' and row.get('smoke') and row.get('complete')
                and row.get('last_round') == row.get('target_round') == 50
                and row.get('manifest_method') == 'v2x_htfl' and row.get('manifest_seed') == row['seed']
                and row.get('history_contiguous') and row.get('all_rounds_applied')
                and row.get('max_skipped_frac', 1) < .01 and row.get('f1_macro') is not None):
            model = {'tabm': 'TabM', 'mlp': 'MLP'}[row['backbone']]
            values[(model, 'v2x_htfl', row['seed'])] = row['f1_macro']

models = ("TabM", "SAE", "MLP", "CNN-BiGRU-Attention")
lines = ["# Macro-F1 cuối theo model và seed — live", "",
         'Sinh bảng: ' + datetime.now(timezone.utc).isoformat() + ' (UTC).', '',
         "Cập nhật từ trạng thái/marker Kaggle hiện tại. Ô trống = RUNNING, QUEUED, chưa xác minh được marker,",
         "hoặc chưa đủ target round. Không dùng điểm best round; CANPerFL dùng kết quả fine-tune round 10.", "",
         "TabM TinyProto dùng aligned-v2 với μ=0,00015. SAE dùng aligned-v2 CE-only/prototype93.",
         "TabM/MLP V2X-HtFL dùng fp32proj-a4 sau kiểm smoke và telemetry đủ 50 round.",
         "Nguồn exact ref, 10 metric và mức kiểm chứng theo seed nằm trong results.csv của từng kịch bản.",
         "Kết quả đã kéo trước được giữ khi API tạm lỗi; không xem lỗi đọc API là mất kết quả.", ""]
lines += [COMMON_CONFIG, ""]
for model in models:
    lines += [f"## {model}", "", MODEL_INFO[model], "", "| Phương pháp | Seed 41 | Seed 42 | Seed 43 |",
              "|---|---:|---:|---:|"]
    for method in METHODS:
        cells = [f"{values[(model, method, seed)]:.5f}" if (model, method, seed) in values else ""
                 for seed in (41, 42, 43)]
        lines.append(f"| {NAMES[method]} | " + " | ".join(cells) + " |")
    lines.append("")
lines += [f"Đã điền {len(values)}/168 ô kết quả.", ""]
OUT.write_text("\n".join(lines), encoding="utf-8")
print(OUT)
print("filled", len(values), "blank", 168 - len(values))
