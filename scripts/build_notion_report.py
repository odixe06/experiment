"""Prepare native Notion page payloads from complete pulled CMs; no network writes.

Upload paper PDFs first into runs/notion-report/uploads.json, then run this script in nckh.
The resulting page JSON files can be sent to the connected Notion create_pages tool.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import metrics

OUT = Path("runs/notion-report")
SPEC = [
    ("centralized", "Centralized DAGSNet", "fl2-central-w1", "global", "afpha_edl_cmso.pdf"),
    ("fedavg", "FedAvg", "fl2-fedavg", "global", "fedavg.pdf"),
    ("fedprox", "FedProx", "fl2-fedprox", "global", "fedprox.pdf"),
    ("afpha", "AFPHA", "fl2-afpha-r2", "global", "afpha_edl_cmso.pdf"),
    ("fd_ids", "FD-IDS", "fl2-fd-ids-r2", "global", "fd_ids.pdf"),
    ("apfed", "APFed", "fl2-apfed-r2", "client", "apfed.pdf"),
    ("lwfednids", "LwFedNIDS", "fl2-lwfednids-r2", "global", "lwfednids.pdf"),
    ("v2x_htfl", "V2X-HtFL", "fl2-v2x-htfl-r2", "client", "v2x_htfl.pdf"),
    ("canperfl", "CANPerFL", "fl2-canperfl", "client", "canperfl.pdf"),
    ("pfedes", "pFedES", "fl2-pfedes-r2", "client", "pfedes.pdf"),
    ("fedtgp", "FedTGP", "fl2-fedtgp-r2", "client_proto", "fedtgp.pdf"),
    ("perfedskd", "PerFed-SKD", "fl2-perfedskd-r2", "client", "perfedskd.pdf"),
    ("fedproto", "FedProto", "fl2-fedproto", "client_proto", "fedproto.pdf"),
    ("nilm", "NILM-FL", "fl2-nilm-r2", "client", "nilm.pdf"),
    ("tinyproto", "TinyProto", "fl2-tinyproto-r4", "client_proto", "tinyproto.pdf"),
]
NOTE = {
    "centralized": "50 epoch centralized; batch 4.096 là deviation đã khóa. Bài PDF mô tả kiến trúc DAGSNet; đây không phải một thuật toán FL.",
    "apfed": "Metric đủ r50; kernel ERROR khi lưu state cuối vì RAM host. Bank private r50 không được giữ; không thể replay model private từ artifact đã pull.",
    "canperfl": "Đúng cơ chế hai pha: FedAvg 50 round (dùng chung run FedAvg), sau đó 10 epoch fine-tune local. Không có 50 round fine-tune độc lập.",
    "nilm": "Nối phiên thành công: session 1 (r1–36) + session 2 (fl2-nilm-r2s2, r37–50); parent preflight r1–36 byte-identical.",
    "tinyproto": "μ=1,5e-4 theo cấu hình tác giả (deviation so với VeReMi); đã khắc phục bùng nổ scale của r3; prototype F1 đạt 0,8534.",
}


def table(rows, label="Round", clients=True):
    cols = metrics.TEN + (["client_f1_macro_mean", "client_f1_macro_p10"] if clients else [])
    lines = ["| " + " | ".join([label] + cols) + " |", "|" + "---:|" * (len(cols) + 1)]
    lines += ["| " + " | ".join([str(r)] + [f"{s[k]:.6f}" for k in cols]) + " |" for r, s in rows]
    return "\n".join(lines)


def cm_rows(root, support, expected):
    logs = [json.loads(s) for s in (root / "rounds.jsonl").read_text().splitlines()]
    assert [r["round"] for r in logs] == list(range(1, expected + 1)), root
    views, data = None, {}
    for row in logs:
        r = row["round"]
        with np.load(root / f"cms_r{r:03d}.npz") as z:
            assert np.array_equal(z["ids"], np.arange(5343))
            current = sorted(k for k in z.files if k != "ids")
            assert views is None or current == views
            views = current
            for v in views:
                cm = z[v]
                assert cm.shape == (5343, 16, 16) and (cm >= 0).all()
                assert np.array_equal(cm.sum(2), support), (root, r, v)
                s = metrics.summary(cm)
                for k in metrics.TEN:
                    np.testing.assert_allclose(s[k], row[f"{v}/{k}"], atol=1e-12, rtol=0)
                data.setdefault(v, []).append((r, s))
    return data, logs


def main():
    uploads = json.loads((OUT / "uploads.json").read_text())
    support = np.load("runs/test_support.npy")
    prepared, headline = [], []
    for alias, title, slug, view, paper in SPEC:
        root = Path("runs/pulls") / slug / slug
        cfg = json.loads((root / "config.json").read_text())
        content = ["VeReMi NextGen v4: **5.343 xe**, train 37.361.273 dòng, test 9.347.679 dòng; 77 đầu vào, 16 lớp.",
                   "DAGSNet W=1; seed 42. Client model chấm trên test của chính xe; global model chấm toàn test. Các số dưới đây tính lại từ confusion matrix đã pull, không chép từ W&B.",
                   f"Bản dựng theo protocol chung; kernel `{slug}`; view headline **{view}**. Các kết quả này không so trực tiếp với số liệu dataset gốc trong bài báo."]
        if alias in NOTE:
            content.append("**Lưu ý:** " + NOTE[alias])
        content += ["## Bài báo", uploads[paper]["suggested_markdown"],
                    "## Cấu hình", "```json\n" + json.dumps(cfg, ensure_ascii=False, indent=2) + "\n```",
                    "accuracy = precision_micro = recall_micro = recall_weighted = f1_micro trong bài toán single-label này; vẫn giữ đủ 10 cột."]
        if alias == "centralized":
            data = []
            logs = [json.loads(s) for s in (root / "epochs.jsonl").read_text().splitlines()]
            assert [x["epoch"] for x in logs] == list(range(1, 51))
            for r, row in enumerate(logs, 1):
                cm = np.load(root / f"cm_e{r:03d}.npy")
                assert np.array_equal(cm.sum(1), support.sum(0))
                s = metrics.ten(cm)
                for k in metrics.TEN:
                    np.testing.assert_allclose(s[k], row[k], atol=1e-12, rtol=0)
                data.append((r, s))
            content += ["## Đủ 50 epoch", table(data, "Epoch", clients=False)]
            final = data[-1][1]
        else:
            expected = 10 if alias == "canperfl" else 50
            data, logs = cm_rows(root, support, expected)
            if alias == "canperfl":
                pre, _ = cm_rows(Path("runs/pulls/fl2-fedavg/fl2-fedavg"), support, 50)
                content += ["## Pha 1 — đủ 50 round FedAvg", table(pre["global"])]
            for v in [view] + [v for v in data if v != view]:
                content += [f"## {'Fine-tune 10 epoch' if alias == 'canperfl' else 'Đủ 50 round'} — {v}",
                            table(data[v], "Epoch fine-tune" if alias == "canperfl" else "Round")]
            final = data[view][-1][1]
        content += ["## Phạm vi bằng chứng",
                    "Đã đối chiếu support nhãn test và các metric với CM. Artifact hiện tại không lưu prediction từng dòng; một số run không giữ bank private từng round, nên chưa replay được toàn bộ suy luận từ model.",
                    "Caveat dataset: benign/timeDelay khó phân biệt; mỗi world chỉ có một loại tấn công có thể tạo shortcut bối cảnh; Sybil lệch prior theo thời gian. Headline là điểm cuối, không chọn round tốt nhất trên test.",
                    f"Artifact local: `{root}`. [Protocol](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/main/docs/research/protocol.md)."]
        payload = dict(properties={"title": title}, content="\n\n".join(content))
        (OUT / f"{alias}.json").write_text(json.dumps(payload, ensure_ascii=False))
        prepared.append(dict(alias=alias, title=title, file=str(OUT / f"{alias}.json"), paper=paper))
        last = logs[-1]
        pb = cfg.get("payload_bytes", {})
        headline.append(dict(alias=alias, title=title, view=view, round="50+10 FT" if alias == "canperfl" else "ep50" if alias == "centralized" else "50",
                             metrics=final, bytes=last.get("bytes_up", pb.get("up", 0)) + last.get("bytes_down", pb.get("down", 0)),
                             note=NOTE.get(alias, "")))
        print(alias, "all rounds/metrics/CM support PASS", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(dict(prepared=prepared, headline=headline), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
