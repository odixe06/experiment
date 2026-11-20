# Git / DVC backup record

## GitHub centralized CNN-LSTM-feature supplement — 20/11/2026

- Remote: `origin` → `https://github.com/odixe06/experiment.git`; branch:
  `codex/mobihet-v4-relkd`; artifact commit:
  `8b1a3d4a360e5431d2be1a426040ad10c4f4d0fc`.
- Git LFS object: `artifacts/central-cnn-lstm-feature-supplement-2026-11-20.zip`,
  77.224.724 bytes; SHA-256
  `1bca385fd7ed168fc630781d2e6674deb19bf54c07ce40370fdae0e1b29562b5`.
- Scope: all 258 safe non-checkpoint files / 478.011.181 source bytes from exact
  `qwerasdf3004/veremi-central-cnn-lstm-feature-e50` v1 output.
- Checks: config identifies CNN-LSTM-feature seed42/50 epochs; epochs1–50 contiguous;
  every CM is 16×16 with 9.347.679 rows; every prediction reconstructs its CM exactly;
  all ten metrics match each CM; ZIP integrity and credential scan PASS.
- Full backup audit after merge: single-model FL 168/168, mixed-model 45/45, and all
  recorded centralized epochs PASS. No checkpoints included; no cleanup performed.

## GitHub mixed-model history supplement — 20/11/2026

- Remote: `origin` → `https://github.com/odixe06/experiment.git`; branch:
  `codex/mobihet-v4-relkd`; artifact commit:
  `7585df08fef18c53bb5eafd8aaf07bd1acca25de`.
- Git LFS object: `artifacts/mixed-history-supplement-2026-11-20.zip`, 1.195.696 bytes;
  SHA-256 `df8032085ba6c0465ae5a85333f0d094ee49ed29b34c4e5c1d38ae676c15e3ba`.
- Scope: complete R1–R50 `rounds.jsonl` for Case2 V2X-HtFL seeds41/42/43 and
  Case3 FedTGP seed41 / TinyProto seeds41/43; CSV contains 300 round rows.
- Checks: exact final kernel/version, non-smoke history, contiguous unique rounds1–50,
  both client/client_proto macro-F1 each round, all ten R50 metrics match final-CM audits,
  ZIP integrity and credential scan PASS; LFS upload 100%.
- No checkpoints included; no local cleanup requested or performed.

## GitHub single-model artifact supplement — 20/11/2026

- Remote: `origin` → `https://github.com/odixe06/experiment.git`; branch:
  `codex/mobihet-v4-relkd`; artifact commit:
  `0af58af785ba24eb2709999d6773880657bb3ff9`.
- Git LFS object: `artifacts/single-fl-final-artifacts-supplement-2026-11-20.zip`,
  88.537.309 bytes; SHA-256
  `46f4ef982fcee7f42ca8d8a8a196396acee252f4534e0f843170336c6c773051`.
- Scope: 390 missing non-checkpoint files / 97.685.945 source bytes from 132 exact final
  runs in the 4-backbone × 14-method × 3-seed matrix.
- Combined with `runs-no-checkpoints-2026-11-20.zip`: 504/504 required files present
  (one final CM, `config.json`, and `rounds.jsonl` for each of 168 runs).
- Checks: 132/132 configs match model/method/seed; round histories contiguous; every CM has
  5.343 clients × 16 × 16, 9.347.679 test rows per view, and all ten final metrics match;
  ZIP integrity PASS; credential/signed-URL scan PASS; LFS upload 100%.
- No checkpoints included; no local cleanup requested or performed.

## GitHub mixed-model results — 20/11/2026

- Remote: `origin` → `https://github.com/odixe06/experiment.git`; branch:
  `codex/mobihet-v4-relkd`.
- Results commit: `5d2648b100e0cb09b1de3ea463553d827acb8663`.
- Scope: `docs/research/mixed128-all-results.md`, its CSV, and
  `scripts/report_mixed_matrix.py`; no unrelated working-tree changes included.
- Checks: merged final-CM audits cover 45/45 unique case/method/seed tuples; all headline
  views contain ten metrics; generated report and CSV contain no missing configuration.
- No local cleanup requested or performed.

## GitHub run artifacts — 20/11/2026

- Canonical remote: `origin` → `https://github.com/odixe06/experiment.git` (GitHub redirected the old
  `fl_dagsnet_veremi_v2` URL during push).
- Branch: `codex/mobihet-v4-relkd`; artifact commit:
  `de5904a5facd2ba5f7d9966242808a76ded91016`.
- Git LFS object: `artifacts/runs-no-checkpoints-2026-11-20.zip`, 903.962.154 bytes;
  SHA-256 `9e5d8eb4573785f418ff710c38de0af0b238cdac0bcfd31714a871ee42f2c012`.
- Archive contains 5.176 files / 1.112.281.698 source bytes from `runs/`. Excluded: 1.036
  `.pt`/`.pth`/`.ckpt` checkpoints and 95 files matching signed-URL/credential gates.
- Manifest: `artifacts/runs-no-checkpoints-2026-11-20.manifest.json`; packaging helper:
  `scripts/package_runs_without_checkpoints.py`.
- Checks: ZIP integrity PASS; Git LFS upload 100% (904 MB); no local cleanup requested or performed.
- DVC remains unconfigured; this selected artifact was explicitly stored in GitHub LFS.

## GitHub code + docs — 20/11/2026

- Remote: `origin` → `https://github.com/odixe06/fl_dagsnet_veremi_v2.git`.
- Branch: `codex/mobihet-v4-relkd`.
- Source commit: `c398d5693cbad7df9762748d122740a6d6e59d4f`; `git push` thành công và
  `git -c http.sslBackend=schannel ls-remote` trả đúng SHA.
- Scope: 177 file code, tests, experiment assignment và tài liệu; 41.619 dòng thêm.
- Loại khỏi Git: `docs/paper/`, `tmp/`, `output/`, credentials, signed URLs, checkpoints và run artifacts.
- Checks: secret scan PASS (chỉ token giả trong unit test), `compileall` PASS,
  `git diff --cached --check` PASS, 3 test assignment PASS.
- Giới hạn test local: full suite chưa chạy vì Python host thiếu `torch`/`pytest` và không tìm thấy env `nckh`;
  không cài dependency mới trong phiên backup.
- DVC: không cấu hình, không được yêu cầu; không upload artifact và không cleanup.

## Deployment riêng — 15/11/2026

Owner yêu cầu hourly cloud dispatcher; đã tạo private repo
https://github.com/dangvanhoang1711/kaggle-fl-dispatcher . Source commit triển khai
`cba65ac42971c2fdc62fcf8581269e61ede5881f`,65files (58training source frozen + tooling).
Ledger queue có commit riêng ban đầu; workflow sẽ ghi tiếp trạng thái. Secrets mã hóa ngoài source.
Credential scan so với toàn bộ saved Kaggle tokens PASS;13tests local PASS; dry-run cloud SUCCESS43619921537.
Không backup toàn repo nghiên cứu, không đổi Git HEAD/index gốc, không DVC hoặc cleanup.
Chi tiết vận hành: [cloud dispatcher](cloud-dispatcher.md); current status: [continue](continue.md).

## Backup lịch sử

Updated: 2026-10-31, Asia/Bangkok — pushed MobiHet v4/RelKD source snapshot to new GitHub branch.
Procedure: repository `repo-backup` skill.

## 1. Destinations

- Git: `origin` → `codex/mobihet-v4-relkd`; source commit `9257cec2c0b4b0886cd30e5f7aff591d64543bc9` verified with `git ls-remote`.
- DVC: not configured or requested in this operation.
- Cloud artifacts: not part of this operation; local `runs/`, checkpoints, and generated notebooks remain ignored by Git.

## 2. Tracked artifacts

No DVC manifest touched.

## 3. Checks that passed

| Check | Scope | Result |
|---|---|---|
| Python `compileall` | `core`, `methods`, `scripts`, `tests` | PASS |
| Documentation link check | repository Markdown links | PASS |
| `git diff --cached --check` | source snapshot | PASS |
| credential pattern scan | committed source and docs | no match |
| `git push` + `git ls-remote` | `codex/mobihet-v4-relkd` | source SHA matched |

## 4. Local cleanup

- None requested; no files deleted.

## 5. Next session

- Push later commits with `git -c http.sslBackend=schannel push` because local HTTPS uses Windows certificate store.
- Re-run checks if source or docs change.
- Artifact backup or restore requires separate DVC configuration and owner request.
