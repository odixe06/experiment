# Git / DVC backup record

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
