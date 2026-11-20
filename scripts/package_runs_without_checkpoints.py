#!/usr/bin/env python3
"""Package run artifacts while excluding checkpoints and credential-like files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SENSITIVE_NAME = re.compile(
    r"(?:^|[-_])(output[-_]?urls?|checkpoint[-_]?urls?|credentials?|kaggle)(?:[-_.]|$)",
    re.IGNORECASE,
)
SENSITIVE_CONTENT = re.compile(
    rb"kaggleusercontent\.com|KGAT_[A-Za-z0-9_-]{12,}|github_pat_[A-Za-z0-9_]+|"
    rb"ghp_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{12,}|"
    rb"WANDB_API_KEY|wandb\.login\s*\(\s*key\s*=|"
    rb"(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[\"']?\s*[:=]\s*[\"'][^\"']{8,}",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusion_reason(path: Path) -> str | None:
    if path.suffix.lower() in CHECKPOINT_SUFFIXES:
        return "checkpoint"
    if SENSITIVE_NAME.search(path.name):
        return "sensitive_filename"
    if path.suffix.lower() in TEXT_SUFFIXES:
        try:
            if SENSITIVE_CONTENT.search(path.read_bytes()):
                return "sensitive_content"
        except OSError:
            return "unreadable"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    runs = args.runs.resolve()
    archive = args.archive.resolve()
    manifest_path = args.manifest.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    included: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    candidates = sorted(path for path in runs.rglob("*") if path.is_file())

    for path in candidates:
        relative = path.relative_to(runs).as_posix()
        reason = exclusion_reason(path)
        if reason:
            excluded.append({"path": relative, "bytes": path.stat().st_size, "reason": reason})
            continue
        included.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})

    manifest = {
        "format": 1,
        "source": "runs/",
        "included_files": len(included),
        "included_bytes": sum(int(item["bytes"]) for item in included),
        "excluded_files": len(excluded),
        "excluded_bytes": sum(int(item["bytes"]) for item in excluded),
        "included": included,
        "excluded": excluded,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        bundle.write(manifest_path, "MANIFEST.json")
        for item in included:
            bundle.write(runs / str(item["path"]), f"runs/{item['path']}")

    archive_hash = sha256(archive)
    print(json.dumps({
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
        "manifest": str(manifest_path),
        "included_files": len(included),
        "excluded_files": len(excluded),
    }, indent=2))


if __name__ == "__main__":
    main()
