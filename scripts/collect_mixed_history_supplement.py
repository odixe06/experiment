#!/usr/bin/env python3
"""Collect complete round histories missing from the mixed-model backup."""

from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath

import requests
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "cloud_dispatch"))
from core import metrics
import dispatcher as d


STAGING = ROOT / "runs/mixed-history-supplement"
ARCHIVE = ROOT / "artifacts/mixed-history-supplement-2026-11-20.zip"
MANIFEST = ROOT / "artifacts/mixed-history-supplement-2026-11-20.manifest.json"
CSV = ROOT / "artifacts/mixed-history-macro-f1-2026-11-20.csv"
AUDIT_SOURCES = (
    "mixed128-final-audit.json",
    "mixed128-v2x-repair-audit.json",
    "mixed128-case1-final-audit.json",
    "mixed128-case3-final-audit.json",
    "mixed128-resume-final-audit.json",
    "mixed128-final-audit-20261119.json",
    "mixed128-final-audit-refresh.json",
)
TARGETS = {
    ("case2", "v2x_htfl", 41),
    ("case2", "v2x_htfl", 42),
    ("case2", "v2x_htfl", 43),
    ("case3", "fedtgp", 41),
    ("case3", "tinyproto", 41),
    ("case3", "tinyproto", 43),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_audits() -> dict[tuple[str, str, int], dict[str, object]]:
    selected = {}
    for name in AUDIT_SOURCES:
        path = ROOT / "runs" / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = (row["case"], row["method"], int(row["seed"]))
            if key in TARGETS and row.get("audit") == "PASS":
                selected[key] = row
    if set(selected) != TARGETS:
        raise RuntimeError(f"missing final audits: {sorted(TARGETS - set(selected))}")
    return selected


def remote_listing(owner: str, slug: str, version: int, token: str) -> dict[str, str]:
    api = d.make_api(token)
    files: dict[str, str] = {}
    page_token = None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.version_label = f"v{version}"
            request.page_size = 200
            if page_token:
                request.page_token = page_token
            reply = d.read_call(
                lambda: client.kernels.kernels_api_client.list_kernel_session_output(request)
            )
            files.update({item.file_name: item.url for item in reply.files or []})
            page_token = reply.next_page_token
            if not page_token:
                return files


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    os.replace(partial, destination)


def collect(item: tuple[tuple[str, str, int], dict[str, object]]) -> dict[str, object]:
    (case, method, seed), audit = item
    owner, slug = audit["ref"].split("/", 1)
    version = int(audit["version"])
    token = (Path.home() / ".kaggle" / "accounts" / f"{owner}.mcp-token").read_text().strip()
    listing = remote_listing(owner, slug, version, token)
    candidates = [
        name
        for name in listing
        if PurePosixPath(name).name == "rounds.jsonl"
        and "smoke" not in {part.lower() for part in PurePosixPath(name).parts}
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"{audit['ref']}/v{version}: history candidates {candidates}")

    destination = STAGING / case / method / f"seed{seed}" / "rounds.jsonl"
    download(listing[candidates[0]], destination)
    rows = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rounds = [int(row["round"]) for row in rows]
    if rounds != list(range(1, 51)):
        raise AssertionError((case, method, seed, rounds))
    for row in rows:
        if any(
            not math.isfinite(value)
            for value in row.values()
            if isinstance(value, (int, float))
        ):
            raise AssertionError((case, method, seed, row["round"], "non-finite"))
        for view in ("client", "client_proto"):
            if f"{view}/f1_macro" not in row:
                raise AssertionError((case, method, seed, row["round"], view))

    final = rows[-1]
    for view in ("client", "client_proto"):
        expected = audit["views"][view]
        for key in metrics.TEN:
            actual = final[f"{view}/{key}"]
            if not math.isclose(actual, expected[key], rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError((case, method, seed, view, key, actual, expected[key]))

    return {
        "case": case,
        "method": method,
        "seed": seed,
        "ref": audit["ref"],
        "version": version,
        "remote_path": candidates[0],
        "archive_path": destination.relative_to(STAGING).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "rounds": 50,
        "views": ["client", "client_proto"],
        "final_f1_macro": {
            "client": final["client/f1_macro"],
            "client_proto": final["client_proto/f1_macro"],
        },
    }


def main() -> None:
    audits = final_audits()
    results = []
    errors = []
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(collect, item): item[0] for item in sorted(audits.items())}
        for future in cf.as_completed(futures):
            key = futures[future]
            try:
                results.append(future.result())
                print("PASS", *key, flush=True)
            except Exception as exc:
                errors.append({"key": key, "error": type(exc).__name__})
    if errors:
        print(json.dumps(errors, indent=2), flush=True)
        raise RuntimeError(f"failed histories: {len(errors)}")

    results.sort(key=lambda row: (row["case"], row["method"], row["seed"]))
    csv_rows = []
    for result in results:
        path = STAGING / result["archive_path"]
        for row in [json.loads(line) for line in path.read_text().splitlines() if line.strip()]:
            csv_rows.append({
                "case": result["case"],
                "method": result["method"],
                "seed": result["seed"],
                "round": row["round"],
                "client_f1_macro": f"{row['client/f1_macro']:.15g}",
                "client_proto_f1_macro": f"{row['client_proto/f1_macro']:.15g}",
            })
    with CSV.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    manifest = {
        "format": 1,
        "scope": "complete R1-R50 histories absent from the prior mixed-model backup",
        "runs": len(results),
        "round_rows": len(csv_rows),
        "history_bytes": sum(row["bytes"] for row in results),
        "checks": [
            "exact final kernel/version",
            "non-smoke rounds.jsonl",
            "contiguous unique rounds 1..50",
            "client and client_proto macro-F1 at every round",
            "all ten round-50 metrics match final confusion-matrix audit",
        ],
        "results": results,
        "csv": {"path": CSV.name, "rows": len(csv_rows), "sha256": sha256(CSV)},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        bundle.write(MANIFEST, "MANIFEST.json")
        bundle.write(CSV, CSV.name)
        for result in results:
            source = STAGING / result["archive_path"]
            bundle.write(source, f"mixed-history-supplement/{result['archive_path']}")
    with zipfile.ZipFile(ARCHIVE) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt ZIP entry: {bad}")

    print(json.dumps({
        "runs": len(results),
        "round_rows": len(csv_rows),
        "history_bytes": manifest["history_bytes"],
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "zip_test": "PASS",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
