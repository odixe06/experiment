#!/usr/bin/env python3
"""Collect and audit missing non-checkpoint artifacts for the 168 final single-FL runs."""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import requests
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "cloud_dispatch"))
from core import metrics
import dispatcher as d


SIZE_INVENTORY = ROOT / "runs/missing-single-fl-artifact-size-inventory.json"
STAGING = ROOT / "runs/single-fl-artifact-supplement"
ARCHIVE = ROOT / "artifacts/single-fl-final-artifacts-supplement-2026-11-20.zip"
MANIFEST = ROOT / "artifacts/single-fl-final-artifacts-supplement-2026-11-20.manifest.json"
BASE_ARCHIVE = ROOT / "artifacts/runs-no-checkpoints-2026-11-20.zip"
TEST_ROWS = 9_347_679


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def download(session: requests.Session, url: str, destination: Path, expected: int) -> None:
    if destination.exists() and destination.stat().st_size == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with session.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    if partial.stat().st_size != expected:
        raise RuntimeError(
            f"size mismatch for {destination.name}: {partial.stat().st_size} != {expected}"
        )
    os.replace(partial, destination)


def materialize_from_base_archive(remote_path: str, destination: Path) -> None:
    suffix = "/" + remote_path
    with zipfile.ZipFile(BASE_ARCHIVE) as bundle:
        matches = [name for name in bundle.namelist() if name == remote_path or name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"base archive matches for {remote_path}: {matches}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".base.part")
        with bundle.open(matches[0]) as source, partial.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        os.replace(partial, destination)


def audit(row: dict[str, object], folder: Path) -> dict[str, object]:
    model, method, seed = row["identity"]
    target = 10 if method == "canperfl" else 50
    cm_name = f"cms_r{target:03d}.npz"
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    history = [
        json.loads(line)
        for line in (folder / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if config["method"] != method:
        raise AssertionError(("method", config["method"], method))
    if config["cfg"]["model"] != model:
        raise AssertionError(("model", config["cfg"]["model"], model))
    if int(config["hp"]["seed"]) != int(seed):
        raise AssertionError(("seed", config["hp"]["seed"], seed))
    if [item["round"] for item in history] != list(range(1, target + 1)):
        raise AssertionError("non-contiguous round history")
    if any(
        not math.isfinite(value)
        for item in history
        for value in item.values()
        if isinstance(value, (int, float))
    ):
        raise AssertionError("non-finite history value")

    final = history[-1]
    checked_views = []
    with np.load(folder / cm_name, allow_pickle=False) as payload:
        ids = payload["ids"]
        if ids.shape != (5343,) or len(np.unique(ids)) != 5343:
            raise AssertionError("invalid client IDs")
        for view in sorted(set(payload.files) - {"ids"}):
            cms = payload[view]
            if cms.shape != (5343, 16, 16):
                raise AssertionError((view, cms.shape))
            if not np.issubdtype(cms.dtype, np.integer) or (cms < 0).any():
                raise AssertionError((view, "invalid confusion dtype/count"))
            if int(cms.sum()) != TEST_ROWS:
                raise AssertionError((view, int(cms.sum()), TEST_ROWS))
            summary = metrics.summary(cms)
            for key in metrics.TEN:
                recorded = final[f"{view}/{key}"]
                if not math.isclose(summary[key], recorded, rel_tol=1e-12, abs_tol=1e-12):
                    raise AssertionError((view, key, summary[key], recorded))
            checked_views.append(view)

    return {
        "target_round": target,
        "history_rows": len(history),
        "views": checked_views,
        "clients": 5343,
        "test_rows_per_view": TEST_ROWS,
    }


def collect(row: dict[str, object]) -> dict[str, object]:
    model, method, seed = row["identity"]
    owner, slug = row["ref"].split("/", 1)
    token = (Path.home() / ".kaggle" / "accounts" / f"{owner}.mcp-token").read_text().strip()
    listing = remote_listing(owner, slug, int(row["version"]), token)
    folder = STAGING / model / method / f"seed{seed}"
    target = 10 if method == "canperfl" else 50
    run_dir = str(PurePosixPath(next(iter(row["files"]))).parent)
    required = [
        f"{run_dir}/cms_r{target:03d}.npz",
        f"{run_dir}/config.json",
        f"{run_dir}/rounds.jsonl",
    ]

    files = []
    with requests.Session() as session:
        for remote_path in required:
            destination = folder / PurePosixPath(remote_path).name
            if remote_path not in row["files"]:
                materialize_from_base_archive(remote_path, destination)
                continue
            if remote_path not in listing:
                raise RuntimeError(f"remote missing {remote_path}")
            expected_size = int(row["files"][remote_path])
            download(session, listing[remote_path], destination, int(expected_size))
            files.append({
                "remote_path": remote_path,
                "archive_path": destination.relative_to(STAGING).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            })

    evidence = audit(row, folder)
    return {
        "model": model,
        "method": method,
        "seed": seed,
        "ref": row["ref"],
        "version": row["version"],
        "files": files,
        "audit": evidence,
    }


def main() -> None:
    inventory = json.loads(SIZE_INVENTORY.read_text(encoding="utf-8"))
    candidates = [row for row in inventory["rows"] if row["files"]]
    if inventory["summary"]["errors"] or len(candidates) != 132:
        raise RuntimeError("size inventory is incomplete")

    STAGING.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []
    with cf.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(collect, row): row for row in candidates}
        for index, future in enumerate(cf.as_completed(futures), 1):
            source = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({
                    "identity": source["identity"],
                    "ref": source["ref"],
                    "version": source["version"],
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if index % 20 == 0:
                print(f"COLLECTED {index}/132", flush=True)
    if errors:
        (STAGING / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(json.dumps(errors[:10], indent=2), flush=True)
        raise RuntimeError(f"collection/audit failures: {len(errors)}")

    results.sort(key=lambda item: (item["model"], item["method"], item["seed"]))
    files = [item for result in results for item in result["files"]]
    manifest = {
        "format": 1,
        "scope": "missing non-checkpoint artifacts for exact final 4-backbone x 14-method x 3-seed runs",
        "source_inventory": "runs/four-backbone-final-checkpoint-size-inventory.json",
        "runs": len(results),
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "audited": len(results),
        "results": results,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        bundle.write(MANIFEST, "MANIFEST.json")
        for item in files:
            source = STAGING / item["archive_path"]
            bundle.write(source, f"single-fl-artifact-supplement/{item['archive_path']}")
    with zipfile.ZipFile(ARCHIVE) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt ZIP entry: {bad}")

    print(json.dumps({
        "runs": len(results),
        "files": len(files),
        "source_bytes": manifest["bytes"],
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "zip_test": "PASS",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
