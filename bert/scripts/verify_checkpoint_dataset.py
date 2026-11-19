"""Download every extracted checkpoint file and verify SHA256 against its frozen local parent."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch_one(ref, name, size, parent, out, run):
    relative = Path(name)
    assert relative.parts[0] == run and '..' not in relative.parts and not relative.is_absolute()
    local = out / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    # Previous verifier passed the parent twice; retain and normalize its already downloaded files.
    old = local.parent / relative
    if old.exists() and not local.exists():
        os.replace(old, local)
    if not local.exists():
        api = KaggleApi(); api.authenticate()
        api.dataset_download_file(ref, name, path=str(out), quiet=True, unzip=True)
    assert local.is_file() and local.stat().st_size == size, name
    actual = sha256(local)
    assert actual == sha256(parent / Path(*relative.parts[1:])), name
    return dict(name=name, size=size, sha256=actual)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    a = ap.parse_args()
    api = KaggleApi(); api.authenticate()
    files, token = [], None
    while True:
        result = api.dataset_list_files(a.ref, page_size=200, page_token=token)
        files.extend((f.name, int(f.total_bytes)) for f in result.dataset_files)
        token = result.next_page_token
        if not token:
            break
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(lambda item: fetch_one(a.ref, *item, a.parent, a.out, a.parent.name), files))
    assert any(x['name'].endswith('/state_rank1.pt') for x in result)
    a.report.write_text(json.dumps(result, indent=2))
    print('DATASET_FILES_HASH_PASS', len(result), flush=True)


if __name__ == '__main__':
    main()
