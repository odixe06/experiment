"""Read-only, version-pinned collection for the TabM/SAE final-round comparison."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import requests
sys.path.insert(0, str(Path(__file__).resolve().parent/'cloud_dispatch'))
import dispatcher as kd
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/comparison-20261115'


def collect(ref):
    owner, slug = ref.split('/')
    token = (Path.home()/'.kaggle/accounts'/f'{owner}.mcp-token').read_text().strip()
    api = kd.make_api(token)
    meta = kd.kernel_metadata(api, ref)
    version = meta.current_version_number
    status = kd.read_call(lambda: api.kernels_status(ref)).status.name
    result = {'ref':ref, 'version':version, 'remote_status':status,
              'checked_at':datetime.now(timezone.utc).isoformat()}
    if status not in ('COMPLETE', 'ERROR'):
        return result
    assert version > 0
    folder = OUT/f'{owner}__{slug}__v{version}'
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder/'COLLECT.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if all((folder/name).is_file() and hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
               for name,sha in previous['sha256'].items()):
            return {**result, 'folder':str(folder), 'files':len(previous['sha256'])}
    files, page = {}, None
    with api.build_kaggle_client() as client:
        for _ in range(100):
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug, req.page_size = owner, slug, 200
            req.version_label = f'v{version}'
            if page:
                req.page_token = page
            response = kd.read_call(lambda: client.kernels.kernels_api_client.list_kernel_session_output(req))
            for f in response.files or []:
                path = Path(f.file_name)
                if path.is_absolute() or '..' in path.parts:
                    raise RuntimeError('unsafe output path')
                if any(p.startswith('smoke') for p in path.parts):
                    continue
                if re.search(r'(^|/)(RUN_STATUS|experiment_manifest|source_manifest|source_snapshot|config)\.json$|(^|/)rounds\.jsonl$|(^|/)cms_r(050|010)\.npz$', f.file_name):
                    files[f.file_name] = f.url
            page = response.next_page_token
            if not page:
                break
        else:
            raise RuntimeError('incomplete output listing')
    hashes = {}
    for name, url in files.items():
        target = folder/name
        target.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        data = response.content
        expected = response.headers.get('Content-Length')
        if expected and not response.headers.get('Content-Encoding'):
            assert len(data) == int(expected)
        temp = target.with_suffix(target.suffix+'.part')
        temp.write_bytes(data)
        temp.replace(target)
        hashes[name] = hashlib.sha256(data).hexdigest()
    marker.write_text(json.dumps({**result, 'sha256':hashes}, indent=2), encoding='utf-8')
    return {**result, 'folder':str(folder), 'files':len(files)}


def safe_collect(ref):
    try:
        result = collect(ref)
    except Exception as exc:
        result = {'ref':ref, 'error_type':type(exc).__name__,
                  'http':getattr(getattr(exc,'response',None),'status_code',None)}
    print(json.dumps(result), flush=True)
    return result


def main():
    state = json.loads((ROOT/'docs/archive/operations/2026-11-15-cloud-dispatcher/check-1102.json').read_text())['queue']
    tabm = [ref for ref in state['inventory_cache'] if '/tabm-fl-' in ref]
    sae = [j['ref'] for j in state['jobs'] if j['backbone']=='sae' and j['state']=='submitted']
    assert len(tabm)==len(sae)==42 and len(set(tabm+sae))==84
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(safe_collect, sorted(tabm+sae)))
    (OUT/'collection.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('COLLECTED',sum('folder' in r for r in results),'of',len(results))
    return int(any('error_type' in r for r in results))


if __name__=='__main__':
    sys.exit(main())
