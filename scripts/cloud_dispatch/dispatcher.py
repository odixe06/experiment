"""Hourly, finite Kaggle submission queue. Never retries an ambiguous write."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetAcceleratorQuotaStatisticsRequest, ApiListKernelSessionOutputRequest, ApiGetKernelRequest,
)
from kagglesdk.security.types.oauth_service import IntrospectTokenRequest

ROOT = Path(__file__).resolve().parent
ACCOUNTS = ['dangthiet', 'hoangdang1711', 'hoangdangnhi', 'hoangdv1711',
            'minhiw', 'phamquang0606', 'ttalgies', 'qwerasdf3004', 'nguynt298']
TERMINAL = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'NO_BATCH'}
ACTIVE = {'RUNNING', 'QUEUED', 'CANCEL_REQUESTED', 'NEW_SCRIPT', 'UNKNOWN_ACTIVE'}
READ_LOCK = threading.Lock()
LAST_READ = 0.0


def read_call(fn):
    global LAST_READ
    for attempt in range(4):
        with READ_LOCK:
            time.sleep(max(0, 0.7 - (time.monotonic() - LAST_READ)))
            LAST_READ = time.monotonic()
        try:
            return fn()
        except requests.HTTPError as exc:
            if exc.response.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep((3, 10, 30)[attempt])


def make_api(token):
    api = KaggleApi()
    api.build_kaggle_client = lambda: KaggleApi.build_kaggle_client_with_params(args=[], api_token=token)
    return api


def quota(api):
    with api.build_kaggle_client() as client:
        q = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
    return max(0, (q.total_time_allowed - q.time_used - q.time_reserved).total_seconds() / 3600)


def kernel_metadata(api, ref):
    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = ref.split('/')
    with api.build_kaggle_client() as client:
        return read_call(lambda: client.kernels.kernels_api_client.get_kernel(req)).metadata


def inventory(owner, token, cache=None):
    api = make_api(token)
    with api.build_kaggle_client() as client:
        req = IntrospectTokenRequest()
        req.token = token
        who = read_call(lambda: client.security.oauth_client.introspect_token(req))
        if not who.active or who.username != owner:
            raise RuntimeError('identity mismatch for ' + owner)
    records, page_token = {}, None
    seen_pages = set()
    for page in range(1, 51):
        response = read_call(lambda: api.kernels_list_with_response(
            mine=True, page_size=100, page_token=page_token, page=None if page_token else page))
        signature = tuple(item.ref for item in response.kernels or [])
        if signature and signature in seen_pages:
            raise RuntimeError('repeated listing page')
        seen_pages.add(signature)
        for item in response.kernels or []:
            if not item.ref and not item.id and item.enable_gpu is False:
                continue  # Kaggle's inaccessible [Private Notebook] placeholder, not an owned session.
            if item.ref.split('/')[0] != owner:
                raise RuntimeError('listing owner mismatch: ' + item.ref)
            records[item.ref] = {'version': item.current_version_number,
                                 'gpu': True,  # List endpoint omits enable_gpu (SDK defaults False).
                                 'last_run_time': str(item.last_run_time)}
        page_token = response.next_page_token
        if not page_token and len(response.kernels or []) < 100:
            break
    else:
        raise RuntimeError('incomplete listing')
    def read_status(ref):
        old = (cache or {}).get(ref, {})
        rec = records[ref]
        if (old.get('status') in TERMINAL and old.get('version') == rec['version']
                and old.get('last_run_time') == rec['last_run_time']):
            return ref, old['status']
        try:
            status = read_call(lambda: api.kernels_status(ref)).status.name
        except Exception as exc:
            if getattr(getattr(exc, 'response', None), 'status_code', None) == 404:
                meta = kernel_metadata(api, ref)
                if not meta.current_version_number:
                    # Confirmed unpublished draft, not a batch session. Interactive allocation,
                    # if any, is still covered by live quota and Kaggle's authoritative push limit.
                    return ref, 'NO_BATCH'
            print('STATUS_UNAVAILABLE', ref, type(exc).__name__, flush=True)
            raise
        if status not in TERMINAL | ACTIVE:
            raise RuntimeError('unknown session status')
        return ref, status
    with ThreadPoolExecutor(max_workers=2) as pool:
        for ref, status in pool.map(read_status, records):
            records[ref]['status'] = status
    active = sum(r['gpu'] and r['status'] in ACTIVE for r in records.values())
    return {'api': api, 'records': records, 'active': active, 'free_hours': quota(api)}


def checked_inventory(owner, token, cache=None):
    try:
        snapshot = inventory(owner, token, cache)
        print('INVENTORY', owner, len(snapshot['records']), 'kernels',
              snapshot['active'], 'active', round(snapshot['free_hours'], 2), 'hours', flush=True)
        return snapshot
    except Exception as exc:
        print('INVENTORY_FAILED', owner, type(exc).__name__,
              getattr(getattr(exc, 'response', None), 'status_code', None),
              str(exc) if isinstance(exc, RuntimeError) else '', flush=True)
        raise


def parent_ready(api, ref, version, job):
    """Check immutable completed FedAvg output; notebook also asserts state round/config."""
    owner, slug = ref.split('/')
    files, page_token = {}, None
    with api.build_kaggle_client() as client:
        for _ in range(100):
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug, req.page_size = owner, slug, 200
            req.version_label = f'v{version}'
            if page_token:
                req.page_token = page_token
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
            files.update({f.file_name: f.url for f in response.files or []})
            page_token = response.next_page_token
            if not page_token:
                break
        else:
            raise RuntimeError('incomplete parent output listing')
    def read_json(name):
        r = requests.get(files[name], timeout=30)
        r.raise_for_status()
        return r.json()
    required = ['RUN_STATUS.json', 'experiment_manifest.json', f'{slug}/state.pt']
    if not all(name in files for name in required):
        return False
    result, manifest = read_json(required[0]), read_json(required[1])
    return (result.get('complete') is True and result.get('last_round') == 50
            and result.get('all_round_metrics') is True and manifest.get('rounds') == 50
            and manifest.get('method') == 'fedavg' and manifest.get('seed') == job['seed']
            and manifest.get('experiment') == f"single-backbone-{job['backbone']}-baselines")


def reconcile(jobs, records):
    for job in jobs:
        hits = [ref for ref in records if ref.split('/')[1] == job['slug']]
        if len(hits) > 1:
            job.update(state='review', reason='duplicate remote slug')
        elif hits:
            ref = hits[0]
            job.update(state='submitted', ref=ref, version=records[ref]['version'] or job.get('version'),
                       remote_status=records[ref]['status'])
        elif job['state'] == 'reserved':
            job.update(state='review', reason='previous write outcome unknown; manual reconciliation required')


def accepted(response, ref):
    # SDK acknowledgements omit ref/version on some successful writes. They are
    # only a rejection signal; confirm_submission proves actual identity/source/version.
    return (not response.error
            and not any(getattr(response, field) for field in
                        ('invalid_kernel_sources', 'invalid_dataset_sources',
                         'invalid_competition_sources', 'invalid_model_sources')))


def confirm_submission(api, ref, folder):
    """Confirm published source and accelerator independently of the push acknowledgement."""
    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = ref.split('/')
    with api.build_kaggle_client() as client:
        result = read_call(lambda: client.kernels.kernels_api_client.get_kernel(req))
    meta = result.metadata
    local_meta = json.loads((folder/'kernel-metadata.json').read_text())
    local = json.loads((folder/local_meta['code_file']).read_text(encoding='utf-8'))
    remote = json.loads(result.blob.source)
    def cells(nb):
        return [(c['cell_type'], ''.join(c['source']) if isinstance(c['source'], list) else c['source'])
                for c in nb['cells']]
    if (meta.ref != ref or not meta.is_private or not meta.enable_gpu or meta.current_version_number < 1
            or cells(local) != cells(remote)):
        raise RuntimeError('published notebook source or metadata mismatch')
    return meta.current_version_number


def save(state, commit):
    state['checked_at'] = datetime.now(timezone.utc).isoformat()
    path = ROOT / 'queue.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)
    if commit:
        for cmd in (['git', 'add', 'queue.json'],
                    ['git', 'commit', '-m', 'Update Kaggle submission queue'],
                    ['git', 'push', 'origin', 'HEAD:main']):
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def build(job, owner, parent=None):
    for name, expected in json.loads((ROOT / 'source_manifest.json').read_text()).items():
        content = (ROOT / name).read_text(encoding='utf-8').encode()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError('frozen source changed: ' + name)
    cmd = [sys.executable, 'scripts/build_tabm_fl.py', '--backbone', job['backbone'],
           '--method', job['method'], '--seed', str(job['seed']), '--owner', owner,
           '--max-hours', '7']
    if parent:
        cmd += ['--parent', parent]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    folder = ROOT / 'notebooks' / job['slug']
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    assert meta['id'] == owner + '/' + job['slug'] and meta['is_private'] is True
    notebook = folder / meta['code_file']
    job['notebook_sha256'] = hashlib.sha256(notebook.read_bytes()).hexdigest()
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    state = json.loads((ROOT / 'queue.json').read_text())
    tokens = json.loads(os.environ['KAGGLE_TOKENS_JSON'])
    assert set(tokens) == set(ACCOUNTS)
    # Fail closed if any account is unreadable: otherwise duplicate detection is incomplete.
    with ThreadPoolExecutor(max_workers=3) as pool:
        snapshots = dict(zip(ACCOUNTS, pool.map(
            lambda o: checked_inventory(o, tokens[o], state.get('inventory_cache')), ACCOUNTS)))
    records = {ref: value for snapshot in snapshots.values() for ref, value in snapshot['records'].items()}
    state['inventory_cache'] = records
    reconcile(state['jobs'], records)
    state['accounts'] = {owner: {k: snap[k] for k in ('active', 'free_hours')}
                         for owner, snap in snapshots.items()}
    print(json.dumps(state['accounts'], indent=2), flush=True)
    if args.execute:
        save(state, commit=True)
    launched, planned = 0, 0
    for job in state['jobs']:
        if job['state'] != 'pending':
            continue
        parent = None
        candidates = list(snapshots)
        if job['method'] == 'canperfl':
            parent_slug = f"{job['backbone']}-fl-fedavg-s{job['seed']}"
            hits = [ref for ref, rec in records.items() if ref.split('/')[1] == parent_slug
                    and rec['status'] == 'COMPLETE']
            if len(hits) != 1:
                continue
            parent = hits[0]
            owner = parent.split('/')[0]
            version = kernel_metadata(snapshots[owner]['api'], parent).current_version_number
            if not version or not parent_ready(snapshots[owner]['api'], parent, version, job):
                continue
            job['parent_version'] = version
            candidates = [owner]
        candidates = [owner for owner in candidates if snapshots[owner]['active'] < 2
                      and snapshots[owner]['free_hours'] >= 8.25]
        if not candidates:
            continue
        owner = max(candidates, key=lambda o: snapshots[o]['free_hours'])
        snap = snapshots[owner]
        ref = owner + '/' + job['slug']
        if not args.execute:
            print('WOULD_PUSH', ref, flush=True)
            planned += 1
            snap['active'] += 1
            snap['free_hours'] -= 8
            continue
        # Recheck quota immediately before reserving capacity, preserving an 8h session + margin.
        snap['free_hours'] = min(snap['free_hours'], quota(snap['api']))
        if snap['free_hours'] < 8.25:
            continue
        folder = build(job, owner, parent)
        job.update(state='reserved', ref=ref)
        save(state, commit=True)  # Durable intent BEFORE network write, including lost-response cases.
        try:
            response = snap['api'].kernels_push(str(folder), timeout='28800')
        except Exception as exc:
            job.update(state='review', reason='ambiguous push: ' + type(exc).__name__)
            snap['active'] = 2
            save(state, commit=True)
            continue
        if accepted(response, ref):
            try:
                version = confirm_submission(snap['api'], ref, folder)
            except Exception as exc:
                job.update(state='review', reason='post-push verification: '+type(exc).__name__)
                snap['active'] = 2
                save(state, commit=True)
                continue
            job.update(state='submitted', version=version)
            snap['active'] += 1
            snap['free_hours'] -= 8
            launched += 1
            print('SUBMITTED', ref, 'version', version, flush=True)
        elif 'Maximum batch GPU session count' in response.error and not response.version_number:
            job.update(state='pending', reason='Kaggle session capacity reached')
            snap['active'] = 2
        else:
            job.update(state='review', reason='push rejected; inspect source/input/account permissions')
            snap['active'] = 2
        save(state, commit=True)
    counts = {s: sum(j['state'] == s for j in state['jobs'])
              for s in ('pending', 'submitted', 'reserved', 'review')}
    summary = {'counts': counts, 'launched': launched, 'planned': planned,
               'checked_at': datetime.now(timezone.utc).isoformat()}
    print(json.dumps(summary), flush=True)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        Path(os.environ['GITHUB_STEP_SUMMARY']).write_text(
            '# Kaggle dispatcher\n\n```json\n' + json.dumps(summary, indent=2) + '\n```\n')
    if args.execute:
        save(state, commit=True)
        if counts['pending'] == 0 and os.getenv('GITHUB_TOKEN'):
            response = requests.put(
                'https://api.github.com/repos/' + os.environ['GITHUB_REPOSITORY']
                + '/actions/workflows/dispatch.yml/disable',
                headers={'Authorization': 'Bearer ' + os.environ['GITHUB_TOKEN'],
                         'Accept': 'application/vnd.github+json'}, timeout=30)
            response.raise_for_status()
            print('No pending jobs; hourly workflow disabled.', flush=True)
    # Report review-needed jobs as workflow failures; never retry these automatically.
    return 1 if counts['review'] else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        # API exception bodies may contain credential material or signed download URLs.
        print('Dispatcher stopped safely:', type(exc).__name__, flush=True)
        sys.exit(1)
