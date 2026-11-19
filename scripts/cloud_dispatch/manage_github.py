"""Owner-operated deployment helper. Credentials stay in memory; never upload this helper."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import requests

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT/'tmp/kaggle-fl-dispatcher'
REPO = 'dangvanhoang1711/kaggle-fl-dispatcher'


def session():
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                            capture_output=True, text=True, check=True,
                            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never'})
    creds = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    api = requests.Session()
    api.headers.update({'Authorization': 'Bearer '+creds['password'],
                        'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
    return api


def call(api, method, path, **kwargs):
    r = api.request(method, 'https://api.github.com'+path, timeout=60, **kwargs)
    if not r.ok:
        raise RuntimeError(f'GitHub {method} {path}: HTTP {r.status_code}')
    return r.json() if r.content else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['inspect', 'deploy', 'secrets', 'dry-run', 'execute', 'runs', 'enable', 'disable', 'logs'])
    parser.add_argument('--run-id')
    args = parser.parse_args()
    api = session()
    who = call(api, 'GET', '/user')['login']
    assert who == REPO.split('/')[0]
    prefix = '/repos/'+REPO
    if args.operation == 'inspect':
        r = api.get('https://api.github.com'+prefix, timeout=30)
        print('repo_http', r.status_code)
        if r.ok:
            print({k:r.json().get(k) for k in ('full_name','private','default_branch')})
        for action, version in [('checkout','v4'), ('setup-python','v5')]:
            obj = call(api, 'GET', f'/repos/actions/{action}/git/ref/tags/{version}')['object']
            print(action, version, obj['sha'])
    elif args.operation == 'deploy':
        r = api.get('https://api.github.com'+prefix, timeout=30)
        if r.status_code == 404:
            repo = call(api, 'POST', '/user/repos', json={'name': REPO.split('/')[1], 'private': True,
                        'auto_init': True, 'description': 'Private hourly Kaggle FL submission dispatcher'})
        else:
            r.raise_for_status()
            repo = r.json()
        assert repo['private'] and repo['default_branch'] == 'main'
        head = call(api, 'GET', prefix+'/git/ref/heads/main')['object']['sha']
        tree_sha = call(api, 'GET', prefix+'/git/commits/'+head)['tree']['sha']
        paths = [*PACKAGE.glob('core/*.py'), *PACKAGE.glob('methods/*.py'), *PACKAGE.glob('knowledge/*.json'),
                 *PACKAGE.glob('scripts/*.py'), *PACKAGE.glob('tests/*.py'),
                 PACKAGE/'dispatcher.py', PACKAGE/'test_dispatcher.py', PACKAGE/'README.md',
                 PACKAGE/'requirements.txt', PACKAGE/'.gitignore', PACKAGE/'source_manifest.json',
                 PACKAGE/'.github/workflows/dispatch.yml']
        # Queue is durable remote state; include it only in the initial source deployment.
        q = api.get('https://api.github.com'+prefix+'/contents/queue.json', timeout=30)
        if q.status_code == 404:
            paths.append(PACKAGE/'queue.json')
        else:
            q.raise_for_status()
        tokens = [p.read_text().strip() for p in (Path.home()/'.kaggle/accounts').glob('*.mcp-token')]
        entries = []
        for path in paths:
            content = path.read_text(encoding='utf-8')
            assert all(not t or t not in content for t in tokens), 'credential scan failed'
            entries.append({'path': path.relative_to(PACKAGE).as_posix(), 'mode':'100644',
                            'type':'blob', 'content':content})
        tree = call(api, 'POST', prefix+'/git/trees', json={'base_tree':tree_sha,'tree':entries})['sha']
        commit = call(api, 'POST', prefix+'/git/commits', json={'message':'Deploy hourly Kaggle submission dispatcher',
                      'tree':tree,'parents':[head]})['sha']
        call(api, 'PATCH', prefix+'/git/refs/heads/main', json={'sha':commit,'force':False})
        print('DEPLOYED', REPO, commit, len(entries), 'files; private=true')
    elif args.operation == 'secrets':
        sys.path.insert(0, str(ROOT/'tmp/cloud-deploy-runtime'))
        from nacl.public import PublicKey, SealedBox
        from dispatcher import ACCOUNTS
        tokens = {u:(Path.home()/'.kaggle/accounts'/f'{u}.mcp-token').read_text().strip() for u in ACCOUNTS}
        public = call(api, 'GET', prefix+'/actions/secrets/public-key')
        encrypted = SealedBox(PublicKey(base64.b64decode(public['key']))).encrypt(json.dumps(tokens).encode())
        call(api, 'PUT', prefix+'/actions/secrets/KAGGLE_TOKENS_JSON', json={
             'key_id':public['key_id'], 'encrypted_value':base64.b64encode(encrypted).decode()})
        print('Secret stored encrypted for', len(tokens), 'accounts')
    elif args.operation in ('dry-run','execute'):
        call(api, 'POST', prefix+'/actions/workflows/dispatch.yml/dispatches',
             json={'ref':'main','inputs':{'execute':str(args.operation=='execute').lower()}})
        print('DISPATCHED', args.operation)
    elif args.operation in ('enable','disable'):
        value = str(args.operation=='enable').lower()
        path = prefix+'/actions/variables/DISPATCH_ENABLED'
        r = api.get('https://api.github.com'+path, timeout=30)
        if r.status_code == 404:
            call(api, 'POST', prefix+'/actions/variables', json={'name':'DISPATCH_ENABLED','value':value})
        else:
            r.raise_for_status()
            call(api, 'PATCH', path, json={'name':'DISPATCH_ENABLED','value':value})
        print('DISPATCH_ENABLED', value)
    elif args.operation == 'runs':
        runs = call(api, 'GET', prefix+'/actions/runs?per_page=5')['workflow_runs']
        for r in runs:
            print({k:r.get(k) for k in ('id','event','status','conclusion','html_url','created_at')})
        q = call(api,'GET',prefix+'/contents/queue.json')
        state = json.loads(base64.b64decode(q['content']))
        print('QUEUE', {s:sum(j['state']==s for j in state['jobs']) for s in ('pending','submitted','reserved','review')})
        print('CHECKED_AT',state.get('checked_at'))
    elif args.operation == 'logs':
        import io, zipfile
        response = api.get('https://api.github.com'+prefix+'/actions/runs/'+args.run_id+'/logs', timeout=60)
        response.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(response.content))
        # GitHub already masks secrets; only emit operational lines, never raw archives.
        for name in z.namelist():
            if name.endswith('.txt'):
                for line in z.read(name).decode(errors='replace').splitlines():
                    if any(mark in line for mark in ('INVENTORY', 'WOULD_PUSH', 'SUBMITTED', 'counts',
                                                     'stopped safely', 'STATUS_UNAVAILABLE', 'Error:', 'FAILED', 'OK')):
                        if 'KAGGLE_TOKENS' not in line:
                            print(line[:500])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('Deployment stopped:', type(exc).__name__, str(exc) if isinstance(exc, RuntimeError) else '')
        sys.exit(1)
