"""Freeze only the authorized training source into a separate deployment directory."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp/kaggle-fl-dispatcher'
METHODS = ['fedavg', 'fedprox', 'afpha', 'fd_ids', 'apfed', 'lwfednids', 'tinyproto',
           'pfedes', 'nilm', 'fedtgp', 'perfedskd', 'fedproto', 'v2x_htfl', 'canperfl']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [*ROOT.glob('core/*.py'), *ROOT.glob('methods/*.py'), *ROOT.glob('knowledge/*.json'),
             ROOT/'scripts/run_fl.py', ROOT/'scripts/build_tabm_fl.py',
             ROOT/'tests/test_sae_fl.py', ROOT/'tests/test_dense_recurrent_fl.py']
    hashes = {}
    for path in files:
        rel = path.relative_to(ROOT)
        dst = OUT/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Match builder's UTF-8/text reading to retain its already-tested source snapshot.
        content = path.read_text(encoding='utf-8')
        if path.suffix == '.py':
            ast.parse(content)
        dst.write_text(content, encoding='utf-8', newline='\n')
        hashes[rel.as_posix()] = hashlib.sha256(content.encode()).hexdigest()
    for name in ('dispatcher.py', 'test_dispatcher.py', 'README.md', 'requirements.txt'):
        shutil.copyfile(ROOT/'scripts/cloud_dispatch'/name, OUT/name)
    (OUT/'source_manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    workflow = OUT/'.github/workflows/dispatch.yml'
    workflow.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/'scripts/cloud_dispatch/dispatch.yml', workflow)
    (OUT/'.gitignore').write_text('notebooks/\n__pycache__/\n*.tmp\n.env*\n')
    queue = OUT/'queue.json'
    if not queue.exists():
        jobs = [{'slug': f'{b}-fl-{m.replace("_", "-")}-s{s}', 'backbone': b,
                 'method': m, 'seed': s, 'state': 'pending'}
                for b in ('sae', 'mlp', 'cnn-bigru-attention') for m in METHODS for s in (41, 42, 43)]
        queue.write_text(json.dumps({'jobs': jobs}, indent=2)+'\n')
    print('Frozen', len(hashes), 'source files;', len(json.loads(queue.read_text())['jobs']), 'queue entries')


if __name__ == '__main__':
    main()
