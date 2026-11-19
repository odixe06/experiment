"""CPU-only support audit; embed exact previously audited CMs, never checkpoints."""
import ast
import base64
import hashlib
import json
import io
from pathlib import Path

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SLUG = 'mobihet-v4-stats-client-diagnostic'


def main():
    out = ROOT/'notebooks'/SLUG
    out.mkdir(parents=True, exist_ok=True)
    source = {name: (ROOT/name).read_text(encoding='utf-8') for name in
              ['scripts/analyze_stats_clients.py', 'core/metrics.py', 'core/__init__.py']}
    pull = ROOT/'runs/pulls/mobihet-v4-stats-nogauss-dangthiet-v1'
    files = {'control': pull/'mobihet-v4-control-realbn-val-r10/cms_r010.npz',
             'nogauss': pull/'mobihet-v4-stats-nogauss-val-r10/cms_r010.npz'}
    assert json.loads((pull/'CM_AUDIT.json').read_text())['metrics'] == 'PASS'
    payload, hashes = {}, {}
    original_hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()}
    for k,p in files.items():
        with np.load(p) as z:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, ids=z['ids'], private=z['private'])
        raw = buffer.getvalue()
        payload[k] = base64.b64encode(raw).decode()
        hashes[k] = hashlib.sha256(raw).hexdigest()
    code = f'''import base64, hashlib, json, sys
from pathlib import Path
source = {source!r}
payload = {payload!r}
hashes = {hashes!r}
original_hashes = {original_hashes!r}
work = Path('/kaggle/working')
src = Path('/kaggle/temp/diagnostic')
for name, content in source.items():
    p = src/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
evidence = work/'evidence'
evidence.mkdir(exist_ok=True)
for name, encoded in payload.items():
    raw = base64.b64decode(encoded)
    assert hashlib.sha256(raw).hexdigest() == hashes[name]
    (evidence/(name+'.npz')).write_bytes(raw)
(work/'source_snapshot.json').write_text(json.dumps(source))
(work/'original_cm_hashes.json').write_text(json.dumps(original_hashes))
(work/'source_manifest.json').write_text(json.dumps({{k:hashlib.sha256(v.encode()).hexdigest() for k,v in source.items()}}))
sys.path.insert(0,str(src))
from scripts.analyze_stats_clients import analyze
roots = [p.parent for p in Path('/kaggle/input').rglob('label_mapping.json') if (p.parent/'train').is_dir()]
assert len(roots)==1, roots
analyze(roots[0], evidence, work/'analysis')
'''
    ast.parse(code)
    for name, content in source.items(): ast.parse(content, filename=name)
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        '# Train-support diagnostic\nCPU only. No training, no test split, no W&B. '
        'Scan train-fold counts and verify full validation counts against embedded audited R10 confusion matrices. '
        'Bins fixed before scanning; descriptive results do not prove causal effects.'), nbformat.v4.new_code_cell(code)])
    nb.metadata.update(kernelspec={'name':'python3','display_name':'Python 3','language':'python'},
        kaggle={'isGpuEnabled':False,'isInternetEnabled':True,'language':'python','sourceType':'notebook'})
    nbformat.validate(nb)
    path=out/(SLUG+'.ipynb'); nbformat.write(nb,path)
    meta=json.loads((ROOT/'notebooks/mobihet-v4-stats-prior-val-r10/kernel-metadata.json').read_text())
    meta.update(id='dangthiet/'+SLUG, title=SLUG, code_file=path.name, enable_gpu=False)
    meta.pop('machine_shape')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print('SHA256',hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
