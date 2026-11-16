"""Check collected baseline source against frozen plan commit; never alters training artifacts."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/maintenance-20261116'
records = json.loads((OUT / 'kernel-inventory.json').read_text())
expected = {}
checked = []
for x in records:
    if x['status'] != 'COMPLETE':
        continue
    r = x['row']
    batch = Path(x['out'])
    assert (batch / (r['slug'] + '.done')).exists(), r['slug']
    sha = json.loads((batch / 'plan.json').read_text())['head']
    manifests = list((ROOT / 'runs/pulls' / r['slug']).rglob('source_manifest.json'))
    assert len(manifests) == 1, r['slug']
    p = manifests[0]
    spec = json.loads(p.read_text())
    assert spec['base_sha'] == sha and not spec['overlay'], r['slug']
    for name, digest in spec['files'].items():
        key = (sha, name)
        if key not in expected:
            expected[key] = hashlib.sha256(subprocess.check_output(['git', 'show', f'{sha}:{name}'], cwd=ROOT)).hexdigest()
        assert digest == expected[key], (r['slug'], name)
        assert hashlib.sha256((p.parent / 'source' / name).read_bytes()).hexdigest() == digest, (r['slug'], name)
    checked.append(dict(kernel=f"{r['account']}/{r['slug']}/{r['version']}", files=len(spec['files']), base_sha=sha))
(OUT / 'baseline-provenance.json').write_text(json.dumps(dict(checked=checked, status='PASS'), indent=2))
print('BASELINE_SOURCE_PROVENANCE_PASS', len(checked))
