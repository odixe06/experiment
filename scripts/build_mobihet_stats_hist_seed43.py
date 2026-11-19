"""Seed43 replication using the exact audited seed42 histogram source snapshot."""
import ast
import hashlib
import json

import nbformat
import build_mobihet_stats_realbn_val as builder


if __name__ == '__main__':
    builder.template.OWNER = 'dangthiet'
    builder.SLUG = 'mobihet-v4-stats-hist-s43-r10'
    builder.CANDIDATE = 'mobihet_stats_hist'
    builder.DESCRIPTION = (
        'Seed43 replication of frozen histogram256 sampler versus original Stats. '
        'Exact source snapshot from audited seed42 histogram run; only run seed changes. '
        'Same train/validation, 5343 clients, R10 with cosine horizon50, fixed quality/cost gate. '
        'No missingness projection. Report alongside BOTH prior seed42 measurements. '
        'New seed checks stochastic robustness on the same validation data, not a new holdout. '
        'No automatic R50 launch, no W&B.'
    )
    builder.main()
    root = builder.template.ROOT
    evidence = root/'runs/pulls/mobihet-v4-stats-hist-dangthiet-v1'
    source = json.loads((evidence/'source_snapshot.json').read_text())
    hashes = json.loads((evidence/'source_manifest.json').read_text())
    assert all(hashlib.sha256(source[k].encode()).hexdigest()==v for k,v in hashes.items())
    assert json.loads((evidence/'CM_AUDIT.json').read_text())['metrics']=='PASS'
    path = root/'notebooks'/builder.SLUG/(builder.SLUG+'.ipynb')
    nb = nbformat.read(path,4)
    lines=nb.cells[2].source.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('source = '): lines[i]='source = '+repr(source)
        if line.startswith('hashes = '): lines[i]='hashes = '+repr(hashes)
    nb.cells[2].source='\n'.join(lines)
    for i in (5,6):
        assert '"--rounds", "50",' in nb.cells[i].source
        nb.cells[i].source=nb.cells[i].source.replace('"--rounds", "50",','"--rounds", "50", "--set", "seed=43",')
    nb.cells[5].source += '\nassert state["round"] == 2\nassert json.loads(Path("/kaggle/working/smoke-stats/config.json").read_text())["hp"]["seed"] == 43\n'
    nb.cells[7].source += '''
for run in ('mobihet-v4-control-realbn-val-r10', 'mobihet-v4-stats-hist-s43-r10'):
    cfg = json.loads(Path(f'/kaggle/working/{run}/config.json').read_text())
    assert cfg['hp']['seed'] == 43 and cfg['hp']['rounds'] == 50
    assert cfg['hp']['train_fold'] == 'train' and cfg['hp']['eval_fold'] == 'val'
gate['seed'] = 43
Path('/kaggle/working/STATS_GATE.json').write_text(json.dumps(gate, indent=2))
'''
    for c in nb.cells:
        if c.cell_type=='code': ast.parse(c.source)
    nbformat.validate(nb)
    nbformat.write(nb,path)
    print('FROZEN SEED43 SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
