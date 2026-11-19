"""Controlled precision diagnostic on seed42; no full training or test tuning."""
import ast
import json
import subprocess
import sys
from pathlib import Path
import nbformat

slug = 'tabm-v2x-precision-probe-s42'
owner = 'haivohoanggg'
subprocess.run([sys.executable, 'scripts/build_tabm_fl.py', '--owner', owner,
                '--method', 'v2x_htfl', '--seed', '42', '--slug', slug], check=True)
p = Path('notebooks') / slug / (slug + '.ipynb')
nb = nbformat.read(p, as_version=4)
probe = '''results = {}
for name, amp, scale, compiled in [
    ('eager_amp4096', True, 4096, False),
    ('eager_amp1', True, 1, False),
    ('eager_fp32', False, 1, False),
    ('compiled_amp1', True, 1, True)]:
    cmd = ['torchrun','--standalone','--nproc_per_node=2','scripts/run_fl.py',
           '--method','v2x_htfl','--backbone','tabm','--rounds','50',
           '--stop-after','1','--subset','32','--train-fold','train','--eval-fold','val',
           '--cache','','--out','/kaggle/working/'+name,
           '--set','seed=42','server_epochs=1','amp='+str(amp).lower(),'loss_scale='+str(scale)]
    if compiled:
        cmd.append('--compile')
    subprocess.run(cmd,cwd='/kaggle/temp/src',check=True)
    row=json.loads(Path('/kaggle/working/'+name+'/rounds.jsonl').read_text().splitlines()[-1])
    results[name]={k:row[k] for k in ('train_loss','applied_steps','skipped_steps','skipped_frac')}
    print('PRECISION_RESULT',name,results[name],flush=True)
    Path('/kaggle/working/PRECISION.json').write_text(json.dumps(results,indent=2))
'''
nb.cells = nb.cells[:4] + [nbformat.v4.new_code_cell(probe)]
for c in nb.cells:
    if c.cell_type == 'code':
        ast.parse(c.source)
nbformat.write(nb, p)
print(p)
