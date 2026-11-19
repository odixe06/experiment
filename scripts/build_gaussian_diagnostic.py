"""Build CPU Gaussian fidelity diagnostic with immutable embedded source."""
import ast
import hashlib
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SLUG = 'mobihet-v4-gaussian-fidelity'


def main():
    source={n:(ROOT/n).read_text(encoding='utf-8') for n in
            ['scripts/diagnose_gaussian.py','core/metrics.py','core/__init__.py','knowledge/input_order.json']}
    code=f'''import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_MODE='disabled')
import hashlib, json, sys
from pathlib import Path
source={source!r}
src=Path('/kaggle/temp/gaussian-diagnostic')
for name,content in source.items():
    p=src/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(content,encoding='utf-8')
work=Path('/kaggle/working')
(work/'source_snapshot.json').write_text(json.dumps(source))
(work/'source_manifest.json').write_text(json.dumps({{n:hashlib.sha256(s.encode()).hexdigest() for n,s in source.items()}}))
sys.path.insert(0,str(src))
from scripts.diagnose_gaussian import main
roots=[p.parent for p in Path('/kaggle/input').rglob('label_mapping.json') if (p.parent/'train').is_dir()]
assert len(roots)==1,roots
order=json.loads(source['knowledge/input_order.json'])['inputs']
main(roots[0],work/'diagnostic',order)
'''
    ast.parse(code)
    for n,s in source.items():
        if n.endswith('.py'): ast.parse(s,filename=n)
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        '# Gaussian fidelity diagnostic\nCPU only; no W&B or official test. Full train moments for classes0/11/13/15. '
        'Fixed sampled domain discrimination and real-vs-synthetic binary transfer, seed42. '
        'Exploratory evidence, not a replacement for paired FL validation. No auto training launch.'),nbformat.v4.new_code_cell(code)])
    nb.metadata.update(kernelspec={'name':'python3','display_name':'Python 3','language':'python'},
        kaggle={'isGpuEnabled':False,'isInternetEnabled':True,'language':'python','sourceType':'notebook'})
    nbformat.validate(nb)
    out=ROOT/'notebooks'/SLUG;out.mkdir(parents=True,exist_ok=True)
    path=out/(SLUG+'.ipynb'); nbformat.write(nb,path)
    meta=json.loads((ROOT/'notebooks/mobihet-v4-stats-client-diagnostic/kernel-metadata.json').read_text())
    meta.update(id='dangthiet/'+SLUG,title=SLUG,code_file=path.name)
    assert not meta['enable_gpu'] and meta['is_private'] and not meta['kernel_sources']
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print('SHA256',hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':
    main()
