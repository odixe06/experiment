"""Build original-source checkpoint replay on the failed kernel's account."""
import ast
import argparse
import json
from pathlib import Path
import nbformat
from build_mixed_fl import IMAGE, DATASET

parser=argparse.ArgumentParser()
parser.add_argument('--owner',default='haivohoanggg')
owner=parser.parse_args().owner
parent='mix128-case2-v2x-htfl-s42-a1'
slug='mix128-case2-v2x-s42-precision-probe'
wrapper=Path('scripts/probe_mixed_checkpoint.py').read_text()
urls=json.loads(Path('runs/mixed128-checkpoint-urls.json').read_text())
cells=[f'''import os,sys,json,hashlib,shutil,subprocess
from pathlib import Path
import torch
assert torch.cuda.device_count()==2
assert all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
os.environ.update(OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',PYTHONUNBUFFERED='1',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',WANDB_MODE='disabled',PYTHONPATH='/kaggle/temp/src')
import requests
urls={urls!r}
for name,url in urls.items():
    target=Path('/kaggle/temp/parent')/name
    target.parent.mkdir(parents=True,exist_ok=True)
    with requests.get(url,stream=True,timeout=120) as response:
        response.raise_for_status()
        with target.open('wb') as handle:
            for chunk in response.iter_content(1<<20):handle.write(chunk)
        assert target.stat().st_size==int(response.headers['Content-Length'])
    print('RESTORED',name,target.stat().st_size,flush=True)
snapshots=list(Path('/kaggle/temp/parent').glob('**/source_snapshot.json'))
assert len(snapshots)==1,snapshots
src=snapshots[0].parent
source=json.loads(snapshots[0].read_text())
hashes=json.loads((src/'source_manifest.json').read_text())
for name,body in source.items():
    assert hashlib.sha256(body.encode()).hexdigest()==hashes[name]
    p=Path('/kaggle/temp/src')/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
Path('/kaggle/temp/src/scripts/probe_mixed_checkpoint.py').write_text({wrapper!r})
old=src/{parent!r}
receipt=json.loads((old/'checkpoint.json').read_text());assert receipt['round']==15,receipt
out=Path('/kaggle/working/replay');out.mkdir()
for name in ('checkpoint.json','rounds.jsonl'):
    shutil.copyfile(old/name,out/name)
folder=out/'checkpoint-r015';folder.mkdir()
for rank in (0,1):
    name=f'state_rank{{rank}}.pt';shutil.copyfile(old/'checkpoint-r015'/name,folder/name)
sys.path.insert(0,'/kaggle/temp/src')
from core import data as D
D.build_npy(D.find_root(),'/kaggle/temp/v4npy',splits=('train','test'))
os.environ['V4_NPY']='/kaggle/temp/v4npy'
''', '''cmd=['torchrun','--standalone','--nproc_per_node=2','scripts/probe_mixed_checkpoint.py',
    '--case','case2','--method','v2x_htfl','--seed','42','--compile','--out','/kaggle/working/replay',
    '--stop-after','18','--max-hours','2']
p=subprocess.run(cmd,cwd='/kaggle/temp/src')
reports=list(Path('/kaggle/working').glob('precision_rank*.json'))
assert len(reports)==2, (p.returncode,reports)
print('DIAGNOSTIC_CAPTURED', [json.loads(p.read_text()) for p in reports],flush=True)
''']
for cell in cells: ast.parse(cell)
nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(c) for c in cells])
nb.metadata.update(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
    kaggle=dict(accelerator='nvidiaTeslaT4',isGpuEnabled=True,isInternetEnabled=True))
dest=Path('notebooks')/slug;dest.mkdir(exist_ok=True)
nbformat.write(nb,dest/(slug+'.ipynb'))
meta=dict(id=owner+'/'+slug,title=slug,code_file=slug+'.ipynb',language='python',kernel_type='notebook',
    is_private=True,enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',docker_image=IMAGE,
    dataset_sources=[DATASET],kernel_sources=[],competition_sources=[])
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print(dest)
