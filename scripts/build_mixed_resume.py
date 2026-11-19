"""Resume immutable mixed-model source and full checkpoint without changing protocol."""
import ast
import json
from pathlib import Path
import nbformat
from build_mixed_fl import IMAGE, DATASET

ROOT = Path(__file__).resolve().parents[1]

def build(job, owner, hours, reserve_hours=.25, min_train_hours=.5):
    assert reserve_hours>=.1 and min_train_hours>=.2
    assert job['status'] in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED') and not job['marker']['complete']
    old = job.get('artifact_slug',job['slug']); round_no = job['checkpoint']['round']
    assert round_no == job['last_round'] and 0 < round_no < 50
    folder = ROOT/'runs/mixed128-progress'/job['slug']
    files = json.loads((folder/'output-urls.json').read_text())
    names = ['source_snapshot.json','source_manifest.json','experiment_manifest.json',
             old+'/checkpoint.json',old+'/rounds.jsonl']
    names += [old+f'/checkpoint-r{round_no:03d}/state_rank{i}.pt' for i in (0,1)]
    urls = {n:files[n] for n in names}
    slug = job['slug']+f'-r{round_no:02d}'
    helper=(ROOT/'.agents/skills/kaggle-training-notebook/scripts/pull_kernel_output.py').read_text(encoding='utf-8')
    fetch_node=next(n for n in ast.parse(helper).body if isinstance(n,ast.FunctionDef) and n.name=='fetch')
    fetch_source=ast.get_source_segment(helper,fetch_node)
    code = f'''import os, sys, json, time, hashlib, shutil, subprocess
from pathlib import Path
import requests, torch, re
{fetch_source}
t0=time.monotonic()
assert torch.cuda.device_count()==2
assert all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
os.environ.update(OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',PYTHONUNBUFFERED='1',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',WANDB_MODE='disabled',PYTHONPATH='/kaggle/temp/src')
urls={urls!r}
parent=Path('/kaggle/temp/parent')
for name,url in urls.items():
    target=parent/name;target.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        try:
            with requests.Session() as session:fetch(session,url,target,120)
            break
        except (requests.RequestException,OSError) as exc:
            print('DOWNLOAD_RETRY',name,attempt+1,type(exc).__name__,flush=True)
            if attempt==3:raise RuntimeError('Checkpoint download failed: '+name) from None
            time.sleep(2**attempt)
    print('RESTORED',name,target.stat().st_size,flush=True)
source=json.loads((parent/'source_snapshot.json').read_text())
hashes=json.loads((parent/'source_manifest.json').read_text())
for name,body in source.items():
    assert hashlib.sha256(body.encode()).hexdigest()==hashes[name]
    path=Path('/kaggle/temp/src')/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body)
for name in ('source_snapshot.json','source_manifest.json','experiment_manifest.json'):
    shutil.copyfile(parent/name,Path('/kaggle/working')/name)
out=Path('/kaggle/working')/{old!r};out.mkdir()
for name in ('checkpoint.json','rounds.jsonl'):
    shutil.copyfile(parent/{old!r}/name,out/name)
receipt=json.loads((out/'checkpoint.json').read_text())
assert receipt=={job['checkpoint']!r}
cp=out/'checkpoint-r{round_no:03d}';cp.mkdir()
for rank in (0,1):
    name=f'state_rank{{rank}}.pt'
    shutil.copyfile(parent/{old!r}/cp.name/name,cp/name)
    state=torch.load(cp/name,map_location='cpu',weights_only=True)
    assert state['round']==receipt['round'] and state['config_sha256']==receipt['config_sha256']
    del state
Path('/kaggle/working/resume_lineage.json').write_text(json.dumps({dict(parent=job['ref'],version=job['version'],round=round_no)!r}))
sys.path.insert(0,'/kaggle/temp/src')
from core import data as D
D.build_npy(D.find_root(),'/kaggle/temp/v4npy',splits=('train','test'))
os.environ['V4_NPY']='/kaggle/temp/v4npy'
remaining={hours!r}-(time.monotonic()-t0)/3600-{reserve_hours!r}
assert remaining>{min_train_hours!r}, 'Insufficient time for measured next round'
cmd=['torchrun','--standalone','--nproc_per_node=2','/kaggle/temp/src/scripts/run_mixed_fl.py',
    '--case',{job['case']!r},'--method',{job['method']!r},'--seed',{str(job['seed'])!r},
    '--compile','--out',str(out),'--max-hours',str(remaining)]
subprocess.run(cmd,cwd='/kaggle/temp/src',check=True)
status=json.loads((out/'RUN_STATUS.json').read_text())
assert status['last_round']>{round_no}
Path('/kaggle/working/RUN_STATUS.json').write_text(json.dumps(status))
print(status,flush=True)
'''
    ast.parse(code)
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)])
    nb.metadata.update(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
                       kaggle=dict(accelerator='nvidiaTeslaT4',isGpuEnabled=True,isInternetEnabled=True))
    dest=ROOT/'notebooks'/slug;dest.mkdir(exist_ok=True)
    nbformat.write(nb,dest/(slug+'.ipynb'))
    meta=dict(id=owner+'/'+slug,title=slug,code_file=slug+'.ipynb',language='python',kernel_type='notebook',
              is_private=True,enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',docker_image=IMAGE,
              dataset_sources=[DATASET],kernel_sources=[],competition_sources=[])
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    return dict(case=job['case'],method=job['method'],seed=job['seed'],owner=owner,slug=slug,
                artifact_slug=old,parent=job['ref'],parent_version=job['version'],resume_round=round_no,
                directory=str(dest),ref=owner+'/'+slug,session_budget_hours=hours)
