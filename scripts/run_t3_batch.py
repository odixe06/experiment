"""Build/push frozen T3 FeatureBERT seeds41/42/43; baseline plans stay independent."""
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/t3'
BASE='cb193e6fd56fc2f0c6810745cc57d8242c77c366'
OWNERS={41:'minhtran0601',42:'annhtnt',43:'trietbackup'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ref=json.loads((ROOT/'knowledge/t3-dagsnet-reference.json').read_text())[0]
    overlays=json.loads((OUT/'port.json').read_text())['files']+['core/local_t3.py','scripts/run_t3.py']
    batch=[]
    for seed,owner in OWNERS.items():
        slug=f'flb-t3-s{seed}'
        hp=[f'seed={seed}' if x.startswith('seed=') else x for x in ref['hp']]
        args=shlex.join(['scripts/run_t3.py','--method','mobihet_compute','--out',f'/kaggle/working/{slug}','--width','1','--rounds','50','--until','50','--save-every','5','--compile','--max-hours','BUDGET','--cache','','--set',*hp])
        subprocess.run([sys.executable,'scripts/gen_notebook.py','--owner',owner,'--slug',slug,'--sha',BASE,'--args',args,'--max-hours','11.5','--overlay',*overlays],cwd=ROOT,check=True,capture_output=True)
        folder=ROOT/'notebooks'/slug; nb=folder/(slug+'.ipynb'); doc=json.loads(nb.read_text())
        for i,cell in enumerate(doc['cells']):
            cell['id']=f't3-cell-{i:03d}'
            if 'BEGIN INLINE WANDB CREDENTIAL' in ''.join(cell['source']): cell['source']=['print("W&B off: same T3 DAGSNet policy")']
            if cell['cell_type']=='code': compile(''.join(cell['source']),str(nb),'exec')
        nb.write_text(json.dumps(doc,indent=1)+'\n')
        meta=json.loads((folder/'kernel-metadata.json').read_text());assert meta['is_private'] and meta['enable_gpu'] and meta['docker_image']
        batch.append(dict(arm=f'T3B{seed}',seed=seed,account=owner,slug=slug,version=1,hp=hp,base_sha=BASE,cap_hours=11.5,endpoint=50,overlay_sha256={n:sha(ROOT/n) for n in overlays},notebook_sha256=sha(nb)))
    if (OUT/'batch.json').exists(): assert json.loads((OUT/'batch.json').read_text())==batch
    else: (OUT/'batch.json').write_text(json.dumps(batch,indent=2))
    if sys.argv[1]=='build': return
    assert sys.argv[1]=='launch'
    for row in batch:
        slug=row['slug']; intent=OUT/(slug+'.intent'); marker=OUT/(slug+'.pushed')
        if marker.exists(): continue
        assert not intent.exists(),'ambiguous push: inspect first'
        # Only accounts reserved from baseline allocation; live quota checked immediately before push.
        helper=ROOT/'.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py'
        inv=subprocess.run([sys.executable,str(helper),row['account'],'--',sys.executable,'scripts/kaggle_inventory.py',row['account']],cwd=ROOT,capture_output=True,text=True,timeout=600)
        (OUT/(slug+'.inventory.log')).write_text(inv.stdout+inv.stderr)
        assert inv.returncode == 0, f'inventory unavailable: see {slug}.inventory.log'
        q=json.loads(inv.stdout); assert q['gpu_left']>=11.5*1.15 and len(q['busy'])<2
        assert slug not in q['versions'], 'slug already exists: inspect first'
        (OUT/(slug+'.gate.json')).write_text(json.dumps(q,indent=2))
        intent.write_text(json.dumps(row,indent=2))
        r=subprocess.run([sys.executable,str(ROOT/'.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py'),row['account'],'--','kaggle','kernels','push','-p',str(ROOT/'notebooks'/slug)],capture_output=True,text=True)
        (OUT/(slug+'.push-reply')).write_text(r.stdout+r.stderr)
        assert r.returncode==0 and 'Kernel version 1 successfully pushed' in r.stdout,slug
        marker.write_text(r.stdout);print('T3_BERT_PUSHED',row['account'],slug,flush=True)
if __name__=='__main__': main()
