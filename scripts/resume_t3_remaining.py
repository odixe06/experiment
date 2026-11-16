"""Resume audited T3 BERT seeds42/43; retain frozen source, HP, RNG and cosine50."""
import hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
sys.path.insert(0,str(ROOT/'scripts'))
import run_batch as B
OUT=ROOT/'runs/t3'
def call(args):
    return subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True,capture_output=True,text=True)
batch=[]
for seed,cap in [(42,6.5),(43,7.5)]:
    b=next(x for x in json.loads((OUT/'batch.json').read_text()) if x['seed']==seed)
    owner=b['account'];run=b['slug'];slug=run+'-p2'
    audit=json.loads((OUT/f'audit-{run}.json').read_text());state=json.loads((OUT/f'state-audit-{run}.json').read_text())
    assert audit['status']=='PASS_BOUNDED' and audit['rounds']==state['round']<50
    assert state['RNG_two_rank'] and state['source_hp'] and state['finite_banks']
    assert not (OUT/(slug+'.intent')).exists(),'inspect intent before retry'
    for n,h in b['overlay_sha256'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    live=B.inventory(owner);(OUT/(slug+'.inventory.json')).write_text(json.dumps(live,indent=2))
    assert len(live['busy'])<2 and slug not in live['versions']
    cap=min(cap,int((live['gpu_left']/1.03-.1)*100)/100)
    assert cap>=1.5, f'{owner}: insufficient quota {live["gpu_left"]}'
    args=shlex.join(['scripts/run_t3.py','--method','mobihet_compute','--out',f'/kaggle/working/{run}','--width','1','--rounds','50','--until','50','--save-every','5','--compile','--max-hours','BUDGET','--cache','','--set',*b['hp']])
    call(['scripts/gen_notebook.py','--owner',owner,'--slug',slug,'--run',run,'--method','mobihet_compute','--sha',b['base_sha'],'--args',args,'--max-hours',str(cap),'--resume-kernel',f'{owner}/{run}','--overlay',*b['overlay_sha256']])
    p=ROOT/'notebooks'/slug/(slug+'.ipynb');nb=json.loads(p.read_text())
    for i,c in enumerate(nb['cells']):
        c['id']=f't3-resume-{i:03d}'
        s=''.join(c['source'])
        if 'BEGIN INLINE WANDB CREDENTIAL' in s:s='print("W&B off: same T3 policy")'
        if 'prev = [p for p in glob.glob' in s:
            s=s.replace('del s, other',f'assert s["round"] == {state["round"]} and s["hp"]["seed"] == {seed}\ndel s, other')
            assert 's["method"] == "mobihet_compute"' in s
        c['source']=s.splitlines(True)
        if c['cell_type']=='code':compile(s,str(p),'exec')
    p.write_text(json.dumps(nb,indent=1)+'\n')
    meta=json.loads((p.parent/'kernel-metadata.json').read_text())
    assert meta['is_private'] and meta['kernel_sources']==[f'{owner}/{run}']
    r=dict(b,slug=slug,version=1,parent=f'{owner}/{run}/1',resume_round=state['round'],cap_hours=cap,notebook_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (OUT/(slug+'.intent')).write_text(json.dumps(r,indent=2))
    cp=subprocess.run([sys.executable,str(B.SK/'kaggle_as.py'),owner,'--','kaggle','kernels','push','-p',str(p.parent)],cwd=ROOT,capture_output=True,text=True)
    (OUT/(slug+'.push-reply')).write_text(cp.stdout+cp.stderr)
    assert cp.returncode==0 and 'Kernel version 1 successfully pushed' in cp.stdout
    (OUT/(slug+'.pushed')).write_text(cp.stdout);batch.append(r)
    (OUT/'remaining-resume-batch.json').write_text(json.dumps(batch,indent=2))
    print('T3_RESUMED',owner,slug,'round',state['round'],'cap',cap,flush=True)
print('T3_REMAINING_PUSHED',flush=True)
subprocess.run([sys.executable,'scripts/watch_kaggle_batch.py','--batch',str(OUT/'remaining-resume-batch.json'),'--out',str(OUT/'remaining-resume-watch'),'--interval','1200','--stop-on-error'],cwd=ROOT,check=True)
