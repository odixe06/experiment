"""Resume frozen T3 BERT seed41 from audited r36; no scientific changes."""
import json,subprocess,sys,shlex,os
from pathlib import Path
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_batch as B
out=ROOT/'runs/t3'; slug='flb-t3-s41-p3'; owner='minhtran0601'
subprocess.run([sys.executable,'scripts/audit_t3_s41_state.py'],cwd=ROOT,check=True)
a=json.loads((out/'audit-s41.json').read_text()); assert a['rounds']==36
assert json.loads((out/'state-audit-s41.json').read_text())['round']==36
assert not (out/(slug+'.intent')).exists(),'inspect existing intent before retry'
live=B.inventory(owner); (out/(slug+'.inventory.json')).write_text(json.dumps(live))
assert len(live['busy'])<2 and live['gpu_left']>=6*1.03 and slug not in live['versions'], 'insufficient quota/slot or existing slug'
b=json.loads((out/'batch.json').read_text())[0]
args=shlex.join(['scripts/run_t3.py','--method','mobihet_compute','--out','/kaggle/working/flb-t3-s41','--width','1','--rounds','50','--until','50','--save-every','5','--compile','--max-hours','BUDGET','--cache','','--set',*b['hp']])
subprocess.run([sys.executable,'scripts/gen_notebook.py','--owner',owner,'--slug',slug,'--run','flb-t3-s41','--method','mobihet_compute','--sha',b['base_sha'],'--args',args,'--max-hours','6','--resume-kernel',owner+'/flb-t3-s41','--overlay',*b['overlay_sha256']],cwd=ROOT,check=True,capture_output=True)
p=ROOT/'notebooks'/slug/(slug+'.ipynb'); nb=json.loads(p.read_text())
for i,c in enumerate(nb['cells']):
 c['id']=f't3-resume-{i:03d}'
 if 'BEGIN INLINE WANDB CREDENTIAL' in ''.join(c['source']): c['source']=['print("W&B off: same T3 policy")']
 if c['cell_type']=='code': compile(''.join(c['source']),str(p),'exec')
assert any('s["method"] == "mobihet_compute"' in ''.join(c['source']) for c in nb['cells'])
p.write_text(json.dumps(nb,indent=1)+'\n')
r=dict(b,slug=slug,version=1,parent=owner+'/flb-t3-s41/1',resume_round=36,cap_hours=6)
(out/(slug+'.intent')).write_text(json.dumps(r,indent=2))
cp=subprocess.run([sys.executable,str(B.SK/'kaggle_as.py'),owner,'--','kaggle','kernels','push','-p',str(p.parent)],cwd=ROOT,capture_output=True,text=True)
(out/(slug+'.push-reply')).write_text(cp.stdout+cp.stderr)
assert cp.returncode==0 and 'Kernel version 1 successfully pushed' in cp.stdout
(out/(slug+'.pushed')).write_text(cp.stdout); (out/'resume-batch.json').write_text(json.dumps([r],indent=2))
print('T3_BERT_S41_RESUME_PUSHED',flush=True)
subprocess.run([sys.executable,str(ROOT/'scripts/watch_kaggle_batch.py'),'--batch',str(out/'resume-batch.json'),'--out',str(out/(slug+'-watch')),'--interval','1200','--stop-on-error'],check=True)
