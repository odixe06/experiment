"""Retry owner-cancelled queued T3 sessions; collect pFedES41, then watch exact retries."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/t3'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
sys.path.insert(0,str(ROOT/'scripts'))
import run_batch as B
from watch_kaggle_batch import query
SK=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
def run(args):
 subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
rows=json.loads((OUT/'remaining-resume-batch.json').read_text());batch=[]
for old in rows:
 slug=old['slug'].replace('-p2','-p3');owner=old['account']
 assert not (OUT/(slug+'.intent')).exists(),'inspect existing intent before retry'
 st=query(old);assert st['status']=='CANCEL_ACKNOWLEDGED',st
 (OUT/(old['slug']+'.cancel-status.json')).write_text(json.dumps(st,indent=2))
 # Cancelled queued versions may have no output; preserve whatever was committed.
 run([SK/'kaggle_as.py',owner,'--',sys.executable,SK/'pull_kernel_output.py',f'{owner}/{old["slug"]}/1','-p',OUT/'pulls'/old['slug'],'-j','8'])
 assert not list((OUT/'pulls'/old['slug']).rglob('state.pt')),'new state exists: audit before choosing parent'
 base=f'flb-t3-s{old["seed"]}'
 a=json.loads((OUT/f'state-audit-{base}.json').read_text());assert a['round']==old['resume_round'] and a['RNG_two_rank'] and a['source_hp']
 src=ROOT/'notebooks'/old['slug'];nbp=src/(old['slug']+'.ipynb')
 assert hashlib.sha256(nbp.read_bytes()).hexdigest()==old['notebook_sha256']
 live=B.inventory(owner);(OUT/(slug+'.inventory.json')).write_text(json.dumps(live,indent=2))
 assert len(live['busy'])<2 and live['gpu_left']>=old['cap_hours']*1.03 and slug not in live['versions'],live
 dest=ROOT/'notebooks'/slug;dest.mkdir(exist_ok=True)
 nb=json.loads(nbp.read_text())
 for c in nb['cells']:
  if c['cell_type']=='code':compile(''.join(c['source']),slug,'exec')
 assert any(f's["round"] == {old["resume_round"]}' in ''.join(c['source']) for c in nb['cells'])
 p=dest/(slug+'.ipynb');p.write_bytes(nbp.read_bytes())
 meta=json.loads((src/'kernel-metadata.json').read_text());meta.update(id=f'{owner}/{slug}',title=slug,code_file=p.name)
 assert meta['is_private'] and meta['kernel_sources']==[old['parent'].rsplit('/',1)[0]]
 (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=1)+'\n')
 r=dict(old,slug=slug,retry_of=f'{owner}/{old["slug"]}/1')
 (OUT/(slug+'.intent')).write_text(json.dumps(r,indent=2))
 cp=subprocess.run([sys.executable,str(SK/'kaggle_as.py'),owner,'--','kaggle','kernels','push','-p',str(dest)],cwd=ROOT,capture_output=True,text=True)
 (OUT/(slug+'.push-reply')).write_text(cp.stdout+cp.stderr)
 assert cp.returncode==0 and 'Kernel version 1 successfully pushed' in cp.stdout
 (OUT/(slug+'.pushed')).write_text(cp.stdout);batch.append(r)
 (OUT/'retry-p3-batch.json').write_text(json.dumps(batch,indent=2))
 print('T3_RETRY_PUSHED',owner,slug,flush=True)
# pFedES41 completion explicitly requested by owner.
spec=dict(arm='pFedES41',account='emtrangtri',slug='flb-pfedes-s41-p3',version=1)
st=query(spec);assert st['status']=='COMPLETE',st
out=ROOT/'runs/pulls'/spec['slug']
run([SK/'kaggle_as.py',spec['account'],'--',sys.executable,SK/'pull_kernel_output.py','emtrangtri/flb-pfedes-s41-p3/1','-p',out,'-j','16'])
run(['scripts/audit_run.py',out/'flb-pfedes-s41','--seed','41','--support','runs/test_support.npy','--out','runs/batch-s41/audit/flb-pfedes-s41.json'])
print('PFEDES41_PULLED_AUDITED',flush=True)
run(['scripts/watch_kaggle_batch.py','--batch',OUT/'retry-p3-batch.json','--out',OUT/'retry-p3-watch','--interval','1200','--stop-on-error'])
