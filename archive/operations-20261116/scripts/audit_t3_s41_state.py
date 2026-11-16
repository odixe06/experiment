import sys,json,hashlib
from pathlib import Path
import torch
sys.path.insert(0,'/home/odixe/project/fl_veremi_bert')
from core.flat import Layout
root=Path('/home/odixe/project/fl_veremi_bert'); p=root/'runs/t3/pulls/flb-t3-s41'; run=p/'flb-t3-s41'
torch.set_num_threads(2)
b=json.loads((root/'runs/t3/batch.json').read_text())[0]; manifest=json.loads((p/'provenance/source_manifest.json').read_text())
assert manifest['base_sha']==b['base_sha'] and manifest['overlay']==b['overlay_sha256']
for n,h in b['overlay_sha256'].items(): assert hashlib.sha256((p/'provenance/source'/n).read_bytes()).hexdigest()==h,n
cfg=json.loads((run/'config.json').read_text()); rows=[json.loads(x) for x in (run/'rounds.jsonl').read_text().splitlines()]; end=rows[-1]['round']
for item in b['hp']:
 k,v=item.split('=',1)
 try: expected=json.loads(v)
 except json.JSONDecodeError: expected=v
 assert cfg['hp'][k]==expected,k
lay=Layout(cfg['cfg']); sizes=[]
for rank,name in enumerate(['state.pt','state_rank1.pt']):
 s=torch.load(run/name,map_location='cpu',weights_only=True,mmap=True); assert s['round']==end
 assert 'rng_cpu' in s and 'rng_cuda' in s
 if rank==0: assert s['hp']==cfg['hp'] and len(s['gen'])==2
 m=s['method_state']; assert m['version']==end
 sizes.append(m['priv_p'].shape[0]); assert m['priv_p'].shape[1]==77904
 for k,v in m.items():
  if torch.is_tensor(v): assert torch.isfinite(v).all(),k
 model=lay.make(cfg['cfg']);model.load_state_dict(lay.state_dict(m['priv_p'][0],m['priv_b'][0]),strict=True)
assert sum(sizes)==5343
for r in rows: assert r['skipped_frac']<=.01 and r['clients_without_applied']==0
out=dict(round=end,rank_clients=sizes,source_hp=True,RNG_two_rank=True,finite_banks=True,strict_rebuild=True,complete=end==50)
(root/'runs/t3/state-audit-s41.json').write_text(json.dumps(out,indent=2));print(out)
