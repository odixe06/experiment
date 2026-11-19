"""Audit cached-real T3 BERT smoke: committed rounds, metrics, model reconstruction and resume state."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from core.metrics import TEN,summary
from core.flat import Layout
p=Path(sys.argv[1]); end=int(sys.argv[2])
rows=[json.loads(x) for x in (p/'rounds.jsonl').read_text().splitlines()]
assert [r['round'] for r in rows]==list(range(1,end+1))
cfg=json.loads((p/'config.json').read_text()); lay=Layout(cfg['cfg']);assert lay.P==77904 and lay.B==0
support=np.diff(np.load('cache/test_sub100_83466334_off.npy'))
for r in rows:
    assert np.isfinite(r['train_loss']) and r['skipped_frac']<=.01 and r['clients_without_applied']==0
    cm=np.load(p/f"cms_r{r['round']:03d}.npz")['private']
    assert np.array_equal(cm.sum((1,2)),support)
    metrics=summary(cm)
    for k in TEN: assert abs(metrics[k]-r['private/'+k])<1e-10,k
s=torch.load(p/'state.pt',weights_only=True,map_location='cpu',mmap=True)
assert s['round']==end and s['method']=='mobihet_compute' and s['hp']['rounds']==50
assert all(k in s for k in ['rng_cpu','rng_cuda','gen']) and len(s['gen'])==1
m=s['method_state'];assert m['version']==end and m['priv_p'].shape==(100,77904) and m['priv_b'].shape==(100,0)
for k,v in m.items():
    if torch.is_tensor(v):assert torch.isfinite(v).all(),k
model=lay.make(cfg['cfg']);model.load_state_dict(lay.state_dict(m['priv_p'][0],m['priv_b'][0]),strict=True)
report={'rounds':end,'finite':True,'CM_metrics':True,'strict_rebuild':True,'RNG':True,'scope':'local 1GPU cached100; not T4 2rank proof'}
(p/'audit.json').write_text(json.dumps(report,indent=2));print('T3_SMOKE_AUDIT_PASS',end)
