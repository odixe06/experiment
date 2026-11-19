import sys,json,base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'scripts/cloud_dispatch')
import dispatcher as d
import manage_github as gh
api=gh.session(); prefix='/repos/'+gh.REPO
wf=gh.call(api,'GET',prefix+'/actions/workflows/dispatch.yml'); var=gh.call(api,'GET',prefix+'/actions/variables/DISPATCH_ENABLED')
assert wf['state']=='disabled_manually' and var['value']=='false'
queue=gh.call(api,'GET',prefix+'/contents/queue.json'); state=json.loads(base64.b64decode(queue['content']))
cache=state.get('inventory_cache',{})
accounts=sorted(p.stem.removesuffix('.mcp-token') for p in (Path.home()/'.kaggle/accounts').glob('*.mcp-token'))
assert len(accounts) == 21, f'Expected 21 Kaggle accounts, got {accounts}'
def one(o):
    try:
        return d.checked_inventory(o,(Path.home()/'.kaggle/accounts'/f'{o}.mcp-token').read_text().strip(),cache)
    except Exception as exc:
        print('INVENTORY_ERROR', o, type(exc).__name__, flush=True)
        old = json.loads(Path('runs/aligned-v2-inventory.json').read_text()).get(o)
        if old is None:
            raise
        return old
with ThreadPoolExecutor(max_workers=4) as pool: snaps=dict(zip(accounts,pool.map(one,accounts)))
out={o:{k:v for k,v in s.items() if k!='api'} for o,s in snaps.items()}
Path('runs/aligned-v2-inventory.json').write_text(json.dumps(out,indent=2))
print('SNAPSHOT SAVED; CLOUD DISABLED')
