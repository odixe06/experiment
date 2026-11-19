"""Launch corrected V2X seeds together; preserve original results and frozen assignments."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from build_mixed_fl import build
sys.path.insert(0,str(Path(__file__).parent/'cloud_dispatch'))
import dispatcher as d
import manage_github as gh

ROOT=Path(__file__).resolve().parents[1]
evidence=json.loads((ROOT/'runs/mixed128-precision-evidence.json').read_text())
assert len(evidence)==2
for row in evidence:
    modes={r['mode']:r for r in row['rows']}
    assert modes['old_amp']['bad'] and not modes['fp32']['bad']
api_gh=gh.session();prefix='/repos/'+gh.REPO
assert gh.call(api_gh,'GET',prefix+'/actions/workflows/dispatch.yml')['state']=='disabled_manually'
assert gh.call(api_gh,'GET',prefix+'/actions/variables/DISPATCH_ENABLED')['value']=='false'
path=ROOT/'runs/mixed128-v2x-repair-launch.json'
ledger=json.loads(path.read_text()) if path.exists() else []
for seed,owner in ((42,'huygia1611'),(41,'hiulhuy'),(43,'honghunhnh')):
    if any(row['seed']==seed for row in ledger):continue
    token=(Path.home()/'.kaggle/accounts'/f'{owner}.mcp-token').read_text().strip()
    snap=d.checked_inventory(owner,token)
    if snap['active']>=2 or snap['free_hours']<11:
        print('UNAVAILABLE',owner,flush=True);continue
    row=build('case2','v2x_htfl',seed,owner,revision='a2')
    row.update(ref=owner+'/'+row['slug'],state='reserved',time=datetime.now(timezone.utc).isoformat(),
        numerical_fix='SAE local training fp32; other backbone AMP unchanged',fresh_start=True)
    ledger.append(row);path.write_text(json.dumps(ledger,indent=2))
    try:
        reply=snap['api'].kernels_push(row['directory'],timeout='37800')
        if not d.accepted(reply,row['ref']):raise RuntimeError('Push requires review')
        row.update(version=d.confirm_submission(snap['api'],row['ref'],Path(row['directory'])),state='submitted')
        row['status']=snap['api'].kernels_status(row['ref']).status.name
    except Exception as exc:
        row.update(state='review',error=type(exc).__name__)
    path.write_text(json.dumps(ledger,indent=2))
    print(json.dumps(row),flush=True)
