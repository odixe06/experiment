"""Manually launch authorized immutable checkpoint resumes within live quota."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'cloud_dispatch'))
import dispatcher as d
from build_mixed_resume import build

ROOT=Path(__file__).resolve().parents[1]
def main():
    import manage_github as gh
    api=gh.session();prefix='/repos/'+gh.REPO
    assert gh.call(api,'GET',prefix+'/actions/workflows/dispatch.yml')['state']=='disabled_manually'
    assert gh.call(api,'GET',prefix+'/actions/variables/DISPATCH_ENABLED')['value']=='false'
    quota=json.loads((ROOT/'runs/mixed128-resume-quota.json').read_text())
    path=ROOT/'runs/mixed128-resume-launch.json'
    ledger=json.loads(path.read_text()) if path.exists() else []
    jobs=[j for j in json.loads((ROOT/'runs/mixed128-progress.json').read_text())
          if j.get('marker',{}).get('complete') is False and j['status']=='COMPLETE']
    jobs.sort(key=lambda j:-j['last_round'])
    used=set()
    for job in jobs:
        if any(r['parent']==job['ref'] for r in ledger):continue
        planned={o:sum(r['session_budget_hours']+.3 for r in ledger if r['owner']==o and r['state'] in ('submitted','reserved','review')) for o in quota}
        candidates=[o for o,q in quota.items() if o not in used and q.get('active',99)<2 and q.get('free_hours',0)-planned[o]>=2.5]
        if not candidates:break
        # Near-complete jobs use smaller accounts, preserving long allocations.
        near=job['last_round']>=40
        candidates.sort(key=lambda o:quota[o]['free_hours']-planned[o],reverse=not near)
        for owner in candidates:
            snap=d.checked_inventory(owner,(Path.home()/'.kaggle/accounts'/f'{owner}.mcp-token').read_text().strip(),quota[owner].get('records'))
            used.add(owner)
            available=snap['free_hours']-planned[owner]
            if snap['active']>=2 or available<2.5:continue
            hours=min(3.5 if near else 10.5,available-.6)
            entry=build(job,owner,hours)
            entry.update(state='reserved',time=datetime.now(timezone.utc).isoformat())
            ledger.append(entry);path.write_text(json.dumps(ledger,indent=2))
            try:
                reply=snap['api'].kernels_push(entry['directory'],timeout=str(int((hours+.3)*3600)))
                assert d.accepted(reply,entry['ref'])
                entry.update(version=d.confirm_submission(snap['api'],entry['ref'],Path(entry['directory'])),state='submitted')
                entry['status']=d.read_call(lambda:snap['api'].kernels_status(entry['ref'])).status.name
            except Exception as exc:entry.update(state='review',error=type(exc).__name__)
            path.write_text(json.dumps(ledger,indent=2))
            print(json.dumps(entry),flush=True)
            break
    print('SUBMITTED',sum(r['state']=='submitted' for r in ledger),flush=True)

if __name__=='__main__':main()
