"""Match remaining quota to measured round time; reject insufficient setup time remotely."""
import json
import sys
from pathlib import Path
from datetime import datetime,timezone
sys.path[:0]=[str(Path(__file__).parent/'cloud_dispatch'),str(Path(__file__).parent)]
import dispatcher as d
from build_mixed_resume import build

ROOT=Path(__file__).resolve().parents[1]
def main():
    snapshots=json.loads((ROOT/'runs/mixed128-capacity-20261119-b.json').read_text())['accounts']
    jobs=[j for j in json.loads((ROOT/'runs/mixed128-progress-20261119-b.json').read_text())
          if j.get('status')=='COMPLETE' and j.get('marker',{}).get('complete') is False and not j.get('error')]
    def need(j):
        r=j['last_metrics'];return max(.2,1.1*(r['t_train']+r['t_eval'])/3600)
    jobs.sort(key=need,reverse=True)
    path=ROOT/'runs/mixed128-launch-20261119-b.json';ledger=json.loads(path.read_text())
    used={r['owner'] for r in ledger}
    for j in jobs:
        if any(r.get('parent')==j['ref'] for r in ledger):continue
        # 9min setup allowance, 6min checkpoint reserve, 3min quota headroom.
        required=need(j)+.3
        candidates=sorted((o for o,s in snapshots.items() if o not in used and s.get('active',99)==0
                           and s.get('free_hours',0)>=required),key=lambda o:snapshots[o]['free_hours'])
        for owner in candidates:
            s=d.checked_inventory(owner,(Path.home()/'.kaggle/accounts'/f'{owner}.mcp-token').read_text().strip(),snapshots[owner].get('records'))
            used.add(owner)
            if s['active'] or s['free_hours']<required:continue
            r=build(j,owner,s['free_hours']-.05,reserve_hours=.1,min_train_hours=need(j))
            r.update(state='reserved',time=datetime.now(timezone.utc).isoformat(),minimum_train_hours=need(j))
            ledger.append(r);path.write_text(json.dumps(ledger,indent=2))
            try:
                reply=d.guarded_push(owner, s['api'], r['directory'],timeout=str(int(r['session_budget_hours']*3600)))
                assert d.accepted(reply,r['ref'])
                r.update(version=d.confirm_submission(s['api'],r['ref'],Path(r['directory'])),state='submitted')
                r['status']=d.read_call(lambda:s['api'].kernels_status(r['ref'])).status.name
            except Exception as exc:r.update(state='review',error=type(exc).__name__)
            path.write_text(json.dumps(ledger,indent=2));print(json.dumps(r),flush=True)
            break
    print('SUBMITTED',sum(r['state']=='submitted' for r in ledger),flush=True)

if __name__=='__main__':main()
