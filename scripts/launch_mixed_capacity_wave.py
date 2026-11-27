"""One manual wave: unique pending repairs and committed checkpoint resumes."""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).parent/'cloud_dispatch'))
import dispatcher as d
from build_mixed_fl import build as fresh
from build_mixed_resume import build as resume

ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inventory',default='runs/mixed128-capacity-latest.json')
    ap.add_argument('--progress',default='runs/mixed128-next-progress.json')
    ap.add_argument('--ledger',default='runs/mixed128-capacity-wave-launch.json')
    ap.add_argument('--resume-only',action='store_true')
    ap.add_argument('--min-free-hours',type=float,default=1.3)
    args=ap.parse_args()
    assert args.min_free_hours>=1.1
    snapshots=json.loads((ROOT/args.inventory).read_text())['accounts']
    prior=json.loads((ROOT/'runs/mixed128-bert-repair-launch.json').read_text())
    tasks=[dict(case=c,method='v2x_htfl',seed=s) for c in ('case1','case3') for s in (41,42,43)
           if not any((r['case'],r['seed'])==(c,s) for r in prior)]
    if args.resume_only:tasks=[]
    pending=[r for r in json.loads((ROOT/args.progress).read_text())
             if r.get('status')=='COMPLETE' and r.get('marker',{}).get('complete') is False and not r.get('error')]
    pending.sort(key=lambda r:-r['last_round'])
    path=ROOT/args.ledger
    ledger=json.loads(path.read_text()) if path.exists() else []
    used={r['owner'] for r in ledger}
    for job in tasks+pending:
        if any((r['case'],r['method'],r['seed'])==(job['case'],job['method'],job['seed']) for r in ledger):continue
        is_resume='checkpoint' in job
        candidates=[o for o,s in snapshots.items() if o not in used and s.get('active',99)==0
                    and s.get('free_hours',0)>=(args.min_free_hours if is_resume else 1.9)]
        candidates.sort(key=lambda o:snapshots[o]['free_hours'],reverse=args.resume_only or not is_resume)
        for owner in candidates:
            snap=d.checked_inventory(owner,(Path.home()/'.kaggle/accounts'/f'{owner}.mcp-token').read_text().strip(),snapshots[owner].get('records'))
            used.add(owner)
            if snap['active'] or snap['free_hours']<(args.min_free_hours if is_resume else 1.9):continue
            hours=min(10.5,snap['free_hours']-(.15 if is_resume else .25))
            entry=resume(job,owner,hours) if is_resume else fresh(job['case'],job['method'],job['seed'],owner,session_hours=hours,revision='a2')
            entry.update(ref=owner+'/'+entry['slug'],state='reserved',time=datetime.now(timezone.utc).isoformat(),session_budget_hours=hours)
            ledger.append(entry);path.write_text(json.dumps(ledger,indent=2))
            try:
                reply=d.guarded_push(owner, snap['api'], entry['directory'],timeout=str(int((hours+.15)*3600)))
                assert d.accepted(reply,entry['ref'])
                entry.update(version=d.confirm_submission(snap['api'],entry['ref'],Path(entry['directory'])),state='submitted')
                entry['status']=d.read_call(lambda:snap['api'].kernels_status(entry['ref'])).status.name
            except Exception as exc:entry.update(state='review',error=type(exc).__name__)
            path.write_text(json.dumps(ledger,indent=2));print(json.dumps(entry),flush=True)
            break
    print('WAVE_SUBMITTED',sum(r['state']=='submitted' for r in ledger),flush=True)

if __name__=='__main__':main()
