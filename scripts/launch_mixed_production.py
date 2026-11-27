"""Launch explicitly authorized mixed128 runs after matching pilot gates and fresh quota."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from build_mixed_fl import build, METHODS
sys.path.insert(0, str(Path(__file__).parent/"cloud_dispatch"))
import dispatcher as d
import manage_github as gh

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, default=45)
    ap.add_argument('--session-hours', type=float, default=10.5)
    ap.add_argument('--idle-only', action='store_true')
    ap.add_argument('--reserve-existing-wave', action='store_true')
    args = ap.parse_args()
    assert 3 <= args.session_hours <= 10.5
    required = args.session_hours + .5
    api_gh = gh.session(); prefix = '/repos/'+gh.REPO
    assert gh.call(api_gh, 'GET', prefix+'/actions/workflows/dispatch.yml')['state'] == 'disabled_manually'
    assert gh.call(api_gh, 'GET', prefix+'/actions/variables/DISPATCH_ENABLED')['value'] == 'false'
    passed = json.loads((ROOT/'runs/mixed128-pilots-live.json').read_text())['rows']
    gates = {(r['case'], r['method']):r for r in passed
             if r.get('status') == 'COMPLETE' and r.get('gpu_smoke') == 'PASS' and r.get('cm_metric_audit') == 'PASS'}
    snapshots = json.loads((ROOT/'runs/accounts-results-check.json').read_text())['accounts']
    path = ROOT/'runs/mixed128-production-launch.json'
    ledger = json.loads(path.read_text()) if path.exists() else []
    used = {row['owner'] for row in ledger if row['state'] in ('reserved','review')}
    launched = 0
    for case in ('case2','case1','case3'):
        for seed in (42,41,43):
            for method in METHODS:
                if launched >= args.limit: break
                if (case,method) not in gates: continue
                if any((r['case'],r['method'],r['seed']) == (case,method,seed) for r in ledger): continue
                candidates = [o for o,s in snapshots.items() if o not in used and s.get('free_hours',0) >= required]
                for owner in sorted(candidates, key=lambda o:-snapshots[o]['free_hours']):
                    token = (Path.home()/'.kaggle/accounts'/f'{owner}.mcp-token').read_text().strip()
                    try:
                        snap = d.checked_inventory(owner,token,snapshots[owner].get('records',{}))
                    except Exception:
                        used.add(owner); continue
                    snapshots[owner] = {k:v for k,v in snap.items() if k != 'api'}
                    if args.reserve_existing_wave:
                        # Conservative when Kaggle has not reserved queued GPU time yet.
                        planned = sum(r.get('session_budget_hours', 10.5) for r in ledger
                                      if r['owner'] == owner and r['case'] in ('case1', 'case3')
                                      and r['state'] in ('reserved', 'submitted', 'review'))
                        snap['free_hours'] = max(0, snap['free_hours'] - planned)
                    if snap['active'] >= (1 if args.idle_only else 2) or snap['free_hours'] < required:
                        used.add(owner); continue
                    entry = build(case,method,seed,owner,session_hours=args.session_hours)
                    entry.update(ref=f"{owner}/{entry['slug']}",state='reserved',time=datetime.now(timezone.utc).isoformat(),
                                 pilot_ref=gates[(case,method)]['ref'],pilot_version=gates[(case,method)]['version'],
                                 session_budget_hours=args.session_hours)
                    ledger.append(entry); path.write_text(json.dumps(ledger,indent=2))
                    try:
                        reply = d.guarded_push(owner, snap['api'], entry['directory'],timeout='37800')
                        if not d.accepted(reply,entry['ref']): raise RuntimeError('Push rejected')
                        entry.update(version=d.confirm_submission(snap['api'],entry['ref'],Path(entry['directory'])),state='submitted')
                        entry['status']=d.read_call(lambda:snap['api'].kernels_status(entry['ref'])).status.name
                        snapshots[owner]['free_hours'] -= required
                        snapshots[owner]['active'] += 1
                        launched += 1
                    except Exception as exc:
                        entry.update(state='review',error=type(exc).__name__); used.add(owner)
                    path.write_text(json.dumps(ledger,indent=2))
                    print(json.dumps(entry),flush=True)
                    break
    print('NEW_SUBMITTED',launched,flush=True)


if __name__ == '__main__': main()
