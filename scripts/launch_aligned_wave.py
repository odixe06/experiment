"""Owner-authorized first aligned wave; durable local ledger, no cloud scheduling."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent/'cloud_dispatch'))
import dispatcher as d

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    ledger_path = ROOT/f'runs/aligned-v2-launch-wave{args.wave}.json'
    snapshots = json.loads((ROOT/'runs/aligned-v2-inventory.json').read_text())
    assert (datetime.now().timestamp() - (ROOT/'runs/aligned-v2-inventory.json').stat().st_mtime) < 900
    records = {r:v for snap in snapshots.values() for r,v in snap['records'].items()}
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    def save():
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    jobs = [('tabm','tinyproto',s) for s in (41,42,43)]
    jobs += [('sae',m,s) for m in ('fedavg','fedprox','apfed','fedproto') for s in (41,42,43)]
    jobs += [('sae','fedtgp',s) for s in (41,42)]
    if args.wave == 2:
        jobs = [('sae','fedtgp',43)]
        jobs += [('sae',m,s) for m in ('tinyproto','fd_ids','afpha','lwfednids','pfedes') for s in (41,42,43)]
        jobs += [('sae','nilm',41)]
    elif args.wave == 3:
        jobs = [('sae','nilm',s) for s in (42,43)]
        jobs += [('sae','perfedskd',s) for s in (41,42,43)]
        jobs += [('sae','canperfl',s) for s in (41,42,43)]
    elif args.wave == 4:
        jobs = [('sae','v2x_htfl',s) for s in (41,42,43)]
    assert len(jobs) == {1:17, 2:17, 3:8, 4:3}[args.wave]
    for b,m,s in jobs:
        slug = f'{b}-fl-{m.replace("_","-")}-s{s}-a2'
        if (any(r.split('/')[1]==slug for r in records)
                or any(j['slug']==slug and j['state'] in ('submitted','reserved') for j in ledger)):
            continue
        parent = None
        candidates = list(snapshots)
        if m == 'canperfl':
            parent_slug = f'sae-fl-fedavg-s{s}-a2'
            hits = [r for r,v in records.items() if r.split('/')[1] == parent_slug and v['status'] == 'COMPLETE']
            if len(hits) != 1:
                raise RuntimeError(f'Expected one COMPLETE parent for {slug}, got {hits}')
            parent = hits[0]
            parent_owner = parent.split('/')[0]
            api_parent = d.make_api((Path.home()/'.kaggle/accounts'/f'{parent_owner}.mcp-token').read_text().strip())
            parent_version = d.kernel_metadata(api_parent, parent).current_version_number
            if not parent_version or not d.parent_ready(
                    api_parent, parent, parent_version,
                    {'seed':s, 'backbone':'sae'}):
                raise RuntimeError(f'Parent artifact gate failed for {parent}')
            candidates = [parent_owner]
        candidates = [o for o in candidates if snapshots[o]['active'] < 2 and snapshots[o]['free_hours'] >= 8.25]
        if not candidates:
            break
        owner = max(candidates,key=lambda o:snapshots[o]['free_hours'])
        snap = snapshots[owner]
        api = d.make_api((Path.home()/'.kaggle/accounts'/f'{owner}.mcp-token').read_text().strip())
        snap['free_hours'] = min(snap['free_hours'], d.quota(api))
        if snap['free_hours'] < 8.25:
            continue
        build_cmd = [sys.executable,'scripts/build_tabm_fl.py','--owner',owner,'--backbone',b,
                     '--method',m,'--seed',str(s),'--slug',slug,'--max-hours','7']
        if parent:
            build_cmd += ['--parent', parent]
        subprocess.run(build_cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        folder = ROOT/'notebooks'/slug
        entry = dict(slug=slug,ref=owner+'/'+slug,backbone=b,method=m,seed=s,state='reserved',
                     time=datetime.now(timezone.utc).isoformat(), parent=parent,
                     sha256=hashlib.sha256((folder/f'{slug}.ipynb').read_bytes()).hexdigest())
        ledger.append(entry); save()
        try:
            reply = api.kernels_push(str(folder),timeout='28800')
            if not d.accepted(reply,entry['ref']):
                raise RuntimeError('push rejected')
            entry.update(version=d.confirm_submission(api,entry['ref'],folder),state='submitted')
            entry['status']=d.read_call(lambda:api.kernels_status(entry['ref'])).status.name
            snap['active'] += 1
            snap['free_hours'] -= 8
        except Exception as exc:
            entry.update(state='review',error=type(exc).__name__)
            snap['active']=2
        save()
        print(json.dumps(entry),flush=True)
    print('SUBMITTED',sum(j['state']=='submitted' for j in ledger),'REVIEW',sum(j['state']=='review' for j in ledger),flush=True)


if __name__=='__main__':
    main()
