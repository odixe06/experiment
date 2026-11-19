"""Collect exact a4 versions and their full round telemetry."""
import json
from datetime import datetime, timezone
from pathlib import Path
from check_aligned_backbones import safe_check, ROOT, OUT, d

rows=[]
jobs=[]
for owner, seed in [('buidinh0303',41),('haivohoanggg',42),('anduyvo06',43)]:
    slug=f'tabm-fl-v2x-htfl-s{seed}-fp32proj-a4'
    jobs.append(dict(ref=f'{owner}/{slug}',version=1,slug=slug,backbone='tabm',method='v2x_htfl',seed=seed))
mlp_ledger=ROOT/'runs/mlp-v2x-fp32proj-launch.json'
if mlp_ledger.exists():
    jobs.extend(j for j in json.loads(mlp_ledger.read_text()) if j['state']=='submitted')
for job in jobs:
    owner=job['ref'].split('/')[0]
    slug=job['slug']
    row=safe_check(job)
    if row['status']=='COMPLETE':
        try:
            api=d.make_api((Path.home()/'.kaggle/accounts'/f'{owner}.mcp-token').read_text().strip())
            api.kernels_output(job['ref']+'/'+str(job['version']),str(OUT/slug),file_pattern=r'.*/rounds\.jsonl$',force=True,quiet=True)
            history=[json.loads(s) for s in (OUT/slug/slug/'rounds.jsonl').read_text().splitlines()]
            row['history_contiguous']=[r['round'] for r in history]==list(range(1,51))
            row['max_skipped_frac']=max(r['skipped_frac'] for r in history)
            row['all_rounds_applied']=all(r['applied_steps']>0 for r in history)
        except Exception as exc:
            row['history_error']=type(exc).__name__
    rows.append(row)
    print(json.dumps(row),flush=True)
(ROOT/'runs/v2x-a4-live.json').write_text(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),rows=rows),indent=2))
