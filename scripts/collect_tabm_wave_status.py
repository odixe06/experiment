"""Pull small status/metrics artifacts from newly completed wave runs."""
import concurrent.futures
import subprocess
import sys
import json
from pathlib import Path

JOBS = [('hoangdangnhi','tinyproto',42), ('hoangdv1711','pfedes',41),
        ('hoangdv1711','pfedes',42), ('minhiw','pfedes',43),
        ('phamquang0606','nilm',42), ('phamquang0606','nilm',43), ('ttalgies','fedtgp',41)]


def collect(job):
    owner, method, seed = job
    slug = f'tabm-fl-{method}-s{seed}'
    out = Path('runs/pulls')/(slug+'-status-check')
    cmd = [sys.executable,'.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py',owner,'--',
           sys.executable,'.agents/skills/kaggle-training-notebook/scripts/pull_kernel_output.py',
           owner+'/'+slug,'-p',str(out),'-j','4','--pattern',
           r'(^|/)(RUN_STATUS|experiment_manifest|SMOKE_PASS|config)\.json$|(^|/)rounds\.jsonl$']
    p = subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    if p.returncode:
        return {'job':slug,'error':p.stderr[-700:]}
    status = json.loads((out/'RUN_STATUS.json').read_text())
    rows = [json.loads(s) for s in (out/slug/'rounds.jsonl').read_text().splitlines()]
    assert [r['round'] for r in rows] == list(range(1,51))
    assert status['complete'] and status['last_round'] == 50
    final = rows[-1]
    return {'job':slug,'round':50,'metrics':{k:v for k,v in final.items() if k.endswith('/f1_macro')}}


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(collect,JOBS):
            print(json.dumps(item),flush=True)
