"""Collect exact-version lightweight progress artifacts for continuation planning."""
import concurrent.futures as cf
import json
import sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/cloud_dispatch'), str(ROOT/'.agents/skills/kaggle-training-notebook/scripts')]
import dispatcher as d
from pull_kernel_output import fetch
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def collect(job):
    row = dict(job)
    try:
        api = d.make_api((Path.home()/'.kaggle/accounts'/f"{job['owner']}.mcp-token").read_text().strip())
        row['status'] = d.read_call(lambda: api.kernels_status(job['ref'])).status.name
        if row['status'] not in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'):
            return row
        files = {}; token = None
        with api.build_kaggle_client() as client:
            while True:
                req = ApiListKernelSessionOutputRequest()
                req.user_name, req.kernel_slug = job['owner'], job['slug']
                req.version_label, req.page_size = 'v'+str(job['version']), 200
                if token: req.page_token = token
                reply = d.read_call(lambda: client.kernels.kernels_api_client.list_kernel_session_output(req))
                files.update({f.file_name: f.url for f in reply.files or []})
                token = reply.next_page_token
                if not token: break
        dest = ROOT/'runs/mixed128-progress'/job['slug']
        wanted = {'RUN_STATUS.json','checkpoint.json','rounds.jsonl','config.json','experiment_manifest.json','source_manifest.json'}
        for name, url in files.items():
            if Path(name).name in wanted:
                with requests.Session() as session: fetch(session,url,dest/name,60)
        row['output_count'] = len(files)
        for name, key in [('RUN_STATUS.json','marker'),('checkpoint.json','checkpoint')]:
            path = dest/job.get('artifact_slug',job['slug'])/name
            if path.exists(): row[key] = json.loads(path.read_text())
        path = dest/job.get('artifact_slug',job['slug'])/'rounds.jsonl'
        if path.exists():
            history = [json.loads(line) for line in path.read_text().splitlines()]
            assert [r['round'] for r in history] == list(range(1,len(history)+1))
            row['last_round'] = len(history)
            row['last_metrics'] = history[-1] if history else {}
        # Private, ignored URLs refreshed each collection; never print these.
        (dest/'output-urls.json').write_text(json.dumps(files))
    except Exception as exc: row['error'] = type(exc).__name__
    return row

if __name__ == '__main__':
    jobs = [j for j in json.loads((ROOT/'runs/mixed128-production-launch.json').read_text()) if j['case'] != 'case2']
    rows = []
    with cf.ThreadPoolExecutor(4) as pool:
        for row in pool.map(collect,jobs):
            rows.append(row)
            (ROOT/'runs/mixed128-progress.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps({k:row[k] for k in ('ref','status','last_round','marker','error') if k in row}),flush=True)
