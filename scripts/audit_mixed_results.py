"""Collect final-round artifacts, verify fixed coverage and independently recompute metrics."""
import concurrent.futures
import csv
import json
from pathlib import Path
import sys
import requests
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/cloud_dispatch'),str(ROOT/'.agents/skills/kaggle-training-notebook/scripts')]
import dispatcher as d
from pull_kernel_output import fetch
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
from core import metrics
from core.model_assignment import load_assignment

def audit(job):
    row={k:job[k] for k in ('case','method','seed','ref','version')}
    api=d.make_api((Path.home()/'.kaggle/accounts'/f"{job['owner']}.mcp-token").read_text().strip())
    try:
        assert d.kernel_metadata(api,job['ref']).current_version_number==job['version']
        row['status']=d.read_call(lambda:api.kernels_status(job['ref'])).status.name
        if row['status']!='COMPLETE': return row
        slug=job.get('artifact_slug',job['slug']);dest=ROOT/'runs/mixed128-final-audit'/job['slug']
        needed={slug+'/'+n for n in ('cms_r050.npz','round_r050.json','RUN_STATUS.json','config.json')}
        needed.add('experiment_manifest.json')
        urls={};token=None
        with api.build_kaggle_client() as client:
            while True:
                req=ApiListKernelSessionOutputRequest()
                req.user_name,req.kernel_slug,req.page_size=job['owner'],job['slug'],200
                req.version_label='v'+str(job['version'])
                if token:req.page_token=token
                reply=d.read_call(lambda:client.kernels.kernels_api_client.list_kernel_session_output(req))
                urls.update({f.file_name:f.url for f in reply.files or [] if f.file_name in needed})
                token=reply.next_page_token
                if not token:break
        assert set(urls)==needed, 'Missing final artifacts'
        def download(item):
            name,url=item
            with requests.Session() as session:fetch(session,url,dest/name,60)
        with concurrent.futures.ThreadPoolExecutor(5) as pool:list(pool.map(download,urls.items()))
        marker=json.loads((dest/slug/'RUN_STATUS.json').read_text())
        cfg=json.loads((dest/slug/'config.json').read_text())
        manifest=json.loads((dest/'experiment_manifest.json').read_text())
        for obj in (marker,manifest):
            assert all(obj[k]==job[k] for k in ('case','method','seed'))
        assert marker['complete'] and marker['last_round']==50
        assert cfg['smoke_rows']==0 and len(cfg['clients'])==5343
        assert cfg['assignment_sha256']==manifest['assignment_sha256']
        history=json.loads((dest/slug/'round_r050.json').read_text());assert history['round']==50
        mapping=load_assignment(ROOT/'experiments/mixed-model-v1',job['case'])
        row['views']={};row['by_model']={}
        with np.load(dest/slug/'cms_r050.npz') as arr:
            ids=arr['ids'];assert len(ids)==5343 and set(ids)==set(mapping)
            for view in arr.files:
                if view=='ids':continue
                cms=arr[view]
                assert cms.shape==(5343,16,16) and (cms>=0).all() and int(cms.sum())==9347679
                assert (cms.sum((1,2))>0).all()
                stat=metrics.summary(cms)
                for key in metrics.TEN:assert abs(stat[key]-history[f'{view}/{key}'])<1e-10
                row['views'][view]=stat
                for model in set(mapping.values()):
                    sub=metrics.summary(cms[[mapping[int(i)]==model for i in ids]])
                    for key in metrics.TEN:assert abs(sub[key]-history[f'model/{model}/{view}/{key}'])<1e-10
                    row['by_model'].setdefault(model,{})[view]=sub
        row['audit']='PASS'
    except Exception as exc:row['audit_error']=type(exc).__name__
    return row

def main():
    jobs=[j for j in json.loads((ROOT/'runs/mixed128-production-launch.json').read_text()) if j['case']=='case2']
    repairs=ROOT/'runs/mixed128-v2x-repair-launch.json'
    if repairs.exists():
        for replacement in json.loads(repairs.read_text()):
            if replacement['state']=='submitted':
                jobs=[j for j in jobs if (j['method'],j['seed'])!=(replacement['method'],replacement['seed'])]
                jobs.append(replacement)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for row in pool.map(audit,jobs):
            results.append(row)
            (ROOT/'runs/mixed128-final-audit.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k not in ('views','by_model')}),flush=True)
    dest=ROOT/'docs/research/mixed/mixed128-case2-results.csv'
    with dest.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=['method','seed','round','group','view']+metrics.TEN)
        writer.writeheader()
        for row in results:
            if row.get('audit')!='PASS':continue
            groups={'all':row['views'],**row['by_model']}
            for group,views in groups.items():
                for view,stat in views.items():
                    writer.writerow(dict(method=row['method'],seed=row['seed'],round=50,group=group,view=view,
                                         **{k:f'{stat[k]:.5f}' for k in metrics.TEN}))
    lines=['# Case 2 — kết quả mixed128 round 50','',
        '30% TabM, 30% SAE, 40% MLP; giữ nguyên dữ liệu từng xe. '+
        'Ô có số đã qua final-CM audit: đủ5343xe/9347679dòng và10metric. Ô chưa đạt để trống.',
        '', '| Phương pháp | View | Seed41 | Seed42 | Seed43 |', '|---|---|---:|---:|---:|']
    for method in ('fedproto','fedtgp','tinyproto','v2x_htfl','pfedes'):
        view='client_proto' if method in ('fedproto','fedtgp','tinyproto') else 'client'
        vals=[]
        for seed in (41,42,43):
            row=next(r for r in results if r['method']==method and r['seed']==seed)
            vals.append(f"{row['views'][view]['f1_macro']:.5f}" if row.get('audit')=='PASS' else '')
        lines.append('| '+' | '.join([method,view]+vals)+' |')
    lines+=['','Macro-F1 headline theo view của từng phương pháp; không đổi sang classifier head để chọn điểm cao hơn.',
        'CSV kèm đủ10metric cho mỗi seed, mỗi model và cả hai view nếu có. V2X dùng a2 (SAE trainFP32), các phương pháp khác dùng a1.']
    (ROOT/'docs/research/mixed/mixed128-case2-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT_WRITTEN', len(results), flush=True)

if __name__=='__main__':main()
