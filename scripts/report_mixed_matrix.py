"""Report audited round-50 scores separately from partial progress."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT_DIR=ROOT/'docs/research/mixed'
TEN=['accuracy','precision_macro','precision_micro','precision_weighted','recall_macro','recall_micro','recall_weighted','f1_macro','f1_micro','f1_weighted']
METHODS=['fedproto','fedtgp','tinyproto','v2x_htfl','pfedes']
def main():
    rows={}
    for name in ('mixed128-final-audit.json','mixed128-v2x-repair-audit.json','mixed128-case1-final-audit.json','mixed128-case3-final-audit.json','mixed128-resume-final-audit.json','mixed128-final-audit-20261119.json','mixed128-final-audit-refresh.json'):
        path=ROOT/'runs'/name
        if path.exists():
            for r in json.loads(path.read_text()):
                if r.get('audit')=='PASS':rows[(r['case'],r['method'],r['seed'])]=r
    provisional={}
    provisional_path=ROOT/'runs/mixed128-provisional-final.json'
    if provisional_path.exists():
        for r in json.loads(provisional_path.read_text()):
            key=(r['case'],r['method'],r['seed'])
            if key not in rows:
                provisional[key]=r
    progress={(r['case'],r['method'],r['seed']):r for r in json.loads((ROOT/'runs/mixed128-progress.json').read_text())}
    latest=ROOT/'runs/mixed128-next-progress.json'
    if latest.exists():
        progress.update({(r['case'],r['method'],r['seed']):r for r in json.loads(latest.read_text())})
    today=ROOT/'runs/mixed128-progress-20261119.json'
    if today.exists():
        progress.update({(r['case'],r['method'],r['seed']):r for r in json.loads(today.read_text())})
    latest_check=ROOT/'runs/mixed128-progress-20261119-b.json'
    if latest_check.exists():
        progress.update({(r['case'],r['method'],r['seed']):r for r in json.loads(latest_check.read_text())})
    refresh=ROOT/'runs/mixed128-results-refresh.json'
    if refresh.exists():
        for r in json.loads(refresh.read_text()):
            key=(r['case'],r['method'],r['seed'])
            progress[key]={**progress.get(key,{}),**r}
            if 'last_round' not in r and 'resume_round' in r:
                progress[key]['last_round']=r['resume_round']
    latest_v2x=ROOT/'runs/mixed128-two-new-accounts-progress.json'
    if latest_v2x.exists():
        for r in json.loads(latest_v2x.read_text()):
            key=(r['case'],r['method'],r['seed'])
            progress[key]={**progress.get(key,{}),**r}
    launch=ROOT/'runs/mixed128-resume-launch.json'
    resumed={(r['case'],r['method'],r['seed']):r for r in json.loads(launch.read_text())} if launch.exists() else {}
    repairs=ROOT/'runs/mixed128-bert-repair-launch.json'
    repaired={(r['case'],r['method'],r['seed']):r for r in json.loads(repairs.read_text())} if repairs.exists() else {}
    repair_resumes=ROOT/'runs/mixed128-bert-repair-resume-launch.json'
    if repair_resumes.exists():
        repaired.update({(r['case'],r['method'],r['seed']):r for r in json.loads(repair_resumes.read_text())})
    wave=ROOT/'runs/mixed128-capacity-wave-launch.json'
    if wave.exists():
        for r in json.loads(wave.read_text()):
            (repaired if r['method']=='v2x_htfl' else resumed)[(r['case'],r['method'],r['seed'])]=r
    today_wave=ROOT/'runs/mixed128-launch-20261119.json'
    if today_wave.exists():
        for r in json.loads(today_wave.read_text()):
            (repaired if r['method']=='v2x_htfl' else resumed)[(r['case'],r['method'],r['seed'])]=r
    latest_wave=ROOT/'runs/mixed128-launch-20261119-b.json'
    if latest_wave.exists():
        for r in json.loads(latest_wave.read_text()):
            (repaired if r['method']=='v2x_htfl' else resumed)[(r['case'],r['method'],r['seed'])]=r
    new_accounts=ROOT/'runs/mixed128-two-new-accounts-launch.json'
    if new_accounts.exists():
        for r in json.loads(new_accounts.read_text()):
            (repaired if r['method']=='v2x_htfl' else resumed)[(r['case'],r['method'],r['seed'])]=r
    wave_27=ROOT/'runs/mixed128-27accounts-launch.json'
    if wave_27.exists():
        for r in json.loads(wave_27.read_text()):
            (repaired if r['method']=='v2x_htfl' else resumed)[(r['case'],r['method'],r['seed'])]=r
    final_v2x=ROOT/'runs/mixed128-final-v2x-launch.json'
    if final_v2x.exists():
        for r in json.loads(final_v2x.read_text()):
            repaired[(r['case'],r['method'],r['seed'])]=r
    final3=ROOT/'runs/mixed128-final3-launch.json'
    if final3.exists():
        for r in json.loads(final3.read_text()):
            (repaired if r['method']=='v2x_htfl' else resumed)[(r['case'],r['method'],r['seed'])]=r
    status_line=f'Đã audit round 50: **{len(rows)}/45** cấu hình.'
    if provisional:
        status_line+=f' Có **{len(provisional)}** kết quả complete50 tạm điền từ marker/history, chờ audit confusion matrix.'
    metric_line=('Macro-F1 round 50; '+('số không có dấu `*` ' if provisional else 'mọi số ')+
           'đã kiểm tra confusion matrix: 5.343 xe, 9.347.679 dòng test. '+
           'Giữ dữ liệu từng xe và mapping model cố định; seed 41/42/43. Ô trống = chưa có kết quả round 50.')
    lines=['# Mixed-model — kết quả 3 case','',status_line,'',metric_line,
           '', 'Case 1: 30% DAGSNet / 30% BERT / 40% CNN-BiGRU-Attention. '+
           'Case 2: 30% TabM / 30% SAE / 40% MLP. '+
           'Case 3: mỗi model DAGSNet/BERT/CNN-BiGRU-Attention/TabM/SAE 20%.',
           '', 'FedProto/FedTGP/TinyProto: client_proto; V2X-HtFL/pFedES: client. Projection 128D cho prototype; pFedES giữ head native.','']
    for case in ('case1','case2','case3'):
        lines += [f'## {case}','','| Phương pháp | Seed 41 | Seed 42 | Seed 43 |','|---|---:|---:|---:|']
        for method in METHODS:
            view='client_proto' if method in METHODS[:3] else 'client'
            values=[]
            for seed in (41,42,43):
                r=rows.get((case,method,seed))
                if r:
                    values.append(f"{r['views'][view]['f1_macro']:.5f}")
                elif (case,method,seed) in provisional:
                    values.append(f"{provisional[(case,method,seed)]['metrics']['f1_macro']:.5f}*")
                else:
                    values.append('')
            lines.append('| '+' | '.join([method]+values)+' |')
        lines.append('')
    pending={key:r for key,r in progress.items() if key not in rows and key not in provisional}
    if pending:
        lines+=['## Chưa đủ 50 round','','| Case | Phương pháp | Seed | Round đã lưu | Trạng thái |','|---|---|---:|---:|---|']
    for key,r in sorted(pending.items()):
        state='Lỗi, cần xử lý' if r['status']=='ERROR' else 'Dừng ngân sách giờ; nối checkpoint'
        if key in resumed:
            run=resumed[key]
            state=f"Nối tiếp: {run.get('status') or run['state']} — {run['owner']}"
        if key in repaired:
            run=repaired[key]
            state=f"Bản sửa a2: {run.get('status') or run['state']} — {run['owner']}"
        active=repaired.get(key,resumed.get(key))
        if active and active['ref']==r['ref']:
            state=f"{r['status']} — {active['owner']}" if r['status']!='COMPLETE' else 'Dừng ngân sách giờ; chờ quota nối tiếp'
        lines.append('| '+' | '.join(map(str,[*key,r.get('last_round',''),state]))+' |')
    if provisional:
        lines+=['','`*` = complete round 50 và đủ 10 metric trong history, nhưng tải confusion matrix đang bị Kaggle timeout; chưa tính vào kết quả audit. CSV chỉ chứa kết quả đã audit.']
    lines+=['','CSV kèm đủ 10 metric, từng seed/model/view. Kết quả dở dang không dùng thay round 50.']
    (REPORT_DIR/'mixed128-all-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (REPORT_DIR/'mixed128-all-results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['case','method','seed','round','group','view']+TEN,
                         lineterminator='\n');w.writeheader()
        for (case,method,seed),r in sorted(rows.items()):
            for group,views in {'all':r['views'],**r['by_model']}.items():
                for view,stat in views.items():
                    w.writerow(dict(case=case,method=method,seed=seed,round=50,group=group,view=view,**{k:f'{stat[k]:.5f}' for k in TEN}))
    print('AUDITED_FINAL_RUNS',len(rows))

if __name__=='__main__':main()
