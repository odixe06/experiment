"""Final-CM bounded audit and tables; never infer completion from Kaggle status alone."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import metrics

BASE = ROOT/'runs/comparison-20261115'
REPORT = ROOT/'docs/research/backbones/tabm-sae-dagsnet-comparison-20261115.md'
METHODS = ['fedavg','fedprox','afpha','fd_ids','apfed','lwfednids','canperfl',
           'tinyproto','pfedes','nilm','fedtgp','perfedskd','fedproto','v2x_htfl']
NAMES = ['FedAvg','FedProx','AFPHA','FD-IDS','APFed','LwFedNIDS','CANPerFL',
         'TinyProto','pFedES','NILM-FL','FedTGP','PerFed-SKD','FedProto','V2X-HtFL']
VIEW = {m:('global' if m in ('fedavg','fedprox','afpha','fd_ids','lwfednids') else
           'client_proto' if m in ('tinyproto','fedtgp','fedproto') else 'client') for m in METHODS}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(item):
    folder = Path(item['folder'])
    manifest = read(folder/'experiment_manifest.json')
    source, hashes = read(folder/'source_snapshot.json'), read(folder/'source_manifest.json')
    assert {k:hashlib.sha256(v.encode()).hexdigest() for k,v in source.items()}==hashes
    configs = list(folder.glob('*/config.json'))
    assert len(configs)==1
    run = configs[0].parent
    cfg = read(configs[0])
    rows = [json.loads(line) for line in (run/'rounds.jsonl').read_text().splitlines() if line.strip()]
    status = read(folder/'RUN_STATUS.json')
    method, seed = manifest['method'], manifest['seed']
    backbone = cfg['cfg']['model']
    target = 10 if method=='canperfl' else 50
    assert manifest['rounds']==target and cfg['method']==method and cfg['hp']['seed']==seed
    assert backbone in ('tabm','sae') and cfg['clients']==5343
    assert manifest['parameters']=={'tabm':6548,'sae':7613}[backbone]
    arch = cfg['cfg']
    h = arch['head_hidden']
    if backbone == 'tabm':
        k = arch['members']
        expected = 77*h + k*77 + 2*k*h + h*h + 3*k*h + k*h*16 + k*16
        assert h == (10 if method == 'lwfednids' else 32)
    else:
        eh, dh = arch['encoder_hidden'], arch['decoder_hidden']
        expected = 78*eh + (eh+1)*h + (h+1)*dh + (dh+1)*77 + (77+h+1)*16
        assert (eh,h,dh) == ((10,5,10) if method == 'lwfednids' else (32,16,32))
    if method == 'v2x_htfl':
        expected += h*h + 2*h
    assert cfg['n_params'] == expected, (cfg['n_params'], expected)
    assert manifest['dataset']=='odixe0502/veremi-nextgen-v4-scaled-per-vehicle'
    assert [r['round'] for r in rows]==list(range(1,target+1))
    assert status['complete'] and status['last_round']==target and status['all_round_metrics']
    for r in rows:
        assert all(math.isfinite(x) for x in r.values() if isinstance(x,(float,int)))
        if 'skipped_frac' in r:
            assert r['skipped_frac']<0.01
    final = rows[-1]
    summaries = {}
    with np.load(run/f'cms_r{target:03d}.npz',allow_pickle=False) as z:
        ids = z['ids']
        assert ids.shape==(5343,) and len(np.unique(ids))==5343
        for view in set(z.files)-{'ids'}:
            cm = z[view]
            assert cm.shape==(5343,16,16) and np.issubdtype(cm.dtype,np.integer)
            assert (cm>=0).all() and cm.sum()==9347679
            summary = metrics.summary(cm)
            for key in metrics.TEN+['client_f1_macro_mean','client_f1_macro_p10','client_f1_macro_min']:
                assert math.isclose(summary[key],final[f'{view}/{key}'],rel_tol=1e-12,abs_tol=1e-12), (view,key)
            assert np.allclose(summary['f1_per_class'],final[f'{view}/f1_per_class'],atol=1e-12,rtol=1e-12)
            summaries[view] = summary
        support = z[VIEW[method]].sum(2)[np.argsort(ids)]
    result = {'backbone':backbone,'method':method,'seed':seed,'round':target,'view':VIEW[method],
              'ref':item['ref'],'version':item['version'],'parameters':cfg['n_params'],
              'metrics':summaries[VIEW[method]],'views':summaries,'hp':cfg['hp'],
              'step_health_coverage':sum('skipped_frac' in r for r in rows),
              'source_hashes':hashes,'support_sha256':hashlib.sha256(support.tobytes()).hexdigest(),
              'ids_sha256':hashlib.sha256(np.sort(ids).tobytes()).hexdigest(),
              'scope':'final CM and metrics; contiguous complete round history; source hashes/config; no weights or all-round CM replay'}
    (folder/'COMPARISON_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result, rows


def dagsnet():
    text = (BASE/'dagsnet/docs__research__results.md').read_text(encoding='utf-8').split('## Lịch sử')[0]
    found = {}
    for line in text.splitlines():
        if not line.startswith('| '): continue
        columns = [c.strip() for c in line.strip('|').split('|')]
        if columns[0] not in METHODS: continue
        assert '(headline)' in columns[1]
        found[columns[0]] = dict(zip(metrics.TEN,[float(v.replace(',','.')) for v in columns[3:13]]))
    assert set(found)==set(METHODS)
    return found


def cell(value):
    return f'{value:.5f}'


def main():
    collection = read(BASE/'collection.json')
    audited, histories, failures = [], {}, []
    for item in collection:
        try:
            row, history = audit(item)
            audited.append(row)
            histories[(row['backbone'],row['method'],row['seed'])] = history
            print('PASS',row['backbone'],row['method'],row['seed'],row['round'],flush=True)
        except Exception as exc:
            failures.append({'ref':item['ref'],'error':type(exc).__name__+': '+str(exc)})
            print('FAIL',item['ref'],type(exc).__name__,str(exc),flush=True)
    (BASE/'audit_summary.json').write_text(json.dumps({'audited':audited,'failures':failures},indent=2),encoding='utf-8')
    lookup = {(r['backbone'],r['method'],r['seed']):r for r in audited}
    assert len(lookup)==len(audited)
    assert len({r['support_sha256'] for r in audited})==1, 'test class counts differ between runs'
    assert len({r['ids_sha256'] for r in audited})==1, 'test client IDs differ between runs'
    baseline = dagsnet()
    def group(b,m,key):
        rows = [lookup.get((b,m,s)) for s in (41,42,43)]
        if not all(rows): return 'N/A'
        values = [r['metrics'][key] for r in rows]
        return f'{np.mean(values):.5f} ± {np.std(values,ddof=1):.5f}'
    lines = ['# DAGSNet – TabM – SAE: 14 baseline FL', '',
             'Ngày15/11/2026. Lịch cloud giữ tắt; chỉ collect/audit kết quả.', '',
             'Round50; CANPerFL = FedAvg50 + fine-tune10. Cùng dataset v4,5343client,9347679 dòng test.',
             'DAGSNet: seed42, số từ bảng đã audit trên GitHub main commit68a01b29cb3dd47d221d48a16289a8c9f1dfa155; nguồn đã làm tròn4số lẻ.',
             'TabM/SAE: mean ± sample standard deviation (ddof=1), seeds41/42/43;5số lẻ. Không chọn best round/seed.', '',
             '## Macro-F1 cuối — tổng hợp3seed cho TabM/SAE', '',
             '| Phương pháp | View | DAGSNet seed42 | TabM mean ± SD | SAE mean ± SD |',
             '|---|---|---:|---:|---:|']
    for m,name in zip(METHODS,NAMES):
        lines.append(f'| {name} | {VIEW[m]} | {baseline[m]["f1_macro"]:.4f} | {group("tabm",m,"f1_macro")} | {group("sae",m,"f1_macro")} |')
    lines += ['', '## Cùng seed42 — Macro-F1', '', '| Phương pháp | DAGSNet | TabM | SAE |', '|---|---:|---:|---:|']
    for m,name in zip(METHODS,NAMES):
        vals = [cell(lookup[(b,m,42)]['metrics']['f1_macro']) if (b,m,42) in lookup else 'N/A' for b in ('tabm','sae')]
        lines.append(f'| {name} | {baseline[m]["f1_macro"]:.4f} | '+ ' | '.join(vals)+' |')
    lines += ['', '## Giới hạn so sánh', '',
              '- DAGSNet hiện dùng1seed theo owner; chưa khôi phục đủ artifacts3seed từ máy cũ. Không suy ý nghĩa thống kê từ bảng này.',
              '- TinyProto: DAGSNet r4 dùng μ=0.00015; bản TabM/SAE thiếu overrideμ và lấy heuristic từ train counts. Đây không phải phép so sánh chỉ đổi backbone.',
              '- Headline của FedProto/FedTGP/TinyProto là client_proto; các phương pháp personalized còn lại là client, không đổi sang view có điểm cao hơn.',
              '- Audit mới chỉ kiểm CM cuối,10metric/per-class/client distribution,config/source và history đủround; chưa replay weights hoặc audit mọi CM từnground.',
              '- Không chạy lại hoặc sửa model/hyperparameter trong lượt báo cáo.', '',
              '## Đủ10metric cuối — mỗi model/phương pháp', '',
              'Single-label multiclass: accuracy = precision_micro = recall_micro = recall_weighted = f1_micro.', '',
              '| Model | Method | Seeds | '+' | '.join(metrics.TEN)+' |',
              '|---|---|---|'+ '---:|'*len(metrics.TEN)]
    for m in METHODS:
        lines.append('| DAGSNet | '+m+' | 42 | '+' | '.join(f'{baseline[m][k]:.4f}' for k in metrics.TEN)+' |')
        for b in ('tabm','sae'):
            lines.append('| '+b+' | '+m+' | 41,42,43 | '+' | '.join(group(b,m,k) for k in metrics.TEN)+' |')
    lines += ['', '## Nguồn TabM/SAE từngseed', '', '| Model | Method | Seed | Kernel/version | Final macro-F1 |', '|---|---|---:|---|---:|']
    for r in audited:
        ref = r['ref']+'/'+str(r['version'])
        lines.append(f'| {r["backbone"]} | {r["method"]} | {r["seed"]} | [{ref}](https://www.kaggle.com/code/{ref}) | {cell(r["metrics"]["f1_macro"])} |')
    lines += ['', 'Raw results/audit: `runs/comparison-20261115/audit_summary.json`.',
              'Bảng từnground đủ10metrics: [histories](tabm-sae-round-history-20261115.md).', '',
              'DAGSNet source: [results.md tại commit cố định](https://github.com/odixe06/fl_dagsnet_veremi_v2/blob/68a01b29cb3dd47d221d48a16289a8c9f1dfa155/docs/research/results.md).', '']
    if failures:
        lines += ['## Chưa xác minh được', '', *[f'- {r["ref"]}: {r["error"]}' for r in failures], '']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    detailed = ['# Lịch sử test từnground — TabM/SAE', '', 'Headline view cố định theo phương pháp; số từ artifacts history, CM cuối đã kiểm độc lập.',
                'accuracy = precision_micro = recall_micro = recall_weighted = f1_micro.', '']
    for (b,m,s),rows in sorted(histories.items()):
        detailed += [f'## {b} / {m} / seed{s} / view={VIEW[m]}', '', '| Round | '+' | '.join(metrics.TEN)+' |', '|---:|'+ '---:|'*len(metrics.TEN)]
        for row in rows:
            detailed.append('| '+str(row['round'])+' | '+' | '.join(cell(row[VIEW[m]+'/'+k]) for k in metrics.TEN)+' |')
        detailed.append('')
    REPORT.with_name('tabm-sae-round-history-20261115.md').write_text('\n'.join(detailed),encoding='utf-8')
    print('REPORT',REPORT,'AUDITED',len(audited),'FAILURES',len(failures),flush=True)
    return int(bool(failures))


if __name__=='__main__':
    sys.exit(main())
