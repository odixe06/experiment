"""Rebuild Proposal 2 tables from audited endpoints and explicit cost estimates; no training/network."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics

MOB = ROOT.parent / 'mobihet'
DAG = ROOT.parent / 'fl_dagsnet_veremi_v2'
OUT = ROOT / 'runs/maintenance-20261116'
NAMES = dict(fedavg='FedAvg', fedprox='FedProx', afpha='AFPHA', fd_ids='FD-IDS', apfed='APFed',
             lwfednids='LwFedNIDS', v2x_htfl='V2X-HtFL', canperfl='CANPerFL', tinyproto='TinyProto',
             pfedes='pFedES', nilm='NILM-FL', fedtgp='FedTGP', perfedskd='PerFed-SKD', fedproto='FedProto', mobihet='MobiHet')
HEAD = dict(fedproto='client_proto', fedtgp='client_proto', tinyproto='client_proto')
KEYS = ['accuracy', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall_macro', 'recall_micro',
        'recall_weighted', 'f1_macro', 'f1_micro', 'f1_weighted', 'client_f1_macro_p10', 'client_f1_macro_min']


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(x) for x in (p / 'rounds.jsonl').read_text().splitlines() if x.strip()]


def audit_metric(p):
    a = read(p)
    assert a['status'] == 'PASS_BOUNDED' and (a['complete'] or (a['method'] == 'canperfl' and a['rounds'] == 10)), p
    m = a['method']; v = HEAD.get(m, 'global' if 'global' in a['last_metrics'] and m not in ['apfed', 'canperfl'] else 'client')
    if v not in a['last_metrics']: v = next(iter(a['last_metrics']))
    return m, v, a['last_metrics'][v]


def bold(m, cells):
    return ['**' + str(x) + '**' for x in cells] if m == 'mobihet' else list(map(str, cells))


def table(header, data):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in data])


def build():
    evidence = dict(BERT={}, DAGSNet={}); runpaths = dict(BERT={}, DAGSNet={})
    for seed in [41, 42, 43]:
        batch = ROOT / ('runs/batch' if seed == 42 else f'runs/batch-s{seed}')
        for p in sorted((batch / 'audit').glob('*.json')):
            m, v, stats = audit_metric(p)
            evidence['BERT'].setdefault(m, []).append(dict(seed=seed, view=v, metrics=stats, source=str(p)))
        for p in sorted((DAG / 'runs/seeds/audit').glob(f'*-s{seed}.json')) if seed != 42 else []:
            if 'scaffold' in p.name: continue
            m, v, stats = audit_metric(p)
            evidence['DAGSNet'].setdefault(m, []).append(dict(seed=seed, view=v, metrics=stats, source=str(p)))
    for c in (OUT / 'dagsnet42').glob('*/**/config.json'):
        cfg = read(c); m = cfg['method']; p = c.parent
        if m not in NAMES: continue
        endpoint = 10 if m == 'canperfl' else 50
        z = np.load(p / f'cms_r{endpoint:03d}.npz'); v = HEAD.get(m, 'global' if 'global' in z.files and m not in ['apfed', 'canperfl'] else 'client')
        cm = z[v]; assert np.array_equal(cm.sum(2), np.load(ROOT / 'runs/test_support.npy'))
        evidence['DAGSNet'].setdefault(m, []).append(dict(seed=42, view=v, metrics=metrics.summary(cm), source=str(p)))
        runpaths['DAGSNet'][m] = p
    for seed, slug in [(41, 'flb-t3-s41-p3'), (42, 'flb-t3-s42-p3'), (43, 'flb-t3-s43-p4')]:
        a = read(ROOT / 'runs/t3' / f'audit-{slug}.json'); st = read(ROOT / 'runs/t3' / f'state-audit-{slug}.json')
        assert a['complete'] and st['complete']
        v = 'private'; evidence['BERT'].setdefault('mobihet', []).append(dict(seed=seed, view=v, metrics=a['last_metrics'][v], source=slug))
        runpaths['BERT'].setdefault('mobihet', []).append(ROOT / 'runs/t3/pulls' / slug / f'flb-t3-s{seed}')
    for key, p in [('T341', MOB / 'runs/t3-s41/audit.json'), ('T3P', MOB / 'runs/b21-r50/audit.json'), ('T343', MOB / 'runs/b21-r50/audit.json')]:
        a = read(p)[key]; assert a['checks']['pass']
        seed = {'T341':41, 'T3P':42, 'T343':43}[key]
        evidence['DAGSNet'].setdefault('mobihet', []).append(dict(seed=seed, view='private', metrics=a['endpoint'], source=str(p)))
    runpaths['DAGSNet']['mobihet'] = [MOB / 'artifacts/t3-s41/mh-k-b21-t341-r50/mh-k-b21-t341-r50',
        MOB / 'artifacts/b21-chain/mh-k-b21-t3p-r20-s36/mh-k-b21-t3p-r20', MOB / 'artifacts/b21-chain/mh-k-b21-t343-r50-s7/mh-k-b21-t343-r50']
    for model in evidence:
        assert set(evidence[model]) == set(NAMES), (model, set(NAMES) - set(evidence[model]))
        assert all(sorted(x['seed'] for x in ss) == [41,42,43] for ss in evidence[model].values())
    # Latest audited baseline BERT paths (seed42 communication estimate only).
    sys.path.insert(0, str(ROOT / 'scripts')); import run_batch as batch
    _, sessions = batch.sessions()
    for m in NAMES:
        if m == 'mobihet': continue
        rr = [r for r in sessions.values() if r['method'] == m and (batch.OUT / (r['slug']+'.done')).exists()]
        r = max(rr, key=lambda r:r['part'])
        runpaths['BERT'][m] = next(p.parent for p in (ROOT/'runs/pulls'/r['slug']).rglob('rounds.jsonl') if p.parent.name == r['run'])
    sizes = np.load(MOB / 'cache/train_sizes.npy'); counts = np.load(MOB / 'cache/fold_counts.npy').sum(1)
    assert int(sizes.sum()) == 37361273 and counts.shape == (5343,16)
    assert np.array_equal(counts.sum(1), sizes)
    steps = (sizes + 63) // 64; tier = np.digitize(sizes, [4096,16384]); doses = np.array([64,48,32])[tier]
    # Match B21's step-normalized logical-work convention, not padded GPU work.
    synth_ratio = float((np.maximum(steps,16) * doses / 64).sum() / steps.sum())
    perf, comm, client, server = {}, {}, {}, {}
    for model in evidence:
        perf[model] = []; comm[model] = []; client[model] = []; server[model] = []
        cost = read(ROOT / 'docs/research/cost.json' if model == 'BERT' else DAG / 'docs/research/cost.json')
        cdata = cost['methods']; fwd = cost.get('forward_flops_per_row', cost.get('dagsnet_forward_flops_per_row'))
        profile = read(OUT / ('fl_veremi_bert-server.json' if model == 'BERT' else 'fl_dagsnet_veremi_v2-server.json'))
        for m, ss in evidence[model].items():
            avg = {k:float(np.mean([s['metrics'][k] for s in ss])) for k in KEYS}
            std = {k:float(np.std([s['metrics'][k] for s in ss], ddof=1)) for k in KEYS}
            perf[model].append(dict(method=m, view=ss[0]['view'], mean=avg, std=std))
            if m == 'mobihet':
                ps = runpaths[model][m]; cfgs = [read(p/'config.json') for p in ps]
                up = np.mean([np.mean([x['bytes_up'] for x in rows(p)]) for p in ps]); down = np.mean([np.mean([x['bytes_down'] for x in rows(p)]) for p in ps])
                boot = np.mean([sum(c['payload_bytes']['bootstrap'].values()) for c in cfgs]); masks = 0; frac = 1.
                flops = cdata['fedavg']['client_flops_per_row'] * (1+synth_ratio)
                ops = 2*5343*16*16 + 5343*16 + 3*16*16  # weight decoded means, sum counts, normalize Q
                detail = '2 K C² + K C + 3 C²; weighted Q aggregation/normalization, excludes codec'
            else:
                pb = cdata['fedavg' if m == 'canperfl' else m]['payload_bytes']; up, down = pb['up'], pb['down']; boot = pb.get('masks_once',0)
                frac = np.mean([x['selected_frac'] for x in rows(runpaths[model][m])]) if m == 'perfedskd' else 1.
                flops = cdata['fedavg' if m == 'canperfl' else m]['client_flops_per_row'] * (1.2 if m == 'canperfl' else 1.)
                d = 64 if model == 'BERT' else 256; k = 5343; p = 77904 if model == 'BERT' else 398672
                if m == 'lwfednids':p = round(up/4)
                if m == 'pfedes':p = 351
                if m in ['fedproto','tinyproto']:
                    dim = d if m == 'fedproto' else (d+9)//10
                    ops = int((counts>0).sum())*dim + 16*dim; detail = 'Np d_eff + C d_eff; aggregation only'
                elif m in ['fedtgp','v2x_htfl']:
                    ops = profile['methods'][m]['server_matmul_flops_per_round']; detail = 'FlopCounter: setup/final +100×epoch; excludes elementwise/optimizer'
                else:
                    ops = int(2*k*p*frac+p); detail = '2 K_selected (P+B)+(P+B); weighted aggregation'
                    if m == 'afpha':detail += '; lower bound, excludes adaptive clustering/distances'
            total = float(50*(up+down)*frac)
            comm[model].append(dict(method=m, up=float(up), down=float(down), fraction=float(frac), bootstrap=float(boot), iterative50=total, total50=total+float(boot)))
            client[model].append(dict(method=m, flops_per_row=float(flops), x_forward=float(flops/fwd)))
            server[model].append(dict(method=m, ops_per_round=int(ops), scope=detail))
    result=dict(evidence=evidence, performance=perf, communication=comm, client_compute=client, server_compute=server, t3_synth_ratio=synth_ratio)
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    text=['# Proposal 2 — MobiHet (T3)', '', METHOD_TEXT, '', '## 1. Performance', '', 'Endpoint50; CANPerFL=FedAvg50 +10epoch fine-tune. Mean ± sample SD,3seed41/42/43. Ranking Macro-F1 giảm dần. Global/own-test/prototype khác luật suy luận; không gọi các view đồng nhất.']
    for model in evidence:
        data=[]
        for i,r in enumerate(sorted(perf[model],key=lambda r:-r['mean']['f1_macro']),1):
            cells=[i,NAMES[r['method']],r['view']]+[f"{r['mean'][k]:.4f} ± {r['std'][k]:.4f}" for k in ['accuracy','recall_macro','f1_macro','client_f1_macro_p10','client_f1_macro_min']]
            data.append(bold(r['method'],cells))
        text+=['',f'### {model}', '',table(['Hạng','Phương pháp','View','Accuracy','Macro recall','Macro F1','Client P10 F1','Client min F1'],data)]
    text+=['','## 2. Communication budget','','Đơn vị B, MB=10⁶B; unicast từng xe, cả upload+download. Tổng50=bootstrap+Σround(up+down); tổng mạng=5343×tổng/xe. Chưa tính header mạng/retransmission. Baseline=payload_bytes(fp32), PerFed-SKD nhân selected_frac đo seed42; TinyProto thêm mask một lần. MobiHet=bytes_up/down50round, bootstrap trong config, trung bình3seed. Ranking tổng gồmbootstrap tăng dần; bootstrap luôn tách riêng.']
    for model in evidence:
        data=[]
        for i,r in enumerate(sorted(comm[model],key=lambda r:r['total50']),1):
            data.append(bold(r['method'],[i,NAMES[r['method']],f"{r['up']:.1f}",f"{r['down']:.1f}",f"{r['fraction']:.4f}",f"{r['bootstrap']/1e6:.6f}",f"{r['iterative50']/1e6:.6f}",f"{r['total50']/1e6:.6f}",f"{r['total50']*5343/1e9:.4f}"]))
        text+=['',f'### {model}', '',table(['Hạng','Phương pháp','Up B/xe/round','Down B/xe/round','Tỷ lệ chọn','Bootstrap MB/xe','50round không bootstrap MB/xe','Tổng MB/xe','Mạng GB'],data)]
    text+=['','## 3. Computational cost','','Client: cost.py đo một round2 toy CPU không đệm bằng FlopCounterMode (matmul/conv/bmm/SDPA), chia dòng train. Không tính elementwise/optimizer/codec/eval/bootstrap; không suy ra thời gian T4. BERT forward10.225664MFLOP, DAGSNet4.452096MFLOP. Baseline giữ cùng quy ước đo; PerFed-SKD trên dòng xe được chọn. CANPerFL=(50+10)/50×FedAvg, phân bổ fine-tune vào50round.', '', f'MobiHet ước lượng công việc hữu ích theo B21: c_train×(1+ρ), ρ=Σ max(ceil(n_k/64),16)×mean_dose_k/(64Σceil(n_k/64))={synth_ratio:.6f}; mean_dose S/M/L=64/48/32. Dùng c_train riêng từng model, không nhân4.94 DAGSNet sang BERT. Chưa cộng sinh copula/codec/bootstrap. Dòng mask/đệm vẫn có thể được kernel thực thi; đây không phải phép đo total GPU FLOPs.']
    for model in evidence:
        data=[]
        for i,r in enumerate(sorted(client[model],key=lambda r:r['flops_per_row']),1):
            data.append(bold(r['method'],[i,NAMES[r['method']],f"{r['flops_per_row']/1e6:.3f}",f"{r['x_forward']:.3f}",'ước lượng T3' if r['method']=='mobihet' else 'toy measured; CANPerFL amortized' if r['method']=='canperfl' else 'toy measured']))
        text+=['',f'### Client — {model}', '',table(['Hạng','Phương pháp','MFLOP/dòng/round','×forward','Loại bằng chứng'],data)]
    text+=['','Server: steady-state một round; loại client training/eval và bootstrap. K=5343,C=16,d=64(BERT)/256(DAGSNet), Np=count ô train+val có lớp. Aggregation ước lượng số phép nhân/cộng/chia từ code; server neural step đo matmul bằng FlopCounterMode,100epoch,20batch/epoch như implementation. Ranking theo con số trong phạm vi nêu; không là ranking latency thực. AFPHA là cận dưới do chưa tính clustering; không suy thứ hạng tuyệt đối khi cost bỏ sót có thể đổi thứ tự. Bootstrap copula/KM16 cần gộp histogram/moment, Lloyd4lượt và factorization cụm; chưa đo, không tính là0 hoặc<5ms.']
    for model in evidence:
        data=[]
        for i,r in enumerate(sorted(server[model],key=lambda r:r['ops_per_round']),1):
            data.append(bold(r['method'],[i,NAMES[r['method']],f"{r['ops_per_round']/1e6:.3f}",r['scope']]))
        text+=['',f'### Server — {model}', '',table(['Hạng trong phạm vi','Phương pháp','Triệu FLOP/operation mỗi round','Cách tính/phạm vi'],data)]
    text+=['','## Nguồn và giới hạn','','CM/support→10metric audit PASS_BOUNDED; T3 bổ sung finite private banks, strict rebuild và RNG/source/HP hai rank khi có state audit. Chưa stored prediction→CM replay toàn mạng; chưa full-content dataset fingerprint. Không tuyên bố mixed-model full-scale, secure aggregation/DP hoặc significance từ3seed.', '', '[BERT source/results](https://github.com/odixe06/fl_veremi_bert) · [MobiHet source/design](https://github.com/odixe06/mobihet) · [Baseline DAGSNet](https://github.com/odixe06/fl_dagsnet_veremi_v2)', '', 'Chi tiết số liệu12metric/seed và nguồn từng hàng: `runs/maintenance-20261116/comparison.json` ở repo BERT; bảng này tái sinh bằng `scripts/build_project_comparison.py`.']
    (OUT/'proposal2.md').write_text('\n'.join(text)+'\n')
    print('COMPARISON_TABLES_READY')


METHOD_TEXT = r'''MobiHet hiện tại là **T3**, owner chọn15/11/2026; các bảng chỉ dùng T3, không dùng kết quả Core/CBB/KM16 cũ thay T3.
VeReMi NextGen v4:5343xe nhận,77đầu vào,16lớp; train+val37361273dòng, test9347679dòng temporal holdout.50round×1epoch thật, batch64, AdamW cosine1e-3→1e-5, clip1, AMP; seed41/42/43.

Bootstrap một lần: histogram256bin/lớp/feature→CDF toàn mạng; normal-score mid-rank; moment lớp và k-means liên kết16cụm/lớp,4lượt Lloyd, covariance shrink w0=.5, loại cụm<20dòng. Lite6 nén moment/tâm int8, Cholesky6bit, CDF lossless. Xe tự sinh từ mixture copula; không gửi dòng thật/model/gradient/BN.

T3 phân tầng theo n_k(train+val): S<4096, M4096–16383, L≥16384. Synthetic/bước r1–25→r26–50: **S64→64; M64→32; L48→16**. Floor16bước; thiếu bước thật thì chỉ thêm synthetic, không thêm epoch thật. Loss synthetic lấy mean riêng, trọng số1; giảm số dòng không giảm hệ số loss.

Prior Laplace π_k,c=(n_k,c+1)/(n_k+16), π_g,c=(Σn_k,c+1)/(Σn_k+16). Loss thật=CE(z+logπ_k−logπ_g,y), synthetic dùng logits thô. Round1 synthetic hardCE; round2+ KL(Q_(r−1)[y] || softmax(f_k(x_syn))).

Online Q_r,c=ΣkΣi:y=c softmax(f_k(x_i))/Σkn_k,c, thu trên dòng thật ngay lúc train, dùng từr+1; không thêm lượt forward riêng. QQ8 gửi xác suất8bit, uplink chỉ lớp hiện diện. Mỗi xe giữ model riêng; schema độc lập kiến trúc. BERT/DAGSNet là hai thí nghiệm riêng, chưa full-scale trộn model.'''

if __name__ == '__main__':
    build()
