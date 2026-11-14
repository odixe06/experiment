import json
from pathlib import Path
import numpy as np

FL2 = Path('/home/odixe/project/fl_dagsnet_veremi_v2')
MH = Path('/home/odixe/project/mobihet')

specs = [
    ('fedavg', 'global', 'fl2-fedavg'),
    ('fedprox', 'global', 'fl2-fedprox'),
    ('afpha', 'global', 'fl2-afpha-r2'),
    ('fd_ids', 'global', 'fl2-fd-ids-r2'),
    ('apfed', 'client', 'fl2-apfed-r2'),
    ('lwfednids', 'global', 'fl2-lwfednids-r2'),
    ('v2x_htfl', 'client', 'fl2-v2x-htfl-r2'),
    ('canperfl', 'client', 'fl2-canperfl'),
    ('pfedes', 'client', 'fl2-pfedes-r2'),
    ('fedtgp', 'client_proto', 'fl2-fedtgp-r2'),
    ('perfedskd', 'client', 'fl2-perfedskd-r2'),
    ('fedproto', 'client_proto', 'fl2-fedproto'),
    ('tinyproto', 'client_proto', 'fl2-tinyproto-r4'),
]

ten_cols = ['accuracy', 'precision_macro', 'precision_micro', 'precision_weighted',
            'recall_macro', 'recall_micro', 'recall_weighted',
            'f1_macro', 'f1_micro', 'f1_weighted',
            'client_f1_macro_mean', 'client_f1_macro_p10', 'client_f1_macro_min']

results = {}
for m, v, dir_name in specs:
    p42 = list((FL2 / 'runs/pulls').glob(f'{dir_name}/**/rounds.jsonl'))[0]
    l42 = [json.loads(l) for l in p42.read_text().splitlines()][-1]
    m42 = {c: l42[f'{v}/{c}'] for c in ten_cols}
    s = m.replace('_', '-')
    d41 = json.loads((FL2 / f'runs/seeds/audit/fl2-{s}-s41.json').read_text())
    m41 = {c: d41['last_metrics'][v][c] for c in ten_cols}
    d43 = json.loads((FL2 / f'runs/seeds/audit/fl2-{s}-s43.json').read_text())
    m43 = {c: d43['last_metrics'][v][c] for c in ten_cols}
    
    stats = {}
    for c in ten_cols:
        vals = [m41[c], m42[c], m43[c]]
        stats[c] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    results[m] = stats

with open(MH / 'runs/cbb-s41/audit.json') as f:
    mh = json.load(f)

mh_stats = {}
for c in ten_cols:
    vals = [mh['r50'][str(seed)][c] for seed in [41, 42, 43]]
    mh_stats[c] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
results['mobihet'] = mh_stats

def fmt(m, c):
    mean, sd = results[m][c]
    return f"{mean:.4f} ± {sd:.4f}"

# Generate new table 1
with open('/home/odixe/.gemini/antigravity-cli/brain/23911446-f552-4d01-b120-52d83befda76/.system_generated/steps/554/output.txt') as f:
    p1_data = json.loads(f.read())
content1 = p1_data['text']

table1_start = content1.find('<table header-row="true">')
table1_end = content1.find('</table>') + len('</table>')
old_table1 = content1[table1_start:table1_end]

t1_map = {
    'FedAvg': 'fedavg',
    'FedProx': 'fedprox',
    'AFPHA': 'afpha',
    'FD-IDS': 'fd_ids',
    'APFed': 'apfed',
    'LwFedNIDS': 'lwfednids',
    'V2X-HtFL': 'v2x_htfl',
    'CANPerFL': 'canperfl',
    'pFedES': 'pfedes',
    'FedTGP': 'fedtgp',
    'PerFed-SKD': 'perfedskd',
    'FedProto': 'fedproto',
    'TinyProto': 'tinyproto',
    'MobiHet KM16 - 50 round': 'mobihet',
}

t1_cols = ['accuracy', 'precision_macro', 'precision_micro', 'precision_weighted',
           'recall_macro', 'recall_micro', 'recall_weighted',
           'f1_macro', 'f1_micro', 'f1_weighted',
           'client_f1_macro_mean', 'client_f1_macro_p10']

t1_rows = old_table1.split('<tr>\n')
new_t1_rows = [t1_rows[0]]
for r_str in t1_rows[1:]:
    found_m = None
    for k in t1_map:
        if f">{k}<" in r_str or f">{k} " in r_str or f">{k}\n" in r_str:
            found_m = t1_map[k]
            break
    if not found_m:
        new_t1_rows.append(r_str)
        continue
    tds = r_str.split('<td>\n')
    new_tds = [tds[0], tds[1], tds[2], tds[3]]
    for idx, c in enumerate(t1_cols):
        new_tds.append(f"{fmt(found_m, c)}\n</td>\n")
    new_row = '<td>\n'.join(new_tds)
    if not new_row.endswith('</tr>\n') and not new_row.endswith('</tr>'):
        new_row = new_row.rstrip() + '\n</tr>\n'
    new_t1_rows.append(new_row)

new_table1 = '<tr>\n'.join(new_t1_rows)

# Generate new table 2
with open('/home/odixe/.gemini/antigravity-cli/brain/23911446-f552-4d01-b120-52d83befda76/.system_generated/steps/558/output.txt') as f:
    p2_data = json.loads(f.read())
content2 = p2_data['text']

table2_start = content2.find('<table header-row="true">')
table2_end = content2.find('</table>') + len('</table>')
old_table2 = content2[table2_start:table2_end]

t2_map = {
    'APFed': 'apfed',
    'CANPerFL': 'canperfl',
    'MobiHet (KM16+Codec)': 'mobihet',
    'FedAvg': 'fedavg',
    'FedProx': 'fedprox',
    'FD-IDS': 'fd_ids',
    'AFPHA': 'afpha',
    'LwFedNIDS': 'lwfednids',
    'FedTGP': 'fedtgp',
    'PerFed-SKD': 'perfedskd',
    'FedProto': 'fedproto',
    'V2X-HtFL': 'v2x_htfl',
    'TinyProto': 'tinyproto',
    'pFedES': 'pfedes',
}

t2_cols = ['accuracy', 'recall_macro', 'f1_macro', 'client_f1_macro_p10', 'client_f1_macro_min']

t2_rows = old_table2.split('<tr>\n')
new_t2_rows = [t2_rows[0]]
for r_str in t2_rows[1:]:
    found_m = None
    for k in t2_map:
        if f"{k}\n" in r_str or f"<b>{k}</b>" in r_str or f">{k}<" in r_str:
            found_m = t2_map[k]
            break
    if not found_m:
        new_t2_rows.append(r_str)
        continue
    tds = r_str.split('<td>\n')
    new_tds = [tds[0], tds[1], tds[2], tds[3]]
    is_bold = (found_m == 'mobihet')
    for idx, c in enumerate(t2_cols):
        val_str = fmt(found_m, c)
        if is_bold:
            new_tds.append(f"<b>{val_str}</b>\n</td>\n")
        else:
            new_tds.append(f"{val_str}\n</td>\n")
    new_row = '<td>\n'.join(new_tds)
    if not new_row.endswith('</tr>\n') and not new_row.endswith('</tr>'):
        new_row = new_row.rstrip() + '\n</tr>\n'
    new_t2_rows.append(new_row)

new_table2 = '<tr>\n'.join(new_t2_rows)

Path('/tmp/old_table1.txt').write_text(old_table1)
Path('/tmp/new_table1.txt').write_text(new_table1)
Path('/tmp/old_table2.txt').write_text(old_table2)
Path('/tmp/new_table2.txt').write_text(new_table2)

print("SUCCESS: Generated new_table1 and new_table2")
