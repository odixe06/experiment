"""Train-support diagnostics for audited stats versus no-Gaussian validation CMs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from core.metrics import summary, ten


def analyze(root, evidence, output):
    import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=True)
    counts = np.zeros((5343, 16), dtype=np.int64)
    val = np.zeros_like(counts)
    files = sorted((root / "train").rglob("*.parquet"))
    assert files
    for i, path in enumerate(files):
        table = pq.ParquetFile(path).read(columns=["client_id", "label", "fold"])
        ids = table['client_id'].to_numpy().astype(np.int64)
        labels = table['label'].to_numpy().astype(np.int64)
        folds = table['fold'].to_numpy(zero_copy_only=False)
        assert ((ids >= 0) & (ids < 5343)).all()
        assert ((labels >= 0) & (labels < 16)).all()
        assert np.isin(folds, ['train', 'val']).all()
        for fold, target in [('train', counts), ('val', val)]:
            use = folds == fold
            target += np.bincount(ids[use] * 16 + labels[use], minlength=5343*16).reshape(5343,16)
        if i % 50 == 0:
            print('counted files', i + 1, '/', len(files), flush=True)
    assert counts.sum() == 32685166 and val.sum() == 4676107
    matrices = {}
    for arm in ['control', 'nogauss']:
        with np.load(evidence / (arm + '.npz')) as z:
            assert np.array_equal(z['ids'], np.arange(5343))
            cm = z['private']
            assert np.array_equal(cm.sum(2), val), 'CM true labels must match scanned val counts per client/class'
            matrices[arm] = cm
    np.savez_compressed(output / 'train_val_class_counts.npz', train=counts, val=val, ids=np.arange(5343))
    f1 = {arm: np.array([ten(cm)['f1_macro'] for cm in cms]) for arm,cms in matrices.items()}
    delta = f1['nogauss'] - f1['control']
    size = counts.sum(1)
    missing = (counts == 0).sum(1)
    with (output / 'clients.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['client_id','train_rows','missing_train_classes','control_f1','nogauss_f1','delta'])
        writer.writerows(zip(range(5343), size, missing, f1['control'], f1['nogauss'], delta))
    groups = {}
    for key, bins in [('train_rows', [0, 256, 1024, 4096, np.inf]), ('missing_classes', [0, 1, 5, 9, 17])]:
        values = size if key == 'train_rows' else missing
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (values >= lo) & (values < hi)
            if mask.any():
                groups[f'{key}:[{lo},{hi})'] = {'clients': int(mask.sum()), 'mean_delta': float(delta[mask].mean()),
                    'fraction_harmed': float((delta[mask] < 0).mean()),
                    'metrics': {arm: summary(cms[mask]) for arm,cms in matrices.items()}}
    per_class = {}
    for c in range(16):
        detail = {}
        for label, mask in [('absent', counts[:,c] == 0), ('1_to_63', (counts[:,c]>0)&(counts[:,c]<64)),
                            ('at_least_64', counts[:,c]>=64)]:
            mask &= val[:,c] > 0
            if mask.any():
                detail[label] = {'clients': int(mask.sum()), 'arms': {}}
                for arm, cms in matrices.items():
                    pooled = cms[mask].sum(0)
                    metrics = ten(pooled)
                    detail[label]['arms'][arm] = {'f1': metrics['f1_per_class'][c],
                        'recall': float(pooled[c,c]/pooled[c].sum()), 'support': int(pooled[c].sum())}
        per_class[str(c)] = detail
    confusions = {}
    for arm, cms in matrices.items():
        cm = cms.sum(0)
        confusions[arm] = {}
        for c in [11,13,15]:
            row, col = cm[c].copy(), cm[:,c].copy()
            row[c] = col[c] = 0
            confusions[arm][str(c)] = {'missed_as': [(int(j),int(row[j])) for j in row.argsort()[::-1][:5]],
                'false_positives_from': [(int(j),int(col[j])) for j in col.argsort()[::-1][:5]]}
    result = {'scope': 'train support; paired R10 full own-client validation; exploratory, not causal',
        'source_kernel': 'dangthiet/mobihet-v4-stats-nogauss-val-r10/1', 'files_scanned': len(files),
        'train_rows': int(counts.sum()), 'val_rows': int(val.sum()),
        'evidence_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence.glob('*.npz')},
        'groups': groups, 'class_support': per_class, 'confusions': confusions,
        'r10_metrics': {arm: summary(cms) for arm,cms in matrices.items()}}
    (output/'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'groups': {k: {n:v for n,v in g.items() if n!='metrics'} for k,g in groups.items()},
                      'focus_classes': {str(c): per_class[str(c)] for c in [11,13,15]}}, indent=2), flush=True)
    (output/'ANALYSIS_PASS.json').write_text(json.dumps({'train_counts': True, 'val_cm_identity': True, 'complete': True}))
