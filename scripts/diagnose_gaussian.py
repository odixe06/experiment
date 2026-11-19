"""Full-train moments, sampled real/synthetic transfer and domain diagnostics; no official test."""
import json
from pathlib import Path

import numpy as np

from core.metrics import ten


CLASSES = (0, 11, 13, 15)
CAP = {'train': 20000, 'val': 10000}


def retain(old, x, ids, priority, cap):
    """Uniform reservoir via smallest independent random priorities, bounded memory."""
    if old is not None:
        x = np.concatenate([old[0], x])
        ids = np.concatenate([old[1], ids])
        priority = np.concatenate([old[2], priority])
    if len(x) > cap:
        use = np.argpartition(priority, cap - 1)[:cap]
        x, ids, priority = x[use], ids[use], priority[use]
    return x, ids, priority


def main(root, output, order):
    import pyarrow.parquet as pq
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import confusion_matrix, roc_auc_score

    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    pools = {(f,c): None for f in CAP for c in CLASSES}
    n = np.zeros(16, np.int64)
    totals = {'train': 0, 'val': 0}
    s1 = {c: np.zeros(77) for c in CLASSES}
    s2 = {c: np.zeros((77,77)) for c in CLASSES}
    offset, manifest = 0, []
    files = sorted((root/'train').rglob('*.parquet'))
    assert len(order) == 77 and files
    for index,path in enumerate(files):
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=order+['label','fold']):
            y = batch.column(batch.schema.get_field_index('label')).to_numpy()
            fold = batch.column(batch.schema.get_field_index('fold')).to_numpy(zero_copy_only=False)
            assert np.isin(fold, list(CAP)).all() and ((y>=0)&(y<16)).all()
            # Match resident training quantization before fitting moments.
            x = np.column_stack([batch.column(batch.schema.get_field_index(k)).to_numpy() for k in order]).astype(np.float16).astype(np.float32)
            assert np.isfinite(x).all()
            ids = np.arange(offset, offset+len(y), dtype=np.int64)
            for f in CAP:
                totals[f] += int((fold==f).sum())
                for c in CLASSES:
                    use = (fold==f)&(y==c)
                    if not use.any():
                        continue
                    xc = x[use]
                    if f == 'train':
                        xd = xc.astype(np.float64)
                        n[c] += len(xc); s1[c] += xd.sum(0); s2[c] += xd.T@xd
                    pools[f,c] = retain(pools[f,c], xc, ids[use], rng.random(len(xc)), CAP[f])
            count += len(y); offset += len(y)
        manifest.append({'file': path.relative_to(root).as_posix(), 'rows':count})
        if index % 25 == 0:
            print('scan', index+1, '/', len(files), totals, flush=True)
    assert totals == {'train':32685166, 'val':4676107}, totals
    assert all(len(pools[f,c][0]) == CAP[f] for f,c in pools)
    arrays, gaussian, flag_report = {}, {}, {}
    flags = [i for i,k in enumerate(order) if k.endswith('_nan')]
    continuous = [i for i in range(77) if i not in flags]
    assert len(flags) == 6
    for c in CLASSES:
        mu = s1[c]/n[c]
        cov = s2[c]/n[c] - np.outer(mu,mu) + 1e-4*np.eye(77)
        chol = np.linalg.cholesky(cov).astype(np.float32)
        raw = mu.astype(np.float32) + rng.standard_normal((20000,77)).astype(np.float32)@chol.T
        gaussian[c] = np.clip(raw,-4,4).astype(np.float16).astype(np.float32)
        arrays[f'mu_{c}'], arrays[f'chol_{c}'] = mu, chol
        arrays[f'gaussian_{c}'] = gaussian[c]
        for f in CAP:
            x, ids, _ = pools[f,c]
            arrays[f'{f}_x_{c}'], arrays[f'{f}_ids_{c}'] = x, ids
        real_flags, fake_flags = pools['train',c][0][:,flags], gaussian[c][:,flags]
        flag_report[str(c)] = {'flags': [order[i] for i in flags],
            'real_nonbinary_fraction': ((real_flags!=0)&(real_flags!=1)).mean(0).tolist(),
            'gaussian_nonbinary_fraction': ((fake_flags!=0)&(fake_flags!=1)).mean(0).tolist(),
            'gaussian_outside_0_1_fraction': ((fake_flags<0)|(fake_flags>1)).mean(0).tolist(),
            'preclip_outside_4_fraction': float((np.abs(raw)>4).mean())}
    np.savez_compressed(output/'samples_moments.npz', **arrays)
    (output/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'SCAN_PASS.json').write_text(json.dumps({'totals':totals,'classes':list(CLASSES),'counts':n.tolist()}))
    print('full train moments and sampling PASS', flush=True)

    def classifier():
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1,
                                              early_stopping=False, random_state=42)

    result = {'scope':'sampled balanced binary diagnostics, not full FL metrics', 'seed':42,
              'sklearn':sklearn.__version__, 'totals':totals, 'class_train_counts':n.tolist(),
              'sampler':'full train moments, numpy RNG; distributional match, not bit-exact torch replay',
              'cap_per_class':CAP, 'flags':flag_report, 'domain':{}, 'transfer':{}}
    predictions = {}
    for c in CLASSES:
        real, fake = pools['train',c][0], gaussian[c]
        # Disjoint row samples; train-domain split is not a client-held-out generalization test.
        xfit, xeval = np.concatenate([real[:10000],fake[:10000]]), np.concatenate([real[10000:],fake[10000:]])
        y = np.repeat([0,1],10000)
        result['domain'][str(c)] = {}
        for view,cols in [('all',list(range(77))), ('without_nan_flags',continuous)]:
            model = classifier().fit(xfit[:,cols], y)
            prob = model.predict_proba(xeval[:,cols])[:,1]
            cm = confusion_matrix(y,prob>=0.5,labels=[0,1])
            result['domain'][str(c)][view] = {'auc':float(roc_auc_score(y,prob)), 'cm':cm.tolist(),'metrics':ten(cm)}
            predictions[f'domain_{c}_{view}_prob'] = prob
        print('domain',c,result['domain'][str(c)]['all']['auc'],flush=True)
    predictions['domain_y'] = np.repeat([0,1],10000)
    for a,b in [(0,13),(11,15)]:
        key=f'{a}_{b}'
        vx=np.concatenate([pools['val',a][0],pools['val',b][0]])
        vy=np.repeat([0,1],10000)
        result['transfer'][key] = {}
        predictions[key+'_val_y'] = vy
        for source in ['real','gaussian']:
            tx=np.concatenate([pools['train',c][0] if source=='real' else gaussian[c] for c in [a,b]])
            ty=np.repeat([0,1],20000)
            model=classifier().fit(tx,ty)
            prob=model.predict_proba(vx)[:,1]
            cm=confusion_matrix(vy,prob>=0.5,labels=[0,1])
            result['transfer'][key][source]={'auc':float(roc_auc_score(vy,prob)), 'cm':cm.tolist(), 'metrics':ten(cm)}
            predictions[key+'_'+source+'_prob']=prob
            print('transfer',key,source,result['transfer'][key][source]['metrics']['f1_macro'],flush=True)
    np.savez_compressed(output/'predictions.npz', **predictions)
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2))
    (output/'DIAGNOSTIC_PASS.json').write_text(json.dumps({'complete':True,'official_test_read':False}))
