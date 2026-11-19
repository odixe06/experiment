"""Three-arm R10: original stats, histogram, histogram missingness repair."""
import ast
import hashlib

import nbformat
import build_mobihet_stats_realbn_val as builder


if __name__ == '__main__':
    builder.template.OWNER = 'dangthiet'
    builder.SLUG = 'mobihet-v4-stats-histmask-val-r10'
    builder.CANDIDATE = 'mobihet_stats_histmask'
    builder.DESCRIPTION = (
        'Histogram candidate additionally sets each of six zero-imputed features to zero when its NaN flag is one. '
        'No other changes to sampling RNG, loss, BN, model, optimizer or schedule. '
        'Three matched arms: original stats, unmodified histogram, repaired histogram. '
        'Primary locked gate remains relative to original stats; histogram comparison isolates repair. '
        'Additionally require positive macro delta vs histogram and mean/p10 >=-0.005, class13 >=-0.01, '
        'training overhead <=20% vs histogram. No automatic R50 launch. '
        'Bootstrap payload equals histogram; zero per-round bytes. No official test or W&B.'
    )
    builder.main()
    path = builder.template.ROOT/'notebooks'/builder.SLUG/(builder.SLUG+'.ipynb')
    nb = nbformat.read(path, 4)
    prod = nb.cells[6].source
    prod = prod.replace('("mobihet_stats_histmask",', '("mobihet_stats_hist", "mobihet-v4-histmask-control-hist-r10"), ("mobihet_stats_histmask",')
    prod = prod.replace('if method == "mobihet_stats":', 'if method != "mobihet_stats_histmask":')
    nb.cells[6].source = prod
    nb.cells[7].source += '''
hist = [json.loads(s) for s in Path('/kaggle/working/mobihet-v4-histmask-control-hist-r10/rounds.jsonl').read_text().splitlines()]
h = hist[-1]
hist_time = sum(r['t_train'] for r in hist[2:])/len(hist[2:])
repair = {'macro':s['private/f1_macro']-h['private/f1_macro'],
          'mean':s['private/client_f1_macro_mean']-h['private/client_f1_macro_mean'],
          'p10':s['private/client_f1_macro_p10']-h['private/client_f1_macro_p10'],
          'class13':s['private/f1_per_class'][13]-h['private/f1_per_class'][13],
          'time_overhead':candidate_time/hist_time-1}
repair_pass = repair['macro']>0 and repair['mean']>=-0.005 and repair['p10']>=-0.005 and repair['class13']>=-0.01 and repair['time_overhead']<=0.2
Path('/kaggle/working/REPAIR_GATE.json').write_text(json.dumps({'vs_histogram':repair,'repair_pass':repair_pass,'original_stats_gate':gate['pass'],'pass':gate['pass'] and repair_pass},indent=2))
print('REPAIR GATE', repair, repair_pass, flush=True)
'''
    for c in nb.cells:
        if c.cell_type == 'code': ast.parse(c.source)
    nbformat.validate(nb)
    nbformat.write(nb,path)
    print('THREE ARM SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
