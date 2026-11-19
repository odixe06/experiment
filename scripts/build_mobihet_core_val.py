"""Build matched Core comparisons, train fold only, fixed last-round evaluation."""
import ast
import json
import sys
import nbformat
import build_mobihet_stats_val as template

template.OWNER = "dangthiet"
template.SLUG = "mobihet-core-copula-q-r10"
blend = "--blend-q" in sys.argv
alternate = "--alternate-q" in sys.argv or blend
if alternate:
    template.SLUG = "mobihet-core-mixed-q-every2-r10"
if blend:
    template.SLUG = "mobihet-core-mixed-q-blend-r10"
template.main()
folder = template.ROOT / "notebooks" / template.SLUG
path = folder / (template.SLUG + ".ipynb")
nb = nbformat.read(path,4)
nb.cells[0].source = """# MobiHet Core, train to validation
Five R10 arms, seed42, cosine50. CE versus Q at W1 and mixed rank widths W1/W0.5.
Fifth arm disables step floor with Q at W1. Compare Q against same-width CE.
Bootstrap train only, normal-score midranks and regularized correlation. R1 hard labels.
Claim improvement only for macro gain >=0.005, mean/p10 >=-0.005, class13 >=-0.01
and train overhead <=20%. Report modelled wire bytes separately. No automatic final run."""
nb.cells[2].source = nb.cells[2].source.replace('"-k", "mobihet_stats"', '"-k", "mobihet_core"')
import hashlib
embedded = {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(nb.cells[2].source).body if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('source','hashes')}
assert {k: hashlib.sha256(v.encode()).hexdigest() for k,v in embedded['source'].items()} == embedded['hashes']
smoke = nb.cells[5].source.replace('"mobihet_stats"', '"mobihet_core"')
smoke = smoke.replace('assert all(r["bytes_up"] == 0 and r["bytes_down"] == 0 for r in rows)', 'assert [(r["bytes_up"],r["bytes_down"]) for r in rows] == [(0,0),(576,512)]')
nb.cells[5].source = smoke.replace('"round_bytes": 0','"round2_bytes": 1088')
prod = nb.cells[6].source.split('arms =')[0]
nb.cells[6].source = prod + """
arms = [('ce', 'mobihet_core_ce', [1,1],16), ('q','mobihet_core',[1,1],16),
        ('mixed_ce','mobihet_core_ce',[1,0.5],16), ('mixed_q','mobihet_core',[1,0.5],16),
        ('q_no_floor','mobihet_core',[1,1],0)]
for name, method, widths, floor in arms:
    out = Path('/kaggle/working')/name
    cmd = common + ['--method',method,'--out',str(out),'--set',
                    'rank_widths='+json.dumps(widths),'synthetic_step_floor='+str(floor)]
    print(name,cmd,flush=True)
    subprocess.run(cmd,cwd='/kaggle/temp/src',check=True)
    rows = [json.loads(l) for l in (out/'rounds.jsonl').read_text().splitlines()]
    assert [r['round'] for r in rows]==list(range(1,11))
    for r in rows:
        use = method=='mobihet_core' and r['round']>1
        assert r['bytes_up']==576*use and r['bytes_down']==512*use
        assert r['q_forward_rows']==32685166*use
        assert r['skipped_frac'] < .01
    for checkpoint in out.glob('state*.pt'):
        if name != 'q_no_floor':
            checkpoint.unlink()
"""
nb.cells[7].source = """
import numpy as np
root=Path('/kaggle/working')
def read(name):
    return [json.loads(l) for l in (root/name/'rounds.jsonl').read_text().splitlines()]
results={}
for control,candidate in [('ce','q'),('mixed_ce','mixed_q'),('q_no_floor','q')]:
    a,b=read(control),read(candidate)
    delta={k:b[-1]['private/'+k]-a[-1]['private/'+k] for k in
           ['f1_macro','client_f1_macro_mean','client_f1_macro_p10']}
    delta['class13']=b[-1]['private/f1_per_class'][13]-a[-1]['private/f1_per_class'][13]
    delta['time_overhead']=np.mean([r['t_train'] for r in b[2:]])/np.mean([r['t_train'] for r in a[2:]])-1
    delta['pass']=bool(delta['f1_macro']>=.005 and delta['client_f1_macro_mean']>=-.005 and
                       delta['client_f1_macro_p10']>=-.005 and delta['class13']>=-.01 and delta['time_overhead']<=.2)
    results[candidate+' vs '+control]=delta
(root/'CORE_RESULTS.json').write_text(json.dumps(results,indent=2))
from core.metrics import summary
cache=Path('/kaggle/temp/v4npy')
off=np.load(cache/'train_off.npy')
client_ids=np.load(cache/'train_ids.npy')
fold=np.load(cache/'train_F.npy',mmap_mode='r')
labels=np.load(cache/'train_Y.npy',mmap_mode='r')
sizes=np.array([np.count_nonzero(fold[a:b]==0) for a,b in zip(off[:-1],off[1:])])
present=np.array([np.any((fold[a:b]==0)&(labels[a:b]==13)) for a,b in zip(off[:-1],off[1:])])
shard=np.zeros(len(sizes),int)
shard[np.argsort(-np.diff(off),kind='stable')[1::2]]=1
groups={'small_lt256':sizes<256,'other_ge256':sizes>=256,'class13_absent':~present,
        'class13_present':present,'rank0':shard==0,'rank1':shard==1}
reports={}
for name,method,widths,floor in arms:
    cmfile=np.load(root/name/'cms_r010.npz')
    assert np.array_equal(cmfile['ids'],client_ids)
    reports[name]={}
    for key,keep in groups.items():
        if not keep.any(): continue
        cms=cmfile['private'][keep]
        metrics=summary(cms)
        steps=(sizes[keep]+63)//64
        reports[name][key]={'clients':int(keep.sum()),'class13_support':int(cms[:,13,:].sum()),
            'real_rows':int(sizes[keep].sum()),'synthetic_rows':int(np.maximum(steps,floor).sum()*64),
            'macro':metrics['f1_macro'],'p10':metrics['client_f1_macro_p10'],'class13':metrics['f1_per_class'][13]}
(root/'CORE_GROUPS.json').write_text(json.dumps(reports,indent=2))
"""
nb.cells[-1].source="print((root/'CORE_RESULTS.json').read_text())"
if alternate:
    nb.cells[0].source = """# Mixed-width MobiHet Core, Q refreshed every second round
    Seed 42, floor 16, train to validation. Compare R10 with archived mixed CE and daily-round Q
    from dangthiet/mobihet-core-copula-q-r10 version 2, using the same gates."""
    smoke = nb.cells[5].source.replace('for stop in (1, 2):',
        'base += ["--set","q_interval=2","rank_widths=[1,0.5]"]\nfor stop in (2, 3):')
    smoke = smoke.replace('== [1, 2]', '== [1, 2, 3]').replace('[(0,0),(576,512)]','[(0,0),(576,512),(0,0)]')
    smoke = smoke.replace('for rank in (0, 1):','assert rows[1]["q13"] == rows[2]["q13"] and rows[2]["q_forward_rows"] == 0\nfor rank in (0, 1):')
    nb.cells[5].source = smoke.replace('== 2\n','== 3\n').replace('"gen_g"}','"gen_g","q"}').replace('"rounds": 2','"rounds": 3')
    run = nb.cells[6].source
    run = run[:run.index('arms =')] + "arms = [('mixed_q2','mobihet_core',[1,0.5],16)]\n" + run[run.index('for name, method'):run.index('    for checkpoint')]
    nb.cells[6].source = run.replace("'rank_widths='+", "'q_interval=2','rank_widths='+").replace("r['round']>1","r['round']%2==0") + """
    for i in (2,4,6,8):
        assert rows[i]['q13']==rows[i-1]['q13']
"""
    prior = template.ROOT/'runs/pulls/mobihet-core-copula-q-r10-v2'
    controls = {arm:[{k:v for k,v in json.loads(line).items() if k in
        ('private/f1_macro','private/client_f1_macro_mean','private/client_f1_macro_p10','private/f1_per_class','t_train')}
        for line in (prior/arm/'rounds.jsonl').read_text().splitlines()] for arm in ('mixed_ce','mixed_q')}
    report = nb.cells[7].source.replace('results={}', 'controls='+repr(controls)+'\nresults={}')
    nb.cells[7].source = report.replace("[('ce','q'),('mixed_ce','mixed_q'),('q_no_floor','q')]", "[('mixed_ce','mixed_q2'),('mixed_q','mixed_q2')]").replace('read(control),read(candidate)','controls[control],read(candidate)')
if blend:
    nb.cells[0].source = """# MobiHet Core mixed widths, 50% hard + 50% Q targets
Seed42, floor16, Q refresh even rounds, train-only to validation R10/cosine50.
Hypothesis: anchoring the synthetic target preserves class identity when pooled Q is uncertain.
Primary gate versus archived mixed CE: macro >= +.005, mean/p10 >= -.005,
class13 >= -.01, train overhead <=20%. Also compare archived every2 Q.
Same-seed exploratory validation. No automatic R50. No W&B."""
    nb.cells[5].source = nb.cells[5].source.replace('"q_interval=2"', '"q_interval=2","q_hard_weight=0.5"')
    nb.cells[6].source = nb.cells[6].source.replace("'q_interval=2'", "'q_interval=2','q_hard_weight=0.5'").replace('mixed_q2','mixed_blend')
    prior_q2 = template.ROOT/'runs/pulls/mobihet-core-mixed-q-every2-r10-v1/mixed_q2/rounds.jsonl'
    q2_rows = [json.loads(line) for line in prior_q2.read_text().splitlines()]
    controls['mixed_q2'] = [{k:row[k] for k in controls['mixed_ce'][0]} for row in q2_rows]
    report = nb.cells[7].source
    begin = report.index('controls=')
    end = report.index('\nresults={}',begin)
    report = report[:begin] + 'controls='+repr(controls) + report[end:]
    nb.cells[7].source = report.replace("[('mixed_ce','mixed_q2'),('mixed_q','mixed_q2')]", "[('mixed_ce','mixed_blend'),('mixed_q2','mixed_blend')]")
for cell in nb.cells:
    if cell.cell_type=='code':
        ast.parse(cell.source)
nbformat.validate(nb)
nbformat.write(nb,path)
print(path,json.loads((folder/'kernel-metadata.json').read_text())['id'])
