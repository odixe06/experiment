"""Explicit nine-job SAE wave into newly free account slots."""
import subprocess
import sys
from pathlib import Path

JOBS = [('dangthiet','fd_ids',43),('hoangdang1711','apfed',41),
        ('hoangdang1711','apfed',42),('hoangdangnhi','apfed',43),
        ('minhiw','lwfednids',41),('phamquang0606','lwfednids',42),
        ('ttalgies','lwfednids',43),('qwerasdf3004','tinyproto',41),
        ('qwerasdf3004','tinyproto',42)]

def launch(jobs):
    cli = str(Path(sys.executable).parent/'Scripts/kaggle.exe')
    for owner, method, seed in jobs:
        slug = f'sae-fl-{method.replace("_", "-")}-s{seed}'
        prefix = [sys.executable,'.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py',owner,'--',cli,'kernels']
        result = subprocess.run(prefix+['list','--mine','--page-size','100','--search',slug],
                                capture_output=True,text=True,check=True)
        assert len(result.stdout.splitlines()) < 100
        if owner+'/'+slug in result.stdout:
            print('EXISTS; SKIP',owner+'/'+slug,flush=True)
            continue
        subprocess.run([sys.executable,'scripts/build_tabm_fl.py','--backbone','sae',
                        '--owner',owner,'--method',method,'--seed',str(seed),'--max-hours','7'],check=True)
        subprocess.run(prefix+['push','-p','notebooks/'+slug,'-t','28800'],check=True)
        subprocess.run(prefix+['status',owner+'/'+slug],check=True)


if __name__ == '__main__':
    launch(JOBS)
