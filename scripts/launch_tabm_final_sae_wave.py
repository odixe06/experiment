"""Authorized final TabM + initial SAE wave; preserve process-scoped credentials."""
import subprocess
import sys
from pathlib import Path

JOBS = [('dangthiet','tabm','canperfl',41),('dangthiet','tabm','canperfl',42),
        ('hoangdang1711','tabm','canperfl',43),('ttalgies','tabm','v2x_htfl',42),
        ('hoangdang1711','sae','fedavg',43),('hoangdangnhi','sae','fedavg',41),
        ('hoangdangnhi','sae','fedavg',42),('hoangdv1711','sae','fedprox',41),
        ('hoangdv1711','sae','fedprox',42),('minhiw','sae','fedprox',43),
        ('minhiw','sae','afpha',41),('phamquang0606','sae','afpha',42),
        ('phamquang0606','sae','afpha',43),('ttalgies','sae','fd_ids',41),
        ('qwerasdf3004','sae','fd_ids',42)]

if __name__ == '__main__':
    cli = str(Path(sys.executable).parent/'Scripts/kaggle.exe')
    for owner, backbone, method, seed in JOBS:
        slug = f'{backbone}-fl-{method.replace("_", "-")}-s{seed}'
        prefix = [sys.executable,'.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py',owner,'--',cli,'kernels']
        listing = subprocess.run(prefix+['list','--mine','--page-size','100','--search',slug],capture_output=True,text=True,check=True)
        if len(listing.stdout.splitlines()) >= 100:
            raise RuntimeError('Listing may be truncated')
        if owner+'/'+slug in listing.stdout:
            print('EXISTS; SKIP', owner+'/'+slug, flush=True)
            continue
        cmd = [sys.executable,'scripts/build_tabm_fl.py','--owner',owner,'--backbone',backbone,
               '--method',method,'--seed',str(seed)]
        if method == 'canperfl':
            parent = f'{owner}/tabm-fl-fedavg-s{seed}'
            if seed == 41:
                parent += '-probe'
            check = subprocess.run(prefix+['status',parent],capture_output=True,text=True,check=True)
            assert 'COMPLETE' in check.stdout
            cmd += ['--parent',parent]
        subprocess.run(cmd,check=True)
        subprocess.run(prefix+['push','-p','notebooks/'+slug,'-t','43200'],check=True)
        subprocess.run(prefix+['status',owner+'/'+slug],check=True)
