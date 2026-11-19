"""Explicit authorized seven-job wave; build, check absence, push and report status."""
import subprocess
import sys
from pathlib import Path

JOBS = [('hoangdv1711','perfedskd',41),('hoangdv1711','perfedskd',42),
        ('phamquang0606','perfedskd',43),('hoangdangnhi','fedproto',41),
        ('phamquang0606','fedproto',42),('minhiw','fedproto',43),('ttalgies','v2x_htfl',41)]

if __name__ == '__main__':
    cli = str(Path(sys.executable).parent/'Scripts/kaggle.exe')
    for owner, method, seed in JOBS:
        slug = f'tabm-fl-{method.replace("_", "-")}-s{seed}'
        prefix = [sys.executable,'.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py',owner,'--',cli,'kernels']
        check = subprocess.run(prefix+['status',owner+'/'+slug],capture_output=True,text=True)
        result = check.stdout+check.stderr
        if check.returncode == 0:
            print('EXISTS; SKIP',owner+'/'+slug,result,flush=True)
            continue
        if '404' not in result and 'not found' not in result.lower():
            listing = subprocess.run(prefix+['list','--mine','--page-size','100'],
                                     capture_output=True,text=True,check=True)
            if owner+'/'+slug in listing.stdout:
                raise RuntimeError('Existing inaccessible kernel: '+owner+'/'+slug)
            if len(listing.stdout.splitlines()) >= 100:
                raise RuntimeError('Listing may be truncated')
        subprocess.run([sys.executable,'scripts/build_tabm_fl.py','--owner',owner,
                        '--method',method,'--seed',str(seed)],check=True)
        subprocess.run(prefix+['push','-p','notebooks/'+slug,'-t','43200'],check=True)
        subprocess.run(prefix+['status',owner+'/'+slug],check=True)
