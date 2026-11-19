"""Read-only status check for the last launched TabM wave."""
import concurrent.futures
import subprocess
import sys
from build_sae_matrix import OWNERS


def check(job):
    owner, slug = job
    cmd = [sys.executable, '.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py', owner,
           '--', str(__import__('pathlib').Path(sys.executable).parent/'Scripts/kaggle.exe'),
           'kernels', 'status', owner+'/'+slug]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return owner+'/'+slug, p.returncode, p.stdout.strip(), p.stderr[-1500:]


if __name__ == '__main__':
    jobs = [(owner, f'tabm-fl-{method.replace("_", "-")}-s{seed}')
            for method in ('lwfednids', 'tinyproto', 'pfedes', 'nilm', 'fedtgp')
            for seed, owner in zip((41,42,43), OWNERS[method])]
    jobs += [('qwerasdf3004', 'tabm-fl-v2x-htfl-s43'),
             ('qwerasdf3004', 'veremi-central-cnn-lstm-feature-e50')]
    from launch_tabm_remaining_wave import JOBS
    jobs += [(owner, f'tabm-fl-{method.replace("_", "-")}-s{seed}') for owner, method, seed in JOBS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(check, jobs):
            print(result, flush=True)
