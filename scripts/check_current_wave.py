"""Read-only live status for final TabM and initial SAE wave."""
from concurrent.futures import ThreadPoolExecutor
from check_tabm_wave import check
from launch_tabm_final_sae_wave import JOBS

if __name__ == '__main__':
    jobs = [(o, f'{b}-fl-{m.replace("_", "-")}-s{s}') for o,b,m,s in JOBS]
    jobs.append(('qwerasdf3004','veremi-central-cnn-lstm-feature-e50'))
    from launch_sae_wave2 import JOBS as WAVE2
    jobs += [(o, f'sae-fl-{m.replace("_", "-")}-s{s}') for o,m,s in WAVE2]
    from launch_sae_wave3 import JOBS as WAVE3
    jobs += [(o, f'sae-fl-{m.replace("_", "-")}-s{s}') for o,m,s in WAVE3]
    from launch_sae_nguynt298 import JOBS as NEW_ACCOUNT
    jobs += [(o, f'sae-fl-{m.replace("_", "-")}-s{s}') for o,m,s in NEW_ACCOUNT]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(check,jobs):
            print(result,flush=True)
