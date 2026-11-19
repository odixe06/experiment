"""Collect/resume authorized baseline chains for all three seeds; deterministic 1200s watcher."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/three-watch'

def main():
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        failures=[]; complete=[]; state={}
        for seed in (42,41,43):
            env=dict(os.environ,FLB_SEED=str(seed))
            cp=subprocess.run([sys.executable,'scripts/run_batch.py','collect'],cwd=ROOT,env=env,capture_output=True,text=True)
            (OUT/f'collect-s{seed}.log').write_text(cp.stdout+cp.stderr)
            print('COLLECT',seed,'rc',cp.returncode,flush=True)
            state[str(seed)]={'collect_rc':cp.returncode,'all_terminal':'ALL_TERMINAL' in cp.stdout}
            if cp.returncode:
                failures.append(seed)
                continue
            if state[str(seed)]['all_terminal']:
                print('AUDIT',seed,'start',flush=True)
                audit=subprocess.run([sys.executable,'scripts/run_batch.py','audit'],cwd=ROOT,env=env,capture_output=True,text=True)
                (OUT/f'audit-s{seed}.log').write_text(audit.stdout+audit.stderr)
                state[str(seed)]['audit_rc']=audit.returncode
                print('AUDIT',seed,'rc',audit.returncode,flush=True)
                if audit.returncode: failures.append(seed)
                else: complete.append(seed)
        (OUT/'state.json').write_text(json.dumps({'at':time.strftime('%FT%T%z'),'seeds':state},indent=2))
        if failures: raise RuntimeError(f'collect/audit failures seeds {failures}; inspect per-seed logs')
        if len(complete)==3:
            print('THREE_BASELINE_SEEDS_AUDITED',flush=True)
            return
        print('WAIT_1200S',flush=True)
        time.sleep(1200)
if __name__=='__main__': main()
