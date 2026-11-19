"""Resume the authorized DVC pushes only after the actual OneDrive mount is ready."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/maintenance-20261116/reboot-backup'
OUT.mkdir(exist_ok=True)
os.environ['PATH'] = str(Path(sys.executable).parent) + ':' + os.environ['PATH']
mount = Path('/home/odixe/Cloud/OneDrive')
deadline = time.monotonic() + 6 * 3600
while not os.path.ismount(mount):
    if time.monotonic() > deadline:
        raise RuntimeError('OneDrive mount did not become ready; no files written under unmounted path')
    time.sleep(10)
print('ONEDRIVE_MOUNT_READY', flush=True)
for repo, targets in [(ROOT, ['artifacts/baseline-pulls.dvc', 'artifacts/t3-pulls.dvc', 'artifacts/research-evidence-20261116.dvc']),
                      (ROOT.parent / 'mobihet', json.loads((ROOT.parent / 'mobihet/runs/maintenance-20261116/new-targets.json').read_text()))]:
    assert os.path.ismount(mount)
    with (OUT / (repo.name + '-push.log')).open('w') as log:
        subprocess.run(['dvc', 'push', '-j', '16', *targets], cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (OUT / (repo.name + '-status.log')).open('w') as log:
        subprocess.run(['dvc', 'status', '-c', *targets], cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert 'in sync' in (OUT / (repo.name + '-status.log')).read_text()
    print('REBOOT_DVC_VERIFIED', repo.name, flush=True)
print('ALL_REBOOT_DVC_PUSHES_VERIFIED', flush=True)
