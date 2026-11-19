"""Recover the authorized last five rounds after reboot, with a verified archive input; no scientific changes."""
import ast
import glob
import shutil
import tempfile
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ['PATH'] = str(Path(sys.executable).parent) + ':' + os.environ['PATH']
os.environ['FLB_SEED'] = '41'
import run_batch as B

OUT = ROOT / 'runs/maintenance-20261116/pfedes-reboot'
OUT.mkdir(exist_ok=True)
ACCOUNT, SLUG, RUN = 'outlookcheck', 'flb-pfedes-s41-p4', 'flb-pfedes-s41'
REF = ACCOUNT + '/flb-pfedes-s41-r45-final'
PARENT = ROOT / 'runs/pulls/flb-pfedes-s41-p3' / RUN


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ka(*args):
    try:
        return B.sh([sys.executable, str(B.SK / 'kaggle_as.py'), ACCOUNT, '--', *args], timeout=1800)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stdout or '') + '\n' + (exc.stderr or '')
        (OUT / 'last-cli-error.log').write_text(re.sub(r'(https?://[^\s?]+)\?[^\s]+', r'\1?<REDACTED>', detail))
        raise


def main():
    assert not (B.OUT / (SLUG + '.intent')).exists(), 'Inspect existing push intent; never duplicate'
    torch.set_num_threads(2)
    clients = []
    for rank, name in enumerate(['state.pt', 'state_rank1.pt']):
        s = torch.load(PARENT / name, weights_only=True, map_location='cpu', mmap=True)
        assert s['round'] == 45
        if rank == 0:
            assert s['method'] == 'pfedes' and s['hp']['seed'] == 41 and s['hp']['rounds'] == 50 and len(s['gen']) == 2
        m = s['method_state']; clients.append(len(m['bank_p']))
        for value in m.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all()
    assert sum(clients) == 5343
    upload = OUT / 'upload'; upload.mkdir(exist_ok=True)
    archive = upload / (RUN + '.zip')
    if not archive.exists():
        tmp = archive.with_suffix('.zip.part')
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            for p in sorted(PARENT.rglob('*')):
                if not p.is_file() or 'inductor' in p.relative_to(PARENT).parts or p.name == 'inductor-cache.tar':
                    continue
                z.write(p, str(Path(RUN) / p.relative_to(PARENT)))
        os.replace(tmp, archive)
    archive_sha = digest(archive)
    metadata = dict(id=REF, title='FeatureBERT pFedES seed41 checkpoint round45 final', licenses=[dict(name='CC0-1.0')])
    (upload / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2))
    intent = OUT / 'dataset-create.intent'
    if not intent.exists():
        intent.write_text(json.dumps(dict(ref=REF, parent='emtrangtri/flb-pfedes-s41-p3/1', round=45, archive_sha256=archive_sha)))
        (OUT / 'dataset-create.reply').write_text(ka('kaggle', 'datasets', 'create', '-p', str(upload), '--keep-tabular'))
    for _ in range(80):
        try:
            status = ka('kaggle', 'datasets', 'status', REF).strip()
            if status == 'ready':
                break
        except subprocess.SubprocessError:
            pass
        time.sleep(15)
    else:
        raise RuntimeError('Checkpoint dataset not ready; inspect receipt, do not relaunch blindly')
    verify = OUT / 'verify'; verify.mkdir(exist_ok=True)
    # Kaggle expands uploaded zip files. Verify the actual extracted files, not a nonexistent archive endpoint.
    (OUT / 'dataset-download.log').write_text(ka(sys.executable, 'scripts/verify_checkpoint_dataset.py', '--ref', REF,
        '--parent', str(PARENT), '--out', str(verify), '--report', str(OUT / 'dataset-file-hashes.json')))
    # Exercise the emitted import guard against the real downloaded parent, including nested path resolution.
    from gen_notebook import cells
    from types import SimpleNamespace
    cfg_args = SimpleNamespace(run=RUN, slug=SLUG, method='pfedes', rounds=50, args='', set=['seed=41'], overlay=[],
                               resume_kernel='', resume_dataset=REF, max_hours=3.8)
    source = next(code for kind, code in cells(cfg_args, 'unused-preflight') if kind=='code' and code.startswith('prev ='))
    with tempfile.TemporaryDirectory(prefix='pfedes-import-', dir=OUT) as tmp:
        source = source.replace('/kaggle/input', str(verify)).replace('/kaggle/working', tmp)
        exec(source, dict(glob=glob, Path=Path, torch=torch, os=os, shutil=shutil))
        assert (Path(tmp) / RUN / 'state_rank1.pt').stat().st_size == (PARENT / 'state_rank1.pt').stat().st_size
    (OUT / 'dataset-verified.json').write_text(json.dumps(dict(ref=REF, archive_sha256=archive_sha, round=45, clients=clients,
                                                              extracted_file_hashes=True, emitted_import_guard=True), indent=2))
    print('CHECKPOINT_DATASET_VERIFIED', REF, flush=True)
    live = B.inventory(ACCOUNT)
    assert len(live['busy']) < 2 and live['gpu_left'] >= 4.2 and SLUG not in live['versions'], live
    (OUT / 'inventory.json').write_text(json.dumps(live, indent=2))
    plan = json.loads((B.OUT / 'plan.json').read_text())
    B.sh([sys.executable, 'scripts/gen_notebook.py', '--owner', ACCOUNT, '--slug', SLUG, '--method', 'pfedes', '--run', RUN,
          '--rounds', '50', '--max-hours', '3.8', '--sha', plan['head'], '--set', 'seed=41', '--resume-dataset', REF])
    folder = ROOT / 'notebooks' / SLUG; path = folder / (SLUG + '.ipynb')
    nb = json.loads(path.read_text())
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    path.write_text(json.dumps(nb, indent=1) + '\n')
    B.sh([sys.executable, str(B.SK / 'embed_wandb_key.py'), str(path), '--metadata', str(folder / 'kernel-metadata.json')])
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    assert meta['is_private'] and meta['machine_shape'] == 'NvidiaTeslaT4' and meta['dataset_sources'].count(REF) == 1
    row = dict(account=ACCOUNT, slug=SLUG, run=RUN, method='pfedes', part=4, max_hours=3.8, set=[],
               resume_dataset=REF, transfer_from='emtrangtri/flb-pfedes-s41-p3/1')
    (OUT / 'notebook-verified.json').write_text(json.dumps(dict(sha256=digest(path), source=plan['head'], method='pfedes', seed=41,
                                                               resume_round=45, until=50, archive_sha256=archive_sha), indent=2))
    (B.OUT / (SLUG + '.intent')).write_text(json.dumps(row, indent=2))
    reply = ka('kaggle', 'kernels', 'push', '-p', str(folder))
    (B.OUT / (SLUG + '.push-reply')).write_text(reply)
    version = re.search(r'Kernel version (\d+) successfully', reply)
    assert version, 'Inspect accepted push with unknown version'
    row['version'] = int(version[1]); (B.OUT / (SLUG + '.intent')).write_text(json.dumps(row, indent=2))
    (B.OUT / (SLUG + '.pushed')).write_text(reply)
    (OUT / 'launched.json').write_text(json.dumps(row, indent=2))
    (OUT / 'batch.json').write_text(json.dumps([dict(arm='pfedes41', account=ACCOUNT, slug=SLUG, version=row['version'])], indent=2))
    print('PFEDES41_PUSH_VERIFIED', ACCOUNT, SLUG, row['version'], flush=True)


if __name__ == '__main__':
    main()
