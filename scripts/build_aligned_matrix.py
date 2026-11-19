"""Build separate v2 notebooks; never overwrite/push the existing matrix."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import nbformat
from build_sae_matrix import OWNERS

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for backbone in ('tabm', 'sae', 'mlp', 'cnn-bigru-attention'):
        for method, owners in {**OWNERS, 'canperfl': OWNERS['fedavg']}.items():
            for seed, owner in zip((41, 42, 43), owners):
                slug = f'{backbone}-fl-{method.replace("_", "-")}-s{seed}-a2'
                cmd = [sys.executable, 'scripts/build_tabm_fl.py', '--owner', owner,
                       '--backbone', backbone, '--method', method, '--seed', str(seed), '--slug', slug]
                if method == 'canperfl':
                    cmd += ['--parent', f'{owner}/{backbone}-fl-fedavg-s{seed}-a2']
                subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
                path = ROOT/'notebooks'/slug/f'{slug}.ipynb'
                nb = nbformat.read(path, as_version=4)
                nbformat.validate(nb)
                for cell in nb.cells:
                    if cell.cell_type == 'code':
                        ast.parse(cell.source)
                tree = ast.parse(nb.cells[2].source)
                source, hashes = [ast.literal_eval(n.value) for n in tree.body[:2]]
                for name, content in source.items():
                    assert hashlib.sha256(content.encode()).hexdigest() == hashes[name]
                    assert (ROOT/name).read_text(encoding='utf-8') == content
                rows.append(dict(backbone=backbone, method=method, seed=seed, owner=owner,
                                 slug=slug, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(backbone, '42/42 static PASS', flush=True)
    out = ROOT/'runs/aligned-v2-build.json'
    out.write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
