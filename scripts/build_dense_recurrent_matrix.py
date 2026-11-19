"""Build and statically audit 84 private FL notebooks; does not launch runs."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import nbformat
from build_sae_matrix import OWNERS

ROOT = Path(__file__).resolve().parents[1]


def main():
    count = 0
    for backbone in ("mlp", "cnn-bigru-attention"):
        for method, owners in {**OWNERS, "canperfl": OWNERS["fedavg"]}.items():
            for seed, owner in zip((41, 42, 43), owners):
                cmd = [sys.executable, str(ROOT/"scripts/build_tabm_fl.py"), "--backbone", backbone,
                       "--method", method, "--seed", str(seed), "--owner", owner]
                parent = f"{owner}/{backbone}-fl-fedavg-s{seed}"
                if method == "canperfl":
                    cmd += ["--parent", parent]
                subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
                slug = f"{backbone}-fl-{method.replace('_', '-')}-s{seed}"
                folder = ROOT/"notebooks"/slug
                meta = json.loads((folder/"kernel-metadata.json").read_text())
                assert meta["id"] == f"{owner}/{slug}"
                assert meta["is_private"] and meta["enable_gpu"]
                assert meta["kernel_sources"] == ([parent] if method == "canperfl" else [])
                nb = nbformat.read(folder/meta["code_file"], as_version=4)
                nbformat.validate(nb)
                for cell in nb.cells:
                    if cell.cell_type == "code":
                        ast.parse(cell.source)
                tree = ast.parse(nb.cells[2].source)
                source, hashes = [ast.literal_eval(n.value) for n in tree.body[:2]]
                for name, content in source.items():
                    assert hashlib.sha256(content.encode()).hexdigest() == hashes[name]
                    assert (ROOT/name).read_text(encoding="utf-8") == content
                    if name.endswith(".py"):
                        ast.parse(content)
                count += 1
        print(backbone, "42/42 built and static-audited", flush=True)
    assert count == 84


if __name__ == "__main__":
    main()
