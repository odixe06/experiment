"""Build a private, seconds-long Kaggle 2xT4 entitlement probe."""
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OWNER = "iamabighotdog"
SLUG = "gpu-2xt4-probe-iamabighotdog"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

code = '''import json
from pathlib import Path
import torch

count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(count)]
capabilities = [list(torch.cuda.get_device_capability(i)) for i in range(count)]
assert count == 2, count
assert names == ["Tesla T4", "Tesla T4"], names
assert capabilities == [[7, 5], [7, 5]], capabilities
Path("/kaggle/working/GPU_PROBE_PASS.json").write_text(json.dumps({
    "count": count, "names": names, "capabilities": capabilities
}, indent=2))
print("GPU PROBE PASS", count, names, capabilities, flush=True)
'''
nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)])
nb.metadata.update({
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python", "version": "3.12"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
               "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
})
folder = ROOT / "notebooks" / SLUG
folder.mkdir(parents=True, exist_ok=True)
notebook = folder / f"{SLUG}.ipynb"
nbformat.validate(nb)
nbformat.write(nb, notebook)
metadata = {
    "id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": notebook.name,
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "docker_image": IMAGE, "dataset_sources": [], "kernel_sources": [],
    "competition_sources": [],
}
(folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(folder)
