"""Static audit for the locally built SAE baseline notebook matrix."""
import ast
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
METHODS = {"fedavg", "fedprox", "afpha", "fd-ids", "apfed", "lwfednids", "canperfl", "tinyproto",
           "pfedes", "nilm", "fedtgp", "perfedskd", "fedproto", "v2x-htfl"}


def main():
    dirs = sorted((ROOT / "notebooks").glob("sae-fl-*-s*"))
    assert len(dirs) == 42, len(dirs)
    direct = can = 0
    seen = set()
    for folder in dirs:
        nb_path = folder / f"{folder.name}.ipynb"
        meta_path = folder / "kernel-metadata.json"
        nb = nbformat.read(nb_path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        tree = ast.parse(nb.cells[2].source)
        source = ast.literal_eval(tree.body[0].value)
        hashes = ast.literal_eval(tree.body[1].value)
        assert set(source) == set(hashes)
        for name, content in source.items():
            assert (ROOT / name).read_text(encoding="utf-8") == content, name
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["id"].endswith("/" + folder.name)
        assert meta["code_file"] == nb_path.name
        assert meta["is_private"] and meta["enable_gpu"] and not meta["enable_internet"]
        assert meta["machine_shape"] == "NvidiaTeslaT4"
        assert meta["dataset_sources"] == [DATASET]
        seed = int(folder.name.rsplit("s", 1)[1])
        assert seed in (41, 42, 43)
        method = folder.name[len("sae-fl-"):].rsplit("-s", 1)[0]
        key = method, seed
        assert key not in seen
        seen.add(key)
        text = "\n".join(c.source for c in nb.cells)
        assert "tests/test_sae_fl.py" in text
        assert "'backbone': 'SAE'" in text and "'parameters': 7613" in text
        assert "--backbone', 'sae'" in text
        if method == "canperfl":
            can += 1
            assert len(meta["kernel_sources"]) == 1
            assert meta["kernel_sources"][0].endswith(f"/sae-fl-fedavg-s{seed}")
        else:
            direct += 1
            assert meta["kernel_sources"] == []
    assert (direct, can) == (39, 3)
    assert len(seen) == 42
    assert {method for method, _ in seen} == METHODS
    print("SAE NOTEBOOK AUDIT PASS: 42/42 (39 direct + 3 CANPerFL)")


if __name__ == "__main__":
    main()
