"""Generate a notebook gate for the verified VeReMi v4 NPY kernel output."""

import json


ARRAYS = {f"{split}_{field}.npy" for split in ("train", "test")
          for field in ("X", "Y", "off", "ids", "F")}


def cache_input(receipt_path, identity=None):
    receipt = json.loads(receipt_path.read_text())["cache"]
    if set(receipt["files"]) != ARRAYS | {"done"}:
        raise ValueError("prebuilt cache must cover both full v4 splits")
    if identity is not None:
        expected = {name: receipt["files"][name]["sha256"] for name in ARRAYS}
        if identity.keys() != ARRAYS | {"label_mapping.json", "scaler.json"} or any(
                identity[name] != digest for name, digest in expected.items()):
            raise ValueError("prebuilt cache differs from the parent data identity")
    exact_ref = f'{receipt["ref"]}/{receipt["version"]}'
    code = f'''cache_started = time.monotonic()
cache_manifest, = Path("/kaggle/input").rglob("v4npy/manifest.json")
manifest_bytes = cache_manifest.read_bytes()
manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
assert manifest_sha256 == {receipt["manifest_sha256"]!r}, "prebuilt NPY manifest"
cache_files = json.loads(manifest_bytes)["files"]
assert cache_files == {receipt["files"]!r}, "prebuilt NPY file list"
mounted_npy = cache_manifest.parent
for name, expected_file in cache_files.items():
    path = mounted_npy / name
    assert path.stat().st_size == expected_file["bytes"], name
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == expected_file["sha256"], name
'''
    if identity is not None:
        code += f'''sys.path.insert(0, str(src))
from core import data as D
raw_v4 = Path(D.find_root())
identity = {{name: entry["sha256"] for name, entry in cache_files.items() if name.endswith(".npy")}}
for name in ("label_mapping.json", "scaler.json"):
    with (raw_v4 / name).open("rb") as handle:
        identity[name] = hashlib.file_digest(handle, "sha256").hexdigest()
assert identity == {identity!r}, "parent v4 data identity"
cache.mkdir()
for name in cache_files:
    (cache / name).symlink_to(mounted_npy / name)
(cache / "identity.json").write_text(json.dumps(identity, sort_keys=True))
'''
    code += f'''elapsed = time.monotonic() - cache_started
cache_receipt = dict(ref={exact_ref!r}, manifest_sha256=manifest_sha256, seconds=elapsed)
Path("/kaggle/working/npy_cache_receipt.json").write_text(json.dumps(cache_receipt, indent=2))
print(f"Verified prebuilt v4 NPY in {{elapsed:.1f}}s", flush=True)
'''
    return receipt["ref"], code
