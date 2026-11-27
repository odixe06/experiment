"""List only DVC v3 MD5 cache objects reachable from selected .dvc pointers."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import yaml


def object_path(cache, digest):
    if not re.fullmatch(r"[0-9a-f]{32}(\.dir)?", digest):
        raise ValueError(f"Unsupported DVC object ID: {digest!r}")
    return cache / digest[:2] / digest[2:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="DVC cache files/md5 directory")
    parser.add_argument("lists", type=Path, help="Existing directory for the three output lists")
    parser.add_argument("pointers", type=Path, nargs="+")
    args = parser.parse_args()
    blobs = set()
    dirs = set()
    for pointer in args.pointers:
        if pointer.suffix != ".dvc":
            parser.error(f"Expected a .dvc pointer: {pointer}")
        for output in yaml.safe_load(pointer.read_text())["outs"]:
            if output.get("hash", "md5") != "md5":
                parser.error(f"Unsupported hash in {pointer}")
            digest = output["md5"]
            if not digest.endswith(".dir"):
                blobs.add(digest)
                continue
            raw = object_path(args.cache, digest).read_bytes()
            if hashlib.md5(raw).hexdigest() + ".dir" != digest:
                parser.error(f"Directory manifest hash mismatch: {pointer}")
            entries = json.loads(raw)
            if len(entries) != output["nfiles"] or any(entry["md5"].endswith(".dir") for entry in entries):
                parser.error(f"Unsupported directory manifest: {pointer}")
            dirs.add(digest)
            blobs.update(entry["md5"] for entry in entries)

    def relative_paths(digests):
        for digest in sorted(digests):
            path = object_path(args.cache, digest)
            if not path.is_file():
                parser.error(f"Object missing from local cache: {digest}")
            yield path.relative_to(args.cache).as_posix()

    blob_paths, dir_paths = list(relative_paths(blobs)), list(relative_paths(dirs))
    for name, paths in (("blobs", blob_paths), ("dirs", dir_paths), ("all", blob_paths + dir_paths)):
        (args.lists / f"{name}.txt").write_text("".join(f"{path}\n" for path in paths))
    size = sum((args.cache / path).stat().st_size for path in blob_paths + dir_paths)
    print(f"{len(blob_paths)} blobs, {len(dir_paths)} directory manifests, {size} unique bytes")


if __name__ == "__main__":
    main()
