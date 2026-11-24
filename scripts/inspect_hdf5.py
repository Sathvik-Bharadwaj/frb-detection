"""List the groups, datasets and attributes in an HDF5 file (writes JSON)."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", type=Path)
parser.add_argument("--out", type=Path)
args = parser.parse_args()

digest = hashlib.sha256()
with args.path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)

objects = []
with h5py.File(args.path, "r") as h:
    def visit(name, obj):
        row = {"path": name, "type": type(obj).__name__,
               "attrs": {k: str(v)[:1000] for k, v in obj.attrs.items()}}
        if isinstance(obj, h5py.Dataset):
            row.update(shape=list(obj.shape), dtype=str(obj.dtype))
        objects.append(row)
    h.visititems(visit)
    attrs = {k: str(v)[:1000] for k, v in h.attrs.items()}

info = {"file": str(args.path), "bytes": args.path.stat().st_size,
        "sha256": digest.hexdigest(), "root_attributes": attrs, "objects": objects}
text = json.dumps(info, indent=2)
if args.out:
    args.out.write_text(text + "\n")
print(text)
