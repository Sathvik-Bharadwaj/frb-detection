"""Zip the code, data and results for submission (output/FRB_Project_Code.zip)."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parents[1]
name = "FRB_Project_Code"
archive = root / "output" / f"{name}.zip"
include = ["README.md", "requirements.txt", "requirements-lock.txt", "src", "scripts", "tests",
           "demo", "data", "results/final", "results/real_preview"]

paths = []
for item in include:
    path = root / item
    if path.is_file():
        paths.append(path)
    else:
        paths += [p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and not p.name.startswith(".") and p.suffix != ".pyc"]

archive.parent.mkdir(exist_ok=True)
with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
    for path in sorted(paths):
        bundle.write(path, f"{name}/{path.relative_to(root).as_posix()}")
print(f"{archive} ({len(paths)} files, {archive.stat().st_size / 1e6:.1f} MB)")
