"""Download the CHIME waterfall for FRB20180725A and check its SHA-256."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = ("https://cadc-west-01.canfar.net/vault/files/AstroDataCitationDOI/CISTI.CANFAR/"
       "21.0007/data/waterfalls/data/FRB20180725A_waterfall.h5")
SIZE = 35527920
SHA256 = "783ddfbfbaf4ac1ef0b903adfb08564b015c1de9c3e93051bc2e2ca3e617c0b4"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("data/raw/FRB20180725A_waterfall.h5"))
args = parser.parse_args()
args.out.parent.mkdir(exist_ok=True, parents=True)

if args.out.exists():
    if hashlib.sha256(args.out.read_bytes()).hexdigest() == SHA256:
        print("Already downloaded:", args.out)
        raise SystemExit(0)
    raise SystemExit("Existing file has the wrong checksum, delete it and run again.")

# download to a .part file so an interrupted download can be resumed
part = args.out.with_suffix(args.out.suffix + ".part")
start = part.stat().st_size if part.exists() else 0
if start > SIZE:
    raise SystemExit("Partial file is larger than expected, delete it and run again.")
if start < SIZE:
    headers = {"Range": f"bytes={start}-"} if start else {}
    request = urllib.request.Request(URL, headers=headers)
    with urllib.request.urlopen(request, timeout=40) as response:
        if start and response.status != 206:
            raise SystemExit("Server did not accept the resume request")
        with part.open("ab" if start else "wb") as out:
            while chunk := response.read(256 * 1024):
                out.write(chunk)

if part.stat().st_size != SIZE:
    raise SystemExit("Download incomplete, run again to resume")
if hashlib.sha256(part.read_bytes()).hexdigest() != SHA256:
    raise SystemExit("Checksum mismatch")
part.replace(args.out)
print("Downloaded:", args.out)
