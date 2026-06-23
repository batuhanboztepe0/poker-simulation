"""
fetch_phh.py
------------
Download a bounded, reproducible subset of the PHH poker-hand-history dataset
straight out of the 1.88 GB Zenodo zip — WITHOUT downloading the whole archive.

Zenodo serves HTTP range requests, so we read only the zip's central directory
(~4.6 MB at the tail), pick the entries we want, and range-fetch just those
compressed members. The raw hands land in `data/phh/` which is gitignored — the
repo commits only the aggregated `results/tilt_realdata.json`, never the hands.

Dataset: A Dataset of Poker Hand Histories (Kim, J., 2024).
  DOI: 10.5281/zenodo.13997158   License: CC-BY-4.0
  The ~21.6 M NLHE hands originate from a July 2009 HandHQ scrape, redistributed
  under CC-BY-4.0; see REFERENCES.md. Used here for the OPPONENT MODEL ONLY.

    # default: 200 PokerStars 25NL files (~200k human hands), chronological
    python -m scripts.fetch_phh
    python -m scripts.fetch_phh --prefix data/handhq/PS-2009-07-01_2009-07-23_25NLH_OBFU --max-files 200
    python -m scripts.fetch_phh --prefix data/pluribus --max-files 30   # bot control
"""

import argparse
import os
import re
import struct
import zlib

import requests

ZENODO_FILE = ("https://zenodo.org/records/13997158/files/"
               "poker-hand-histories.zip")
DOI = "10.5281/zenodo.13997158"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "phh")

DEFAULT_PREFIX = "data/handhq/PS-2009-07-01_2009-07-23_25NLH_OBFU"


def _range(a, b):
    r = requests.get(ZENODO_FILE, headers={"Range": f"bytes={a}-{b}"},
                     timeout=180)
    r.raise_for_status()
    return r.content


def _total_size():
    r = requests.get(ZENODO_FILE, headers={"Range": "bytes=0-0"}, timeout=60)
    r.raise_for_status()
    return int(r.headers["content-range"].split("/")[1])


def _central_directory(size):
    """Return [(name, method, csize, local_header_offset), ...] from the zip CD."""
    tail = _range(size - 131072, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("EOCD not found (zip64 not handled)")
    _, _, _, _, _, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH",
                                                      tail[i:i + 22])
    cd = _range(cd_off, cd_off + cd_size - 1)
    entries, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (_, _, _, _, method, _, _, _, csize, _usize, nlen, elen, clen,
         _ds, _ia, _ea, lho) = struct.unpack("<IHHHHHHIIIHHHHHII",
                                             cd[p:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        entries.append((name, method, csize, lho))
        p += 46 + nlen + elen + clen
    return entries


def _extract(name, method, csize, lho):
    lh = _range(lho, lho + 30 - 1)
    nlen, elen = struct.unpack("<HH", lh[26:30])
    data = _range(lho + 30 + nlen + elen, lho + 30 + nlen + elen + csize - 1)
    return zlib.decompress(data, -15) if method == 8 else data


def _file_index(name):
    """Numeric handhq_<n> index, so files come out in chronological order."""
    m = re.search(r"handhq_(\d+)", name)
    return int(m.group(1)) if m else 1 << 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=DEFAULT_PREFIX,
                    help="zip path prefix to select (network/stake or 'data/pluribus')")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    print(f"PHH dataset (Kim 2024, DOI {DOI}, CC-BY-4.0) — subset fetch")
    size = _total_size()
    print(f"  archive {size / 1e9:.2f} GB; reading central directory...")
    entries = _central_directory(size)

    chosen = [e for e in entries
              if e[0].startswith(args.prefix) and e[0].endswith(".phhs")]
    chosen.sort(key=lambda e: _file_index(e[0]))
    chosen = chosen[:args.max_files]
    if not chosen:
        print(f"  no .phhs entries under prefix {args.prefix!r}")
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"  fetching {len(chosen)} files -> {args.out}")
    total = 0
    for k, (name, method, csize, lho) in enumerate(chosen):
        raw = _extract(name, method, csize, lho)
        out = os.path.join(args.out, os.path.basename(name))
        with open(out, "wb") as f:
            f.write(raw)
        total += len(raw)
        if (k + 1) % 25 == 0 or k + 1 == len(chosen):
            print(f"    {k + 1}/{len(chosen)}  ({total / 1e6:.1f} MB)")
    print(f"  done: {total / 1e6:.1f} MB uncompressed in {args.out}")


if __name__ == "__main__":
    main()
