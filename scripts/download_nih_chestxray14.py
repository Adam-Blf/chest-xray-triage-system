"""Download NIH ChestX-ray14 (~42 GB) from the official NIH mirror.

The dataset is split across 12 tar.gz archives hosted at
https://nihcc.app.box.com/v/ChestXray-NIHCC. Each archive is ~3-5 GB.
Metadata CSV (BBox_List_2017.csv + Data_Entry_2017.csv) is small and always
fetched first so the project can run on metadata before image bulk arrives.

Usage ·
    python -m scripts.download_nih_chestxray14 --metadata-only        # fast (<5 MB)
    python -m scripts.download_nih_chestxray14 --images              # full pull (~42 GB)
    python -m scripts.download_nih_chestxray14 --images --start 1 --end 3
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from src.config import DATA_DIR

NIH_DIR = DATA_DIR / "nih_chestxray14"

METADATA_URLS = [
    ("https://nihcc.box.com/shared/static/7jiyc1g0e2y6gtmddhdmgm9hbpcjqndy.csv",
     "Data_Entry_2017.csv"),
    ("https://nihcc.box.com/shared/static/6m1xfzvxocsdb1szymq8j9kx7ihm5lec.csv",
     "BBox_List_2017.csv"),
    ("https://nihcc.box.com/shared/static/d65qmrlfsf9ihm59ig5rh1m4nbnq0qx0.csv",
     "train_val_list.txt"),
    ("https://nihcc.box.com/shared/static/9xqxjlx32evfrpsblay0p5dttiqkr3lt.csv",
     "test_list.txt"),
]

IMAGE_TARBALLS = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
]


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"   exists · {dest.name}")
        return
    print(f"   GET {url}\n        -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.rename(dest)


def fetch_metadata() -> None:
    print(f">> fetching NIH ChestX-ray14 metadata into {NIH_DIR}")
    for url, name in METADATA_URLS:
        _download(url, NIH_DIR / name)


def fetch_images(start: int, end: int, extract: bool) -> None:
    images_dir = NIH_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    if end > len(IMAGE_TARBALLS):
        end = len(IMAGE_TARBALLS)
    print(f">> fetching NIH images tarballs {start}..{end} (1-indexed) into {NIH_DIR}")
    for i in range(start - 1, end):
        url = IMAGE_TARBALLS[i]
        dest = NIH_DIR / f"images_{i+1:02d}.tar.gz"
        _download(url, dest)
        if extract:
            print(f"   extracting {dest.name} ...")
            with tarfile.open(dest, "r:gz") as tf:
                tf.extractall(images_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-only", action="store_true", default=False)
    p.add_argument("--images", action="store_true", default=False)
    p.add_argument("--extract", action="store_true", default=False,
                   help="extract tarballs after download")
    p.add_argument("--start", type=int, default=1, help="first tarball index (1-12)")
    p.add_argument("--end", type=int, default=12, help="last tarball index (1-12)")
    args = p.parse_args()

    NIH_DIR.mkdir(parents=True, exist_ok=True)
    fetch_metadata()

    if args.images or not args.metadata_only:
        if not args.images:
            print("\nMetadata fetched. To pull the ~42 GB image tarballs run ·")
            print("  python -m scripts.download_nih_chestxray14 --images --extract")
            return
        fetch_images(args.start, args.end, args.extract)

    print("\nDone. NIH ChestX-ray14 contents under", NIH_DIR)


if __name__ == "__main__":
    sys.exit(main())
