"""Download Open-i Indiana University chest X-ray collection.

Open-i provides ~7,470 frontal/lateral chest radiographs paired with
de-identified radiology reports (XML). It is the lightweight option for
the multimodal image + text component required by the project.

Source · https://openi.nlm.nih.gov/faq · the canonical download URLs are ·
- images.tgz   (~1.6 GB · DICOM-derived PNGs)
- ecgen-radiology.tar.gz (~16 MB · per-study XML reports)
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from src.config import DATA_DIR

OPENI_DIR = DATA_DIR / "openi"

URLS = {
    "images.tgz": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
    "reports.tgz": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz",
}


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extract", action="store_true", default=False)
    p.add_argument("--reports-only", action="store_true", default=False,
                   help="only fetch the XML reports (~16 MB)")
    args = p.parse_args()

    OPENI_DIR.mkdir(parents=True, exist_ok=True)
    print(f">> Open-i downloads into {OPENI_DIR}")

    targets = ["reports.tgz"] if args.reports_only else list(URLS.keys())
    for name in targets:
        url = URLS[name]
        dest = OPENI_DIR / name
        _download(url, dest)
        if args.extract:
            print(f"   extracting {dest.name} ...")
            with tarfile.open(dest, "r:gz") as tf:
                tf.extractall(OPENI_DIR)

    print("\nDone. Open-i contents under", OPENI_DIR)
    print("Reports XML files end in *.xml under ecgen-radiology/, images in NLMCXR_png/.")


if __name__ == "__main__":
    main()
