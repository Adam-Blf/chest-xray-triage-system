"""One-shot · trigger every dataset download covered by the project brief.

ChestMNIST is fetched in full (only ~100 MB for 64px).
NIH ChestX-ray14 metadata is fetched by default; pass --nih-images for the
full ~42 GB pull. OpenI is fetched fully (~1.6 GB).
MIMIC-CXR is checked but never auto-downloaded (credentialed access).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> int:
    print(f"\n=== running · {' '.join(args)} ===")
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chestmnist-sizes", nargs="+", type=int,
                   default=[64, 128], help="ChestMNIST resolutions")
    p.add_argument("--nih-images", action="store_true",
                   help="also download the ~42 GB NIH image tarballs")
    p.add_argument("--nih-extract", action="store_true",
                   help="extract NIH tarballs after download")
    p.add_argument("--openi-extract", action="store_true",
                   help="extract OpenI archives after download")
    p.add_argument("--skip", nargs="+", default=[],
                   help="dataset names to skip (chestmnist nih openi mimic)")
    args = p.parse_args()

    skip = set(args.skip)

    if "chestmnist" not in skip:
        _run(["-m", "scripts.download_chestmnist",
              "--sizes", *map(str, args.chestmnist_sizes)])

    if "nih" not in skip:
        nih_args = ["-m", "scripts.download_nih_chestxray14"]
        if args.nih_images:
            nih_args += ["--images"]
            if args.nih_extract:
                nih_args += ["--extract"]
        else:
            nih_args += ["--metadata-only"]
        _run(nih_args)

    if "openi" not in skip:
        openi_args = ["-m", "scripts.download_openi"]
        if args.openi_extract:
            openi_args += ["--extract"]
        _run(openi_args)

    if "mimic" not in skip:
        _run(["-m", "scripts.download_mimic_cxr", "--check"])

    print("\nAll requested downloads finished.")


if __name__ == "__main__":
    main()
