"""Download MIMIC-CXR (advanced option, credentialed access required).

MIMIC-CXR-JPG is hosted on PhysioNet behind a credentialed-access agreement ·
1. complete the CITI "Data or Specimens Only Research" course,
2. sign the PhysioNet Credentialed Health Data Use Agreement,
3. associate your PhysioNet account with the dataset listing.

Once credentialed, you can fetch the dataset with the PhysioNet wget recipe
(documented below) or with the ``physionet-cli``.

This script intentionally does NOT bypass authentication. It only checks for
credentials in ``~/.netrc`` and runs the recommended wget command.

Reference · https://physionet.org/content/mimic-cxr-jpg/2.0.0/
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import DATA_DIR

MIMIC_DIR = DATA_DIR / "mimic_cxr_jpg"
PHYSIONET_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/"


def _check_netrc() -> bool:
    netrc = Path.home() / (".netrc" if os.name != "nt" else "_netrc")
    if not netrc.exists():
        return False
    content = netrc.read_text(errors="ignore")
    return "physionet.org" in content


def _wget_available() -> bool:
    return shutil.which("wget") is not None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", default=False,
                   help="only verify credentials and tooling, do not download")
    p.add_argument("--user", default=os.environ.get("PHYSIONET_USER"),
                   help="PhysioNet username (or set PHYSIONET_USER env var)")
    p.add_argument("--targets", nargs="+", default=["files/"],
                   help="subpaths under the dataset to mirror")
    args = p.parse_args()

    print("MIMIC-CXR-JPG download · PhysioNet credentialed access")
    print(f"  dataset URL · {PHYSIONET_BASE}")
    print(f"  local dir   · {MIMIC_DIR}")

    has_netrc = _check_netrc()
    has_wget = _wget_available()
    print(f"  ~/.netrc with physionet.org entry · {'YES' if has_netrc else 'NO'}")
    print(f"  wget on PATH                       · {'YES' if has_wget else 'NO'}")

    if args.check:
        if not (has_netrc and has_wget):
            print("\nMissing prerequisites. See https://physionet.org/about/credentialing/")
            sys.exit(1)
        print("\nAll checks passed.")
        return

    if not (has_netrc and has_wget):
        print("\nMissing prerequisites. Provide credentials with ~/.netrc or _netrc · ")
        print("  machine physionet.org login YOUR_USER password YOUR_PASSWORD")
        print("Install wget (winget install GNU.Wget2 or via WSL) and re-run.")
        sys.exit(1)

    MIMIC_DIR.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        url = PHYSIONET_BASE + target
        cmd = ["wget", "-r", "-N", "-c", "-np",
               "--no-check-certificate", "--directory-prefix", str(MIMIC_DIR), url]
        print("\n>>", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("\nDone. MIMIC-CXR-JPG contents under", MIMIC_DIR)


if __name__ == "__main__":
    main()
