"""Rebuild the `extra/` wheel bundle that `kernel_stage.py` installs from.

Runs on the BUILD kernel, the one with internet. `kernel_stage.py` expects these
eight packages to be sitting on disk at /tmp/extra because the RUN kernel cannot
fetch them itself, and a fresh RUN session starts without them.

Downloaded with `--no-deps`, deliberately. A full resolve pulls pandas, which
requires numpy >= 2.1 and would break the numpy 2.0.2 pin that the PGSR CUDA
extensions were compiled against. `kernel_stage.py` installs with `--no-deps`
for the same reason; downloading the same way keeps the bundle honest about
what it actually contains.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile

DEST = "/kaggle/working/extra"
RELAY = "/kaggle/working/relay"
TARBALL = f"{RELAY}/extra.tar.gz"

# Names as pip knows them. kernel_stage.py matches on the wheel filename, where
# scikit-image becomes scikit_image; that difference is the packaging convention,
# not a typo in either place.
PACKAGES = [
    "evo",
    "pycocotools",
    "scikit-image",
    "tifffile",
    "lazy_loader",
    "networkx",
    "colorama",
    "natsort",
]


def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    os.makedirs(RELAY, exist_ok=True)
    print("python", sys.version.split()[0])

    failed = []
    for pkg in PACKAGES:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "--dest", DEST, pkg],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print(f"  ok    {pkg}")
        else:
            tail = (r.stderr or r.stdout).strip().splitlines()
            print(f"  FAIL  {pkg}: {tail[-1][:160] if tail else 'no output'}")
            failed.append(pkg)

    if failed:
        raise SystemExit(f"could not download: {', '.join(failed)}")

    files = sorted(os.listdir(DEST))
    print(f"{len(files)} files in {DEST}")
    for f in files:
        print("   ", f)

    with tarfile.open(TARBALL, "w:gz") as tf:
        tf.add(DEST, arcname="extra")
    print(f"tarball {TARBALL} {os.path.getsize(TARBALL)} bytes")
    print("BUILD_EXTRA_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
