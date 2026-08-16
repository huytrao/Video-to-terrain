"""Build PGSR's CUDA extensions for a GPU the offline kernel cannot reach PyPI for.

The offline Blackwell kernel has nvcc but no network; the internet-enabled
kernel has no usable GPU. That does not matter — nvcc cross-compiles for any
architecture, and the P100 kernel carries the same Python 3.12 and
torch 2.10+cu128 as the Blackwell one, so a wheel built there imports cleanly
over there.

One source fix is required. ``simple_knn.cu`` uses ``FLT_MAX`` without
including ``<float.h>``; CUDA 12.8 no longer pulls it in transitively, so the
2023 sources fail with "identifier FLT_MAX is undefined" until it is added.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Blackwell RTX PRO 6000 is compute capability 12.0.
DEFAULT_ARCH = "12.0"
PACKAGES = ("simple-knn", "diff-plane-rasterization")


def patch_float_header(submodules_dir: str | Path) -> list[str]:
    """Prepend ``#include <float.h>`` to any .cu that needs FLT_MAX and lacks it."""
    root = Path(submodules_dir)
    patched: list[str] = []
    for cu in root.rglob("*.cu"):
        text = cu.read_text()
        if "FLT_MAX" in text and "<float.h>" not in text and "<cfloat>" not in text:
            cu.write_text("#include <float.h>\n" + text)
            patched.append(str(cu.relative_to(root)))
    return patched


def build_wheels(
    submodules_dir: str | Path,
    wheel_dir: str | Path,
    arch: str = DEFAULT_ARCH,
    max_jobs: int = 4,
) -> list[Path]:
    root = Path(submodules_dir)
    out = Path(wheel_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, TORCH_CUDA_ARCH_LIST=arch, MAX_JOBS=str(max_jobs))
    for pkg in PACKAGES:
        src = root / pkg
        if not src.exists():
            raise FileNotFoundError(f"missing PGSR submodule: {src}")
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "wheel", str(src),
                "-w", str(out), "--no-build-isolation", "--no-deps",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"building {pkg} for sm_{arch.replace('.', '')} failed:\n"
                + (result.stdout + result.stderr)[-4000:]
            )
    return sorted(out.glob("*.whl"))


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submodules", help="PGSR/submodules directory")
    ap.add_argument("--wheels", default="wheels", help="output directory for wheels")
    ap.add_argument("--arch", default=DEFAULT_ARCH, help="TORCH_CUDA_ARCH_LIST value")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)

    patched = patch_float_header(args.submodules)
    print(f"patched float.h into: {patched or 'nothing (already clean)'}")
    for w in build_wheels(args.submodules, args.wheels, args.arch, args.jobs):
        print(f"  {w.name}  {w.stat().st_size/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
