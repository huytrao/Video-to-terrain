"""Install the pipeline's missing Python deps on an offline kernel.

open3d, trimesh and plyfile are absent from the Kaggle image but present as wheels
in the v3d-build-cache-sm120 dataset, together with the two PGSR CUDA extensions
compiled for sm_120. --no-index is the point: this kernel has no internet, so a
pip that silently reaches for PyPI would hang rather than fail, and the failure
would look like a network problem instead of a missing wheel.
"""
import subprocess
import sys
from pathlib import Path

DEPS = Path("/kaggle/input/datasets/traoanhuy/v3d-build-cache-sm120/deps")
PGSR = Path("/kaggle/input/datasets/traoanhuy/v3d-build-cache-sm120/pgsr_sm120")

targets = ["open3d", "trimesh", "plyfile"]
pgsr_wheels = sorted(PGSR.glob("*.whl")) if PGSR.exists() else []


def pip(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--no-index",
           "--find-links", str(DEPS), "-q", *args]
    print("$", " ".join(cmd[-4:]))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode:
        # pip's useful line is near the end; the resolver preamble is noise.
        tail = (r.stdout + r.stderr).strip().splitlines()[-12:]
        print("  FAILED:", *("    " + t for t in tail), sep="\n")
    return r.returncode


rc = pip(targets)
for wheel in pgsr_wheels:
    rc |= pip([str(wheel)])

print("\n=== after install ===")
for mod in ("open3d", "trimesh", "plyfile", "diff_plane_rasterization", "simple_knn"):
    try:
        m = __import__(mod)
        print(f"  {mod:26} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        print(f"  {mod:26} MISSING ({type(exc).__name__}: {exc})")

# open3d's TSDF path is the one the mesh stage actually calls; importing the
# module is not proof the native extension loaded.
try:
    import numpy as np
    import open3d as o3d
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02, sdf_trunc=0.06,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    mesh = vol.extract_triangle_mesh()
    print(f"  open3d TSDF volume constructs, extract -> {len(mesh.vertices)} verts (0 expected)")
    print(f"  numpy {np.__version__}")
except Exception as exc:
    print(f"  open3d TSDF BROKEN: {type(exc).__name__}: {exc}")

sys.exit(rc)
