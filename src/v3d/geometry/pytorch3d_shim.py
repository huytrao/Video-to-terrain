"""The one pytorch3d function PGSR needs, without the 40-minute CUDA build.

`grep -r pytorch3d` over the paper's repository returns a single line:

    PGSR/scene/gaussian_model.py:23:from pytorch3d.transforms import quaternion_to_matrix

That function is pure tensor arithmetic — no CUDA kernels, no compiled
extension. Building all of pytorch3d against torch 2.10 / sm_120 to obtain it
would be a large risk for no benefit, so it is reproduced here.

**Convention matters more than usual here.** pytorch3d takes quaternions
*real part first* — (w, x, y, z) — while scipy, ROS and most of the graphics
world put the real part last. Getting this backwards produces rotations that
look plausible and are wrong, which is exactly the class of bug that survives
every "does it load?" check. `verify()` proves the convention numerically
against scipy rather than trusting the comment above.
"""

from __future__ import annotations

import torch


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from quaternions given real part first (w, x, y, z).

    Mirrors pytorch3d.transforms.quaternion_to_matrix, including its
    normalisation by the squared norm rather than requiring unit input.

    Args:
        quaternions: (..., 4) tensor, real part first.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def verify(n: int = 512, tol: float = 1e-6) -> dict[str, float]:
    """Check against scipy, which orders quaternions real part *last*.

    Returns the worst deviation for the correct ordering and, for contrast,
    for the swapped ordering — the second number should be large, which is
    what makes the first number meaningful.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    wxyz = rng.normal(size=(n, 4))
    wxyz /= np.linalg.norm(wxyz, axis=1, keepdims=True)

    ours = quaternion_to_matrix(torch.from_numpy(wxyz)).numpy()

    # scipy wants (x, y, z, w)
    xyzw = np.concatenate([wxyz[:, 1:], wxyz[:, :1]], axis=1)
    theirs = Rotation.from_quat(xyzw).as_matrix()

    # deliberately wrong ordering, as a control
    swapped = Rotation.from_quat(wxyz).as_matrix()

    return {
        "max_abs_diff_correct_order": float(np.abs(ours - theirs).max()),
        "max_abs_diff_swapped_order": float(np.abs(ours - swapped).max()),
        "orthonormal_error": float(
            np.abs(np.einsum("nij,nkj->nik", ours, ours) - np.eye(3)).max()
        ),
        "tol": tol,
    }


_TRANSFORMS_MODULE = '''"""Minimal stand-in for pytorch3d.transforms.

Provides only quaternion_to_matrix, which is the sole symbol the PGSR sources
import. Real part first (w, x, y, z), matching pytorch3d.
"""

import torch

from v3d.geometry.pytorch3d_shim import quaternion_to_matrix  # noqa: F401

__all__ = ["quaternion_to_matrix"]
'''


def install(target_dir: str = "/kaggle/working") -> str:
    """Expose a minimal `pytorch3d.transforms` package for PGSR to import.

    PGSR imports the real module path, so rather than editing the paper's
    source we satisfy that import with a package that re-exports the function
    verified above. Requires v3d itself to be importable.
    """
    from pathlib import Path

    pkg = Path(target_dir) / "pytorch3d"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        '"""Stand-in providing only what the PGSR sources import."""\n'
    )
    (pkg / "transforms.py").write_text(_TRANSFORMS_MODULE)
    return str(pkg)
