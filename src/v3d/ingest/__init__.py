"""Video in, a ScanNet++-shaped scene out.

`frames` decides which frames are worth using; `poses` recovers the geometry and
writes the one file every later stage reads. Split that way because the first is
testable without a GPU and the second is not.
"""

from __future__ import annotations

from .frames import FrameSpec, FrameStat, Selection, ingest, measure, select, extract
from .poses import (
    PoseSpec,
    build_transforms,
    consensus_intrinsic,
    rescale_intrinsic,
    to_4x4,
    trajectory_report,
    transform_matrix_to_w2c,
    verify_roundtrip,
    w2c_to_transform_matrix,
    write_scene,
)

__all__ = [
    "FrameSpec", "FrameStat", "Selection", "ingest", "measure", "select", "extract",
    "PoseSpec", "build_transforms", "consensus_intrinsic", "rescale_intrinsic",
    "to_4x4", "trajectory_report", "transform_matrix_to_w2c", "verify_roundtrip",
    "w2c_to_transform_matrix", "write_scene",
]
