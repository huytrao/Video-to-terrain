"""Holi-Spatial ground-truth boxes.

The released annotation is a list of instances, each carrying the eight corner
points of an oriented box. Everything downstream wants centre + extent + axes,
so that conversion lives here and is done once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Labels that describe the room shell rather than a movable object.
SHELL_LABELS = frozenset({"wall", "floor", "ceiling"})


@dataclass(frozen=True)
class GTBox:
    ins_id: str
    label: str
    center: np.ndarray  # (3,)
    size: np.ndarray  # (3,) full extent along the box axes, descending
    axes: np.ndarray  # (3, 3) unit row vectors matching ``size``

    @property
    def is_shell(self) -> bool:
        return self.label in SHELL_LABELS

    def volume_m3(self) -> float:
        return float(np.prod(self.size))


def box_from_corners(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, extents and axes of an oriented box given its 8 corners.

    Uses the three shortest edges leaving one corner. They are mutually
    orthogonal for a well-formed box, so their lengths are the extents and
    their directions the axes.
    """
    if corners.shape != (8, 3):
        raise ValueError(f"expected 8 corners, got {corners.shape}")

    center = corners.mean(axis=0)
    origin = corners[0]
    deltas = corners[1:] - origin
    lengths = np.linalg.norm(deltas, axis=1)

    picked: list[int] = []
    for idx in np.argsort(lengths):
        if lengths[idx] < 1e-9:
            continue
        d = deltas[idx] / lengths[idx]
        # keep only directions independent of the ones already taken
        if all(abs(float(d @ (deltas[j] / lengths[j]))) < 0.5 for j in picked):
            picked.append(int(idx))
        if len(picked) == 3:
            break

    if len(picked) < 3:  # degenerate annotation; fall back to the AABB
        size = corners.max(axis=0) - corners.min(axis=0)
        return center, size, np.eye(3)

    size = lengths[picked]
    axes = np.stack([deltas[i] / lengths[i] for i in picked])

    order = np.argsort(-size)
    return center, size[order], axes[order]


def load_gt(path_or_bytes: str | Path | bytes) -> list[GTBox]:
    if isinstance(path_or_bytes, (str, Path)):
        raw = json.loads(Path(path_or_bytes).read_text())
    else:
        raw = json.loads(path_or_bytes)

    out: list[GTBox] = []
    for item in raw:
        pts = np.array([[c["x"], c["y"], c["z"]] for c in item["bounding_box"]], dtype=float)
        center, size, axes = box_from_corners(pts)
        out.append(
            GTBox(
                ins_id=str(item["ins_id"]),
                label=str(item["label"]),
                center=center,
                size=size,
                axes=axes,
            )
        )
    return out


def room_dims_from_gt(boxes: list[GTBox]) -> dict[str, float | str | int]:
    """Room footprint and height implied by the shell annotations.

    The footprint comes from the floor box's *own* extents, not from an
    axis-aligned wrap of its corners. None of the ScanNet++ shell boxes are
    axis aligned, and wrapping a rotated box inflates it badly — on the pilot
    scene a 6.42 x 6.22 m floor wraps to 11.49 x 11.44 m, which would fail a
    correct reconstruction by three and a half metres.

    Because ``GTBox.size`` is sorted descending along the box's own axes, the
    two largest extents of a slab-shaped floor are its footprint regardless of
    how the room is rotated in the world frame.

    Height is the floor-to-ceiling centre separation along the up axis, which
    is frame-independent and therefore directly comparable to a mesh.
    """
    floors = [b for b in boxes if b.label == "floor"]
    ceils = [b for b in boxes if b.label == "ceiling"]
    dims: dict[str, float | str | int] = {}

    if floors:
        # widest floor annotation; fragments are common in these scenes
        floor = max(floors, key=lambda b: b.size[0] * b.size[1])
        dims["footprint_a_m"] = float(floor.size[0])
        dims["footprint_b_m"] = float(floor.size[1])
        dims["footprint_source"] = "floor_obb"
    elif ceils:
        ceiling = max(ceils, key=lambda b: b.size[0] * b.size[1])
        dims["footprint_a_m"] = float(ceiling.size[0])
        dims["footprint_b_m"] = float(ceiling.size[1])
        dims["footprint_source"] = "ceiling_obb"

    if floors and ceils:
        sep = np.abs(
            np.mean([c.center for c in ceils], axis=0)
            - np.mean([f.center for f in floors], axis=0)
        )
        up = int(np.argmax(sep))
        dims["height_m"] = float(sep[up])
        dims["up_axis_index"] = up
    return dims
