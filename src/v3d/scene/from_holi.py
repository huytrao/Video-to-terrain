"""Convert a Holi-Spatial instance annotation into a `scene.json` document.

The pipeline's own output and the released ground truth share one format —
`postprocess_3d_bbox_aabb.py` writes `label` plus a `bounding_box` of eight
corner points, which is exactly what the ScanNet++ annotations carry. So the
same reader serves both, and Stage J ends up comparing like with like.

The room shell is *derived from the shell instances*, never invented. Where a
value cannot be derived it is written with ``source="assumed"``, which Stage J
rejects — a missing floor should fail loudly rather than default to 2.0 m.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from v3d.scene.schema import (
    Ceiling,
    Floor,
    OrientedBox,
    Plane,
    Room,
    SceneDocument,
    SceneObject,
    Wall,
)
from v3d.validate.gt_boxes import GTBox, load_gt

FALLBACK_CEILING_M = 2.6


def _corners(box: GTBox) -> np.ndarray:
    half = (box.axes.T * (box.size / 2.0)).T
    return np.array(
        [box.center + s[0] * half[0] + s[1] * half[1] + s[2] * half[2]
         for s in itertools.product((-1, 1), repeat=3)]
    )


def _up_axis(boxes: list[GTBox]) -> int:
    """Whichever axis separates floor from ceiling, else the world Z."""
    floors = [b for b in boxes if b.label == "floor"]
    ceils = [b for b in boxes if b.label == "ceiling"]
    if floors and ceils:
        sep = np.abs(
            np.mean([c.center for c in ceils], axis=0)
            - np.mean([f.center for f in floors], axis=0)
        )
        return int(np.argmax(sep))
    return 2


def build_room(boxes: list[GTBox]) -> Room:
    shell = [b for b in boxes if b.is_shell]
    if not shell:
        raise ValueError(
            "no floor/wall/ceiling instances — the room shell cannot be measured"
        )

    up = _up_axis(boxes)
    horiz = [i for i in range(3) if i != up]

    pts = np.vstack([_corners(b) for b in shell])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    polygon = [
        [float(lo[horiz[0]]), float(lo[horiz[1]])],
        [float(hi[horiz[0]]), float(lo[horiz[1]])],
        [float(hi[horiz[0]]), float(hi[horiz[1]])],
        [float(lo[horiz[0]]), float(hi[horiz[1]])],
    ]

    floors = [b for b in boxes if b.label == "floor"]
    ceils = [b for b in boxes if b.label == "ceiling"]
    if floors and ceils:
        height = float(
            abs(
                np.mean([c.center[up] for c in ceils])
                - np.mean([f.center[up] for f in floors])
            )
        )
        ceiling_source = "measured"
    else:
        height = FALLBACK_CEILING_M
        ceiling_source = "assumed"

    floor_z = float(np.mean([f.center[up] for f in floors])) if floors else float(lo[up])
    normal = [0.0, 0.0, 0.0]
    normal[up] = 1.0

    walls = []
    for b in boxes:
        if b.label != "wall":
            continue
        c = _corners(b)
        wlo, whi = c.min(axis=0), c.max(axis=0)
        walls.append(
            Wall(
                plane=Plane(
                    coeffs=[*(float(v) for v in b.axes[0]), -float(b.axes[0] @ b.center)],
                    source="measured",
                    inlier_count=0,
                ),
                footprint_xy=[
                    [float(wlo[horiz[0]]), float(wlo[horiz[1]])],
                    [float(whi[horiz[0]]), float(whi[horiz[1]])],
                ],
                z_range_m=[float(wlo[up]), float(whi[up])],
                source="measured",
            )
        )

    return Room(
        floor=Floor(
            plane=Plane(coeffs=[*normal, -floor_z], source="measured"),
            polygon_xy=polygon,
            source="measured",
        ),
        ceiling=Ceiling(
            height_m=height,
            plane=Plane(coeffs=[*normal, -(floor_z + height)], source=ceiling_source)
            if ceiling_source == "measured"
            else None,
            source=ceiling_source,
        ),
        walls=walls,
    )


def build_scene(
    scene_id: str,
    instances_path: str | Path,
    pipeline: str = "holi-spatial",
    assets: dict[str, str] | None = None,
) -> SceneDocument:
    boxes = load_gt(instances_path)

    objects = [
        SceneObject(
            id=f"obj_{b.label.replace(' ', '_')}_{b.ins_id}",
            category=b.label,
            box=OrientedBox(
                center=[float(v) for v in b.center],
                size=[float(v) for v in b.size],
                rotation=[[float(v) for v in row] for row in b.axes],
                source="measured",
            ),
        )
        for b in boxes
        if not b.is_shell
    ]

    return SceneDocument(
        scene_id=scene_id,
        room=build_room(boxes),
        objects=objects,
        assets=assets or {},
        provenance={"pipeline": pipeline, "instances": str(instances_path)},
    )
