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

    floors = [b for b in boxes if b.label == "floor"]
    ceils = [b for b in boxes if b.label == "ceiling"]

    # The footprint is the floor slab's own rectangle, kept rotated. Wrapping
    # it to the world axes would inflate a rotated room — on the pilot scene a
    # 6.42 x 6.22 m floor wraps to 11.49 x 11.44 m.
    slab = max(floors, key=lambda b: b.size[0] * b.size[1]) if floors else None
    if slab is None:
        slab = max(ceils, key=lambda b: b.size[0] * b.size[1]) if ceils else None
    if slab is not None:
        half_a = slab.axes[0] * (slab.size[0] / 2.0)
        half_b = slab.axes[1] * (slab.size[1] / 2.0)
        corners = [
            slab.center - half_a - half_b,
            slab.center + half_a - half_b,
            slab.center + half_a + half_b,
            slab.center - half_a + half_b,
        ]
        polygon = [[float(c[horiz[0]]), float(c[horiz[1]])] for c in corners]
    else:
        pts = np.vstack([_corners(b) for b in shell])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        polygon = [
            [float(lo[horiz[0]]), float(lo[horiz[1]])],
            [float(hi[horiz[0]]), float(lo[horiz[1]])],
            [float(hi[horiz[0]]), float(hi[horiz[1]])],
            [float(lo[horiz[0]]), float(hi[horiz[1]])],
        ]
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


def room_from_mesh(objects_json: str | Path) -> Room:
    """Build the room shell from the mesh-derived shell planes, not from lifted boxes.

    Why this exists. `build_room` takes the floor polygon from whatever the
    semantic pipeline lifted as `floor` instances. On the factory scene that produced
    a polygon spanning 5.90 x 1.43 while the fused mesh's floor plane held 3.8 M
    points across **96.3%** of the footprint — the lifted floor captured a walkable
    strip, not the floor. Stage J then reported two thirds of the objects as "stray"
    and two machines as "larger than the room", both artefacts of an undersized
    footprint rather than errors in the boxes.

    `v3d.scene.decompose` finds the shell by RANSAC over the mesh itself, requires a
    shell plane to sit at the scene's boundary, and its audit passes. Its floor extent
    is therefore the better witness, and this reads it.

    The polygon is still a rectangle, which is a real approximation for an L-shaped
    room — recorded as `source="derived"` so a reader can tell it from a
    measured polygon.
    """
    import json as _json

    doc = _json.loads(Path(objects_json).read_text())
    up_axis = np.asarray(doc["up_axis"], dtype=float)
    up = int(np.argmax(np.abs(up_axis)))
    horiz = [i for i in range(3) if i != up]

    planes = doc.get("shell_planes", [])
    floors = [q for q in planes if q["role"] == "floor"]
    ceils = [q for q in planes if q["role"] == "ceiling"]
    if not floors:
        raise ValueError(
            f"{objects_json} has no floor among its shell planes "
            f"({[q['role'] for q in planes]}); the mesh shell cannot supply a room"
        )

    # Position, not just size. Building the polygon from `room_extent` alone forces
    # the assumption that the scene is centred on the origin; it is not, and that
    # error displaced the whole footprint and made eleven objects read as strays.
    if doc.get("room_aabb_min") and doc.get("room_aabb_max"):
        lo = np.asarray(doc["room_aabb_min"], dtype=float)
        hi = np.asarray(doc["room_aabb_max"], dtype=float)
    else:
        raise ValueError(
            f"{objects_json} predates room_aabb_min/max. Re-run the object cut: a "
            f"floor polygon cannot be positioned from an extent alone, and guessing "
            f"the origin is what this field exists to prevent."
        )
    a, b = horiz
    polygon = [[float(lo[a]), float(lo[b])], [float(hi[a]), float(lo[b])],
               [float(hi[a]), float(hi[b])], [float(lo[a]), float(hi[b])]]
    half = [float(hi[a] - lo[a]) / 2.0, float(hi[b] - lo[b]) / 2.0]

    floor_off = float(doc["floor_offset"])
    normal = [0.0, 0.0, 0.0]
    normal[up] = 1.0

    if ceils:
        height = float(doc["up_axis_evidence"]["room_height_estimate"])
        ceiling_source = "derived"
        ceiling_plane = Plane(coeffs=[*normal, -(floor_off + height)],
                              source="derived")
    else:
        height = FALLBACK_CEILING_M
        ceiling_source = "assumed"
        ceiling_plane = None

    walls = []
    for q in planes:
        if q["role"] != "wall":
            continue
        walls.append(Wall(
            plane=Plane(coeffs=[*(float(v) for v in q["normal"]), float(q["offset"])],
                        source="derived",
                        inlier_count=int(q.get("points", 0))),
            footprint_xy=[[float(lo[a]), float(lo[b])],
                          [float(hi[a]), float(hi[b])]],
            z_range_m=[floor_off, floor_off + height],
            source="derived",
        ))

    return Room(
        floor=Floor(
            plane=Plane(coeffs=[*normal, -floor_off], source="derived",
                        inlier_count=int(floors[0].get("points", 0))),
            polygon_xy=polygon,
            source="derived",
        ),
        ceiling=Ceiling(height_m=height, plane=ceiling_plane, source=ceiling_source),
        walls=walls,
    )


def build_scene(
    scene_id: str,
    instances_path: str | Path,
    pipeline: str = "holi-spatial",
    assets: dict[str, str] | None = None,
    objects_json: str | Path | None = None,
) -> SceneDocument:
    """Convert lifted instances into the scene document.

    `objects_json` is `decompose`'s output. When given, the room shell comes from the
    mesh rather than from lifted floor/wall instances — see `room_from_mesh` for the
    measurement that motivates it.
    """
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

    room = build_room(boxes)
    room_source = "lifted_instances"
    if objects_json is not None:
        room = room_from_mesh(objects_json)
        room_source = "mesh_shell"

    return SceneDocument(
        scene_id=scene_id,
        room=room,
        objects=objects,
        assets=assets or {},
        provenance={"pipeline": pipeline, "instances": str(instances_path),
                    "room_source": room_source,
                    "objects_json": str(objects_json) if objects_json else None},
    )
