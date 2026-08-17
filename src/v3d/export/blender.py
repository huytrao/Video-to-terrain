"""Build a .blend from the reconstruction.

Run under Blender's own Python, which is where ``bpy`` lives:

    blender --background --python src/v3d/export/blender.py -- \\
        --mesh room.glb --boxes ground_truth_boxes.json --out room.blend

`scene.json` is the intended input once the lifting stage produces one; until
then the same file is built from the mesh plus a Holi-Spatial box annotation,
which is the identical eight-corner format.

Two things this gets right that an exported GLB does not: the file opens with
**metric units and real dimensions**, so a wall measures its actual metres in
the N-panel, and every object is a **named** object in a collection rather than
one anonymous soup of triangles.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy  # noqa: F401  (only available inside Blender)
from mathutils import Matrix, Vector


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0


def import_mesh(path: Path) -> list:
    """Import by extension. GLB keeps vertex colours; PLY is the raw surface."""
    suffix = path.suffix.lower()
    before = set(bpy.data.objects)
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"unsupported mesh format: {suffix}")
    return [o for o in bpy.data.objects if o not in before]


def box_corners_to_transform(corners: list[list[float]]):
    """Centre, per-axis extent and rotation from eight oriented-box corners.

    Same derivation as v3d.validate.gt_boxes.box_from_corners, repeated here
    because this module runs inside Blender's interpreter and cannot import the
    project's package.
    """
    pts = [Vector(c) for c in corners]
    centre = sum(pts, Vector((0, 0, 0))) / len(pts)

    origin = pts[0]
    deltas = [p - origin for p in pts[1:]]
    order = sorted(range(len(deltas)), key=lambda i: deltas[i].length)

    picked: list[int] = []
    for i in order:
        if deltas[i].length < 1e-9:
            continue
        d = deltas[i].normalized()
        if all(abs(d.dot(deltas[j].normalized())) < 0.5 for j in picked):
            picked.append(i)
        if len(picked) == 3:
            break
    if len(picked) < 3:
        return centre, Vector((0.1, 0.1, 0.1)), Matrix.Identity(3)

    axes = [deltas[i].normalized() for i in picked]
    size = Vector([deltas[i].length for i in picked])
    rot = Matrix((axes[0], axes[1], axes[2])).transposed()
    return centre, size, rot


def add_box(name: str, centre, size, rot, collection) -> None:
    """One wireframe cube per object, carrying its true dimensions."""
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "CUBE"
    empty.empty_display_size = 0.5
    empty.matrix_world = (
        Matrix.Translation(centre) @ rot.to_4x4() @ Matrix.Diagonal(size.to_4d())
    )
    collection.objects.link(empty)


def build(
    mesh_path: Path | None,
    boxes_path: Path | None,
    out_path: Path,
    shell_labels=("wall", "floor", "ceiling"),
) -> Path:
    clear_scene()

    if mesh_path is not None:
        imported = import_mesh(mesh_path)
        for obj in imported:
            obj.name = "reconstruction"

    if boxes_path is not None:
        raw = json.loads(Path(boxes_path).read_text())
        instances = raw if isinstance(raw, list) else raw.get("objects", [])

        structure = bpy.data.collections.new("structure")
        objects = bpy.data.collections.new("objects")
        bpy.context.scene.collection.children.link(structure)
        bpy.context.scene.collection.children.link(objects)

        for item in instances:
            corners = item.get("bounding_box")
            if not corners or len(corners) != 8:
                continue
            pts = [[c["x"], c["y"], c["z"]] for c in corners]
            centre, size, rot = box_corners_to_transform(pts)
            label = str(item.get("label", "object"))
            name = f"{label}_{item.get('ins_id', '?')}"
            add_box(name, centre, size, rot,
                    structure if label in shell_labels else objects)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
    return out_path


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path)
    ap.add_argument("--boxes", type=Path, help="Holi-Spatial instance json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    path = build(args.mesh, args.boxes, args.out)
    counts = {
        c.name: len(c.objects) for c in bpy.data.collections
    }
    print(f"wrote {path} ({path.stat().st_size/1e6:.1f} MB), collections: {counts}")


if __name__ == "__main__":
    main()
