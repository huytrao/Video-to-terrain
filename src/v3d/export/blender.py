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
    """Import by extension, into the *source* coordinate frame.

    glTF is defined Y-up, so Blender's importer rotates +90° about X on the way
    in — and bakes that rotation into the mesh data, leaving rotation_euler at
    zero so nothing looks amiss. The reconstruction's own frame is neither: it
    is whatever COLMAP produced, and the ground-truth boxes live in it
    unrotated. Left alone, the mesh and the boxes end up in different worlds.

    Measured on the pilot scene: the imported mesh had its smallest extent on Y
    (3.42 m, the room height) while ground truth has its smallest on Z. Undoing
    the importer's rotation maps (x, y, z) -> (x, z, -y) and puts the floor back
    at z ~ 0 with the ceiling at z ~ -3.3, agreeing with the annotations.

    PLY and OBJ importers apply no such conversion and are left alone.
    """
    suffix = path.suffix.lower()
    before = set(bpy.data.objects)

    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
        imported = [o for o in bpy.data.objects if o not in before]
        undo = Matrix.Rotation(math.radians(-90.0), 4, "X")
        for obj in imported:
            obj.matrix_world = undo @ obj.matrix_world
        return imported

    if suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"unsupported mesh format: {suffix}")
    return [o for o in bpy.data.objects if o not in before]


def frame_report(mesh_objects: list, boxes: list[dict],
                 correction: Matrix | None = None) -> dict:
    """Bounding boxes of mesh and annotations, so a mismatch is visible.

    Exists because the frames silently disagreed once already. The mesh should
    sit inside the annotation bounds, allowing for unobserved regions.
    """
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for obj in mesh_objects:
        for corner in obj.bound_box:
            w = obj.matrix_world @ Vector(corner)
            for i in range(3):
                lo[i] = min(lo[i], w[i])
                hi[i] = max(hi[i], w[i])

    blo = [1e9] * 3
    bhi = [-1e9] * 3
    for item in boxes:
        for c in item.get("bounding_box", []):
            p = Vector((c["x"], c["y"], c["z"]))
            if correction:
                p = correction @ p
            for i in range(3):
                blo[i] = min(blo[i], p[i])
                bhi[i] = max(bhi[i], p[i])

    return {
        "mesh_min": [round(v, 2) for v in lo],
        "mesh_max": [round(v, 2) for v in hi],
        "boxes_min": [round(v, 2) for v in blo],
        "boxes_max": [round(v, 2) for v in bhi],
        "mesh_inside_boxes": all(
            blo[i] - 1.0 <= lo[i] and hi[i] <= bhi[i] + 1.0 for i in range(3)
        ),
    }


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

    # The three shortest independent edges come out in arbitrary order, so the
    # triple is left-handed about half the time — det = -1 on many of these
    # annotations. That is a mirror, which turns real geometry inside out.
    # Flipping one axis restores a right-handed basis and describes the same box.
    rot = Matrix((axes[0], axes[1], axes[2])).transposed()
    if rot.determinant() < 0:
        axes[2] = -axes[2]
        rot = Matrix((axes[0], axes[1], axes[2])).transposed()

    return centre, size, rot


def add_box(name: str, centre, size, rot, collection, wire: bool = True,
            correction: Matrix | None = None) -> None:
    """One box per object, as real geometry rather than an Empty.

    Empties were the obvious choice and the wrong one: Blender never renders
    them, so a box that sits in the wrong place cannot be seen in any output
    image — only by opening the file. Real mesh cubes render, export, and can be
    measured, and displaying them as wire keeps the room readable.
    """
    mesh = bpy.data.meshes.new(name)
    verts = [
        (x, y, z)
        for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)
    ]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    world = Matrix.Translation(centre) @ rot.to_4x4() @ Matrix.Diagonal(size.to_4d())
    obj.matrix_world = (correction @ world) if correction else world
    if wire:
        obj.display_type = "WIRE"
        obj.show_wire = True
    collection.objects.link(obj)


def world_correction(instances: list[dict]) -> Matrix:
    """Rotation putting the reconstruction the right way up for Blender.

    Reconstruction frames do not agree on which way is up, and this scene's
    points **down**: its annotations place the floor at z = -0.04 and the
    ceiling at z = -3.33. Mesh and boxes agree with each other, so nothing looks
    wrong in any measurement — but Blender treats +Z as up, so the file opens
    with the room inverted, floor above ceiling.

    The direction is derived from the floor and ceiling annotations rather than
    assumed. Returns identity when the scene is already Z-up.
    """
    floors = [i for i in instances if i.get("label") == "floor"]
    ceilings = [i for i in instances if i.get("label") == "ceiling"]
    if not floors or not ceilings:
        return Matrix.Identity(4)

    def mean_z(items):
        zs = [c["z"] for it in items for c in it.get("bounding_box", [])]
        return sum(zs) / len(zs) if zs else 0.0

    # A ceiling below the floor means +z points down.
    if mean_z(ceilings) < mean_z(floors):
        return Matrix.Rotation(math.radians(180.0), 4, "X")
    return Matrix.Identity(4)


def build(
    mesh_path: Path | None,
    boxes_path: Path | None,
    out_path: Path,
    shell_labels=("wall", "floor", "ceiling"),
) -> Path:
    clear_scene()

    instances: list[dict] = []
    if boxes_path is not None:
        raw = json.loads(Path(boxes_path).read_text())
        instances = raw if isinstance(raw, list) else raw.get("objects", [])
    correction = world_correction(instances)
    if correction != Matrix.Identity(4):
        print("scene is Z-down; rotating 180 degrees about X so it opens the right way up")

    imported = []
    if mesh_path is not None:
        imported = import_mesh(mesh_path)
        for obj in imported:
            obj.name = "reconstruction"
            obj.matrix_world = correction @ obj.matrix_world

    if boxes_path is not None:
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
                    structure if label in shell_labels else objects,
                    correction=correction)

    if imported and boxes_path is not None:
        report = frame_report(imported, instances, correction)
        print("frame check:", report)
        if not report["mesh_inside_boxes"]:
            raise SystemExit(
                "mesh and annotations are in different coordinate frames: " + str(report)
            )

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
