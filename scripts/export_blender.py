"""Export pipeline outputs to Blender-friendly formats (glTF/GLB, PLY).

Run bake_texture.py first - this script uses its textured GLB output
(terrain_mesh_textured.glb) when present so the environment scene carries a
real UV image texture instead of flat vertex colors; falls back to
vertex-colored mesh otherwise.

Produces, under outputs/run_mvp/blender/:
  - terrain_mesh.glb      reconstructed room/staircase mesh, vertex colors (fallback/quick preview)
  - point_cloud.ply        raw DA3 point cloud (Blender: File > Import > Stanford PLY)
  - environment_scene.glb  terrain mesh (textured if available) + wall/object collision
                            boxes, one file, each box as a named node so it stays editable
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

RUN = Path("/Users/traoanhuy/source_code/Full-pipeline-scan-terrain/Videos_to_terrain/outputs/run_mvp")
OUT = RUN / "blender"


def export_terrain_mesh() -> None:
    src = RUN / "geometry" / "scene_mesh.ply"
    mesh = trimesh.load(src)
    mesh.export(OUT / "terrain_mesh.glb")
    print(f"terrain_mesh.glb  <- {src}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")


def export_point_cloud() -> None:
    src = RUN / "geometry" / "point_cloud.ply"
    shutil.copy(src, OUT / "point_cloud.ply")
    print(f"point_cloud.ply   <- {src}")


def _box_geom(size, rgba):
    box = trimesh.creation.box(extents=[s * 2 for s in size])
    box.visual.vertex_colors = np.tile(
        (np.array(rgba) * 255).astype(np.uint8), (len(box.vertices), 1)
    )
    return box


def export_environment_scene() -> None:
    """Terrain mesh + collision boxes (walls/objects) as one editable GLB scene."""
    scene = trimesh.Scene()

    textured_path = OUT / "terrain_mesh_textured.glb"
    if textured_path.exists():
        textured_scene = trimesh.load(textured_path)
        terrain = list(textured_scene.geometry.values())[0]
        print(f"environment_scene: using textured terrain from {textured_path.name}")
    else:
        terrain = trimesh.load(RUN / "geometry" / "scene_mesh.ply")
        print("environment_scene: textured terrain not found, using vertex-colored mesh "
              "(run bake_texture.py first for a real UV texture)")
    scene.add_geometry(terrain, node_name="terrain_mesh")

    coll_dir = RUN / "simulation" / "collision"
    wall_rgba = [0.7, 0.7, 0.7, 1.0]
    obj_rgba = [0.8, 0.6, 0.4, 1.0]

    for sub, rgba in [("walls", wall_rgba), ("objects", obj_rgba), ("stairs", [0.6, 0.6, 0.9, 1.0])]:
        d = coll_dir / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*.box.json")):
            data = json.loads(f.read_text())
            if data.get("type") != "box":
                continue
            size = data["size"]
            pos = data["position"]
            rot = data.get("rotation", [0, 0, 0, 1])  # xyzw
            box = _box_geom(size, rgba)
            transform = trimesh.transformations.quaternion_matrix(
                [rot[3], rot[0], rot[1], rot[2]]  # trimesh wants wxyz
            )
            transform[:3, 3] = pos
            scene.add_geometry(box, node_name=f.stem, transform=transform)

    scene.export(OUT / "environment_scene.glb")
    print(f"environment_scene.glb <- terrain + {len(list(coll_dir.rglob('*.box.json')))} collision boxes")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    export_terrain_mesh()
    export_point_cloud()
    export_environment_scene()
    print(f"\nDone. Files in {OUT}")


if __name__ == "__main__":
    main()
