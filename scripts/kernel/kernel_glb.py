"""Export the reconstruction as GLB: the format anything can open.

Reads the mesh, the geometric object cut and the semantic scene.json, and writes
room LODs plus a scene file with labelled boxes. Every file is read back before it
is reported.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np  # noqa: E402

SCENE = os.environ.get("V3D_SCENE", "small_factory")
WORK = Path(os.environ.get("V3D_WORK_ROOT", "/tmp/run"))
OUT = Path(os.environ.get("V3D_DELIVER", "/tmp/deliver"))
SCALE_MODE = os.environ.get("V3D_SCALE_MODE", "assumed")


def main() -> None:
    from v3d.export import glb

    gs = WORK / "gs" / SCENE
    pruned = gs / "mesh" / "tsdf_fusion_pruned.ply"
    mesh_path = pruned if pruned.exists() else gs / "mesh" / "tsdf_fusion_post.ply"
    if not mesh_path.exists():
        raise SystemExit(f"no mesh at {mesh_path}")

    # The up axis comes from the geometric cut, which derived it from the cameras.
    objs_json = OUT / "objects.json"
    if not objs_json.exists():
        raise SystemExit(f"no {objs_json}; run the outputs stage first")
    cut = json.loads(objs_json.read_text())
    up = np.array(cut["up_axis"], dtype=float)
    print(f"mesh      {mesh_path.name} ({mesh_path.stat().st_size/1e6:.0f} MB)")
    print(f"up axis   {np.round(up, 4).tolist()}")

    # Prefer the labelled semantic boxes; fall back to the geometric cut's boxes.
    scene_json = WORK / "scene" / SCENE / "scene.json"
    objects, source = [], None
    if scene_json.exists():
        doc = json.loads(scene_json.read_text())
        objects = doc.get("objects", [])
        source = "scene.json (semantic, labelled)"
    if not objects:
        objects = [{"label": o.get("label", "object"), "centre": o["centre"],
                    "extent": o["extent"], "obb_axes": o.get("obb_axes")}
                   for o in cut["objects"]]
        source = "objects.json (geometric cut)"
    print(f"boxes     {len(objects)} from {source}")

    rep = glb.export(mesh_path, OUT / "glb", up, objects=objects,
                     scale_mode=SCALE_MODE)
    print()
    for f in rep.files:
        print(f"  {f['path']:26} {f['bytes']/1e6:8.1f} MB  "
              f"{f.get('triangles', '')!s:>9}  nodes {f['nodes']:>3}  "
              f"read-back {f['vertices_read_back']:,} verts")
    print()
    for n in rep.notes:
        print(f"  note: {n}")
    print(f"\n  -> {OUT/'glb'/'glb_manifest.json'}")
    print("\nGLB_OK")


main()
