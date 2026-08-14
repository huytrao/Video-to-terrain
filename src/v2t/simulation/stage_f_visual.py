"""Stage F: Visual Asset Generation.

Reads structured_scene.json and geometry outputs, produces:
- CFG.paths.visual/scene_visual.obj (visual mesh for rendering)
- CFG.paths.visual/materials.json (material definitions)
- CFG.paths.visual/visual_assets.json (visual asset metadata)
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.v2t.config import CFG


def load_structured_scene() -> dict[str, Any] | None:
    """Load structured_scene.json from scene directory."""
    scene_path = CFG.paths.scene / "structured_scene.json"
    if not scene_path.exists():
        print(f"WARNING: {scene_path} not found. Producing stub outputs.")
        return None
    with open(scene_path) as f:
        return json.load(f)


def find_geometry_mesh() -> Path | None:
    """Find the main geometry mesh file."""
    # Priority: OBJ > PLY > GLTF/GLB
    for pattern in ["*.obj", "*.ply", "*.gltf", "*.glb"]:
        meshes = list(CFG.paths.geometry.glob(pattern))
        if meshes:
            return meshes[0]
    return None


def copy_or_convert_visual_mesh() -> Path | None:
    """Copy or convert geometry mesh to visual output directory."""
    CFG.paths.visual.mkdir(parents=True, exist_ok=True)

    src_mesh = find_geometry_mesh()
    if src_mesh is None:
        print("WARNING: No geometry mesh found. Visual output will be stub.")
        return None

    dst_mesh = CFG.paths.visual / "scene_visual.obj"

    # If already OBJ, just copy
    if src_mesh.suffix.lower() == ".obj":
        shutil.copy2(src_mesh, dst_mesh)
        print(f"Copied visual mesh: {src_mesh} -> {dst_mesh}")
        return dst_mesh

    # Try to convert using trimesh
    try:
        import trimesh
        mesh = trimesh.load(str(src_mesh), process=False)
        mesh.export(str(dst_mesh))
        print(f"Converted visual mesh: {src_mesh} -> {dst_mesh}")
        return dst_mesh
    except Exception as e:
        print(f"WARNING: Failed to convert mesh: {e}")
        # Fallback: copy as-is
        shutil.copy2(src_mesh, dst_mesh)
        print(f"Copied visual mesh (fallback): {src_mesh} -> {dst_mesh}")
        return dst_mesh


def build_materials_json(scene_data: dict | None) -> dict:
    """Build materials definition for visual assets."""
    materials = {
        "floor": {
            "base_color": [0.5, 0.5, 0.5, 1.0],
            "metallic": 0.0,
            "roughness": 0.8,
            "emissive": [0.0, 0.0, 0.0, 1.0],
        },
        "wall": {
            "base_color": [0.7, 0.7, 0.7, 1.0],
            "metallic": 0.0,
            "roughness": 0.7,
            "emissive": [0.0, 0.0, 0.0, 1.0],
        },
        "table": {
            "base_color": [0.8, 0.6, 0.4, 1.0],
            "metallic": 0.0,
            "roughness": 0.5,
            "emissive": [0.0, 0.0, 0.0, 1.0],
        },
        "chair": {
            "base_color": [0.6, 0.5, 0.4, 1.0],
            "metallic": 0.0,
            "roughness": 0.6,
            "emissive": [0.0, 0.0, 0.0, 1.0],
        },
        "cabinet": {
            "base_color": [0.5, 0.4, 0.3, 1.0],
            "metallic": 0.0,
            "roughness": 0.5,
            "emissive": [0.0, 0.0, 0.0, 1.0],
        },
        "default": {
            "base_color": [0.8, 0.6, 0.4, 1.0],
            "metallic": 0.0,
            "roughness": 0.5,
            "emissive": [0.0, 0.0, 0.0, 1.0],
        },
    }

    # Add materials for any objects in scene
    if scene_data:
        objects = scene_data.get("objects", [])
        for obj in objects:
            category = obj.get("category", "default")
            if category not in materials:
                materials[category] = materials["default"].copy()

    return materials


def build_visual_assets_json(scene_data: dict | None, visual_mesh_path: Path | None) -> dict:
    """Build visual assets metadata."""
    assets = {
        "scene_mesh": {
            "path": "scene_visual.obj" if visual_mesh_path else None,
            "format": "obj",
            "units": "meters",
            "up_axis": CFG.up_axis,
        },
        "materials": build_materials_json(scene_data),
        "objects": [],
        "provenance": {
            "source_mesh": str(visual_mesh_path) if visual_mesh_path else None,
            "structured_scene": "scene/structured_scene.json",
        },
    }

    if scene_data:
        # Regular objects - have position, rotation (quat), size directly
        # These are the actual object instances with proper transforms
        objects = scene_data.get("objects", [])
        for obj in objects:
            obj_id = obj.get("id", 0)
            category = obj.get("category", "default")
            position = obj.get("position", [0.0, 0.0, 0.0])
            rotation = obj.get("rotation", [1.0, 0.0, 0.0, 0.0])
            size = obj.get("size", [1.0, 1.0, 1.0])

            assets["objects"].append({
                "id": obj_id,
                "category": category,
                "name": obj.get("name", f"obj_{obj_id}"),
                "position": position,
                "rotation": rotation,
                "size": size,
                "material": category,
                "visible": True,
                "cast_shadows": True,
            })

        # Optionally add room floor/walls if they have OBB data (future enhancement)
        # Current format uses plane+bounds which requires complex conversion

    return assets


def run_stage_f() -> dict:
    """Run Stage F: Visual Asset Generation."""
    print("=" * 60)
    print("Stage F: Visual Asset Generation")
    print("=" * 60)

    CFG.paths.visual.mkdir(parents=True, exist_ok=True)

    # Load structured scene
    scene_data = load_structured_scene()

    # Copy/convert visual mesh
    visual_mesh_path = copy_or_convert_visual_mesh()

    # Build materials and visual assets
    materials = build_materials_json(scene_data)
    visual_assets = build_visual_assets_json(scene_data, visual_mesh_path)

    # Write materials.json
    materials_path = CFG.paths.visual / "materials.json"
    with open(materials_path, "w") as f:
        json.dump(materials, f, indent=2)
    print(f"Wrote: {materials_path}")

    # Write visual_assets.json
    assets_path = CFG.paths.visual / "visual_assets.json"
    with open(assets_path, "w") as f:
        json.dump(visual_assets, f, indent=2)
    print(f"Wrote: {assets_path}")

    print(f"Stage F complete. Visual mesh: {visual_mesh_path.name if visual_mesh_path else 'None'}")
    return {
        "visual_mesh": str(visual_mesh_path) if visual_mesh_path else None,
        "materials": str(materials_path),
        "visual_assets": str(assets_path),
    }


if __name__ == "__main__":
    run_stage_f()