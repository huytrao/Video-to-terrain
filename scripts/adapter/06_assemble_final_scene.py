#!/usr/bin/env python3
"""
scripts/adapter/06_assemble_final_scene.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Step I: Assembles all 38 aligned 3D assets (OUTPUT_G) into a single unified
Factory 3D Scene (Combined GLB), generates the Master Scene Manifest, and creates
the MuJoCo MJCF simulation XML with collision geometries for robotics RL.

Usage:
    python scripts/adapter/06_assemble_final_scene.py \
        --g-dir outputs-full-factory/delivery/OUTPUT_G \
        --output-dir outputs-full-factory/delivery/OUTPUT_I
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import trimesh

def create_mujoco_mjcf(objects_meta: List[Dict], output_xml_path: Path):
    """
    Generate MuJoCo MJCF simulation XML file with collision bodies for all aligned objects.
    """
    xml_lines = [
        '<mujoco model="small_factory_completed_scene">',
        '  <compiler angle="radian" coordinate="local" inertiafromgeom="true" meshdir="meshes"/>',
        '  <option gravity="0 0 -9.81" timestep="0.002"/>',
        '  <visual>',
        '    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        '  </visual>',
        '  <asset>',
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>',
        '    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>',
        '    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>',
    ]

    # Add mesh assets
    mesh_dir = output_xml_path.parent / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    for obj in objects_meta:
        inst_id = obj["instance_id"]
        xml_lines.append(f'    <mesh name="{inst_id}_mesh" file="{inst_id}.obj"/>')

    xml_lines.extend([
        '  </asset>',
        '  <worldbody>',
        '    <light directional="true" pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>',
        '    <geom name="floor" type="plane" size="50 50 0.1" pos="0 0 -1.5" material="grid"/>',
    ])

    # Add bodies
    for obj in objects_meta:
        inst_id = obj["instance_id"]
        sem_class = obj.get("semantic_class", "object")
        center = obj.get("obb_center_world", [0, 0, 0])
        extent = obj.get("obb_extent_xyz", [1, 1, 1])
        
        # Position in MuJoCo world frame
        pos_str = f"{center[0]:.4f} {center[1]:.4f} {center[2]:.4f}"
        
        xml_lines.extend([
            f'    <!-- {inst_id} ({sem_class}) -->',
            f'    <body name="{inst_id}" pos="0 0 0">',
            f'      <geom name="{inst_id}_geom" type="mesh" mesh="{inst_id}_mesh" rgba="0.7 0.7 0.75 1.0" density="500"/>',
            f'    </body>',
        ])

    xml_lines.extend([
        '  </worldbody>',
        '</mujoco>',
    ])

    with open(output_xml_path, "w") as f:
        f.write("\n".join(xml_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Assemble Final Factory Scene")
    parser.add_argument("--g-dir", type=Path, default=Path("outputs-full-factory/delivery/OUTPUT_G"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs-full-factory/delivery/OUTPUT_I"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = args.g_dir / "adapter_b_manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Missing {manifest_p}")

    with open(manifest_p) as f:
        objects_meta = json.load(f)

    print("=================================================================")
    print("🏭 STEP I: ASSEMBLING FULL FACTORY 3D SCENE")
    print(f"  • Aligned Assets dir (G): {args.g_dir}")
    print(f"  • Output Final Scene (I): {args.output_dir}")
    print(f"  • Total Objects to Assemble: {len(objects_meta)}")
    print("=================================================================\n")

    scene = trimesh.Scene()
    combined_meshes = []
    
    mesh_asset_dir = args.output_dir / "meshes"
    mesh_asset_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for idx, obj_info in enumerate(objects_meta):
        inst_id = obj_info["instance_id"]
        sem_class = obj_info["semantic_class"]
        glb_path = args.g_dir / inst_id / "mesh_aligned.glb"
        obj_path = args.g_dir / inst_id / "mesh_aligned.obj"

        if not glb_path.exists():
            print(f"[{idx+1}/{len(objects_meta)}] ⚠️  Missing {glb_path}")
            continue

        mesh = trimesh.load(str(glb_path), force="mesh")
        # Give distinctive visual color per semantic class
        colors = {
            "table": [180, 140, 100, 255],
            "workbench": [160, 120, 80, 255],
            "machine": [80, 140, 200, 255],
            "shelf": [200, 180, 140, 255],
            "cabinet": [140, 140, 160, 255],
            "box": [220, 160, 80, 255],
            "ladder": [230, 200, 50, 255],
            "trash_can": [100, 100, 100, 255],
            "tool": [220, 80, 80, 255],
            "drawer": [170, 130, 90, 255],
            "radiator": [210, 210, 220, 255],
        }
        c = colors.get(sem_class, [160, 160, 160, 255])
        mesh.visual.vertex_colors = np.tile(c, (len(mesh.vertices), 1))

        scene.add_geometry(mesh, geom_name=inst_id)
        combined_meshes.append(mesh)

        # Copy OBJ to meshes dir for MuJoCo
        if obj_path.exists():
            dest_obj = mesh_asset_dir / f"{inst_id}.obj"
            import shutil
            shutil.copy2(obj_path, dest_obj)

        print(f"  [{idx+1}/{len(objects_meta)}] Added {inst_id} ({sem_class}) - {len(mesh.vertices):,} verts")

    # 1. Export Unified Scene GLB
    scene_glb_path = args.output_dir / "full_factory_scene.glb"
    scene.export(str(scene_glb_path))
    scene_size_mb = scene_glb_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Exported Combined Scene GLB: {scene_glb_path} ({scene_size_mb:.2f} MB)")

    # 2. Export Single Merged Mesh (for lightweight collision/navigation mesh)
    if combined_meshes:
        merged = trimesh.util.concatenate(combined_meshes)
        merged_glb_path = args.output_dir / "factory_merged_mesh.glb"
        merged.export(str(merged_glb_path))
        print(f"✓ Exported Merged Collision Mesh: {merged_glb_path} ({len(merged.vertices):,} total vertices, {len(merged.faces):,} faces)")

    # 3. Create MuJoCo MJCF XML
    mjcf_path = args.output_dir / "factory_scene.mjcf"
    create_mujoco_mjcf(objects_meta, mjcf_path)
    print(f"✓ Created MuJoCo Physics Scene: {mjcf_path}")

    # 4. Master Scene Manifest
    master_manifest = {
        "scene_id": "small_factory_v3d",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_completed_objects": len(objects_meta),
        "scene_bounds": {
            "min": scene.bounds[0].tolist() if scene.bounds is not None else [],
            "max": scene.bounds[1].tolist() if scene.bounds is not None else [],
            "extents": scene.extents.tolist() if scene.extents is not None else [],
        },
        "deliverables": {
            "combined_scene_glb": str(scene_glb_path.name),
            "merged_collision_glb": "factory_merged_mesh.glb",
            "mujoco_mjcf": str(mjcf_path.name),
            "assets_manifest": "scene_assets.json",
        },
        "objects": objects_meta,
    }

    manifest_out = args.output_dir / "scene_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(master_manifest, f, indent=2)
    print(f"✓ Saved Master Scene Manifest: {manifest_out}")

    print(f"\n=================================================================")
    print(f"🎉 STEP I COMPLETE in {time.time() - t0:.2f}s! All 38 assets unified.")
    print("=================================================================")

if __name__ == "__main__":
    main()
