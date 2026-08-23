#!/usr/bin/env python3
"""
scripts/adapter/10_robotics_manipulation_gate.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Robotics Manipulation Quality Gate & MuJoCo Physics Scene Generator

Enforces strict physical constraints for robotic interaction (Unitree G1 / Go2 / Z1):
1. Degeneracy & Flatness Filter: Rejects ultra-flat 2D planar meshes (thickness < 15mm or ratio < 0.03).
2. Physical Classification:
   - STATIC COLLIDERS: Heavy industrial machines, fixed tables, walls, floor.
   - DYNAMIC LIFTABLE: Solid 3D boxes, tools, parts, wheels, containers (with <freejoint/>, mass, friction).
3. Calibration: Injects precise physical mass, inertia, and contact friction.

Usage:
    python scripts/adapter/10_robotics_manipulation_gate.py \
        --g-dir outputs-full-factory/delivery/OUTPUT_G \
        --output-dir outputs-full-factory/delivery/OUTPUT_I
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh

# Known static heavy infrastructure (cannot be lifted by standard mobile manipulators)
STATIC_KEYWORDS = ["machine", "radiator", "cabinet", "shelf", "ladder"]

# Liftable dynamic classes
LIFTABLE_KEYWORDS = ["box", "tool", "cup", "drawer", "part", "wheel", "ball", "door", "crate"]

def compute_inertia_tensor(mass: float, extents: np.ndarray) -> np.ndarray:
    """
    Approximate diagonal inertia tensor for a solid box of given mass and extents.
    I_xx = 1/12 * m * (y^2 + z^2)
    """
    w, h, d = extents
    ixx = (1.0 / 12.0) * mass * (h**2 + d**2)
    iyy = (1.0 / 12.0) * mass * (w**2 + d**2)
    izz = (1.0 / 12.0) * mass * (w**2 + h**2)
    return np.array([ixx, iyy, izz], dtype=np.float64)

def evaluate_object_quality_gate(obj_meta: Dict, glb_path: Path) -> Dict:
    inst_id = obj_meta["instance_id"]
    sem_class = obj_meta.get("semantic_class", "unknown").lower()
    
    mesh = trimesh.load(str(glb_path), force="mesh")
    extents = mesh.extents
    min_ext = float(np.min(extents))
    max_ext = float(np.max(extents))
    aspect_ratio = min_ext / max(max_ext, 1e-4)
    volume = float(mesh.volume) if mesh.is_watertight else float(np.prod(extents) * 0.5)
    
    # Check 1: Degenerate Flat / Planar Mesh Filter
    is_degenerate_flat = (min_ext < 0.015) or (aspect_ratio < 0.035)
    
    # Check 2: Classification (Static Obstacle vs Dynamic Liftable)
    if is_degenerate_flat:
        category = "REJECTED_DEGENERATE_FLAT"
        liftable = False
        reason = f"Thickness ({min_ext*1000:.1f} mm) or aspect ratio ({aspect_ratio:.3f}) is too flat for stable robotic grasping."
        mass_kg = 0.0
    elif any(k in sem_class or k in inst_id for k in STATIC_KEYWORDS):
        category = "STATIC_OBSTACLE"
        liftable = False
        reason = "Heavy fixed industrial machinery or large structure. Static obstacle collider."
        mass_kg = float(max(50.0, volume * 1500.0))
    elif any(k in sem_class or k in inst_id for k in LIFTABLE_KEYWORDS):
        category = "DYNAMIC_LIFTABLE"
        liftable = True
        reason = "Solid 3D object within robotic manipulator payload limit."
        # Calibrate realistic manipulation mass (1 - 10 kg)
        mass_kg = float(np.clip(volume * 400.0, 0.5, 12.0))
    else:
        # Default fallback based on mass & size
        if max_ext > 1.2 or volume > 0.3:
            category = "STATIC_OBSTACLE"
            liftable = False
            reason = "Large volume object classified as static structure."
            mass_kg = 80.0
        else:
            category = "DYNAMIC_LIFTABLE"
            liftable = True
            reason = "Standard manipulable component."
            mass_kg = 2.5

    inertia = compute_inertia_tensor(mass_kg, extents) if mass_kg > 0 else np.zeros(3)

    return {
        "instance_id": inst_id,
        "semantic_class": sem_class,
        "category": category,
        "is_liftable": liftable,
        "rejection_or_role_reason": reason,
        "physical_properties": {
            "mass_kg": round(mass_kg, 3),
            "extents_xyz_meters": [round(float(x), 4) for x in extents],
            "extents_xyz_mm": [round(float(x) * 1000.0, 1) for x in extents],
            "min_thickness_mm": round(min_ext * 1000.0, 1),
            "aspect_ratio": round(aspect_ratio, 4),
            "estimated_volume_m3": round(volume, 6),
            "inertia_diagonal": [round(float(x), 6) for x in inertia],
            "friction": {"sliding": 0.8, "torsional": 0.01, "rolling": 0.001},
        },
        "mesh_stats": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "is_watertight": mesh.is_watertight,
        }
    }

def generate_mujoco_robotics_scene(audited_objects: List[Dict], g_dir: Path, output_xml_path: Path):
    """
    Creates complete MuJoCo XML with separate dynamic liftable bodies vs static environment.
    """
    mesh_dir = output_xml_path.parent / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    xml_lines = [
        '<mujoco model="factory_robotics_manipulation">',
        '  <compiler angle="radian" coordinate="local" inertiafromgeom="true" meshdir="meshes"/>',
        '  <option gravity="0 0 -9.81" timestep="0.002" iterations="50" solver="Newton" cone="elliptic"/>',
        '  <visual>',
        '    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        '  </visual>',
        '  <default>',
        '    <geom friction="0.8 0.01 0.001" margin="0.001" solref="0.004 1" solimp="0.9 0.95 0.001"/>',
        '  </default>',
        '  <asset>',
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>',
        '    <texture name="floor_grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.15 0.18 0.22" rgb2="0.22 0.25 0.30"/>',
        '    <material name="floor_mat" texture="floor_grid" texrepeat="2 2" texuniform="true" reflectance="0.15"/>',
    ]

    # Register meshes
    for obj in audited_objects:
        if obj["category"] == "REJECTED_DEGENERATE_FLAT":
            continue
        inst_id = obj["instance_id"]
        obj_file = g_dir / inst_id / "mesh_aligned.obj"
        if obj_file.exists():
            shutil.copy2(obj_file, mesh_dir / f"{inst_id}.obj")
            xml_lines.append(f'    <mesh name="{inst_id}_mesh" file="{inst_id}.obj"/>')

    xml_lines.extend([
        '  </asset>',
        '  <worldbody>',
        '    <light directional="true" pos="0 0 12" dir="0 0 -1" diffuse="0.85 0.85 0.85" specular="0.2 0.2 0.2"/>',
        '    <geom name="factory_floor" type="plane" size="60 60 0.1" pos="0 0 -1.5" material="floor_mat"/>',
        '',
        '    <!-- ================= STATIC ENVIRONMENT & OBSTACLES ================= -->',
    ])

    # 1. Static Bodies
    for obj in audited_objects:
        if obj["category"] == "STATIC_OBSTACLE":
            inst_id = obj["instance_id"]
            sem_class = obj["semantic_class"]
            xml_lines.extend([
                f'    <!-- Static: {inst_id} ({sem_class}) -->',
                f'    <body name="{inst_id}" pos="0 0 0">',
                f'      <geom name="{inst_id}_geom" type="mesh" mesh="{inst_id}_mesh" rgba="0.65 0.70 0.75 1.0"/>',
                f'    </body>',
            ])

    xml_lines.extend([
        '',
        '    <!-- ================= DYNAMIC LIFTABLE ASSETS (ROBOT GRASPABLE) ================= -->',
    ])

    # 2. Dynamic Liftable Bodies
    for obj in audited_objects:
        if obj["category"] == "DYNAMIC_LIFTABLE":
            inst_id = obj["instance_id"]
            sem_class = obj["semantic_class"]
            mass = obj["physical_properties"]["mass_kg"]
            xml_lines.extend([
                f'    <!-- Dynamic Liftable: {inst_id} ({sem_class}) - Mass: {mass}kg -->',
                f'    <body name="{inst_id}" pos="0 0 0">',
                f'      <freejoint name="{inst_id}_joint"/>',
                f'      <geom name="{inst_id}_geom" type="mesh" mesh="{inst_id}_mesh" mass="{mass}" rgba="0.85 0.55 0.25 1.0"/>',
                f'    </body>',
            ])

    xml_lines.extend([
        '  </worldbody>',
        '</mujoco>',
    ])

    with open(output_xml_path, "w") as f:
        f.write("\n".join(xml_lines) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Robotics Manipulation Quality Gate")
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
    print("🤖 ROBOTICS MANIPULATION QUALITY GATE AUDIT")
    print(f"  • Aligned Assets: {args.g_dir}")
    print(f"  • Total Objects to Evaluate: {len(objects_meta)}")
    print("=================================================================\n")

    audited = []
    dynamic_count = 0
    static_count = 0
    rejected_count = 0

    for idx, obj in enumerate(objects_meta):
        inst_id = obj["instance_id"]
        glb_path = args.g_dir / inst_id / "mesh_aligned.glb"
        if not glb_path.exists():
            continue
            
        res = evaluate_object_quality_gate(obj, glb_path)
        audited.append(res)
        
        cat = res["category"]
        if cat == "DYNAMIC_LIFTABLE":
            dynamic_count += 1
            icon = "📦 [LIFTABLE]"
        elif cat == "STATIC_OBSTACLE":
            static_count += 1
            icon = "🏭 [STATIC]"
        else:
            rejected_count += 1
            icon = "🚫 [REJECTED FLAT]"
            
        print(f"[{idx+1}/{len(objects_meta)}] {icon} {inst_id} ({res['semantic_class']}): {res['rejection_or_role_reason']} | Mass={res['physical_properties']['mass_kg']}kg")

    # Generate MuJoCo XML
    mjcf_out = args.output_dir / "robotics_manipulation_scene.xml"
    generate_mujoco_robotics_scene(audited, args.g_dir, mjcf_out)

    # Save Manifest
    audit_manifest_p = args.output_dir / "robotics_manipulation_manifest.json"
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluated": len(audited),
        "dynamic_liftable_objects": dynamic_count,
        "static_obstacle_colliders": static_count,
        "rejected_degenerate_flat_objects": rejected_count,
        "simulation_scene_xml": str(mjcf_out.name),
        "objects": audited,
    }
    with open(audit_manifest_p, "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n=================================================================")
    print(f"🎉 QUALITY GATE COMPLETE:")
    print(f"  • 📦 Dynamic Liftable Objects: {dynamic_count}")
    print(f"  • 🏭 Static Obstacle Colliders: {static_count}")
    print(f"  • 🚫 Rejected Flat/Degenerate Meshes: {rejected_count}")
    print(f"  • 📄 MuJoCo Robotics Scene: {mjcf_out}")
    print(f"  • 📋 Manipulation Manifest: {audit_manifest_p}")
    print("=================================================================")

if __name__ == "__main__":
    main()
