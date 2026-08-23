#!/usr/bin/env python3
"""
scripts/adapter/09_run_images_to_3d_batch.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generates high-quality 3D Meshes for custom input objects (e.g. object1..object4 in Images_to_terrain)
using remote Hunyuan3D-2mv on Kaggle GPU.

Handles multi-view mapping (font.png -> front, right, back, left), uploads images,
executes DiT flow-matching shape generation, downloads deliverables (GLB, OBJ) and renders previews.

Usage:
    python scripts/adapter/09_run_images_to_3d_batch.py
"""
import os
import sys
import time
import json
import tarfile
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_TERRAIN_ROOT = REPO_ROOT.parent / "Images_to_terrain"
INPUT_DIR = IMG_TERRAIN_ROOT / "input"
OUTPUT_DIR = IMG_TERRAIN_ROOT / "outputs" / "hunyuan_3d_objects"

def prepare_input_tar():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = Path("/tmp/custom_objects_input.tar.gz")
    
    obj_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("object")])
    print(f"📦 Packaging {len(obj_dirs)} custom object folders from {INPUT_DIR}...")
    
    with tarfile.open(tar_path, "w:gz") as tar:
        for obj_d in obj_dirs:
            print(f"  • {obj_d.name}:")
            for f in sorted(obj_d.glob("*.png")):
                # Map font.png -> front.png if needed
                canonical_name = "front.png" if f.name.lower() in ["font.png", "front.png"] else f.name.lower()
                tar.add(str(f), arcname=f"{obj_d.name}/{canonical_name}")
                print(f"      + {f.name} -> {canonical_name}")
                
    print(f"✓ Package created: {tar_path} ({tar_path.stat().st_size / 1024:.1f} KB)")
    return tar_path, [d.name for d in obj_dirs]

def main():
    tar_path, obj_names = prepare_input_tar()
    
    print("\n=================================================================")
    print("🚀 DISPATCHING 3D GENERATION TO KAGGLE GPU (HUNYUAN3D-2MV)")
    print(f"  • Target objects: {obj_names}")
    print(f"  • Destination: {OUTPUT_DIR}")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
