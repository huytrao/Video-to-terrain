#!/usr/bin/env python3
"""
Step F: Run Hunyuan3D-DiT (Shape Generation)

This script loads the prepared images from Step E (Adapter A) in OUTPUT_D/adapter_a/
and runs the Hunyuan3D-2 / Hunyuan3D-2mv DiT pipeline to generate bare 3D meshes (mesh_raw.glb).

Supports both:
1. Multi-View Mode (Hunyuan3D-2mv): Reads canonical views (front, left, back, right) from mv_inputs/
2. Single-View Mode (Hunyuan3D-2): Reads canonical single view from hunyuan_input.png
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

# Ensure local Hunyuan3D-2 package can be imported
HUNYUAN_ROOT = Path(__file__).resolve().parents[2] / "Hunyuan3D-2"
if str(HUNYUAN_ROOT) not in sys.path:
    sys.path.insert(0, str(HUNYUAN_ROOT))

try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    HAS_HY3D = True
except ImportError as e:
    HAS_HY3D = False
    IMPORT_ERR = e


def run_shape_generation(
    manifest_path: Path,
    output_dir: Path,
    mode: str = "multiview",
    model_name: Optional[str] = None,
    num_inference_steps: int = 50,
    octree_resolution: int = 380,
    target_object: Optional[str] = None,
    limit: Optional[int] = None,
    device: str = "cuda",
):
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if not HAS_HY3D:
        raise RuntimeError(f"Could not import hy3dgen from {HUNYUAN_ROOT}: {IMPORT_ERR}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    objects = manifest.get("objects", [])
    if target_object:
        objects = [o for o in objects if o.get("instance_id") == target_object]
        if not objects:
            raise ValueError(f"Target object '{target_object}' not found in manifest.")

    if limit and limit > 0:
        objects = objects[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine default model subfolder
    if mode == "multiview":
        model_id = model_name or "tencent/Hunyuan3D-2mv"
        subfolder = "hunyuan3d-dit-v2-mv"
    else:
        model_id = model_name or "tencent/Hunyuan3D-2"
        subfolder = "hunyuan3d-dit-v2-0"

    print(f"\n=======================================================")
    print(f"HUNYUAN3D-DiT SHAPE GENERATION")
    print(f"  Mode: {mode.upper()}")
    print(f"  Model ID: {model_id} (subfolder: {subfolder})")
    print(f"  Total objects to generate: {len(objects)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Device: {device}")
    print(f"=======================================================\n")

    # Load Pipeline
    print("Loading Hunyuan3DDiTFlowMatchingPipeline model weights...")
    start_load = time.time()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_id,
        subfolder=subfolder,
        variant="fp16",
    )
    if device == "cuda" and torch.cuda.is_available():
        pipeline.to("cuda")
    print(f"Model loaded successfully in {time.time() - start_load:.2f}s.\n")

    summary_results = []

    for idx, obj in enumerate(objects):
        obj_id = obj["instance_id"]
        sclass = obj.get("semantic_class", "unknown")
        obj_out_dir = output_dir / obj_id
        obj_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx+1}/{len(objects)}] Processing {obj_id} ({sclass})...")
        t0 = time.time()

        # Prepare Input Image(s)
        if mode == "multiview":
            mv_dir = Path(obj.get("mv_inputs_dir", ""))
            input_dict = {}
            if mv_dir.is_dir():
                for slot in ["front", "left", "back", "right"]:
                    slot_file = mv_dir / f"{slot}.png"
                    if slot_file.is_file():
                        input_dict[slot] = Image.open(slot_file).convert("RGBA")

            if not input_dict:
                # Fallback to single hunyuan_input.png
                single_path = Path(obj["hunyuan_input_path"])
                input_data = Image.open(single_path).convert("RGBA")
            else:
                input_data = input_dict
        else:
            single_path = Path(obj["hunyuan_input_path"])
            input_data = Image.open(single_path).convert("RGBA")

        # Inference
        try:
            mesh = pipeline(
                image=input_data,
                num_inference_steps=num_inference_steps,
                octree_resolution=octree_resolution,
                output_type="trimesh",
            )[0]

            out_mesh_glb = obj_out_dir / "mesh_raw.glb"
            out_mesh_obj = obj_out_dir / "mesh_raw.obj"

            mesh.export(str(out_mesh_glb))
            mesh.export(str(out_mesh_obj))

            duration = time.time() - t0
            print(f"  -> Successfully generated 3D mesh ({len(mesh.vertices)} verts, {len(mesh.faces)} faces) in {duration:.2f}s")

            meta_data = {
                "instance_id": obj_id,
                "semantic_class": sclass,
                "mode": mode,
                "generation_time_sec": round(duration, 2),
                "mesh_stats": {
                    "num_vertices": len(mesh.vertices),
                    "num_faces": len(mesh.faces),
                    "bounds": mesh.bounds.tolist() if hasattr(mesh, "bounds") else None,
                },
                "mesh_files": {
                    "glb": str(out_mesh_glb),
                    "obj": str(out_mesh_obj),
                },
            }
            with open(obj_out_dir / "generation_meta.json", "w", encoding="utf-8") as gf:
                json.dump(meta_data, gf, indent=2)

            summary_results.append(meta_data)
        except Exception as e:
            print(f"  -> ERROR generating {obj_id}: {e}")
            meta_data = {
                "instance_id": obj_id,
                "semantic_class": sclass,
                "status": "FAILED",
                "error": str(e),
            }
            with open(obj_out_dir / "generation_meta.json", "w", encoding="utf-8") as gf:
                json.dump(meta_data, gf, indent=2)

    # Write Step F manifest
    manifest_f = output_dir / "stage05_manifest.json"
    with open(manifest_f, "w", encoding="utf-8") as mf:
        json.dump({
            "total_requested": len(objects),
            "total_succeeded": len([s for s in summary_results if "mesh_stats" in s]),
            "results": summary_results,
        }, mf, indent=2)

    print(f"\nStep F Complete! Manifest saved to: {manifest_f}\n")


def main():
    parser = argparse.ArgumentParser(description="Step F: Run Hunyuan3D-DiT 3D Shape Generation")
    parser.add_argument(
        "--adapter-manifest",
        type=Path,
        default=Path("outputs-full-factory/delivery/OUTPUT_D/adapter_a/adapter_a_manifest.json"),
        help="Path to adapter_a_manifest.json from Step E",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs-full-factory/delivery/OUTPUT_F"),
        help="Directory to save generated 3D meshes",
    )
    parser.add_argument(
        "--mode",
        choices=["multiview", "single"],
        default="multiview",
        help="Generation mode (multiview uses Hunyuan3D-2mv, single uses Hunyuan3D-2)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace model ID or local path (default: tencent/Hunyuan3D-2mv for multiview)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion inference steps (default: 50)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=380,
        help="Octree resolution for marching cubes (default: 380)",
    )
    parser.add_argument(
        "--target-object",
        type=str,
        default=None,
        help="Specific instance_id to generate (e.g. obj_0219_table)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of objects to process (e.g. 1 for quick test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda, mps, cpu)",
    )

    args = parser.parse_args()

    run_shape_generation(
        manifest_path=args.adapter_manifest,
        output_dir=args.output_dir,
        mode=args.mode,
        model_name=args.model_name,
        num_inference_steps=args.steps,
        octree_resolution=args.resolution,
        target_object=args.target_object,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
