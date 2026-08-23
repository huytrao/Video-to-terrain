#!/usr/bin/env python3
"""
scripts/adapter/05_align_and_verify_adapter_b.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adapter B: Step G (Initial Alignment) & Step H (Multi-View Verification)

Aligns Hunyuan3D raw meshes (OUTPUT_F) with HoliSpatial 3D OBBs (OUTPUT_C1)
and selects optimal canonical rotation (0°, 90°, 180°, 270°) by evaluating
multi-view silhouette projection IoU against ground-truth camera masks.

Usage:
    python scripts/adapter/05_align_and_verify_adapter_b.py \
        --c1-dir outputs-full-factory/delivery/OUTPUT_C1 \
        --f-dir outputs-full-factory/delivery/OUTPUT_F \
        --output-dir outputs-full-factory/delivery/OUTPUT_G
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import trimesh
import cv2
from PIL import Image

def rot_z(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

def rot_y(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float64)

def rot_x(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c]
    ], dtype=np.float64)

def project_points(pts_world: np.ndarray, cam_pos: np.ndarray, cam_rot: np.ndarray, fx: float, fy: float, cx: float, cy: float, w: int, h: int) -> np.ndarray:
    """
    Project world 3D points onto 2D image plane using camera extrinsics and intrinsics.
    cam_rot is OpenCV camera-to-world rotation or world-to-cam.
    """
    # Convert points to camera frame: p_cam = R_w2c * (p_world - cam_pos)
    # If cam_rot is c2w: R_w2c = cam_rot.T
    R_w2c = cam_rot.T
    pts_cam = (pts_world - cam_pos) @ R_w2c.T
    
    # Check positive depth (Z > 0.05)
    valid = pts_cam[:, 2] > 0.05
    pts_valid = pts_cam[valid]
    
    u = (pts_valid[:, 0] * fx / pts_valid[:, 2]) + cx
    v = (pts_valid[:, 1] * fy / pts_valid[:, 2]) + cy
    
    uv = np.column_stack([u, v])
    # Filter within image bounds
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[in_bounds].astype(np.int32)

def compute_mask_iou(rendered_pts_uv: np.ndarray, gt_mask: np.ndarray) -> float:
    if len(rendered_pts_uv) == 0:
        return 0.0
    h, w = gt_mask.shape[:2]
    # Draw convex hull or filled points
    rendered_mask = np.zeros((h, w), dtype=np.uint8)
    if len(rendered_pts_uv) >= 3:
        hull = cv2.convexHull(rendered_pts_uv)
        cv2.drawContours(rendered_mask, [hull], -1, 255, -1)
    else:
        for pt in rendered_pts_uv:
            rendered_mask[pt[1], pt[0]] = 255
            
    gt_bin = (gt_mask > 0).astype(np.uint8)
    rend_bin = (rendered_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(gt_bin, rend_bin).sum()
    union = np.logical_or(gt_bin, rend_bin).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

def align_single_object(
    obj_id: str,
    mesh_glb_path: Path,
    c1_obj_dir: Path,
    output_dir: Path,
) -> Dict:
    obj_json_path = c1_obj_dir / "object.json"
    if not obj_json_path.exists():
        raise FileNotFoundError(f"Missing {obj_json_path}")
    
    with open(obj_json_path) as f:
        meta = json.load(f)
        
    obb = meta.get("obb", {})
    center_world = np.array(obb.get("center_world", [0, 0, 0]), dtype=np.float64)
    extent_xyz = np.array(obb.get("extent_xyz", [1, 1, 1]), dtype=np.float64)
    rot_world = np.array(obb.get("rotation_world_3x3", np.eye(3)), dtype=np.float64)
    
    # Load raw mesh
    mesh = trimesh.load(str(mesh_glb_path), force="mesh")
    
    # 1. Center raw mesh at origin
    mesh_center = mesh.centroid
    mesh.vertices -= mesh_center
    
    # 2. Derive scale
    raw_extents = mesh.extents
    # Avoid zero division
    raw_extents = np.maximum(raw_extents, 1e-4)
    
    # Match principal extents
    # Scale uniformly or anisotropically
    scale_factors = extent_xyz / raw_extents
    uniform_scale = float(np.median(scale_factors))
    
    # 3. Load camera views and masks for Multi-View Verification
    cam_dir = c1_obj_dir / "cameras"
    mask_dir = c1_obj_dir / "masks"
    
    cameras = []
    if cam_dir.exists() and mask_dir.exists():
        for cam_file in sorted(cam_dir.glob("*.json")):
            stem = cam_file.stem
            mask_file = mask_dir / f"{stem}.png"
            if mask_file.exists():
                with open(cam_file) as f:
                    cdata = json.load(f)
                gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    cameras.append({
                        "cam_pos": np.array(cdata["position"], dtype=np.float64),
                        "cam_rot": np.array(cdata["rotation"], dtype=np.float64),
                        "fx": float(cdata["fx"]),
                        "fy": float(cdata["fy"]),
                        "cx": float(cdata["width"]) / 2.0,
                        "cy": float(cdata["height"]) / 2.0,
                        "width": int(cdata["width"]),
                        "height": int(cdata["height"]),
                        "gt_mask": gt_mask,
                    })
    
    # Sample surface points for fast silhouette rendering
    sample_pts, _ = trimesh.sample.sample_surface(mesh, 4000)
    sample_pts_scaled = sample_pts * scale_factors
    
    # 4. Evaluate Rotation Candidates (0°, 90°, 180°, 270°)
    angles = [0, 90, 180, 270]
    candidate_scores = []
    
    for angle_deg in angles:
        rad = np.radians(angle_deg)
        R_yaw = rot_z(rad)
        R_cand = rot_world @ R_yaw
        
        # Transform points to candidate pose
        pts_world_cand = (sample_pts_scaled @ R_cand.T) + center_world
        
        ious = []
        for cam in cameras:
            uv = project_points(
                pts_world_cand,
                cam["cam_pos"],
                cam["cam_rot"],
                cam["fx"],
                cam["fy"],
                cam["cx"],
                cam["cy"],
                cam["width"],
                cam["height"]
            )
            iou = compute_mask_iou(uv, cam["gt_mask"])
            ious.append(iou)
            
        mean_iou = float(np.mean(ious)) if ious else 0.5
        candidate_scores.append({
            "angle_deg": angle_deg,
            "mean_iou": mean_iou,
            "R_cand": R_cand.tolist(),
        })
        
    # Select best candidate
    best_candidate = max(candidate_scores, key=lambda x: x["mean_iou"])
    best_angle = best_candidate["angle_deg"]
    best_R = np.array(best_candidate["R_cand"], dtype=np.float64)
    best_iou = best_candidate["mean_iou"]
    
    # 5. Apply optimal transformation to mesh
    aligned_mesh = mesh.copy()
    aligned_mesh.vertices = (aligned_mesh.vertices * scale_factors) @ best_R.T + center_world
    
    # 6. Build 4x4 transform matrix
    T4x4 = np.eye(4, dtype=np.float64)
    T4x4[:3, :3] = best_R
    T4x4[:3, 3] = center_world
    
    # 7. Save Deliverables
    obj_out_dir = output_dir / obj_id
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    
    aligned_glb = obj_out_dir / "mesh_aligned.glb"
    aligned_obj = obj_out_dir / "mesh_aligned.obj"
    aligned_mesh.export(str(aligned_glb))
    aligned_mesh.export(str(aligned_obj))
    
    meta_result = {
        "instance_id": obj_id,
        "semantic_class": meta.get("semantic_class", "unknown"),
        "best_yaw_angle_deg": best_angle,
        "mean_mask_iou": best_iou,
        "scale_factors_xyz": scale_factors.tolist(),
        "obb_center_world": center_world.tolist(),
        "obb_extent_xyz": extent_xyz.tolist(),
        "transform_matrix_4x4": T4x4.tolist(),
        "candidate_evaluations": candidate_scores,
        "mesh_stats": {
            "vertices": len(aligned_mesh.vertices),
            "faces": len(aligned_mesh.faces),
            "bounds_min": aligned_mesh.bounds[0].tolist(),
            "bounds_max": aligned_mesh.bounds[1].tolist(),
        }
    }
    
    with open(obj_out_dir / "alignment_meta.json", "w") as f:
        json.dump(meta_result, f, indent=2)
        
    return meta_result

def main():
    parser = argparse.ArgumentParser(description="Adapter B: Align and Verify 3D Meshes")
    parser.add_argument("--c1-dir", type=Path, default=Path("outputs-full-factory/delivery/OUTPUT_C1"))
    parser.add_argument("--f-dir", type=Path, default=Path("outputs-full-factory/delivery/OUTPUT_F"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs-full-factory/delivery/OUTPUT_G"))
    parser.add_argument("--target-object", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.target_object:
        obj_dirs = [args.f_dir / args.target_object]
    else:
        obj_dirs = sorted([d for d in args.f_dir.iterdir() if d.is_dir()])

    print("=================================================================")
    print("🎯 ADAPTER B: STEP G (INITIAL ALIGNMENT) & STEP H (VERIFICATION)")
    print(f"  • C1 Metadata dir: {args.c1_dir}")
    print(f"  • Raw Meshes (F): {args.f_dir}")
    print(f"  • Aligned Meshes (G): {args.output_dir}")
    print(f"  • Total objects to align: {len(obj_dirs)}")
    print("=================================================================\n")

    manifest = []
    t0 = time.time()
    for idx, obj_dir in enumerate(obj_dirs):
        obj_id = obj_dir.name
        mesh_glb = obj_dir / "mesh_raw.glb"
        c1_obj = args.c1_dir / obj_id
        
        if not mesh_glb.exists():
            print(f"[{idx+1}/{len(obj_dirs)}] ⚠️  {obj_id}: Missing mesh_raw.glb, skipping.")
            continue
        if not c1_obj.exists():
            print(f"[{idx+1}/{len(obj_dirs)}] ⚠️  {obj_id}: Missing C1 folder {c1_obj}, skipping.")
            continue
            
        t_start = time.time()
        try:
            res = align_single_object(obj_id, mesh_glb, c1_obj, args.output_dir)
            elapsed = time.time() - t_start
            print(f"[{idx+1}/{len(obj_dirs)}] ✓ {obj_id}: Aligned in {elapsed:.2f}s | Best Yaw={res['best_yaw_angle_deg']}° | Mean IoU={res['mean_mask_iou']:.3f}")
            manifest.append(res)
        except Exception as e:
            print(f"[{idx+1}/{len(obj_dirs)}] ✗ {obj_id}: ERROR: {e}")

    # Write summary manifest
    manifest_path = args.output_dir / "adapter_b_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=================================================================")
    print(f"✅ ADAPTER B COMPLETE: Aligned {len(manifest)} objects in {time.time() - t0:.2f}s")
    print(f"Manifest written to: {manifest_path}")
    print("=================================================================")

if __name__ == "__main__":
    main()
