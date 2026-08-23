#!/usr/bin/env python3
"""
Step C1: Normalize Holi-Spatial Raw Output into Standardized HoliObjectPackage (OUTPUT_C1)

This script reads metadata from Step B1 (ground_truth_boxes.json, cameras_optimised.json)
and the raw RGB frames, decodes the COCO RLE masks, extracts the best view and multi-view
supporting frames/masks/cameras, and saves them into the standardized adapter structure:

OUTPUT_C1/
└── obj_{id:04d}_{label}/
    ├── object.json
    ├── best_rgb.png
    ├── best_mask.png
    ├── views/
    │   ├── {frame_id}.png
    │   └── ...
    ├── masks/
    │   ├── {frame_id}.png
    │   └── ...
    └── cameras/
        ├── {frame_id}.json
        └── ...
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image


def parse_frame_id(path_or_name: str) -> str:
    """Extracts frame ID like 'DSC00921' from a path or file name."""
    match = re.search(r"(DSC\d+)", path_or_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    base = Path(path_or_name).stem
    return base


def find_image_file(images_dir: Path, frame_id: str) -> Optional[Path]:
    """Finds image file by frame_id with case-insensitive extension check."""
    for ext in [".JPG", ".jpg", ".png", ".PNG", ".jpeg", ".JPEG"]:
        candidate = images_dir / f"{frame_id}{ext}"
        if candidate.is_file():
            return candidate
    # Fallback search
    for p in images_dir.glob(f"*{frame_id}*"):
        if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            return p
    return None


def extract_obb_components(obb_transform: List[List[float]], obb_extents: List[float]) -> Dict:
    """Extracts center, rotation_3x3, and extent_xyz from OBB transform matrix."""
    transform = np.array(obb_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 obb_transform, got shape {transform.shape}")
    
    rotation_3x3 = transform[:3, :3].tolist()
    center_world = transform[:3, 3].tolist()
    extent_xyz = [float(x) for x in obb_extents]
    
    return {
        "center_world": center_world,
        "extent_xyz": extent_xyz,
        "rotation_world_3x3": rotation_3x3,
        "raw_transform_4x4": obb_transform,
    }


def select_best_view_smart(
    images: List[str],
    mask_encodings: List[Dict],
    fallback_best_ref: str,
    border_margin: int = 4,
) -> Tuple[int, str]:
    """
    Selects the optimal frame index and frame_id using Smart Best View Selection:
    1. Prioritizes views where object mask does NOT touch image boundaries (margin >= 4px).
    2. Among non-touching views, picks the view with maximum pixel area (sharpest/highest resolution).
    3. If all views touch the border, picks the view with maximum distance to border and largest area.
    """
    if not images or not mask_encodings:
        frame_id = parse_frame_id(fallback_best_ref) if fallback_best_ref else "UNKNOWN"
        return 0, frame_id

    candidates = []
    for i, (img_ref, rle) in enumerate(zip(images, mask_encodings)):
        fid = parse_frame_id(img_ref)
        try:
            m = mask_util.decode(rle)
            ys, xs = np.where(m > 0)
            if len(ys) == 0:
                continue
            h, w = m.shape
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())
            touches = (
                ymin <= border_margin
                or xmin <= border_margin
                or ymax >= h - 1 - border_margin
                or xmax >= w - 1 - border_margin
            )
            area = int((m > 0).sum())
            dist_to_border = int(min(ymin, xmin, h - 1 - ymax, w - 1 - xmax))
            candidates.append({
                "idx": i,
                "frame_id": fid,
                "touches": touches,
                "dist_to_border": dist_to_border,
                "area": area,
            })
        except Exception:
            continue

    if not candidates:
        frame_id = parse_frame_id(fallback_best_ref) if fallback_best_ref else parse_frame_id(images[0])
        return 0, frame_id

    # Sort: First non-touching border, then maximum pixel area, then distance to border
    candidates.sort(key=lambda c: (not c["touches"], c["area"], c["dist_to_border"]), reverse=True)
    best = candidates[0]
    return best["idx"], best["frame_id"]


def select_top_k_views(
    images: List[str],
    mask_encodings: List[Dict],
    best_view_idx: int,
    max_views: int = 10,
    border_margin: int = 4,
) -> List[int]:
    """
    Selects up to max_views indices from images, always including best_view_idx,
    prioritizing non-border-touching views and largest pixel areas.
    """
    if len(images) <= max_views:
        return list(range(len(images)))

    scored_candidates = []
    for i, (img_ref, rle) in enumerate(zip(images, mask_encodings)):
        try:
            m = mask_util.decode(rle)
            ys, xs = np.where(m > 0)
            if len(ys) == 0:
                continue
            h, w = m.shape
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())
            touches = (
                ymin <= border_margin
                or xmin <= border_margin
                or ymax >= h - 1 - border_margin
                or xmax >= w - 1 - border_margin
            )
            area = int((m > 0).sum())
            dist_to_border = int(min(ymin, xmin, h - 1 - ymax, w - 1 - xmax))
            scored_candidates.append({
                "idx": i,
                "touches": touches,
                "area": area,
                "dist": dist_to_border,
            })
        except Exception:
            continue

    # Sort candidates: non-touching first, then max area, then max distance
    scored_candidates.sort(key=lambda c: (not c["touches"], c["area"], c["dist"]), reverse=True)

    selected_indices = []
    if best_view_idx < len(images):
        selected_indices.append(best_view_idx)

    for c in scored_candidates:
        if c["idx"] not in selected_indices:
            selected_indices.append(c["idx"])
        if len(selected_indices) >= max_views:
            break

    # If still not enough, add remainder
    for i in range(len(images)):
        if i not in selected_indices:
            selected_indices.append(i)
        if len(selected_indices) >= max_views:
            break

    return sorted(selected_indices)


def normalize_holi_output(
    boxes_path: Path,
    cameras_path: Path,
    images_dir: Path,
    output_dir: Path,
    exclude_labels: Set[str],
    include_multi_views: bool = True,
    use_smart_best_view: bool = True,
    max_views: int = 10,
) -> List[Dict]:
    """
    Normalizes Holi-Spatial output into Step C1 HoliObjectPackage format.
    """
    if not boxes_path.is_file():
        raise FileNotFoundError(f"ground_truth_boxes.json not found at {boxes_path}")
    if not cameras_path.is_file():
        raise FileNotFoundError(f"cameras_optimised.json not found at {cameras_path}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found at {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(boxes_path, "r", encoding="utf-8") as f:
        boxes_data = json.load(f)

    with open(cameras_path, "r", encoding="utf-8") as f:
        cameras_list = json.load(f)

    # Index cameras by frame_id
    camera_map: Dict[str, Dict] = {}
    for cam in cameras_list:
        frame_id = parse_frame_id(cam.get("img_name", ""))
        if frame_id:
            camera_map[frame_id] = cam

    summary_list = []
    total_processed = 0
    total_skipped = 0

    print(f"Loaded {len(boxes_data)} boxes and {len(camera_map)} camera poses.")
    print(f"Excluding labels: {exclude_labels}")
    print(f"Smart Best View Selection: {'ENABLED' if use_smart_best_view else 'DISABLED'}")
    print(f"Max Multi-Views per Object: {max_views}")

    for idx, box in enumerate(boxes_data):
        raw_ins_id = box.get("ins_id", str(idx + 1))
        try:
            ins_id_num = int(raw_ins_id)
        except ValueError:
            ins_id_num = idx + 1

        label = box.get("label", "unlabelled").strip().lower()
        clean_label = re.sub(r"[^\w\-]", "_", label)

        if label in exclude_labels:
            total_skipped += 1
            continue

        instance_id = f"obj_{ins_id_num:04d}_{clean_label}"
        obj_dir = output_dir / instance_id
        obj_dir.mkdir(parents=True, exist_ok=True)

        images = box.get("images", [])
        mask_encodings = box.get("mask_encodings", [])
        fallback_best_ref = box.get("highest_confidence_mask", images[0] if images else "")

        if use_smart_best_view:
            best_view_idx, best_frame_id = select_best_view_smart(
                images=images,
                mask_encodings=mask_encodings,
                fallback_best_ref=fallback_best_ref,
            )
        else:
            best_frame_id = parse_frame_id(fallback_best_ref) if fallback_best_ref else (parse_frame_id(images[0]) if images else "UNKNOWN")
            best_view_idx = 0
            if fallback_best_ref in images:
                best_view_idx = images.index(fallback_best_ref)
            else:
                for i, img_ref in enumerate(images):
                    if parse_frame_id(img_ref) == best_frame_id:
                        best_view_idx = i
                        break

        # Decode best mask
        best_mask_img = None
        if mask_encodings and best_view_idx < len(mask_encodings):
            rle = mask_encodings[best_view_idx]
            decoded = mask_util.decode(rle)  # binary 0/1 uint8
            best_mask_img = (decoded * 255).astype(np.uint8)
            cv2.imwrite(str(obj_dir / "best_mask.png"), best_mask_img)

        # Copy/Save best RGB
        best_img_src = find_image_file(images_dir, best_frame_id)
        if best_img_src and best_img_src.is_file():
            img_bgr = cv2.imread(str(best_img_src))
            if img_bgr is not None:
                cv2.imwrite(str(obj_dir / "best_rgb.png"), img_bgr)
            else:
                raise RuntimeError(f"Failed to read image at {best_img_src}")
        else:
            print(f"Warning: Could not find image file for best frame {best_frame_id} for {instance_id}")

        # OBB information
        obb_transform = box.get("obb_transform", [])
        obb_extents = box.get("obb_extents", [])
        obb_data = extract_obb_components(obb_transform, obb_extents) if obb_transform and obb_extents else {}

        # Select top-k views (max 10)
        selected_view_indices = select_top_k_views(
            images=images,
            mask_encodings=mask_encodings,
            best_view_idx=best_view_idx,
            max_views=max_views,
        )

        selected_source_views = [parse_frame_id(images[i]) for i in selected_view_indices if i < len(images)]

        # Multi-view subdirectories
        if include_multi_views and selected_view_indices:
            views_dir = obj_dir / "views"
            masks_dir = obj_dir / "masks"
            cameras_dir = obj_dir / "cameras"
            views_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)
            cameras_dir.mkdir(exist_ok=True)

            for i in selected_view_indices:
                if i >= len(images):
                    continue
                img_ref = images[i]
                fid = parse_frame_id(img_ref)
                # Save mask
                if i < len(mask_encodings):
                    m = (mask_util.decode(mask_encodings[i]) * 255).astype(np.uint8)
                    cv2.imwrite(str(masks_dir / f"{fid}.png"), m)

                # Save RGB view
                img_path = find_image_file(images_dir, fid)
                if img_path and img_path.is_file():
                    # Save png view
                    img_data = cv2.imread(str(img_path))
                    if img_data is not None:
                        cv2.imwrite(str(views_dir / f"{fid}.png"), img_data)

                # Save camera JSON
                if fid in camera_map:
                    with open(cameras_dir / f"{fid}.json", "w", encoding="utf-8") as cf:
                        json.dump(camera_map[fid], cf, indent=2)

        # Build object.json
        object_json_data = {
            "instance_id": instance_id,
            "raw_ins_id": raw_ins_id,
            "semantic_class": label,
            "confidence": 1.0,
            "best_frame_id": best_frame_id,
            "obb": obb_data,
            "bounding_box_corners": box.get("bounding_box", []),
            "source_views": selected_source_views,
            "num_views": len(selected_source_views),
            "total_available_views": len(images),
        }

        with open(obj_dir / "object.json", "w", encoding="utf-8") as of:
            json.dump(object_json_data, of, indent=2)

        summary_list.append({
            "instance_id": instance_id,
            "semantic_class": label,
            "best_frame_id": best_frame_id,
            "num_views": len(selected_source_views),
            "directory": str(obj_dir),
        })

        total_processed += 1
        if total_processed % 20 == 0:
            print(f"Processed {total_processed} objects...")

    # Write summary manifest for Step C1
    manifest_path = output_dir / "c1_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump({
            "total_objects": total_processed,
            "total_skipped": total_skipped,
            "objects": summary_list,
        }, mf, indent=2)

    print(f"\nStep C1 Complete! Successfully processed {total_processed} objects ({total_skipped} skipped).")
    print(f"Manifest written to: {manifest_path}")
    return summary_list


def main():
    parser = argparse.ArgumentParser(description="Step C1: Normalize Holi output into HoliObjectPackages")
    parser.add_argument(
        "--boxes",
        type=Path,
        default=Path("outputs-full-factory/delivery/OUTPUT_B1/metadata/ground_truth_boxes.json"),
        help="Path to ground_truth_boxes.json",
    )
    parser.add_argument(
        "--cameras",
        type=Path,
        default=Path("outputs-full-factory/delivery/OUTPUT_B1/metadata/cameras_optimised.json"),
        help="Path to cameras_optimised.json",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("outputs-full-factory/delivery/INPUT/scannetppv2/data/00777c41d4/dslr/resized_undistorted_images"),
        help="Path to input images directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs-full-factory/delivery/OUTPUT_C1"),
        help="Path to output C1 directory",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[
            "floor", "wall", "ceiling", "background", "door", "glass",
            "wheel", "window", "windows", "brush", "sticker", "label",
            "paper", "cord", "knob", "handle", "hinge", "sign",
            "electrical outlet", "electrical_outlet", "wood", "panel",
            "rubber", "plastic", "base", "support", "light", "lamp",
            "pipe", "pipes", "tube", "tubing", "bottle", "bottles",
            "can", "cans"
        ],
        help="Labels to exclude",
    )
    parser.add_argument(
        "--no-multi-views",
        action="store_true",
        help="Do not export multi-view subdirectories",
    )
    parser.add_argument(
        "--no-smart-view",
        action="store_true",
        help="Disable Smart Best View Selection and use default highest_confidence_mask",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=10,
        help="Maximum multi-views to export per object (default: 10)",
    )

    args = parser.parse_args()

    exclude_set = {label.strip().lower() for label in args.exclude}
    normalize_holi_output(
        boxes_path=args.boxes,
        cameras_path=args.cameras,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        exclude_labels=exclude_set,
        include_multi_views=not args.no_multi_views,
        use_smart_best_view=not args.no_smart_view,
        max_views=args.max_views,
    )


if __name__ == "__main__":
    main()
