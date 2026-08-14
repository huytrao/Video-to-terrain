"""Stage D2: SAM3 text-prompted mask generation.

Reads vlm_labels.json from CFG.paths.holi_spatial. For each frame+label,
generates a SAM3 segmentation mask and saves it as .npy to
CFG.paths.sam_masks/mask_<label>_<frame_stem>.npy, plus a metadata.json.

If SAM3 cannot be loaded (heavy, may OOM on M1), FALLBACK to stub:
per label, use depth edges + connected components to create a coarse mask,
with `stub: true` recorded in metadata.json. Keeps the MVP working end-to-end.

Production: Set SAM3_CHECKPOINT env var to use local checkpoint, or let it
auto-download from HuggingFace (facebook/sam3) on first run.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.v2t.config import CFG

# holi-spatial repo location (inspect copy with sam3 package)
HOLI_SPATIAL_REPO = Path("/tmp/holi-spatial-inspect")

# MVP speed limit
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "10"))

# SAM3 checkpoint path (optional, for production)
SAM3_CHECKPOINT = os.getenv("SAM3_CHECKPOINT")  # e.g., /path/to/sam3.pth


def slugify_label(label: str) -> str:
    """Make a filesystem-safe slug from a label."""
    safe = re.sub(r"[^\w\-]+", "_", label.strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "label"


def get_frame_files(frames_dir: Path, max_frames: int = MAX_FRAMES) -> list[Path]:
    """Sorted frame files (jpg or png), capped at max_frames."""
    files = sorted(frames_dir.glob("frame_*.jpg"))
    if not files:
        files = sorted(frames_dir.glob("frame_*.png"))
    return files[:max_frames]


def _to_numpy(arr: Any) -> np.ndarray:
    """Best-effort conversion of tensor/array-like to numpy."""
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "detach"):  # torch tensor
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def normalize_masks(output: dict) -> list[dict]:
    """Normalize SAM3 output into a list of {mask, box, score} dicts.

    Masks end up as (H, W) bool arrays. Handles shapes (N,H,W) and (H,W).
    """
    masks = output.get("masks")
    if masks is None:
        masks = output.get("out_binary_masks")
    if masks is None:
        return []

    masks = _to_numpy(masks)
    boxes = output.get("boxes")
    scores = output.get("scores")
    if boxes is not None:
        boxes = _to_numpy(boxes)
    if scores is not None:
        scores = _to_numpy(scores)

    if masks.ndim == 4:
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    elif masks.ndim == 3:
        pass
    elif masks.ndim == 2:
        masks = masks[None, ...]
    else:
        return []

    n = len(masks)
    if boxes is not None and boxes.ndim >= 2:
        boxes = boxes.reshape(-1, boxes.shape[-1])
    if scores is not None:
        scores = scores.reshape(-1)

    items: list[dict] = []
    for i in range(n):
        m = masks[i]
        mask_bool = m.astype(bool) if m.dtype != bool else (m > 0)
        box = None
        if boxes is not None and i < len(boxes):
            box = boxes[i].tolist()
        score = None
        if scores is not None and i < len(scores):
            score = float(scores[i])
        items.append({"mask": mask_bool, "box": box, "score": score})
    return items


def _try_load_sam3():
    """Attempt to build the SAM3 model + processor. Returns (processor, err).

    Supports SAM3_CHECKPOINT env var for production local checkpoints.
    Falls back to auto-download from HuggingFace (facebook/sam3).
    """
    try:
        if HOLI_SPATIAL_REPO.exists():
            repo_str = str(HOLI_SPATIAL_REPO)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
        from sam3.model_builder import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore

        checkpoint_path = SAM3_CHECKPOINT if SAM3_CHECKPOINT else None
        if checkpoint_path:
            print(f"Loading SAM3 model from local checkpoint: {checkpoint_path}")
        else:
            print("Loading SAM3 model (auto HF download from facebook/sam3, may take a moment)...")

        model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        processor = Sam3Processor(model, confidence_threshold=CFG.sam3_confidence_threshold)
        print("SAM3 model loaded successfully.")
        return processor, None
    except ImportError as e:
        return None, f"SAM3 package not available: {e}. Ensure holi-spatial repo is at {HOLI_SPATIAL_REPO}"
    except Exception as e:  # noqa: BLE001 - broad on purpose for fallback
        return None, e


def create_stub_masks(
    per_image: dict[str, list[str]],
    frame_files: list[Path],
    sam_masks_dir: Path,
) -> dict:
    """Fallback: create masks using depth edges + connected components.

    For each frame+label, load corresponding depth map, compute edges via
    Sobel gradient, then use connected components to find regions.
    """
    sam_masks_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for frame_path in frame_files:
        frame_name = frame_path.name
        labels = per_image.get(frame_name, [])
        if not labels:
            continue

        # Try to load depth map for edge-based fallback
        depth_path = CFG.paths.depth / f"{frame_path.stem}.npy"
        depth = None
        if depth_path.exists():
            try:
                depth = np.load(depth_path)
            except Exception:
                depth = None

        try:
            with Image.open(frame_path) as im:
                w, h = im.size
        except Exception:
            w, h = 640, 480

        for label in set(labels):
            safe_label = slugify_label(label)

            # Create mask using depth edges + connected components
            if depth is not None and depth.shape[:2] == (h, w):
                mask_arr = _create_mask_from_depth_edges(depth, label, h, w)
            else:
                # Final fallback: central rectangle covering 60% of frame
                mask_arr = np.zeros((h, w), dtype=np.uint8)
                y1, y2 = int(h * 0.2), int(h * 0.8)
                x1, x2 = int(w * 0.2), int(w * 0.8)
                mask_arr[y1:y2, x1:x2] = 255

            # Save as .npy (binary mask as uint8)
            mask_file = sam_masks_dir / f"mask_{safe_label}_{frame_path.stem}.npy"
            np.save(mask_file, mask_arr.astype(np.uint8))

            # Compute bbox for metadata
            ys, xs = np.where(mask_arr > 0)
            if len(xs) > 0 and len(ys) > 0:
                bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]

            all_results.append({
                "frame": frame_name,
                "label": label,
                "mask_path": f"mask_{safe_label}_{frame_path.stem}.npy",
                "bbox": bbox,
                "score": 0.5,
                "stub": True,
            })

    metadata = {
        "scene": CFG.run_name,
        "image_root": str(CFG.paths.frames),
        "mask_root": str(sam_masks_dir),
        "items": all_results,
        "stub": True,
    }
    metadata_path = sam_masks_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"STUB mask metadata saved: {metadata_path} ({len(all_results)} masks)")
    return {"metadata": str(metadata_path), "num_masks": len(all_results), "stub": True}


def _create_mask_from_depth_edges(depth: np.ndarray, label: str, h: int, w: int) -> np.ndarray:
    """Create a coarse mask from depth edges using connected components."""
    import cv2

    # Normalize depth to 0-255
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Compute edges via Sobel
    sobel_x = cv2.Sobel(depth_norm, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_norm, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold edges
    _, edges_bin = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

    # Invert to get regions (non-edges)
    regions = cv2.bitwise_not(edges_bin)

    # Connected components
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(regions, connectivity=8)

    # Pick the largest component that's not the background (label 0)
    best_label = 1
    best_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_label = i

    # Create mask from best component
    mask = (labels_img == best_label).astype(np.uint8) * 255

    # If mask is too small or covers too much, fall back to central rect
    mask_ratio = mask.sum() / (255 * h * w)
    if mask_ratio < 0.01 or mask_ratio > 0.9:
        mask = np.zeros((h, w), dtype=np.uint8)
        y1, y2 = int(h * 0.2), int(h * 0.8)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        mask[y1:y2, x1:x2] = 255

    return mask


def run_stage_d2() -> dict:
    """Run Stage D2: SAM3 text-prompted mask generation."""
    paths = CFG.paths
    vlm_labels_path = paths.holi_spatial / "vlm_labels.json"
    sam_masks_dir = paths.sam_masks
    sam_masks_dir.mkdir(parents=True, exist_ok=True)

    # Load VLM labels
    if not vlm_labels_path.exists():
        print(f"WARNING: vlm_labels.json not found at {vlm_labels_path}. Using stub labels.")
        vlm_data: dict[str, Any] = {"per_image": {}, "stub": True}
    else:
        with open(vlm_labels_path, "r", encoding="utf-8") as f:
            vlm_data = json.load(f)

    # Handle both old format (list of strings) and new format (list of objects with category)
    per_image_raw = vlm_data.get("per_image", {}) or {}
    per_image: dict[str, list[str]] = {}
    for frame_name, objects in per_image_raw.items():
        if isinstance(objects, list) and objects and isinstance(objects[0], dict):
            # New format: list of {"category": "...", "bbox": ...}
            per_image[frame_name] = [obj.get("category", "") for obj in objects if obj.get("category")]
        elif isinstance(objects, list):
            # Old format: list of strings
            per_image[frame_name] = objects
        else:
            per_image[frame_name] = []
    is_stub_vlm = bool(vlm_data.get("stub", False))

    # Frames: process the same frames Stage D1 (VLM) actually labeled, not an
    # independent first-N-sequential slice - those rarely overlap since VLM
    # uniformly samples across the whole clip (was causing masks for only
    # 1-2 frames out of the ~20 VLM labeled).
    if per_image:
        frame_files = [
            paths.frames / name for name in per_image.keys()
            if (paths.frames / name).exists()
        ]
        frame_files.sort()
    else:
        frame_files = get_frame_files(paths.frames)
    if not frame_files:
        print(f"WARNING: No frames found in {paths.frames}. Writing empty metadata.")
        empty_metadata = {
            "scene": CFG.run_name,
            "image_root": str(paths.frames),
            "mask_root": str(sam_masks_dir),
            "items": [],
            "stub": True,
        }
        metadata_path = sam_masks_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(empty_metadata, f, indent=2, ensure_ascii=False)
        return {"metadata": str(metadata_path), "num_masks": 0, "stub": True}

    # If VLM labels are stubs, skip SAM3 (no real labels) and produce stub masks.
    if is_stub_vlm:
        print("VLM labels are stubs; producing stub masks.")
        return create_stub_masks(per_image, frame_files, sam_masks_dir)

    # Try SAM3
    processor, err = _try_load_sam3()
    if processor is None:
        print(f"SAM3 unavailable ({err}). Falling back to stub masks.")
        return create_stub_masks(per_image, frame_files, sam_masks_dir)

    # Process each frame+label with SAM3
    all_results: list[dict] = []
    for frame_path in frame_files:
        frame_name = frame_path.name
        labels = per_image.get(frame_name, [])
        if not labels:
            print(f"  {frame_name}: no labels, skipping.")
            continue

        print(f"Processing {frame_name} with {len(set(labels))} label(s): {labels}")
        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"  Failed to open {frame_name}: {e}. Skipping.")
            continue

        for label in sorted(set(labels)):
            try:
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=label)
                items = normalize_masks(output)
                if not items:
                    print(f"  No masks returned for label '{label}'.")
                    continue

                safe_label = slugify_label(label)
                multi = len(items) > 1

                for i, item in enumerate(items, 1):
                    # Save mask as .npy (binary bool array)
                    mask_arr = item["mask"].astype(np.uint8) * 255
                    fname = f"mask_{safe_label}_{i:02d}_{frame_path.stem}.npy" if multi else f"mask_{safe_label}_{frame_path.stem}.npy"
                    mask_file = sam_masks_dir / fname
                    np.save(mask_file, mask_arr)

                    all_results.append({
                        "frame": frame_name,
                        "label": label,
                        "mask_path": fname,
                        "bbox": item["box"],
                        "score": item["score"],
                        "stub": False,
                    })
                print(f"  Saved {len(items)} mask(s) for '{label}'.")
            except Exception as e:  # noqa: BLE001
                print(f"  Error processing label '{label}' on {frame_name}: {e}")

    metadata = {
        "scene": CFG.run_name,
        "image_root": str(paths.frames),
        "mask_root": str(sam_masks_dir),
        "items": all_results,
        "stub": False,
    }
    metadata_path = sam_masks_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Mask metadata saved: {metadata_path} ({len(all_results)} masks)")
    return {"metadata": str(metadata_path), "num_masks": len(all_results), "stub": False}


if __name__ == "__main__":
    run_stage_d2()