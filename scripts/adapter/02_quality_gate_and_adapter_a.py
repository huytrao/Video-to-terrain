#!/usr/bin/env python3
"""
Step D & E: Quality Gate and Adapter A (Holi-Spatial -> Hunyuan3D Input Preparation)

1. Quality Gate (Step D):
   - Evaluates camera coverage around each 3D object in 8 azimuth bins (45 deg each).
   - Classifies objects into:
     * COMPLETE_WITH_HUNYUAN: Coverage < 0.50 or missing backside
     * REVIEW_OPTIONAL: 0.50 <= Coverage < 0.75
     * KEEP_HOLI: Coverage >= 0.75
     * SKIP_SMALL_OBJECT: 2D or tiny decorative/detail labels (sticker, label, etc.)
   - Exports coverage_report.csv, complete_objects.json, incomplete_objects.json.

2. Adapter A (Step E):
   - Reads the OBB-anchored SAM 3 masks produced by 03_resegment_obb_sam3.py
     (falls back to C1's original masks when that stage has not been run).
   - Drops instances standing on the object and inpaints the hole they leave.
   - Ranks the surviving views by OBB coverage x sharpness x pixel area and takes
     the top one as the single-view input.
   - Assigns front/left/back/right by azimuth in the object's OWN OBB frame, not
     in world coordinates, because Hunyuan3D-2mv conditions on the slot index.
   - Writes RGBA 518x518 crops with alpha = mask, plus adapter_a_meta.json.

   rembg is deliberately not used: unioning its alpha with the segmentation mask
   is what pulled neighbouring objects and floor into the crop.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image

# Default labels to skip from 3D completion because they are 2D planar details / small attachments
DEFAULT_SMALL_CLASSES = {
    "sticker",
    "label",
    "paper",
    "electrical outlet",
    "cord",
    "knob",
    "handle",
    "hinge",
    "sign",
    "screw",
}


def compute_azimuth_coverage(
    object_center: List[float],
    source_views: List[str],
    cameras_dir: Path,
    num_bins: int = 8,
) -> Tuple[float, Set[int], List[float]]:
    """
    Computes azimuth coverage around object center in the horizontal (X, Y) plane.
    Returns: (coverage_ratio, observed_bins, observed_angles_deg)
    """
    ox, oy, _ = object_center
    observed_bins = set()
    angles_deg = []

    for view_id in source_views:
        cam_file = cameras_dir / f"{view_id}.json"
        if not cam_file.is_file():
            continue
        try:
            with open(cam_file, "r", encoding="utf-8") as f:
                cam_data = json.load(f)
            cpos = cam_data.get("position", [0, 0, 0])
            cx, cy, _ = cpos

            dx = cx - ox
            dy = cy - oy
            angle_rad = math.atan2(dy, dx)
            if angle_rad < 0:
                angle_rad += 2 * math.pi
            
            deg = math.degrees(angle_rad)
            angles_deg.append(deg)

            bin_size_rad = 2 * math.pi / num_bins
            bin_idx = int(angle_rad / bin_size_rad) % num_bins
            observed_bins.add(bin_idx)
        except Exception as e:
            print(f"Warning reading camera {cam_file}: {e}")

    coverage_ratio = len(observed_bins) / float(num_bins)
    return coverage_ratio, observed_bins, sorted(angles_deg)


def obb_corners(obb: Dict) -> np.ndarray:
    """The 8 world-space corners of an OBB from centre/extent/rotation."""
    center = np.asarray(obb["center_world"], dtype=float)
    extent = np.asarray(obb["extent_xyz"], dtype=float)
    rot = np.asarray(obb["rotation_world_3x3"], dtype=float)
    signs = np.array([[sx, sy, sz]
                      for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    return center + (rot @ (signs * extent / 2.0).T).T


def obb_contains(host: Dict, corners: np.ndarray, tol: float = 0.02) -> bool:
    """True when another OBB sits wholly inside `host`, i.e. it is a part of it."""
    center = np.asarray(host["center_world"], dtype=float)
    rot = np.asarray(host["rotation_world_3x3"], dtype=float)
    extent = np.asarray(host["extent_xyz"], dtype=float)
    local = (rot.T @ (corners - center).T).T
    return bool(np.all(np.abs(local) <= extent / 2.0 + tol))


def obb_intersects(a: Dict, b: Dict, tol: float = 0.0) -> bool:
    """True when two OBBs share space - one is standing on or inside the other.

    Corner containment in either direction; approximate, but it only has to
    separate "this box is sitting on the bench" from "this machine happens to be
    behind the bench in the picture".
    """
    for host, guest in ((a, b), (b, a)):
        center = np.asarray(host["center_world"], dtype=float)
        rot = np.asarray(host["rotation_world_3x3"], dtype=float)
        extent = np.asarray(host["extent_xyz"], dtype=float)
        local = (rot.T @ (obb_corners(guest) - center).T).T
        if np.any(np.all(np.abs(local) <= extent / 2.0 + tol, axis=1)):
            return True
    return False


def obb_local_azimuth(obb: Dict, cam_position: Sequence[float]) -> float:
    """Azimuth of a camera around the object's OWN frame, in degrees.

    Hunyuan3D-2mv indexes a learned `view_embed` by slot (conditioner.py:174-180),
    so `front`/`left`/`back`/`right` carry real meaning to the model. Measuring the
    angle in world coordinates — as this stage used to — gives every object a
    "front" that depends on how it happens to be rotated in the room. Measured in
    the OBB's own frame instead, 0 deg is the normal of the object's longest face,
    which is the same place for every instance of a class.
    """
    center = np.asarray(obb["center_world"], dtype=float)
    rot = np.asarray(obb["rotation_world_3x3"], dtype=float)
    extent = np.asarray(obb["extent_xyz"], dtype=float)

    # The local axis most aligned with world up is the object's vertical.
    world_up = np.array([0.0, 0.0, 1.0])
    up_axis = int(np.argmax(np.abs(rot.T @ world_up)))
    horizontal = [a for a in (0, 1, 2) if a != up_axis]
    # Facing the longest side means looking along the shorter horizontal axis.
    front_axis = min(horizontal, key=lambda a: extent[a])
    side_axis = [a for a in horizontal if a != front_axis][0]

    direction = np.asarray(cam_position, dtype=float) - center
    local = rot.T @ direction
    return float(np.degrees(np.arctan2(local[side_axis], local[front_axis])) % 360.0)


def azimuth_slot(angle_deg: float) -> str:
    if angle_deg >= 315.0 or angle_deg < 45.0:
        return "front"
    if angle_deg < 135.0:
        return "left"
    if angle_deg < 225.0:
        return "back"
    return "right"


def sharpness(image: np.ndarray, mask: np.ndarray, norm_size: int = 256) -> float:
    """How in-focus the object is, independent of how far away it was shot.

    Laplacian variance measured on the raw pixels scales with detail-per-pixel,
    so a small distant object scores many times higher than a clean close-up of
    the same thing. Resampling the object's own crop to a fixed size first
    removes that bias and leaves an actual focus measure.
    """
    if not mask.any():
        return 0.0
    ys, xs = np.where(mask)
    crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop_mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if crop.size == 0:
        return 0.0

    crop = cv2.resize(crop, (norm_size, norm_size), interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(crop_mask.astype(np.uint8), (norm_size, norm_size),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    if not crop_mask.any():
        return 0.0

    lap = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    return float(lap[crop_mask].var())


def drop_specks(mask: np.ndarray, keep_ratio: float = 0.10) -> np.ndarray:
    """Keep the object's body and its substantial parts, drop floating debris.

    Segmentation leaves small disconnected blobs — a scrap of the neighbouring
    shelf, a highlight on the floor — which survive every geometric test because
    they sit inside the projected OBB. On a transparent canvas they read to
    Hunyuan3D as detached geometry. Components below a fraction of the largest
    one go; a real part (a ladder rung, a table leg) is far above the threshold.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 2:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    floor = keep_ratio * areas.max()
    keep = {i + 1 for i, area in enumerate(areas) if area >= floor}
    return np.isin(labels, list(keep))


def subtract_occluders(
    mask: np.ndarray,
    rgb: np.ndarray,
    occluder_masks: Sequence[np.ndarray],
    dilate_px: int = 3,
    max_removed_ratio: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove instances standing on the object and paint over the hole they leave.

    Leaving the hole empty would read to Hunyuan3D as a real gap in the surface,
    so the removed region is inpainted from its surroundings instead.
    """
    if not occluder_masks:
        return mask, rgb

    kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
    removed = np.zeros_like(mask)
    for occ in occluder_masks:
        removed |= cv2.dilate(occ.astype(np.uint8), kernel).astype(bool)

    overlap = np.logical_and(mask, removed)
    if not overlap.any():
        return mask, rgb

    # A neighbour resting on the object covers a corner of it, never most of it.
    # If the subtraction would eat the object, the instance data is wrong and it
    # is safer to keep the mask whole than to hand Hunyuan3D a fragment.
    if overlap.sum() > max_removed_ratio * mask.sum():
        return mask, rgb

    filled = cv2.inpaint(rgb, overlap.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    return np.logical_and(mask, ~removed), filled


def build_hunyuan_input(
    rgb_path: Path,
    mask: np.ndarray,
    target_size: int = 518,
    pad_ratio: float = 0.10,
) -> Optional[Dict]:
    """Crop the object out on its own alpha channel and centre it on a square canvas.

    The image is written RGBA with alpha = mask. Hunyuan3D crops and centres on
    alpha; a fully opaque white-background PNG (what this stage produced before)
    makes the model treat the entire 518x518 square as the object.
    """
    rgb = cv2.imread(str(rgb_path))
    if rgb is None or not mask.any():
        return None

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    crop_rgb = rgb[y0:y1 + 1, x0:x1 + 1]
    crop_alpha = mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8) * 255
    # Keep white under the alpha so the file still looks right in an image viewer.
    crop_rgb = np.where(crop_alpha[..., None] > 0, crop_rgb, 255).astype(np.uint8)

    h, w = crop_rgb.shape[:2]
    side = max(h, w) + 2 * int(max(h, w) * pad_ratio)
    canvas = np.full((side, side, 4), (255, 255, 255, 0), dtype=np.uint8)
    off_y, off_x = (side - h) // 2, (side - w) // 2
    canvas[off_y:off_y + h, off_x:off_x + w, :3] = crop_rgb
    canvas[off_y:off_y + h, off_x:off_x + w, 3] = crop_alpha

    resized = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return {
        "hunyuan_input": resized,
        "original_crop": rgb[y0:y1 + 1, x0:x1 + 1],
        "meta": {
            "target_size": [target_size, target_size],
            "crop_xyxy": [x0, y0, x1, y1],
            "box_hw": [h, w],
            "canvas_side": side,
            "padding_offsets_yx": [off_y, off_x],
            "mode": "obb_anchored_sam3_rgba",
        },
    }


def resolve_view(obj_dir: Path, view_id: str) -> Optional[Path]:
    """views/ may hold PNG (local) or JPEG (round-tripped through the GPU kernel)."""
    for ext in (".png", ".jpg", ".jpeg", ".JPG"):
        candidate = obj_dir / "views" / f"{view_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_mask(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if raw is None else raw > 127


def occluders_for_view(
    obj_id: str,
    obb: Dict,
    all_objects: Dict[str, Dict],
    c1_dir: Path,
    resegment_dir: Optional[Path],
    view_id: str,
) -> List[np.ndarray]:
    """Masks of instances that intersect this object but are not parts of it.

    An instance whose OBB lies wholly inside the target's is a component of it
    (the vice bolted to the bench, a drawer handle) and stays. Anything else that
    overlaps is standing on or in front of the object and has to come out.
    """
    out: List[np.ndarray] = []
    for other_id, other in all_objects.items():
        if other_id == obj_id:
            continue
        if obb_contains(obb, obb_corners(other["obb"])):
            continue
        # Overlapping in the image is not enough: the machine behind the bench
        # shares pixels with it but is not on it. Only instances that share 3D
        # space with the target may be cut out of its mask.
        if not obb_intersects(obb, other["obb"]):
            continue
        for candidate in (
            (resegment_dir / other_id / "masks_v2" / f"{view_id}.png") if resegment_dir else None,
            c1_dir / other_id / "masks" / f"{view_id}.png",
        ):
            if candidate is None:
                continue
            mask = load_mask(candidate)
            if mask is not None and mask.any():
                out.append(mask)
                break
    return out


def emit_adapter_a(
    obj_id: str,
    sclass: str,
    obj_data: Dict,
    obj_dir: Path,
    out_obj_dir: Path,
    all_objects: Dict[str, Dict],
    c1_dir: Path,
    resegment_dir: Optional[Path],
    resegment: Optional[Dict],
    coverage: float,
    decision: str,
    target_size: int = 518,
    fallback_leak_max: float = 0.15,
    fallback_frame_max: float = 0.60,
) -> Optional[Dict]:
    """Turn one object's re-segmented views into Hunyuan3D inputs.

    Picks the best single view and fills the four multi-view slots by the object's
    own azimuth, writing RGBA crops with alpha = mask.
    """
    obb = obj_data["obb"]
    report_views = (resegment or {}).get("views", {})

    # Only views that survived the OBB gate are eligible; without a report at all
    # we fall back to every view C1 recorded.
    if report_views:
        eligible = [v for v, d in report_views.items() if d.get("view_ok")]
        if not eligible:
            # Nothing clean. Keep the least-leaky view so the object survives, but
            # never a degenerate one: a mask covering most of the frame is the
            # background, and shipping it would be worse than the mask C1 already
            # had - which the whole scoring pass exists to rule out.
            def usable_fallback(d: Dict) -> bool:
                if not d.get("ok") or d.get("leak", 1.0) > fallback_leak_max:
                    return False
                # hull_area / hull_frame_ratio recovers the frame area without
                # having to open the image.
                ratio = d.get("hull_frame_ratio")
                if not ratio:
                    return True
                frame_area = d.get("hull_area", 0) / ratio
                return d.get("area", 0) <= fallback_frame_max * frame_area

            fallback = sorted(
                (d["leak"], v) for v, d in report_views.items() if usable_fallback(d)
            )
            eligible = [fallback[0][1]] if fallback else []
    else:
        eligible = list(obj_data.get("source_views", []))
    if not eligible:
        return None

    prepared: Dict[str, Dict] = {}
    for view_id in eligible:
        rgb_path = resolve_view(obj_dir, view_id)
        cam_path = obj_dir / "cameras" / f"{view_id}.json"
        if rgb_path is None or not cam_path.is_file():
            continue

        mask = None
        if resegment_dir is not None:
            mask = load_mask(resegment_dir / obj_id / "masks_v2" / f"{view_id}.png")
        if mask is None:
            mask = load_mask(obj_dir / "masks" / f"{view_id}.png")
        if mask is None or not mask.any():
            continue

        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            continue
        mask, rgb = subtract_occluders(
            mask, rgb,
            occluders_for_view(obj_id, obb, all_objects, c1_dir, resegment_dir, view_id),
        )
        mask = drop_specks(mask)
        if not mask.any():
            continue

        with open(cam_path, "r", encoding="utf-8") as cf:
            cam = json.load(cf)
        angle = obb_local_azimuth(obb, cam.get("position", [0, 0, 0]))
        area = int(mask.sum())
        fill = float(report_views.get(view_id, {}).get("fill", 0.0)) or 1.0

        prepared[view_id] = {
            "mask": mask,
            "rgb": rgb,
            "rgb_path": rgb_path,
            "angle_deg": round(angle, 2),
            "slot": azimuth_slot(angle),
            "area": area,
            "fill": fill,
            "sharpness": sharpness(rgb, mask),
        }

    if not prepared:
        return None

    # Pixels on the object are what give Hunyuan3D detail, so area x OBB coverage
    # decides the ranking. Laplacian variance cannot be a factor in it: a distant
    # object packs high-frequency detail into few pixels and scores far higher
    # than a clean close-up of a smooth surface. It is only used to throw out
    # views that are genuinely out of focus relative to this object's own views.
    focus_floor = 0.25 * float(np.median([d["sharpness"] for d in prepared.values()]))
    for data in prepared.values():
        in_focus = data["sharpness"] >= focus_floor
        data["score"] = data["fill"] * data["area"] * (1.0 if in_focus else 0.05)

    out_obj_dir.mkdir(parents=True, exist_ok=True)
    mv_dir = out_obj_dir / "mv_inputs"
    mv_dir.mkdir(parents=True, exist_ok=True)

    def write_rgba(view_id: str, dest: Path) -> Optional[Dict]:
        data = prepared[view_id]
        tmp = out_obj_dir / "_tmp_view.png"
        cv2.imwrite(str(tmp), data["rgb"])
        built = build_hunyuan_input(tmp, data["mask"], target_size=target_size)
        tmp.unlink(missing_ok=True)
        if built is None:
            return None
        cv2.imwrite(str(dest), built["hunyuan_input"])
        return built

    best_view = max(prepared, key=lambda v: prepared[v]["score"])
    best_built = write_rgba(best_view, out_obj_dir / "hunyuan_input.png")
    if best_built is None:
        return None
    cv2.imwrite(str(out_obj_dir / "original_crop.png"), best_built["original_crop"])

    slots: Dict[str, Dict] = {}
    for slot_name in ("front", "left", "back", "right"):
        members = [v for v, d in prepared.items() if d["slot"] == slot_name]
        if not members:
            continue
        pick = max(members, key=lambda v: prepared[v]["score"])
        if write_rgba(pick, mv_dir / f"{slot_name}.png") is None:
            continue
        slots[slot_name] = {
            "view_id": pick,
            "angle_deg": prepared[pick]["angle_deg"],
            "file": f"mv_inputs/{slot_name}.png",
            "pixel_area": prepared[pick]["area"],
        }

    # 2mv runs happily on two or three slots; a single view has no multi-view
    # information to give it, so that object goes to single-view Hunyuan3D-2.
    mode = "multiview" if len(slots) >= 2 else "singleview"

    meta = {
        "instance_id": obj_id,
        "semantic_class": sclass,
        "source_frame": best_view,
        "coverage": coverage,
        "decision": decision,
        "mode": mode,
        "meta": best_built["meta"],
        "eligible_views": sorted(prepared),
        "canonical_multiview_slots": slots,
        "num_canonical_views": len(slots),
        "azimuth_frame": "obb_local",
    }
    with open(out_obj_dir / "adapter_a_meta.json", "w", encoding="utf-8") as af:
        json.dump(meta, af, indent=2)

    return {
        "instance_id": obj_id,
        "semantic_class": sclass,
        "hunyuan_input_path": str(out_obj_dir / "hunyuan_input.png"),
        "mv_inputs_dir": str(mv_dir),
        "available_mv_slots": list(slots),
        "mode": mode,
        "decision": decision,
        "coverage": coverage,
        "best_view": best_view,
    }


def run_quality_gate_and_adapter_a(
    c1_dir: Path,
    output_dir: Path,
    small_classes: Set[str],
    coverage_low_thresh: float = 0.50,
    coverage_high_thresh: float = 0.75,
    include_review_optional: bool = True,
    resegment_dir: Optional[Path] = None,
):
    """
    Executes Quality Gate evaluation and generates Adapter A 518x518 inputs for Hunyuan3D.
    """
    manifest_file = c1_dir / "c1_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"c1_manifest.json not found in {c1_dir}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_a_dir = output_dir / "adapter_a"
    adapter_a_dir.mkdir(parents=True, exist_ok=True)

    # Masks come from the OBB-anchored SAM 3 pass (stage 03) when it has been run.
    # Without it this stage falls back to C1's original masks; rembg is gone either
    # way - unioning it in is what used to pull neighbouring objects and floor into
    # the crop.
    resegment_report: Dict[str, Dict] = {}
    if resegment_dir is not None:
        report_path = resegment_dir / "resegment_report.json"
        if report_path.is_file():
            with open(report_path, "r", encoding="utf-8") as f:
                resegment_report = json.load(f)
            print(f"Using OBB-anchored masks for {len(resegment_report)} objects from {resegment_dir}")
        else:
            print(f"WARNING: no resegment_report.json under {resegment_dir}; falling back to C1 masks")
            resegment_dir = None

    # Every object's OBB, needed to tell a component of an object apart from
    # something merely standing on it.
    all_objects: Dict[str, Dict] = {}
    for obj_json in sorted(c1_dir.glob("*/object.json")):
        with open(obj_json, "r", encoding="utf-8") as f:
            all_objects[obj_json.parent.name] = json.load(f)

    objects_list = manifest.get("objects", [])
    print(f"Starting Quality Gate analysis on {len(objects_list)} objects...")

    csv_rows = []
    complete_objects = []
    incomplete_objects = []
    skipped_objects = []
    adapter_a_manifest = []

    for item in objects_list:
        obj_id = item["instance_id"]
        sclass = item["semantic_class"].strip().lower()
        obj_dir = c1_dir / obj_id
        obj_json_path = obj_dir / "object.json"

        if not obj_json_path.is_file():
            continue

        with open(obj_json_path, "r", encoding="utf-8") as f:
            obj_data = json.load(f)

        center = obj_data.get("obb", {}).get("center_world", [0, 0, 0])
        extent = obj_data.get("obb", {}).get("extent_xyz", [0, 0, 0])
        source_views = obj_data.get("source_views", [])
        best_frame = obj_data.get("best_frame_id", "UNKNOWN")

        # Compute camera coverage
        coverage, obs_bins, angles = compute_azimuth_coverage(
            object_center=center,
            source_views=source_views,
            cameras_dir=obj_dir / "cameras",
            num_bins=8,
        )

        # Quality Gate decision logic
        if sclass in small_classes:
            decision = "SKIP_SMALL_OBJECT"
            reason = f"Class '{sclass}' is 2D planar detail or small attachment"
        elif coverage < coverage_low_thresh:
            decision = "COMPLETE_WITH_HUNYUAN"
            reason = f"Coverage ({coverage:.3f}) < {coverage_low_thresh} (missing backside/views)"
        elif coverage < coverage_high_thresh:
            decision = "REVIEW_OPTIONAL"
            reason = f"Coverage ({coverage:.3f}) in [{coverage_low_thresh}, {coverage_high_thresh})"
        else:
            decision = "KEEP_HOLI"
            reason = f"Coverage ({coverage:.3f}) >= {coverage_high_thresh} (well observed)"

        entry = {
            "instance_id": obj_id,
            "semantic_class": sclass,
            "num_views": len(source_views),
            "observed_bins_count": len(obs_bins),
            "observed_bins": sorted(list(obs_bins)),
            "coverage": round(coverage, 4),
            "decision": decision,
            "reason": reason,
            "best_frame_id": best_frame,
            "center_world": center,
            "extent_xyz": extent,
        }

        csv_rows.append({
            "instance_id": obj_id,
            "semantic_class": sclass,
            "num_views": len(source_views),
            "observed_bins": len(obs_bins),
            "coverage": f"{coverage:.4f}",
            "decision": decision,
            "best_frame_id": best_frame,
            "reason": reason,
        })

        if decision == "SKIP_SMALL_OBJECT":
            skipped_objects.append(entry)
        elif decision == "KEEP_HOLI":
            complete_objects.append(entry)
        else:
            # COMPLETE_WITH_HUNYUAN or REVIEW_OPTIONAL
            incomplete_objects.append(entry)

            # Process Adapter A if selected for completion
            if decision == "COMPLETE_WITH_HUNYUAN" or (decision == "REVIEW_OPTIONAL" and include_review_optional):
                emitted = emit_adapter_a(
                    obj_id=obj_id,
                    sclass=sclass,
                    obj_data=obj_data,
                    obj_dir=obj_dir,
                    out_obj_dir=adapter_a_dir / obj_id,
                    all_objects=all_objects,
                    c1_dir=c1_dir,
                    resegment_dir=resegment_dir,
                    resegment=resegment_report.get(obj_id),
                    coverage=coverage,
                    decision=decision,
                )
                if emitted is not None:
                    adapter_a_manifest.append(emitted)
                    print(f"  {obj_id:22s} {emitted['mode']:11s} "
                          f"best={emitted['best_view']} slots={emitted['available_mv_slots']}")
                else:
                    print(f"  {obj_id:22s} SKIPPED (no usable view survived the OBB gate)")

    # Write CSV report
    csv_path = output_dir / "coverage_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["instance_id", "semantic_class", "num_views", "observed_bins", "coverage", "decision", "best_frame_id", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Write JSON categorizations
    with open(output_dir / "complete_objects.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(complete_objects), "objects": complete_objects}, f, indent=2)

    with open(output_dir / "incomplete_objects.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(incomplete_objects), "objects": incomplete_objects}, f, indent=2)

    with open(output_dir / "skipped_small_objects.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(skipped_objects), "objects": skipped_objects}, f, indent=2)

    # Write Adapter A Manifest
    with open(adapter_a_dir / "adapter_a_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_prepared_for_hunyuan": len(adapter_a_manifest),
            "objects": adapter_a_manifest,
        }, f, indent=2)

    print("\n=======================================================")
    print("QUALITY GATE & ADAPTER A SUMMARY:")
    print(f"  - Total objects evaluated: {len(objects_list)}")
    print(f"  - Skipped small/2D objects: {len(skipped_objects)}")
    print(f"  - Complete (KEEP_HOLI): {len(complete_objects)}")
    print(f"  - Incomplete (candidates for Hunyuan3D): {len(incomplete_objects)}")
    print(f"  - Adapter A inputs successfully generated: {len(adapter_a_manifest)}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Coverage report: {csv_path}")
    print(f"  - Adapter A inputs: {adapter_a_dir}")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Step D & E: Quality Gate & Adapter A Preparation")
    parser.add_argument(
        "--c1-dir",
        type=Path,
        default=Path("outputs-full-factory/delivery/OUTPUT_C1"),
        help="Path to Step C1 normalized packages directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs-full-factory/delivery/OUTPUT_D"),
        help="Path to save Quality Gate reports and Adapter A outputs",
    )
    parser.add_argument(
        "--low-thresh",
        type=float,
        default=0.50,
        help="Low coverage threshold (default: 0.50)",
    )
    parser.add_argument(
        "--high-thresh",
        type=float,
        default=0.75,
        help="High coverage threshold (default: 0.75)",
    )
    parser.add_argument(
        "--resegment-dir",
        type=Path,
        default=Path("outputs-full-factory/delivery/OUTPUT_C3"),
        help="Directory of OBB-anchored SAM 3 masks from 03_resegment_obb_sam3.py",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip REVIEW_OPTIONAL objects from Adapter A image generation",
    )

    args = parser.parse_args()

    run_quality_gate_and_adapter_a(
        c1_dir=args.c1_dir,
        output_dir=args.output_dir,
        small_classes=DEFAULT_SMALL_CLASSES,
        coverage_low_thresh=args.low_thresh,
        coverage_high_thresh=args.high_thresh,
        include_review_optional=not args.skip_optional,
        resegment_dir=args.resegment_dir if args.resegment_dir.is_dir() else None,
    )


if __name__ == "__main__":
    main()
