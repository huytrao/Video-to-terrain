#!/usr/bin/env python3
"""
Step C3: OBB-anchored re-segmentation with SAM 3 (runs on GPU, e.g. Kaggle RTX PRO 6000).

Replaces the text-only prompting that Holi-Spatial uses (`sam3.py` calls
`set_text_prompt(label)` and nothing else) with geometry-grounded prompting:
the metric 3D OBB already stored in `object.json` is projected into every view
and handed to SAM 3 as a positive box, while neighbouring instances that sit on
top of / in front of the target become negative boxes.

For every view of every object this writes a binary mask to `masks_v2/` plus the
metrics the downstream stages gate on:

    fill = |mask ∩ hull| / |hull|      how much of the projected OBB is covered
    leak = |mask \\ hull| / |mask|      how much of the mask escapes the OBB

`leak` is the hard gate: the OBB is the physical boundary of the object, so a
pixel outside it is wrong by definition. `fill` only ranks candidates, because
hollow objects (shelves, ladders, hoses) legitimately fill very little of their
bounding box.

Usage (on the Kaggle kernel):

    python 03_resegment_obb_sam3.py \
        --c1-dir /kaggle/working/C1 \
        --output-dir /kaggle/working/out_c3 \
        --objects obj_0219_table obj_0208_shelf obj_0033_hose obj_0181_box
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Kaggle layout: the SAM 3.1 source tree and checkpoint ship as an attached dataset.
DEFAULT_SAM3_SRC = "/kaggle/input/datasets/traoanhuy/v3d-sam31-multiplex/src"
DEFAULT_CKPT = "/kaggle/input/datasets/traoanhuy/v3d-sam31-multiplex/ckpt/sam3.1_multiplex.pt"
DEFAULT_BPE = "/kaggle/working/assets/bpe_simple_vocab_16e6.txt.gz"

# A view is only usable as a Hunyuan3D input if the whole OBB is comfortably
# inside the frame; anything closer than this to an edge is a truncated object.
BORDER_MARGIN_PX = 16

# `leak` only means something while the projected OBB is small relative to the
# frame. Shot from close up, a large object's OBB covers the whole image, every
# mask is trivially "inside" it, and the gate stops discriminating - so such a
# view carries no geometric constraint and cannot be used.
MAX_HULL_FRAME_RATIO = 0.60


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def obb_corners(obb: Dict) -> np.ndarray:
    """The 8 corners of an OBB in world coordinates, derived from centre/extent/rotation."""
    center = np.asarray(obb["center_world"], dtype=float)
    extent = np.asarray(obb["extent_xyz"], dtype=float)
    rot = np.asarray(obb["rotation_world_3x3"], dtype=float)
    signs = np.array([[sx, sy, sz]
                      for sx in (-1.0, 1.0)
                      for sy in (-1.0, 1.0)
                      for sz in (-1.0, 1.0)])
    return center + (rot @ (signs * extent / 2.0).T).T


def project(points_world: np.ndarray, cam: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Project world points into a camera. Returns (uv, z) with z the camera-space depth.

    `rotation` in the Holi camera JSONs is camera-to-world, so the world-to-camera
    map is its transpose: X_cam = R.T @ (X_world - C).
    """
    cam_pos = np.asarray(cam["position"], dtype=float)
    rot = np.asarray(cam["rotation"], dtype=float)
    cam_pts = (rot.T @ (points_world - cam_pos).T).T
    z = cam_pts[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cam["fx"] * cam_pts[:, 0] / safe + cam["width"] / 2.0
    v = cam["fy"] * cam_pts[:, 1] / safe + cam["height"] / 2.0
    return np.stack([u, v], axis=1), z


def hull_mask(uv: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterise the convex hull of projected points into a boolean image mask."""
    import cv2

    pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(canvas, cv2.convexHull(pts), 1)
    return canvas.astype(bool)


def obb_contains(host: Dict, corners: np.ndarray, tol: float = 0.02) -> bool:
    """True when every corner of another OBB lies inside `host` (it is a part, not an occluder)."""
    center = np.asarray(host["center_world"], dtype=float)
    rot = np.asarray(host["rotation_world_3x3"], dtype=float)
    extent = np.asarray(host["extent_xyz"], dtype=float)
    local = (rot.T @ (corners - center).T).T
    return bool(np.all(np.abs(local) <= extent / 2.0 + tol))


def obb_overlaps(host: Dict, corners: np.ndarray, tol: float = 0.0) -> bool:
    """True when any corner of another OBB falls inside `host`."""
    center = np.asarray(host["center_world"], dtype=float)
    rot = np.asarray(host["rotation_world_3x3"], dtype=float)
    extent = np.asarray(host["extent_xyz"], dtype=float)
    local = (rot.T @ (corners - center).T).T
    return bool(np.any(np.all(np.abs(local) <= extent / 2.0 + tol, axis=1)))


def json_default(value):
    """numpy scalars leak in through the projection maths; json cannot encode them."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def to_cxcywh_norm(box_xyxy: Sequence[float], width: int, height: int) -> List[float]:
    """SAM 3 wants geometric prompts as normalised centre-x/centre-y/width/height."""
    x0, y0, x1, y1 = box_xyxy
    return [((x0 + x1) / 2.0) / width,
            ((y0 + y1) / 2.0) / height,
            (x1 - x0) / width,
            (y1 - y0) / height]


def clip_box(box_xyxy: Sequence[float], width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = box_xyxy
    return [max(0.0, x0), max(0.0, y0), min(width - 1.0, x1), min(height - 1.0, y1)]


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def score_mask(mask: np.ndarray, hull: np.ndarray) -> Dict:
    """fill/leak plus the connectivity and size numbers the gate reports."""
    import cv2

    area = int(mask.sum())
    hull_area = int(hull.sum())
    if area == 0 or hull_area == 0:
        return {"area": area, "hull_area": hull_area, "fill": 0.0, "leak": 1.0, "components": 0}

    inside = int(np.logical_and(mask, hull).sum())
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    # Ignore specks below 1% of the mask when counting components.
    significant = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 0.01 * area)

    return {
        "area": area,
        "hull_area": hull_area,
        "fill": round(inside / hull_area, 4),
        "leak": round((area - inside) / area, 4),
        "components": significant,
    }


def touches_border(mask: np.ndarray, margin: int = BORDER_MARGIN_PX) -> bool:
    if not mask.any():
        return True
    ys, xs = np.where(mask)
    h, w = mask.shape
    return bool(ys.min() < margin or xs.min() < margin
                or ys.max() >= h - margin or xs.max() >= w - margin)


def rank_key(metrics: Dict, leak_max: float) -> Tuple:
    """Sort candidates: inside the leak budget first, then by coverage of the OBB.

    Candidates that blow the budget must be ranked by how badly they leak, never
    by fill. A mask that swallows the whole frame trivially covers 100% of the
    OBB, so ranking failures on fill hands the win to the worst possible mask.
    """
    if metrics["leak"] <= leak_max:
        return (1, metrics["fill"], -metrics["leak"])
    return (0, -metrics["leak"], metrics["fill"])


# --------------------------------------------------------------------------- #
# SAM 3
# --------------------------------------------------------------------------- #

class Segmenter:
    """Thin wrapper over Sam3Processor exposing the box-prompt path Holi never used."""

    def __init__(self, src: str, ckpt: str, bpe: str, device: str = "cuda") -> None:
        if src not in sys.path:
            sys.path.insert(0, src)
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # SAM 3 runs its vision tower in bf16; without this the first Linear
        # raises "mat1 and mat2 must have the same dtype".
        torch.autocast(device, dtype=torch.bfloat16).__enter__()

        started = time.time()
        self.model = build_sam3_image_model(
            bpe_path=bpe, checkpoint_path=ckpt, load_from_HF=False, device=device
        )
        self.processor = Sam3Processor(self.model)
        print(f"[sam3] model ready in {time.time() - started:.1f}s", flush=True)

    def set_image(self, image) -> Dict:
        return self.processor.set_image(image)

    def run(
        self,
        base_state: Dict,
        text: Optional[str],
        positive_boxes: Sequence[Sequence[float]],
        negative_boxes: Sequence[Sequence[float]] = (),
    ) -> List[np.ndarray]:
        """One prompt combination -> the masks SAM 3 returns for it."""
        state = dict(base_state)
        if text:
            state = self.processor.set_text_prompt(text, state)
        for box in positive_boxes:
            state = self.processor.add_geometric_prompt(list(box), True, state)
        for box in negative_boxes:
            state = self.processor.add_geometric_prompt(list(box), False, state)

        masks = state.get("masks")
        if masks is None or len(masks) == 0:
            return []
        return list(masks.squeeze(1).cpu().numpy().astype(bool))


# --------------------------------------------------------------------------- #
# per-view work
# --------------------------------------------------------------------------- #

def neighbour_boxes(
    target_id: str,
    target_obb: Dict,
    others: Dict[str, Dict],
    cam: Dict,
    width: int,
    height: int,
    hull: np.ndarray,
) -> List[Dict]:
    """Projected boxes of instances that overlap the target but are not parts of it.

    An instance whose OBB sits entirely inside the target's OBB is a part
    (a vice bolted to the bench, a drawer handle) and is kept. Anything that
    merely intersects — a box standing on the table — becomes a negative prompt.
    """
    out: List[Dict] = []
    for other_id, other in others.items():
        if other_id == target_id:
            continue
        corners = obb_corners(other["obb"])
        if obb_contains(target_obb, corners):
            continue  # a part of the object, leave it in
        if not obb_overlaps(target_obb, corners):
            continue  # unrelated furniture elsewhere in the room

        uv, z = project(corners, cam)
        if not np.all(z > 1e-6):
            continue
        box = clip_box([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], width, height)
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue
        # Only worth prompting against if it actually lands on the target.
        patch = np.zeros((height, width), dtype=bool)
        patch[int(box[1]):int(box[3]) + 1, int(box[0]):int(box[2]) + 1] = True
        if not np.logical_and(patch, hull).any():
            continue
        out.append({"instance_id": other_id, "box_xyxy": [round(v, 1) for v in box]})
    return out


def segment_view(
    segmenter: Segmenter,
    image,
    old_mask: Optional[np.ndarray],
    obb: Dict,
    cam: Dict,
    semantic_class: str,
    negatives: Sequence[Dict],
    leak_max: float,
) -> Dict:
    """Try several prompt combinations for one view and keep the best-scoring mask."""
    width, height = image.size
    corners = obb_corners(obb)
    uv, z = project(corners, cam)
    all_in_front = bool(np.all(z > 1e-6))

    if not all_in_front:
        return {"ok": False, "reason": "obb_behind_camera"}

    hull = hull_mask(uv, height, width)
    hull_ratio = float(hull.sum()) / float(height * width)
    if hull_ratio > MAX_HULL_FRAME_RATIO:
        return {"ok": False, "reason": f"obb_fills_frame({hull_ratio:.2f})"}

    raw_box = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
    obb_inside_frame = bool(raw_box[0] >= BORDER_MARGIN_PX and raw_box[1] >= BORDER_MARGIN_PX
                            and raw_box[2] <= width - BORDER_MARGIN_PX
                            and raw_box[3] <= height - BORDER_MARGIN_PX)
    box = clip_box(raw_box, width, height)
    pos = [to_cxcywh_norm(box, width, height)]
    neg = [to_cxcywh_norm(n["box_xyxy"], width, height) for n in negatives]

    base_state = segmenter.set_image(image)
    trials = [
        ("obb_box", None, pos, []),
        ("obb_box_neg", None, pos, neg) if neg else None,
        ("text_obb_box", semantic_class, pos, neg),
        ("text_only", semantic_class, [], []),
    ]

    candidates: List[Dict] = []
    for trial in trials:
        if trial is None:
            continue
        name, text, positives, negatives_norm = trial
        try:
            masks = segmenter.run(base_state, text, positives, negatives_norm)
        except Exception as exc:  # a failed prompt must not kill the whole run
            print(f"    ! {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        for idx, mask in enumerate(masks):
            metrics = score_mask(mask, hull)
            candidates.append({"variant": f"{name}#{idx}", "mask": mask, "metrics": metrics})

    # The mask Holi already had competes on exactly the same terms, so this stage
    # can never make an object worse than it is today.
    if old_mask is not None and old_mask.any():
        candidates.append({"variant": "holi_original", "mask": old_mask,
                           "metrics": score_mask(old_mask, hull)})

    if not candidates:
        return {"ok": False, "reason": "no_candidate"}

    candidates.sort(key=lambda c: rank_key(c["metrics"], leak_max), reverse=True)
    best = candidates[0]

    return {
        "ok": True,
        "mask": best["mask"],
        "hull": hull,
        "variant": best["variant"],
        "metrics": best["metrics"],
        "obb_inside_frame": obb_inside_frame,
        "hull_frame_ratio": round(hull_ratio, 4),
        "obb_box_xyxy": [round(v, 1) for v in raw_box],
        "negatives": [n["instance_id"] for n in negatives],
        "all_variants": {c["variant"]: c["metrics"] for c in candidates},
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def run(
    c1_dir: Path,
    output_dir: Path,
    only: Optional[Sequence[str]],
    leak_max: float,
    src: str,
    ckpt: str,
    bpe: str,
) -> Dict:
    import cv2
    from PIL import Image

    objects: Dict[str, Dict] = {}
    for obj_json in sorted(c1_dir.glob("*/object.json")):
        data = json.loads(obj_json.read_text(encoding="utf-8"))
        objects[obj_json.parent.name] = data
    if not objects:
        raise SystemExit(f"no object.json found under {c1_dir}")
    print(f"[c1] {len(objects)} object packages, {sum(1 for _ in c1_dir.glob('*/views'))} with views",
          flush=True)

    targets = [o for o in objects if (only is None or o in only)]
    if only:
        missing = sorted(set(only) - set(targets))
        if missing:
            print(f"[warn] not in C1, skipped: {missing}", flush=True)
    print(f"[plan] re-segmenting {len(targets)} objects", flush=True)

    segmenter = Segmenter(src=src, ckpt=ckpt, bpe=bpe)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Dict] = {}

    for n, obj_id in enumerate(targets, 1):
        obj = objects[obj_id]
        obj_dir = c1_dir / obj_id
        obb = obj["obb"]
        semantic_class = obj.get("semantic_class", "object")
        views = obj.get("source_views", [])
        out_masks = output_dir / obj_id / "masks_v2"
        out_masks.mkdir(parents=True, exist_ok=True)

        print(f"\n[{n}/{len(targets)}] {obj_id}  class={semantic_class}  views={len(views)}", flush=True)
        per_view: Dict[str, Dict] = {}

        for view_id in views:
            # views/ ships as PNG locally but is re-encoded to JPEG before upload
            # to keep the transfer to the offline kernel under ~100 MB.
            rgb_path = next((p for p in (obj_dir / "views" / f"{view_id}{ext}"
                                         for ext in (".png", ".jpg", ".jpeg", ".JPG"))
                             if p.is_file()), None)
            cam_path = obj_dir / "cameras" / f"{view_id}.json"
            if rgb_path is None or not cam_path.is_file():
                continue
            cam = json.loads(cam_path.read_text(encoding="utf-8"))
            image = Image.open(rgb_path).convert("RGB")
            width, height = image.size

            old_path = obj_dir / "masks" / f"{view_id}.png"
            old_mask = None
            if old_path.is_file():
                raw = cv2.imread(str(old_path), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    old_mask = raw > 127

            corners = obb_corners(obb)
            uv, z = project(corners, cam)
            hull = hull_mask(uv, height, width) if np.all(z > 1e-6) else np.zeros((height, width), bool)
            negatives = neighbour_boxes(obj_id, obb, objects, cam, width, height, hull)

            result = segment_view(segmenter, image, old_mask, obb, cam,
                                  semantic_class, negatives, leak_max)
            if not result["ok"]:
                per_view[view_id] = {"ok": False, "reason": result["reason"]}
                print(f"    {view_id}  SKIP ({result['reason']})", flush=True)
                continue

            mask = result["mask"]
            cv2.imwrite(str(out_masks / f"{view_id}.png"), (mask.astype(np.uint8) * 255))

            metrics = result["metrics"]
            entry = {
                "ok": True,
                "variant": result["variant"],
                "obb_inside_frame": result["obb_inside_frame"],
                "hull_frame_ratio": result["hull_frame_ratio"],
                "obb_box_xyxy": result["obb_box_xyxy"],
                "negatives": result["negatives"],
                "touches_border": touches_border(mask),
                "all_variants": result["all_variants"],
                **metrics,
            }
            if old_mask is not None and old_mask.any():
                entry["old_area"] = int(old_mask.sum())
            per_view[view_id] = entry

            flag = "" if result["obb_inside_frame"] else "  [OBB truncated]"
            print(f"    {view_id}  {result['variant']:16s} fill={metrics['fill']:.3f} "
                  f"leak={metrics['leak']:.3f} area={metrics['area']:7d} "
                  f"cc={metrics['components']}{flag}", flush=True)

        # Cross-view voting: a view whose coverage collapses relative to its peers
        # saw something other than the object and must not reach mv_inputs.
        usable = {v: d for v, d in per_view.items() if d.get("ok")}
        fills = np.array([d["fill"] for d in usable.values()], dtype=float)
        if len(fills) >= 3:
            median = float(np.median(fills))
            mad = float(np.median(np.abs(fills - median))) or 1e-6
            for view_id, data in usable.items():
                data["fill_outlier"] = bool(data["fill"] < median - 2.0 * mad)
        else:
            for data in usable.values():
                data["fill_outlier"] = False

        for view_id, data in usable.items():
            data["view_ok"] = bool(
                data["obb_inside_frame"]
                and not data["touches_border"]
                and not data["fill_outlier"]
                and data["leak"] <= leak_max
            )

        good = [v for v, d in usable.items() if d["view_ok"]]
        report[obj_id] = {
            "semantic_class": semantic_class,
            "num_views": len(views),
            "num_segmented": len(usable),
            "num_view_ok": len(good),
            "views_ok": sorted(good),
            "views": per_view,
        }
        print(f"    -> {len(good)}/{len(usable)} views pass the gate", flush=True)

    report_path = output_dir / "resegment_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    print(f"\n[done] {report_path}", flush=True)
    return report


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="OBB-anchored SAM 3 re-segmentation")
    parser.add_argument("--c1-dir", type=Path, default=Path("/kaggle/working/C1"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/out_c3"))
    parser.add_argument("--objects", nargs="*", default=None,
                        help="instance ids to process; default is every object in C1")
    parser.add_argument("--leak-max", type=float, default=0.05,
                        help="hard gate: fraction of mask allowed outside the projected OBB")
    parser.add_argument("--sam3-src", default=DEFAULT_SAM3_SRC)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--bpe", default=DEFAULT_BPE)
    args = parser.parse_args()

    run(
        c1_dir=args.c1_dir,
        output_dir=args.output_dir,
        only=args.objects,
        leak_max=args.leak_max,
        src=args.sam3_src,
        ckpt=args.ckpt,
        bpe=args.bpe,
    )


if __name__ == "__main__":
    main()
