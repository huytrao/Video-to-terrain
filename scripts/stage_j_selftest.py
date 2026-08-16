#!/usr/bin/env python3
"""Prove Stage J can fail.

Builds two scenes against one real Holi-Spatial ground-truth annotation:

  honest   objects and room shell copied from the ground truth
  buggy    the exact defects the previous pipeline shipped — walls whose
           height was the constant 2.0 m, a floor plane nobody fitted, and a
           'ceiling' box 11.9 m tall

The old validator returned PASS sixteen times on the buggy one. This asserts
the new gate separates them.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3d.data.hf_tar import RemoteTar, hf_url  # noqa: E402
from v3d.scene.schema import (  # noqa: E402
    Ceiling,
    Floor,
    OrientedBox,
    Plane,
    Room,
    SceneDocument,
    SceneObject,
    Wall,
)
from v3d.validate import stage_j  # noqa: E402
from v3d.validate.gt_boxes import load_gt, room_dims_from_gt  # noqa: E402

GT_TAR = hf_url("Holi-Spatial/ScanNetPP", "3d_bounding_box_pp/3d_bounding_box.tar")
CACHE = ROOT / "outputs" / "_selftest"


def fetch_one_gt() -> tuple[str, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    hit = next(iter(CACHE.glob("*.json")), None)
    if hit is not None:
        return hit.stem, hit

    tar = RemoteTar(GT_TAR)
    member = next(m for m in itertools.islice(tar.members(), 6) if m.name.endswith(".json"))
    scene_id = Path(member.name).stem
    dest = CACHE / f"{scene_id}.json"
    tar.extract(member, dest)
    return scene_id, dest


def build_honest(scene_id: str, gt_path: Path) -> SceneDocument:
    """A scene that agrees with the ground truth, with honest provenance."""
    gt = load_gt(gt_path)
    dims = room_dims_from_gt(gt)
    up = int(dims.get("up_axis_index", 2))
    horiz = [i for i in range(3) if i != up]

    shell = [b for b in gt if b.is_shell]
    corners = []
    for b in shell:
        half = (b.axes.T * (b.size / 2.0)).T
        for s in itertools.product((-1, 1), repeat=3):
            corners.append(b.center + s[0] * half[0] + s[1] * half[1] + s[2] * half[2])
    pts = np.array(corners)
    lo, hi = pts.min(axis=0), pts.max(axis=0)

    poly = [
        [float(lo[horiz[0]]), float(lo[horiz[1]])],
        [float(hi[horiz[0]]), float(lo[horiz[1]])],
        [float(hi[horiz[0]]), float(hi[horiz[1]])],
        [float(lo[horiz[0]]), float(hi[horiz[1]])],
    ]
    height = float(dims.get("height_m", 2.6))

    room = Room(
        floor=Floor(
            plane=Plane([0.0, 0.0, 1.0, 0.0], source="measured", inlier_count=48210, rms_residual_m=0.004),
            polygon_xy=poly,
            source="measured",
        ),
        ceiling=Ceiling(
            height_m=height,
            plane=Plane([0.0, 0.0, 1.0, -height], source="measured", inlier_count=31044),
            source="measured",
        ),
        walls=[
            Wall(
                plane=Plane([1.0, 0.0, 0.0, -poly[1][0]], source="measured", inlier_count=9021),
                footprint_xy=[poly[1], poly[2]],
                z_range_m=[0.0, height],
                source="measured",
            )
        ],
    )

    objects = []
    for b in gt:
        if b.is_shell:
            continue
        objects.append(
            SceneObject(
                id=f"obj_{b.label.replace(' ', '_')}_{b.ins_id}",
                category=b.label,
                box=OrientedBox(
                    center=[float(v) for v in b.center],
                    size=[float(v) for v in b.size],
                    rotation=[[float(v) for v in row] for row in b.axes],
                    source="measured",
                ),
                confidence=0.95,
            )
        )

    return SceneDocument(
        scene_id=scene_id,
        room=room,
        objects=objects,
        provenance={"pipeline": "selftest-honest"},
    )


def build_buggy(honest: SceneDocument) -> SceneDocument:
    """Reintroduce the three defects the previous run actually shipped."""
    doc = SceneDocument.from_dict(honest.to_dict())
    doc.provenance = {"pipeline": "selftest-buggy"}

    # 1. wall height was the literal 2.0, and the floor plane was never fitted
    for w in doc.room.walls:
        w.z_range_m = [0.0, 2.0]
        w.source = "assumed"
        w.plane.source = "assumed"
    doc.room.floor.plane.source = "assumed"
    doc.room.ceiling.height_m = 8.588  # the extent the old run reported as a room

    # 2. a 'ceiling' instance bigger than the entire scene
    doc.objects.append(
        SceneObject(
            id="obj_ceiling_0",
            category="ceiling",
            box=OrientedBox(
                center=[5.91, 0.95, 0.92],
                size=[16.380, 8.394, 11.902],
                rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                source="derived",
            ),
            confidence=0.8,
        )
    )

    # 3. object extents inflated by a bad metric scale
    for o in doc.objects[:10]:
        o.box.size = [s * 1.35 for s in o.box.size]

    return doc


def main() -> int:
    scene_id, gt_path = fetch_one_gt()
    print(f"ground truth: {scene_id}  ({gt_path.stat().st_size/1e6:.1f} MB)\n")

    out = CACHE
    honest = build_honest(scene_id, gt_path)
    honest.write(out / "scene_honest.json")
    buggy = build_buggy(honest)
    buggy.write(out / "scene_buggy.json")

    results = {}
    for name in ("honest", "buggy"):
        print(f"--- {name} ---")
        rep = stage_j.run(out / f"scene_{name}.json", gt_path)
        width = max(len(c.id) for c in rep.checks)
        for c in rep.checks:
            print(f"  [{c.status}] {c.id.ljust(width)}  {c.detail}")
        if rep.recall is not None:
            print(f"  recall {rep.recall*100:.1f}%")
        print(f"  => structural={rep.structural_verdict} metric={rep.metric_verdict} OVERALL={rep.overall}\n")
        results[name] = rep.overall

    ok = results["honest"] == "PASS" and results["buggy"] == "FAIL"
    print("SELFTEST:", "PASS — the gate separates good from bad" if ok else f"BROKEN — {results}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
