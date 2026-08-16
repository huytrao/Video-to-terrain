"""Stage J — the gate that can actually fail.

The previous pipeline's validator asked "does this file load?" and answered
PASS sixteen times on a scene with an 8.6 m ceiling. This one asks "is this
the right size?" and names the object when it is not.

Two independent verdicts:

  metric     dimensional error against Holi-Spatial ground truth, gated at
             ``DEFAULT_TOL_M``. Only meaningful on benchmark scenes.
  structural schema and plausibility problems that need no ground truth at
             all — assumed room geometry, boxes larger than the room,
             objects floating outside the shell.

A scene with no ground truth still gets a structural verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v3d.scene.schema import SceneDocument  # noqa: E402
from v3d.validate.gt_boxes import GTBox, load_gt, room_dims_from_gt  # noqa: E402

DEFAULT_TOL_M = 0.05
MATCH_RADIUS_M = 1.0

# Plausibility bounds for an indoor room. Not a substitute for ground truth —
# these only catch results that cannot be true of any real room.
CEILING_RANGE_M = (2.0, 3.5)
MAX_OBJECT_FRACTION_OF_ROOM = 0.6


@dataclass
class ObjectError:
    gt_id: str
    gt_label: str
    pred_id: str | None
    size_error_m: float | None
    center_error_m: float | None
    pred_size: list[float] | None
    gt_size: list[float]

    @property
    def matched(self) -> bool:
        return self.pred_id is not None


@dataclass
class Check:
    id: str
    status: str  # PASS | FAIL | SKIP
    detail: str


@dataclass
class StageJReport:
    scene_id: str
    tol_m: float
    metric_verdict: str = "SKIP"
    structural_verdict: str = "SKIP"
    checks: list[Check] = field(default_factory=list)
    object_errors: list[ObjectError] = field(default_factory=list)
    room_errors: dict[str, float] = field(default_factory=dict)
    recall: float | None = None

    @property
    def overall(self) -> str:
        verdicts = {self.metric_verdict, self.structural_verdict}
        if "FAIL" in verdicts:
            return "FAIL"
        return "PASS" if "PASS" in verdicts else "SKIP"

    def room_error_lines(self) -> list[tuple[str, float]]:
        return sorted(self.room_errors.items(), key=lambda kv: -kv[1])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overall"] = self.overall
        return d


def _obb_corners(center: np.ndarray, size: np.ndarray, axes: np.ndarray) -> np.ndarray:
    half = (axes.T * (size / 2.0)).T
    return np.array(
        [
            center + sx * half[0] + sy * half[1] + sz * half[2]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )


def match_objects(scene: SceneDocument, gt: list[GTBox]) -> list[ObjectError]:
    """Greedy nearest-centre matching, closest pairs claimed first."""
    targets = [b for b in gt if not b.is_shell]
    preds = list(scene.objects)

    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(targets):
        for pi, p in enumerate(preds):
            d = float(np.linalg.norm(np.asarray(p.box.center) - g.center))
            if d <= MATCH_RADIUS_M:
                pairs.append((d, gi, pi))
    pairs.sort()

    taken_gt: set[int] = set()
    taken_pred: set[int] = set()
    matches: dict[int, tuple[int, float]] = {}
    for d, gi, pi in pairs:
        if gi in taken_gt or pi in taken_pred:
            continue
        taken_gt.add(gi)
        taken_pred.add(pi)
        matches[gi] = (pi, d)

    out: list[ObjectError] = []
    for gi, g in enumerate(targets):
        if gi in matches:
            pi, dist = matches[gi]
            p = preds[pi]
            pred_size = np.sort(np.asarray(p.box.size, dtype=float))[::-1]
            gt_size = np.sort(g.size)[::-1]
            out.append(
                ObjectError(
                    gt_id=g.ins_id,
                    gt_label=g.label,
                    pred_id=p.id,
                    size_error_m=float(np.max(np.abs(pred_size - gt_size))),
                    center_error_m=dist,
                    pred_size=[round(float(v), 4) for v in pred_size],
                    gt_size=[round(float(v), 4) for v in gt_size],
                )
            )
        else:
            out.append(
                ObjectError(
                    gt_id=g.ins_id,
                    gt_label=g.label,
                    pred_id=None,
                    size_error_m=None,
                    center_error_m=None,
                    pred_size=None,
                    gt_size=[round(float(v), 4) for v in np.sort(g.size)[::-1]],
                )
            )
    return out


def structural_checks(scene: SceneDocument) -> list[Check]:
    checks: list[Check] = []

    errs = scene.structural_errors()
    checks.append(
        Check("schema", "PASS" if not errs else "FAIL", "; ".join(errs) or "schema valid")
    )

    assumed = scene.assumed_room_values()
    checks.append(
        Check(
            "no_assumed_room_geometry",
            "PASS" if not assumed else "FAIL",
            "no invented room values"
            if not assumed
            else f"room geometry marked 'assumed': {', '.join(assumed)}",
        )
    )

    h = scene.room.height_m()
    lo, hi = CEILING_RANGE_M
    checks.append(
        Check(
            "ceiling_plausible",
            "PASS" if lo <= h <= hi else "FAIL",
            f"ceiling height {h:.3f} m (plausible range {lo}-{hi} m)",
        )
    )

    fa, fb = scene.room.footprint_extent_m()
    room_volume = fa * fb * h
    oversized = [
        o.id
        for o in scene.objects
        if room_volume > 0 and o.box.volume_m3() > MAX_OBJECT_FRACTION_OF_ROOM * room_volume
    ]
    checks.append(
        Check(
            "no_object_larger_than_room",
            "PASS" if not oversized else "FAIL",
            f"room volume {room_volume:.1f} m^3"
            + ("" if not oversized else f"; oversized: {', '.join(oversized)}"),
        )
    )

    xs = [p[0] for p in scene.room.floor.polygon_xy]
    ys = [p[1] for p in scene.room.floor.polygon_xy]
    pad = 0.25
    outside = []
    for o in scene.objects:
        cx, cy = o.box.center[0], o.box.center[1]
        if not (min(xs) - pad <= cx <= max(xs) + pad and min(ys) - pad <= cy <= max(ys) + pad):
            outside.append(o.id)
    checks.append(
        Check(
            "objects_inside_room",
            "PASS" if not outside else "FAIL",
            "all object centres inside the footprint"
            if not outside
            else f"outside footprint: {', '.join(outside)}",
        )
    )
    return checks


def run(
    scene_path: str | Path,
    gt_path: str | Path | None = None,
    tol_m: float = DEFAULT_TOL_M,
) -> StageJReport:
    scene = SceneDocument.read(scene_path)
    report = StageJReport(scene_id=scene.scene_id, tol_m=tol_m)

    report.checks = structural_checks(scene)
    report.structural_verdict = (
        "FAIL" if any(c.status == "FAIL" for c in report.checks) else "PASS"
    )

    if gt_path is None:
        return report

    gt = load_gt(gt_path)
    report.object_errors = match_objects(scene, gt)

    matched = [e for e in report.object_errors if e.matched]
    total = len(report.object_errors)
    report.recall = (len(matched) / total) if total else None

    over_tol = [e for e in matched if (e.size_error_m or 0.0) > tol_m]
    report.checks.append(
        Check(
            "object_size_within_tol",
            "PASS" if not over_tol else "FAIL",
            f"{len(matched)}/{total} matched; {len(over_tol)} over {tol_m*100:.0f} cm"
            + (
                ""
                if not over_tol
                else "; worst: "
                + ", ".join(
                    f"{e.gt_label}#{e.gt_id} {e.size_error_m*100:.1f}cm"
                    for e in sorted(over_tol, key=lambda x: -(x.size_error_m or 0))[:5]
                )
            ),
        )
    )

    gt_dims = room_dims_from_gt(gt)
    if gt_dims:
        fa, fb = scene.room.footprint_extent_m()
        pred = {
            "footprint_a_m": max(fa, fb),
            "footprint_b_m": min(fa, fb),
            "height_m": scene.room.height_m(),
        }
        for key, gt_val in gt_dims.items():
            if key in pred:
                report.room_errors[key] = abs(pred[key] - gt_val)
        worst = max(report.room_errors.values()) if report.room_errors else 0.0
        report.checks.append(
            Check(
                "room_dims_within_tol",
                "PASS" if worst <= tol_m else "FAIL",
                "; ".join(f"{k} off by {v*100:.1f}cm" for k, v in report.room_error_lines()),
            )
        )

    metric_ids = {"object_size_within_tol", "room_dims_within_tol"}
    metric_checks = [c for c in report.checks if c.id in metric_ids]
    report.metric_verdict = (
        "FAIL" if any(c.status == "FAIL" for c in metric_checks) else "PASS"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage J — metric and structural gate")
    ap.add_argument("scene", help="path to scene.json")
    ap.add_argument("--gt", help="Holi-Spatial ground-truth box json for this scene")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_M, help="tolerance in metres")
    ap.add_argument("--out", help="write the report json here")
    args = ap.parse_args(argv)

    report = run(args.scene, args.gt, args.tol)

    width = max((len(c.id) for c in report.checks), default=10)
    for c in report.checks:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[c.status]
        print(f"  [{mark}] {c.id.ljust(width)}  {c.detail}")
    if report.recall is not None:
        print(f"\n  recall: {report.recall*100:.1f}% of ground-truth objects matched")
    print(f"\n  structural: {report.structural_verdict}   metric: {report.metric_verdict}")
    print(f"  OVERALL: {report.overall}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"  report -> {args.out}")

    return 0 if report.overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
