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

# An object centre may legitimately sit outside the floor polygon: a window's box
# lies in the wall, and the polygon is a rectangle fitted to a shell that does not
# agree with the other four (see docs/reference/bugs-found.md B1). Measured on the
# released annotation for 00777c41d4, 75 of 218 *correct* object centres fall
# outside, the worst by 0.82 m — so a check demanding containment fails on ground
# truth and could never pass on a reconstruction. What is still worth catching is an
# object placed somewhere else entirely, so the limit is a fraction of the room's
# short edge rather than an absolute pad: it scales with the scene, and 0.20 leaves
# the annotation's own worst case (0.82 m against a 1.24 m limit) inside the bound
# with roughly a third to spare. That calibration is against real correct data, not
# a number chosen to make this scene pass.
STRAY_FRACTION_OF_ROOM = 0.20

# Recall floors, per size band. The metric verdict grades *matched* objects, so
# without this a pipeline that finds one wardrobe and misses the rest scores a
# perfect 5 cm: the tolerance becomes a claim about the easy half of the scene.
# Measured — dropping half this scene's objects at random left every check PASS
# at 50% recall.
#
# A single floor across all objects would repeat the containment mistake above.
# 99 of this annotation's 218 objects are under 30 cm — stickers, labels, cords,
# switches — and no video reconstruction segments a 6 cm sticker, so a flat 80%
# could never pass honestly. The floor therefore applies per band, and the tiny
# band is reported but not gated (absent from this dict = ungated). Large objects
# are both the ones any reconstruction can see and the ones a robot collides
# with, so they are held highest.
#
# These are declared expectations, not measured calibration: the only end-to-end
# run so far feeds the released annotation to itself, where recall is 100% by
# construction and proves nothing. Revisit against the first real DA3+PGSR run
# rather than treating these as validated.
MIN_RECALL_BY_BAND = {"large": 0.80, "small": 0.50}

# Size bands for reporting. A fixed 5 cm tolerance is 2% of a 2.5 m door and
# 80% of a 6 cm cord, and the ScanNet++ annotations are full of the latter —
# stickers, labels, cords, panels. Banding keeps that visible rather than
# letting hundreds of trivially-passed fragments hide a wrong wall.
SIZE_BANDS = (("tiny", 0.0, 0.3), ("small", 0.3, 1.0), ("large", 1.0, float("inf")))

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

    def size_band_summary(self) -> list[tuple[str, int, int, int, float | None]]:
        """Per band: (name, ground-truth count, matched, over tolerance, worst error)."""
        rows = []
        for name, lo, hi in SIZE_BANDS:
            in_band = [e for e in self.object_errors if lo <= max(e.gt_size) < hi]
            matched = [e for e in in_band if e.matched]
            over = [e for e in matched if (e.size_error_m or 0.0) > self.tol_m]
            worst = max((e.size_error_m or 0.0) for e in matched) if matched else None
            rows.append((name, len(in_band), len(matched), len(over), worst))
        return rows

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

    overhangs = footprint_overhang_m(scene)
    stray = [
        o.id for o, d in zip(scene.objects, overhangs)
        if d > STRAY_FRACTION_OF_ROOM * min(fa, fb)
    ]
    limit = STRAY_FRACTION_OF_ROOM * min(fa, fb)
    if overhangs.size:
        out = int((overhangs > 0).sum())
        detail = (f"{out}/{len(overhangs)} objects reach past the footprint, worst "
                  f"{overhangs.max():.2f} m; stray limit {limit:.2f} m")
    else:
        detail = "no objects"
    checks.append(
        Check(
            "no_stray_objects",
            "PASS" if not stray else "FAIL",
            detail + ("" if not stray else f"; stray: {', '.join(stray)}"),
        )
    )
    return checks


def footprint_overhang_m(scene: SceneDocument) -> np.ndarray:
    """Each object centre's signed distance outside the floor polygon, in metres.

    Negative is inside. What this fixes, and what it deliberately does not measure:

    **The polygon, not its bounding box.** A real room is not axis-aligned —
    this scene's footprint is a 6.42 x 6.22 m rectangle rotated about 21
    degrees, whose axis-aligned span is 8.15 x 8.26 m. Testing centres against
    ``min(xs)..max(xs)`` therefore admitted objects a metre outside the room
    along the diagonals, while flagging others merely for sitting near a corner
    of a box the room does not occupy. ``Room.footprint_extent_m`` already
    documents having fixed exactly this mistake for the room's own dimensions;
    the containment test never got the same treatment.

    **The centre, not the box.** Folding in the box's own half-extent would make
    a wardrobe against a wall score worse than a mug in the garden, which is
    backwards for a check asking "is this object in this room". Size is already
    gated, to 5 cm, by ``object_size_within_tol``.
    """
    poly = np.asarray(scene.room.floor.polygon_xy, float)
    if len(poly) < 3 or not scene.objects:
        return np.zeros(len(scene.objects))

    # Outward edge normals, with the winding resolved from the signed area so a
    # clockwise polygon does not invert every normal and report the room inside out.
    shoelace = float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                            - np.roll(poly[:, 0], -1) * poly[:, 1]))
    q = poly if shoelace > 0 else poly[::-1]
    edges = np.roll(q, -1, axis=0) - q
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    keep = lengths > 1e-9
    starts, edges, lengths = q[keep], edges[keep], lengths[keep]
    normals = np.stack([edges[:, 1] / lengths, -edges[:, 0] / lengths], axis=1)

    centres = np.array([o.box.center[:2] for o in scene.objects], float)
    # (n_objects, n_edges): distance outside each edge; the max is the polygon
    # distance for a convex footprint.
    signed = (centres[:, None, :] - starts[None, :, :]) * normals[None, :, :]
    return signed.sum(axis=2).max(axis=1)


def selftest() -> dict:
    """Pin the footprint geometry against analytic cases.

    Every bug this guards against returns a plausible number rather than an
    error: an inverted normal reports the room inside out, and an axis-aligned
    box in place of the polygon silently moves the room's walls outward by up to
    the difference between a rectangle and its rotated bounding box — 1.7 m on
    this scene. Neither is visible in an exit code.
    """
    import math

    class _Floor:
        def __init__(self, poly): self.polygon_xy = poly

    class _Room:
        def __init__(self, poly): self.floor = _Floor(poly)

    class _Box:
        def __init__(self, c): self.center = c

    class _Obj:
        def __init__(self, c): self.box = _Box(c)

    class _Scene:
        def __init__(self, poly, cs):
            self.room, self.objects = _Room(poly), [_Obj(c) for c in cs]

    square = [[0, 0], [4, 0], [4, 4], [0, 4]]
    probes = [[2, 2, 0], [0, 2, 0], [-1, 2, 0], [-1, -1, 0]]
    want = [-2.0, 0.0, 1.0, 1.0]

    ccw = footprint_overhang_m(_Scene(square, probes))
    if not np.allclose(ccw, want, atol=1e-9):
        raise AssertionError(f"axis-aligned footprint wrong: {ccw.round(3)}, wanted {want}")

    # The same square wound the other way must give the same answer; if the
    # winding is not resolved, every normal flips and inside becomes outside.
    cw = footprint_overhang_m(_Scene(square[::-1], probes))
    if not np.allclose(cw, want, atol=1e-9):
        raise AssertionError(
            f"clockwise footprint inverted: {cw.round(3)}, wanted {want} — the room "
            f"is being reported inside out"
        )

    # A rotated room, which is every real room. A point on a wall face is exactly
    # on the boundary, and a point inside the axis-aligned bounding box but
    # outside the room itself must be flagged — that is the region the previous
    # min/max test admitted.
    angle = math.radians(21.0)
    rot = np.array([[math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]])
    rotated = [(rot @ np.array(p)).tolist() for p in ([-2, -2], [2, -2], [2, 2], [-2, 2])]
    on_face = (rot @ np.array([2.0, 0.0])).tolist() + [0.0]
    face_d = float(footprint_overhang_m(_Scene(rotated, [on_face]))[0])
    if abs(face_d) > 1e-9:
        raise AssertionError(f"a point on the wall face measured {face_d:.6f} m, wanted 0")

    span = 2.0 * (math.cos(angle) + math.sin(angle))       # AABB half-span, 2.584 m
    diag = [span - 0.05, span - 0.05, 0.0]                 # inside the AABB
    diag_d = float(footprint_overhang_m(_Scene(rotated, [diag]))[0])
    if diag_d <= 0.0:
        raise AssertionError(
            f"a point inside the bounding box but outside the rotated room measured "
            f"{diag_d:.3f} m — the polygon test has regressed to its bounding box"
        )

    return {
        "axis_aligned_m": [round(float(v), 3) for v in ccw],
        "winding_independent": True,
        "wall_face_m": round(face_d, 9),
        "aabb_corner_outside_by_m": round(diag_d, 3),
        "aabb_inflation_m": round(2 * span - 4.0, 3),
    }


def recall_check(report: StageJReport, floors: dict[str, float]) -> Check:
    """Gate how much of the scene was found, per size band.

    Empty ``floors`` gates nothing and says so — SKIP, not PASS, because "we did
    not check" and "we checked and it was fine" must not print the same word.
    """
    rows: list[str] = []
    short: list[str] = []
    for name, n_gt, n_matched, _over, _worst in report.size_band_summary():
        if not n_gt:
            continue
        got = n_matched / n_gt
        floor = floors.get(name)
        rows.append(
            f"{name} {n_matched}/{n_gt} = {got*100:.0f}%"
            + ("" if floor is None else f" (floor {floor*100:.0f}%)")
        )
        if floor is not None and got < floor:
            short.append(f"{name} short by {(floor - got)*100:.0f} points")
    detail = "; ".join(rows) or "no ground-truth objects"
    if not floors:
        return Check("object_recall", "SKIP", detail + "; no floor applied")
    return Check(
        "object_recall",
        "PASS" if not short else "FAIL",
        detail + ("" if not short else "; " + ", ".join(short)),
    )


def run(
    scene_path: str | Path,
    gt_path: str | Path | None = None,
    tol_m: float = DEFAULT_TOL_M,
    min_size_m: float = 0.0,
    recall_floors: dict[str, float] | None = None,
) -> StageJReport:
    """Grade a scene.

    ``min_size_m`` restricts the metric gate to ground-truth objects whose
    largest dimension reaches it. Zero — the default — gates everything, which
    is what was asked for; raising it is a deliberate, recorded choice, and
    excluded objects still appear in the size-band summary.

    ``recall_floors`` maps size-band name to the fraction of that band's
    ground-truth objects that must be found; ``{}`` gates none of it.
    """
    floors = MIN_RECALL_BY_BAND if recall_floors is None else recall_floors
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

    gated = [e for e in matched if max(e.gt_size) >= min_size_m]
    over_tol = [e for e in gated if (e.size_error_m or 0.0) > tol_m]
    scope = "" if min_size_m <= 0 else f" (gating objects >= {min_size_m*100:.0f} cm)"
    report.checks.append(
        Check(
            "object_size_within_tol",
            "PASS" if not over_tol else "FAIL",
            f"{len(matched)}/{total} matched{scope}; {len(over_tol)} over {tol_m*100:.0f} cm"
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

    report.checks.append(recall_check(report, floors))

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

    metric_ids = {"object_size_within_tol", "object_recall", "room_dims_within_tol"}
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
    ap.add_argument("--min-size", type=float, default=0.0,
                    help="only gate ground-truth objects at least this large (metres)")
    ap.add_argument("--min-recall", type=float, default=None,
                    help="override every gated size band's recall floor with this "
                         "fraction; 0 reports recall without gating it "
                         f"(default per band: {MIN_RECALL_BY_BAND})")
    ap.add_argument("--out", help="write the report json here")
    args = ap.parse_args(argv)

    if args.min_recall is None:
        floors = None
    elif args.min_recall <= 0:
        floors = {}
    else:
        floors = {band: args.min_recall for band in MIN_RECALL_BY_BAND}

    report = run(args.scene, args.gt, args.tol, args.min_size, floors)

    width = max((len(c.id) for c in report.checks), default=10)
    for c in report.checks:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[c.status]
        print(f"  [{mark}] {c.id.ljust(width)}  {c.detail}")

    if report.object_errors:
        print("\n  by object size:")
        print(f"    {'band':<8}{'in GT':>7}{'matched':>9}{'over tol':>10}{'worst':>10}")
        for name, n_gt, n_matched, n_over, worst in report.size_band_summary():
            w = f"{worst*100:.1f}cm" if worst is not None else "-"
            print(f"    {name:<8}{n_gt:>7}{n_matched:>9}{n_over:>10}{w:>10}")
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
