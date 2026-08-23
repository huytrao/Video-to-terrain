#!/usr/bin/env python3
"""Negative controls for Stage J — proof the gate is not a rubber stamp.

A gate that has been made to pass on ground truth has proved nothing yet. The
previous pipeline's validator answered PASS sixteen times on a scene with an
8.6 m ceiling, and it would have kept doing so however carefully it was read.
The only evidence that a gate discriminates is that it *fails* on scenes that
deserve to fail, and passes on ones that do not, with the boundary in between
landing where it was claimed to land.

So each control perturbs a known-good scene one way and asserts the verdict,
including which check fires. A control that fails to fail is a hole in the gate;
a control that fires on the wrong check is a gate that happens to be right for
the wrong reason, which will not survive the next edit.

Perturbations are computed from the scene's own geometry, never from constants
measured on one room. The first version of the mislocalisation control shifted an
object 5.7 m along +x and was recorded as catching a stray object; this room is a
rectangle rotated 21 degrees, so that shift slides *along* the long axis and
leaves the object legitimately inside. The control passed for a reason that had
nothing to do with the distance in its name.

Usage:
    scripts/sim/stage_j_controls.py <honest-scene.json> --gt <ground-truth.json>

Companion to `stage_j_selftest.py`, which is not the same test and should not be
merged into this one: that one *constructs* an honest and a buggy scene from the
annotation and asserts the gate separates the previous pipeline's specific
defects. This one takes a scene the pipeline actually *produced* and perturbs it,
which is the only way to catch a check that stopped applying to real output.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v3d.validate.stage_j import (  # noqa: E402
    MIN_RECALL_BY_BAND,
    footprint_overhang_m,
    run,
    selftest,
)
from v3d.scene.schema import SceneDocument  # noqa: E402


def _outward_push(scene: SceneDocument, index: int, metres: float) -> tuple[float, float]:
    """A shift that moves object `index` `metres` further out of the footprint.

    Along the outward normal of whichever wall the object is nearest, so the
    displacement means the same thing in a rotated room as in an axis-aligned
    one.
    """
    poly = np.asarray(scene.room.floor.polygon_xy, float)
    shoelace = float(
        np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    )
    q = poly if shoelace > 0 else poly[::-1]
    edges = np.roll(q, -1, axis=0) - q
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    normals = np.stack([edges[:, 1] / lengths, -edges[:, 0] / lengths], axis=1)
    centre = np.asarray(scene.objects[index].box.center[:2], float)
    nearest = int(np.argmax(((centre - q) * normals).sum(axis=1)))
    return tuple(normals[nearest] * metres)


def _grade(doc: dict, gt: str) -> tuple:
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(doc))
    try:
        report = run(tmp, gt)
    finally:
        tmp.unlink(missing_ok=True)
    fired = sorted(c.id for c in report.checks if c.status == "FAIL")
    return report, fired


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage J negative controls")
    ap.add_argument("scene", help="a scene.json that should pass")
    ap.add_argument("--gt", required=True, help="ground-truth box json")
    args = ap.parse_args(argv)

    print("geometry selftest:", json.dumps(selftest()))

    base = json.loads(Path(args.scene).read_text())
    scene = SceneDocument.read(args.scene)
    fa, fb = scene.room.footprint_extent_m()
    limit = 0.20 * min(fa, fb)  # STRAY_FRACTION_OF_ROOM * short edge
    head = float(footprint_overhang_m(scene)[0])
    print(f"room {fa:.2f} x {fb:.2f} m, stray limit {limit:.2f} m, "
          f"object[0] currently {head:+.2f} m from the wall")

    def grow(cm):
        def f(d):
            for o in d["objects"]:
                o["box"]["size"] = [v + cm / 100.0 for v in o["box"]["size"]]
        return f

    def push(metres):
        """Move object[0] so its overhang lands `metres` past the stray limit."""
        dx, dy = _outward_push(scene, 0, (limit - head) + metres)

        def f(d):
            d["objects"][0]["box"]["center"][0] += dx
            d["objects"][0]["box"]["center"][1] += dy
        return f

    def keep_fraction_of_band(frac, lo, hi):
        def f(d):
            keep, seen = [], 0
            for o in d["objects"]:
                big = lo <= max(o["box"]["size"]) < hi
                if not big:
                    keep.append(o)
                    continue
                seen += 1
                if (seen * frac) % 1 >= frac and frac < 1.0:
                    continue
                keep.append(o)
            d["objects"] = keep
        return f

    def decimate(step):
        def f(d):
            d["objects"] = d["objects"][::step]
        return f

    def ceiling(h):
        def f(d):
            d["room"]["ceiling"]["height_m"] = h
        return f

    # (name, mutation, expected overall, checks expected to fire)
    controls = [
        ("honest", None, "PASS", []),
        ("sizes +8cm", grow(8), "FAIL", ["object_size_within_tol"]),
        ("sizes +4cm", grow(4), "PASS", []),
        ("object 0.3m inside stray limit", push(-0.3), "PASS", []),
        ("object 0.3m past stray limit", push(0.3), "FAIL", ["no_stray_objects"]),
        ("half the objects gone", decimate(2), "FAIL", ["object_recall"]),
        ("all large objects gone", keep_fraction_of_band(0.0, 1.0, float("inf")),
         "FAIL", ["object_recall"]),
        ("ceiling 8.6m", ceiling(8.6), "FAIL",
         ["ceiling_plausible", "room_dims_within_tol"]),
    ]

    print(f"\nrecall floors: {MIN_RECALL_BY_BAND}\n")
    bad = 0
    for name, mutate, want_overall, want_fired in controls:
        doc = copy.deepcopy(base)
        if mutate:
            mutate(doc)
        report, fired = _grade(doc, args.gt)
        ok = report.overall == want_overall and fired == sorted(want_fired)
        bad += not ok
        print(f"  [{'ok  ' if ok else 'HOLE'}] {name:<32} {report.overall:<4} "
              f"recall {report.recall*100:5.1f}%  fired: {', '.join(fired) or '-'}")
        if not ok:
            print(f"{'':>9} wanted {want_overall} firing {want_fired or ['-']}")

    print(f"\n{len(controls) - bad}/{len(controls)} controls behaved as claimed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
