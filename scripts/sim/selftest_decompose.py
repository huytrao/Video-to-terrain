"""Self-test the object cut and the terrain export against a room we built ourselves.

The value of a synthetic scene is that the answer is known. Everything downstream
of the mesh — plane detection, up-axis derivation, component clustering, the
height map, the MJCF — is code with no ground truth on a real capture, so a bug
in it produces a plausible wrong answer. Here the room has exactly five shell
planes and exactly three objects, at known positions and sizes, so each of those
claims can actually fail.

Two deliberate difficulties, because a test that only uses an easy case proves the
easy case:

  - **Up is +Y**, not +Z. The pipeline has been bitten three times by coordinate
    conventions, including a floor plan that came out upside down and looked
    entirely reasonable. If `derive_up_axis` has a hardcoded axis anywhere, this
    fails.
  - **One object is a flat panel** standing on the floor. It is as planar as a
    wall, and only its *extent* distinguishes it. If shell detection tests
    flatness rather than span, the panel disappears and the test says so.

Run on the kernel, where open3d lives. No GPU needed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

OUT = Path(os.environ.get("V3D_SELFTEST_OUT", "/tmp/selftest_decompose"))

# The room, in a frame where up is +Y so no assumed axis can pass by luck.
ROOM = (6.0, 3.0, 8.0)          # x, y(up), z
OBJECTS = [
    # (centre, half-extent) — a crate, a tall cabinet, and a thin standing panel.
    ((-1.5, 0.4, -2.0), (0.6, 0.4, 0.5)),
    ((2.0, 0.9, 1.5), (0.5, 0.9, 0.4)),
    ((0.0, 0.7, 3.0), (1.1, 0.7, 0.05)),      # the panel: planar, but small in span
]


def build_room() -> o3d.geometry.TriangleMesh:
    """Floor, ceiling, four walls, and three free-standing objects."""
    sx, sy, sz = ROOM
    parts = []

    def slab(centre, half):
        b = o3d.geometry.TriangleMesh.create_box(
            width=2 * half[0], height=2 * half[1], depth=2 * half[2])
        b.translate(np.array(centre) - np.array(half))
        return b

    t = 0.05
    parts.append(slab((0, -t, 0), (sx / 2, t, sz / 2)))           # floor
    parts.append(slab((0, sy + t, 0), (sx / 2, t, sz / 2)))       # ceiling
    parts.append(slab((-sx / 2 - t, sy / 2, 0), (t, sy / 2, sz / 2)))
    parts.append(slab((sx / 2 + t, sy / 2, 0), (t, sy / 2, sz / 2)))
    parts.append(slab((0, sy / 2, -sz / 2 - t), (sx / 2, sy / 2, t)))
    parts.append(slab((0, sy / 2, sz / 2 + t), (sx / 2, sy / 2, t)))
    for centre, half in OBJECTS:
        parts.append(slab(centre, half))

    mesh = parts[0]
    for p in parts[1:]:
        mesh += p
    # Subdivide so RANSAC has samples to work with rather than eight corners per
    # slab: a box mesh has 12 triangles, and plane fitting over 12 vertices is not
    # a test of plane fitting.
    mesh = mesh.subdivide_midpoint(number_of_iterations=4)
    mesh.compute_vertex_normals()
    return mesh


def fake_cameras() -> np.ndarray:
    """Cameras walking the room at eye height — the up-axis sign comes from these."""
    n = 40
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([2.0 * np.cos(ang),
                     np.full(n, 1.6),
                     2.5 * np.sin(ang)], axis=1)


def main() -> None:
    from v3d.scene import decompose as dc
    from v3d.export import terrain as tr

    OUT.mkdir(parents=True, exist_ok=True)
    mesh = build_room()
    mesh_path = OUT / "room.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False)
    print(f"synthetic room: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    print(f"  room {ROOM} with up = +Y, {len(OBJECTS)} objects\n")

    failures = []

    # --- the cut --------------------------------------------------------------
    res = dc.cut(mesh_path, OUT, camera_centres=fake_cameras(), scale_mode="metres")
    up = np.array(res.up_axis)
    print(f"up axis      {np.round(up, 4).tolist()}  ({res.up_axis_evidence['rule']})")
    print(f"             plane source: {res.up_axis_evidence['plane_source']}")
    print(f"room height  {res.up_axis_evidence['room_height_estimate']:.3f} "
          f"(true {ROOM[1]:.3f})")

    # Up must be +Y to within a few degrees.
    cos_y = float(abs(up @ np.array([0.0, 1.0, 0.0])))
    if cos_y < 0.98:
        failures.append(f"up axis is {np.round(up,3).tolist()}, not +/-Y (cos {cos_y:.3f})")
    if up[1] < 0:
        failures.append(f"up axis points down: {np.round(up,3).tolist()}")
    h = res.up_axis_evidence["room_height_estimate"]
    if not 0.85 * ROOM[1] <= h <= 1.15 * ROOM[1]:
        failures.append(f"room height {h:.3f} is not within 15% of {ROOM[1]}")

    print(f"\nshell planes {len(res.shell_planes)}")
    for p in res.shell_planes:
        print(f"  {p.role:30.30} pts {p.points:>7,}  span {p.span_frac:.3f} edge {p.edge_frac:.3f} "
              f"vert {p.verticality:.3f}  removed {p.removed:>7,} "
              f"({p.removed_frac:.1%})  interior {p.interior_signed:+.3f}")

    d = res.diagnostics
    print(f"\nremoval mode  {d.get('removal_mode')}")
    print(f"vertex survival {d.get('vertex_survival_fraction', 0):.2%}")
    print(f"shell removed {d['shell_triangles_removed']:,}/{d['mesh_triangles']:,} "
          f"({d['shell_fraction']:.1%})")
    print(f"components {d['components_after_shell_removal']} -> "
          f"{d['objects_kept']} objects")
    for o in res.objects:
        print(f"  object {o.index}: extent {np.round(o.extent,3).tolist()} "
              f"centre {np.round(o.centre,2).tolist()} "
              f"rests_on_floor {o.rests_on_floor}")

    # The count is the headline claim.
    if len(res.objects) != len(OBJECTS):
        failures.append(
            f"found {len(res.objects)} objects, expected {len(OBJECTS)}. "
            f"Fewer means a shell plane consumed one (check the panel); more means "
            f"an object split or a wall survived."
        )

    # Match found objects to true ones by CENTRE, not by sorted extent. Sorting both
    # lists and zipping them pairs the smallest with the smallest, which happily
    # compared a panel fragment against a crate and reported a 100% error about the
    # wrong pair. Centres are what identify an object; extent is then the thing being
    # measured, so it must not also be the thing used to decide what to compare.
    print("\nmatched by centre:")
    used = set()
    for centre, half in OBJECTS:
        true_ext = tuple(sorted((2 * h for h in half), reverse=True))
        best, best_d = None, None
        for o in res.objects:
            if o.index in used:
                continue
            dd = float(np.linalg.norm(np.array(o.centre) - np.array(centre)))
            if best_d is None or dd < best_d:
                best, best_d = o, dd
        if best is None:
            failures.append(f"no object left to match the one at {centre}")
            continue
        used.add(best.index)
        found_ext = tuple(sorted(best.extent, reverse=True))
        rel = max(abs(a - b) / max(a, 1e-9) for a, b in zip(true_ext, found_ext))
        print(f"  true centre {np.round(centre,2)} extent {np.round(true_ext,3)}")
        print(f"    -> object {best.index} centre {np.round(best.centre,2)} "
              f"extent {np.round(found_ext,3)}  centre err {best_d:.3f}  "
              f"worst extent err {rel:.0%}")
        # Centre is the sharp claim: it is what says the object was found in the
        # right place. Extent is looser on purpose -- the synthetic room is built
        # from SOLID boxes, so shell removal legitimately shaves the faces that
        # touch the floor and walls, and a TSDF surface does not have that property.
        if best_d > 0.25 * max(true_ext):
            failures.append(f"object at {centre} was matched {best_d:.3f} away, "
                            f"more than a quarter of its own longest edge")
        if rel > 0.5:
            failures.append(f"extent {np.round(found_ext,3)} differs from "
                            f"{np.round(true_ext,3)} by {rel:.0%}")

    rep = dc.audit(res)
    print(f"\naudit {rep['overall']}")
    for c in rep["checks"]:
        print(f"  [{c['status']}] {c['check']}: {c['detail'][:100]}")
    if rep["overall"] == "FAIL":
        failures.append(f"the audit failed on a scene built to pass it: "
                        f"{[c['check'] for c in rep['checks'] if c['status']=='FAIL']}")
    dc.write(res, rep, OUT)

    # --- terrain --------------------------------------------------------------
    print("\n--- terrain ---")
    hm = tr.height_map(mesh_path, up, res.floor_offset)
    hm["floor_offset"] = res.floor_offset
    trav = tr.traversability(hm)
    clear = tr.clearances([{"index": o.index, "centre": o.centre, "extent": o.extent}
                           for o in res.objects], res.room_extent, res.up_axis,
                          res.floor_offset)
    print(f"  grid {hm['nu']}x{hm['nv']} cell {hm['cell_size']:.4f} "
          f"holes {hm['hole_fraction']:.2%}")
    print(f"  height range {hm['height_range'][0]:.3f} .. {hm['height_range'][1]:.3f}")
    s = trav["summary"]
    print(f"  floor-like {s['floor_like_fraction']:.1%}, "
          f"flat<5deg {s['curriculum']['flat_fraction_under_5deg']:.1%}")
    print(f"  clearances min {clear.get('gap_min')} median {clear.get('gap_median')}")

    # The ceiling is the top surface everywhere, so the height map should read the
    # ceiling, not the floor. That is a property of a 2.5D map worth asserting
    # rather than being surprised by later.
    if hm["height_range"][1] < 0.8 * ROOM[1]:
        failures.append(f"height map tops out at {hm['height_range'][1]:.3f}, "
                        f"below the {ROOM[1]} ceiling; the up axis or the "
                        f"rasterisation is wrong")
    if hm["hole_fraction"] > 0.05:
        failures.append(f"{hm['hole_fraction']:.1%} of the grid is holes on a closed "
                        f"synthetic room; the sampler is not covering the surface")

    png = tr.write_heightfield_png(hm, OUT / "heightfield.png")
    mjcf = tr.write_mjcf(OUT / "scene.xml", hm,
                         [{"index": o.index, "centre": o.centre,
                           "extent": o.extent, "label": "test"} for o in res.objects],
                         "heightfield.png", scale_mode="metres")
    print(f"  wrote {png.name} and {mjcf.name}")

    # The hfield asset is referenced by a bare filename, so MuJoCo resolves it
    # relative to the XML. Both already sit in OUT, so nothing needs copying —
    # doing so raised SameFileError, which is the check that they are co-located.
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    maxv = float(np.abs(data.qvel).max()) if data.qvel.size else 0.0
    print(f"  MJCF compiles: {model.nbody} bodies, {model.ngeom} geoms")
    print(f"  1000 steps -> finite {finite}, max |qvel| {maxv:.4f}")
    if not finite:
        failures.append("physics diverged on the synthetic scene")
    # Boxes dropped onto a heightfield should settle, not keep accelerating.
    if maxv > 50.0:
        failures.append(f"max |qvel| {maxv:.1f} after 1000 steps; the objects are "
                        f"not settling, which usually means they spawned inside "
                        f"the terrain")

    # --- verdict --------------------------------------------------------------
    print("\n" + "=" * 70)
    if failures:
        print(f"SELFTEST FAIL — {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("SELFTEST PASS — objects, up axis, extents, height map and MJCF all correct")


main()
