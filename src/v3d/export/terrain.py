"""Height map, traversability parameters, and a MuJoCo scene a robot can stand in.

The visual mesh is not a simulation asset. The pilot produced 18.5 M triangles,
not watertight, no convex decomposition, no material semantics; a locomotion policy
cannot collide against that at any useful rate. What a controls stack actually
consumes is much smaller and much more specific, and this module produces it:

- **a height map** — one float per cell, the highest surface in that column. This
  is the terrain representation locomotion RL trains on, and it is also what makes
  slope and step statistics computable at all.
- **traversability parameters** — slope, step height and roughness per cell, plus
  the distribution of each over the room. Curriculum generation needs the
  distribution, not one number: "the hardest step is 18 cm" is what sets the top of
  a curriculum, and "82% of the floor is under 5 degrees" is what sets its bottom.
- **clearances** — for a robot the number that matters is often not an object's
  size but the gap between it and the wall. A scene can pass every dimensional
  check and still be untraversable.
- **an MJCF scene** — heightfield for the terrain, one box per cut object, with
  physics values that each carry a `measured | derived | assumed` tag so a policy
  that only works because a friction coefficient was guessed at 0.8 is visible
  before it reaches hardware.

Two honest limits, stated because they change how the output should be used.

*Scale.* With `scale_mode="assumed"` every length here is in DA3's units. A
height map in unknown units still gives correct *relative* terrain — slopes are
dimensionless and transfer exactly — but "18 cm step" is not a claim this pipeline
can make from video alone. One measured distance fixes all of it.

*A height map is 2.5D.* It cannot represent an overhang, a table you reach under,
or a shelf. Everything below the top surface in a column is invisible to it. So
the height map is the terrain, and the cut objects remain separate bodies rather
than being baked into it — which is also what lets a manipulation policy see them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class TerrainPolicy:
    """Resolution as a count of cells, never as a cell size in metres."""

    # Cells across the room's longer footprint axis. A count transfers between
    # scenes; a 2 cm cell does not, and in DA3 units it does not even mean 2 cm.
    cells_across: int = 512
    # Surface samples per grid cell. This has to be a ratio to the grid, not a
    # property of the mesh: sizing the sample count from the triangle count gave a
    # 512x388 grid roughly one sample per cell, and Poisson statistics then left
    # 58% of the cells empty on a *closed* synthetic room — holes that are an
    # artefact of undersampling and read downstream as pits in the terrain.
    # Sixteen samples per cell puts the expected empty-cell fraction below 1e-7.
    samples_per_cell: int = 16
    max_samples: int = 24_000_000
    # Columns holding fewer samples than this are holes, not floor. Filled by
    # nearest-neighbour and reported, because an unfilled hole reads to a
    # simulator as a pit the robot falls into.
    min_samples_per_cell: int = 1
    # A cell's slope is computed over this many cells, so the gradient is measured
    # over a robot-relevant distance rather than between adjacent samples where
    # reconstruction noise dominates.
    slope_stride: int = 3
    # Step height is the largest jump to any neighbour within this radius.
    step_radius: int = 2
    # Fraction of height range above the floor treated as "obstacle" rather than
    # terrain, for the traversability summary.
    obstacle_height_frac: float = 0.12
    # Percentiles reported for every distribution. A curriculum is built from the
    # tail, not from the mean.
    percentiles: tuple[float, ...] = (50.0, 75.0, 90.0, 95.0, 99.0, 100.0)


@dataclass
class PhysicsDefaults:
    """Every value tagged with where it came from. All of these are `assumed`."""

    floor_friction: float = 0.9
    object_friction: float = 0.7
    restitution: float = 0.05
    object_density: float = 300.0        # kg/m^3, plausible for crates and panels
    source: str = "assumed"
    note: str = ("no material was measured or inferred; these are plausible "
                 "defaults and a policy that depends on them is not transferable "
                 "until they are")


def _basis_from_up(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two axes spanning the ground plane, built without a degenerate cross product."""
    up = up / np.linalg.norm(up)
    helper = np.eye(3)[int(np.argmin(np.abs(up)))]
    x = np.cross(up, helper)
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    return x, y


def height_map(mesh_path: str | Path, up: np.ndarray, floor_offset: float,
               policy: TerrainPolicy | None = None) -> dict:
    """Rasterise the highest surface per column onto a regular grid.

    Sampled from the mesh *surface*, not from its vertices. Vertex density follows
    curvature, so a large flat floor contributes few vertices and would rasterise
    as mostly holes while a detailed machine contributes thousands — the grid would
    be dense exactly where the terrain does not matter.

    The grid is sized first and the sampling second, which is the opposite of the
    obvious order and the reason it works: the number of samples needed is a
    property of the *grid*, and deriving it from the mesh instead left 58% of the
    cells empty on a closed synthetic room.
    """
    import open3d as o3d

    policy = policy or TerrainPolicy()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"{mesh_path} has no triangles")

    up = np.asarray(up, dtype=float)
    up /= np.linalg.norm(up)
    ax, ay = _basis_from_up(up)

    # Extent from the vertices: exact for a bounding box, and free.
    verts = np.asarray(mesh.vertices)
    vu, vv = verts @ ax, verts @ ay
    u0, u1 = float(vu.min()), float(vu.max())
    v0, v1 = float(vv.min()), float(vv.max())
    span_u, span_v = u1 - u0, v1 - v0
    cell = max(span_u, span_v) / policy.cells_across
    nu = max(2, int(np.ceil(span_u / cell)))
    nv = max(2, int(np.ceil(span_v / cell)))

    n = int(min(policy.max_samples, max(200_000, nu * nv * policy.samples_per_cell)))
    pcd = mesh.sample_points_uniformly(number_of_points=n)
    pts = np.asarray(pcd.points)

    u = pts @ ax
    v = pts @ ay
    h = pts @ up + floor_offset

    iu = np.clip(((u - u0) / cell).astype(int), 0, nu - 1)
    iv = np.clip(((v - v0) / cell).astype(int), 0, nv - 1)
    flat = iv * nu + iu

    # Highest surface per column, and how many samples supported it.
    top = np.full(nu * nv, -np.inf)
    np.maximum.at(top, flat, h)
    count = np.zeros(nu * nv, dtype=np.int64)
    np.add.at(count, flat, 1)
    # The lowest surface too: the gap between them is what says whether a column
    # holds an overhang the height map cannot represent.
    bottom = np.full(nu * nv, np.inf)
    np.minimum.at(bottom, flat, h)

    grid = top.reshape(nv, nu)
    low = bottom.reshape(nv, nu)
    occupancy = count.reshape(nv, nu)
    holes = occupancy < policy.min_samples_per_cell
    hole_frac = float(holes.mean())

    # Fill holes from the nearest observed cell. A hole left as -inf becomes a pit
    # in the simulator, and a robot walking into it is a bug in the export rather
    # than a property of the room.
    filled = grid.copy()
    if holes.any() and (~holes).any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(holes, return_distances=False, return_indices=True)
        filled = grid[tuple(idx)]
    filled[~np.isfinite(filled)] = 0.0

    return {
        "grid": filled,
        "raw": grid,
        "bottom": low,
        "occupancy": occupancy,
        "holes": holes,
        "hole_fraction": hole_frac,
        "samples_per_cell_requested": policy.samples_per_cell,
        "samples_per_cell_actual": float(n / max(nu * nv, 1)),
        "cell_size": float(cell),
        "nu": nu, "nv": nv,
        "origin_uv": [u0, v0],
        "axes": {"u": [float(x) for x in ax], "v": [float(x) for x in ay],
                 "up": [float(x) for x in up]},
        "samples": int(len(pts)),
        "height_range": [float(np.nanmin(filled)), float(np.nanmax(filled))],
    }


def traversability(hm: dict, policy: TerrainPolicy | None = None) -> dict:
    """Slope, step height and roughness per cell, plus their distributions."""
    policy = policy or TerrainPolicy()
    g = hm["grid"]
    cell = hm["cell_size"]
    s = policy.slope_stride

    # Gradient over `slope_stride` cells: between adjacent cells the reconstruction
    # noise is comparable to the real relief, and the resulting "slope" describes
    # the mesh's finish rather than the terrain a robot walks on.
    gu = (np.roll(g, -s, axis=1) - np.roll(g, s, axis=1)) / (2 * s * cell)
    gv = (np.roll(g, -s, axis=0) - np.roll(g, s, axis=0)) / (2 * s * cell)
    slope_deg = np.degrees(np.arctan(np.sqrt(gu ** 2 + gv ** 2)))
    # Rolled edges wrap around, which invents a cliff at every border.
    valid = np.ones_like(g, dtype=bool)
    valid[:s, :] = valid[-s:, :] = valid[:, :s] = valid[:, -s:] = False

    r = policy.step_radius
    step = np.zeros_like(g)
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            if du == dv == 0:
                continue
            step = np.maximum(step, np.abs(np.roll(np.roll(g, du, 1), dv, 0) - g))

    # Roughness as the residual from a local plane, approximated by the deviation
    # from a box mean — it separates "sloped but smooth" from "rubble", which a
    # slope figure alone cannot.
    from scipy.ndimage import uniform_filter
    smooth = uniform_filter(g, size=2 * r + 1, mode="nearest")
    rough = np.abs(g - smooth)

    height_span = float(np.nanmax(g) - np.nanmin(g))
    obstacle_thresh = policy.obstacle_height_frac * height_span
    low_enough = valid & (g - np.nanmin(g) < obstacle_thresh)

    # A cell being low is not enough to call it walkable floor. The base of every
    # machine standing on the floor is also low, and the step from the floor up its
    # side then gets reported as terrain relief: on the factory scene this gave a
    # p95 step of 1.76 units and a p95 slope of 88 degrees, which describes the side
    # of a machine, not ground a robot walks on.
    #
    # So walkable floor is low cells that are also not *next to* an obstacle. The
    # dilation radius is the same neighbourhood the step statistic looks over, which
    # is the consistent choice: a cell whose step value is contaminated by an
    # obstacle is exactly a cell within step_radius of one.
    from scipy.ndimage import binary_dilation
    obstacle = valid & ~low_enough
    k = 2 * policy.step_radius + 1
    near_obstacle = binary_dilation(obstacle, np.ones((k, k), dtype=bool))
    floor_like = low_enough & ~near_obstacle

    def dist(a: np.ndarray, mask: np.ndarray) -> dict:
        vals = a[mask & np.isfinite(a)]
        if vals.size == 0:
            return {"samples": 0}
        return {"samples": int(vals.size), "mean": float(vals.mean()),
                **{f"p{p:g}": float(np.percentile(vals, p)) for p in policy.percentiles}}

    return {
        "slope_deg": slope_deg,
        "step": step,
        "roughness": rough,
        "valid": valid,
        "floor_like": floor_like,
        "summary": {
            "cell_size": cell,
            "height_span": height_span,
            "obstacle_threshold": obstacle_thresh,
            "floor_like_fraction": float(floor_like.sum() / max(valid.sum(), 1)),
            # Both numbers, so the effect of excluding obstacle margins is visible.
            "low_cells_fraction": float(low_enough.sum() / max(valid.sum(), 1)),
            "obstacle_fraction": float(obstacle.sum() / max(valid.sum(), 1)),
            "excluded_as_obstacle_margin": float(
                (low_enough & near_obstacle).sum() / max(valid.sum(), 1)),
            "slope_deg_over_floor": dist(slope_deg, floor_like),
            "slope_deg_everywhere": dist(slope_deg, valid),
            "step_over_floor": dist(step, floor_like),
            "step_everywhere": dist(step, valid),
            "roughness_over_floor": dist(rough, floor_like),
            # The curriculum-facing numbers, named so they can be lifted directly.
            "curriculum": {
                "flat_fraction_under_5deg": float(
                    (slope_deg[floor_like] < 5).mean()) if floor_like.any() else 0.0,
                "traversable_fraction_under_20deg": float(
                    (slope_deg[floor_like] < 20).mean()) if floor_like.any() else 0.0,
                "max_step_p95": float(np.percentile(step[floor_like], 95))
                if floor_like.any() else 0.0,
                "max_slope_p95_deg": float(np.percentile(slope_deg[floor_like], 95))
                if floor_like.any() else 0.0,
            },
        },
    }


def clearances(objects: list[dict], room_extent: list[float], up: list[float],
               floor_offset: float, robot_width: float | None = None) -> dict:
    """Gaps between objects, and between each object and the room bounds.

    For a mobile robot this is frequently the binding constraint, and it is not
    implied by any object's own dimensions — a scene can pass every size check and
    have no route through it. Reported as a distribution plus the narrowest gaps,
    since one number cannot say whether a room is navigable.
    """
    if len(objects) < 2:
        return {"pairs": 0, "note": "fewer than two objects; no gaps to measure"}

    centres = np.array([o["centre"] for o in objects], dtype=float)
    # Half-extent of the object's own bounding sphere: a conservative, orientation
    # free proxy. It under-reports gaps for long thin objects, which is the safe
    # direction for a clearance figure.
    radii = np.array([0.5 * float(np.linalg.norm(o["extent"])) for o in objects])

    gaps = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            d = float(np.linalg.norm(centres[i] - centres[j])) - radii[i] - radii[j]
            gaps.append({"a": objects[i]["index"], "b": objects[j]["index"],
                         "gap": d})
    arr = np.array([g["gap"] for g in gaps])
    positive = arr[arr > 0]
    narrowest = sorted(gaps, key=lambda g: g["gap"])[:12]

    out = {
        "pairs": len(gaps),
        "gap_min": float(arr.min()),
        "gap_median": float(np.median(arr)),
        "overlapping_pairs": int((arr <= 0).sum()),
        "positive_gap_p5": float(np.percentile(positive, 5)) if positive.size else None,
        "narrowest": narrowest,
        "note": ("negative gaps mean the bounding spheres intersect, which is "
                 "expected for objects standing close together and is not by "
                 "itself an error"),
    }
    if robot_width is not None:
        passable = int((arr >= robot_width).sum())
        out["robot_width"] = robot_width
        out["pairs_passable"] = passable
        out["pairs_too_narrow"] = int((arr < robot_width).sum())
    return out


def write_heightfield_png(hm: dict, out: str | Path) -> Path:
    """16-bit PNG of the height map, which is what MuJoCo's hfield asset reads."""
    import cv2

    g = hm["grid"]
    lo, hi = float(np.nanmin(g)), float(np.nanmax(g))
    norm = (g - lo) / max(hi - lo, 1e-9)
    # MuJoCo reads the image as elevation in [0,1] and scales it by the hfield
    # size, so the PNG carries shape and the XML carries magnitude. 16-bit because
    # 8-bit quantises a 3-unit room into 12 mm steps, which a step-height statistic
    # would then report as real relief.
    img = np.clip(norm * 65535.0, 0, 65535).astype(np.uint16)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.flipud(img))
    return out


def write_mjcf(out: str | Path, hm: dict, objects: list[dict],
               hfield_png: str, physics: PhysicsDefaults | None = None,
               scale_mode: str = "assumed", robot_spawn_clearance: float = 0.15) -> Path:
    """An MJCF scene: heightfield terrain plus one box per cut object.

    Boxes rather than the visual meshes. A box is convex, has an analytic inertia,
    and costs a physics engine almost nothing; the 18-million-triangle visual mesh
    is for looking at. The per-object convex hulls `decompose` writes are the
    middle option when a box is too coarse, and are referenced in the JSON rather
    than inlined here so this file stays loadable without them.
    """
    physics = physics or PhysicsDefaults()
    up = np.array(hm["axes"]["up"], dtype=float)
    ax = np.array(hm["axes"]["u"], dtype=float)
    ay = np.array(hm["axes"]["v"], dtype=float)

    g = hm["grid"]
    lo, hi = float(np.nanmin(g)), float(np.nanmax(g))
    size_u = hm["nu"] * hm["cell_size"]
    size_v = hm["nv"] * hm["cell_size"]
    elevation = max(hi - lo, 1e-6)

    # World frame: x along u, y along v, z up. Objects are expressed in that frame
    # rather than in DA3's, so the MJCF has gravity along -z as MuJoCo expects and
    # no consumer has to know which axis this reconstruction happened to call up.
    R = np.stack([ax, ay, up])

    body_xml = []
    for o in objects:
        c = R @ np.array(o["centre"], dtype=float)
        c[2] += float(hm.get("floor_offset", 0.0))
        ext = np.array(o["extent"], dtype=float)
        half = np.maximum(ext * 0.5, 1e-4)
        mass = max(physics.object_density * float(np.prod(ext)), 1e-3)
        name = f"obj_{o['index']:04d}_{o.get('label', 'unlabelled')}"
        body_xml.append(f"""
    <body name="{name}" pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}">
      <freejoint/>
      <geom type="box" size="{half[0]:.6f} {half[1]:.6f} {half[2]:.6f}"
            mass="{mass:.6f}" friction="{physics.object_friction} 0.005 0.0001"
            solref="0.01 1" rgba="0.7 0.7 0.75 1"/>
    </body>""".rstrip())

    # Spawn point: the flattest, most open cell, lifted clear of the surface. A
    # robot dropped into geometry is the commonest way a generated scene "explodes"
    # on the first step, and it reads as a physics bug rather than a placement one.
    spawn_u = size_u * 0.5
    spawn_v = size_v * 0.5
    spawn_z = hi + robot_spawn_clearance

    xml = f"""<mujoco model="v3d_scene">
  <!-- Generated from a reconstructed video. Scale mode: {scale_mode}.
       {"Lengths are in DA3's arbitrary units, NOT metres: gravity, mass and" if scale_mode == "assumed" else "Lengths are metric."}
       {"friction below are therefore not physically calibrated, and a policy" if scale_mode == "assumed" else ""}
       {"trained here transfers only in proportion, not in scale." if scale_mode == "assumed" else ""}
       Physics values source: {physics.source} - {physics.note} -->
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <!-- MuJoCo's offscreen framebuffer defaults to 640x480, and mujoco.Renderer
       refuses any larger request with "Image width N > framebuffer width 640".
       The size has to be declared in the model, not passed to the renderer, so a
       scene that will ever be rendered headless has to carry it. -->
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <hfield name="terrain" file="{hfield_png}"
            size="{size_u * 0.5:.6f} {size_v * 0.5:.6f} {elevation:.6f} 0.05"/>
  </asset>

  <worldbody>
    <light pos="{size_u * 0.5:.3f} {size_v * 0.5:.3f} {hi + 3:.3f}" dir="0 0 -1"/>
    <geom name="terrain" type="hfield" hfield="terrain"
          pos="{size_u * 0.5:.6f} {size_v * 0.5:.6f} {lo:.6f}"
          friction="{physics.floor_friction} 0.005 0.0001"
          solref="0.01 1" rgba="0.55 0.55 0.55 1"/>
    <site name="robot_spawn" pos="{spawn_u:.6f} {spawn_v:.6f} {spawn_z:.6f}" size="0.03"/>
{chr(10).join(body_xml)}
  </worldbody>
</mujoco>
"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    return out


def summary(hm: dict, trav: dict, clear: dict, scale_mode: str) -> dict:
    """Everything numeric, without the arrays, so it fits in a report."""
    return {
        "scale_mode": scale_mode,
        "units": "DA3 units, not metres" if scale_mode == "assumed" else "metres",
        "height_map": {
            "cells": [hm["nu"], hm["nv"]],
            "cell_size": hm["cell_size"],
            "hole_fraction": hm["hole_fraction"],
            "height_range": hm["height_range"],
            "surface_samples": hm["samples"],
        },
        "traversability": trav["summary"],
        "clearances": clear,
        "caveats": [
            (f"{hm['hole_fraction']:.1%} of grid cells had no surface sample and were "
             f"filled from the nearest observed cell; slope and step there describe "
             f"the fill, not the room")
            if hm["hole_fraction"] > 0.02 else
            "every grid cell had a surface sample",
            "a height map is 2.5D: overhangs, tables and shelves are invisible to "
            "it, which is why the cut objects stay separate bodies",
            ("slopes and fractions are dimensionless and transfer exactly; step "
             "heights and clearances are in DA3 units and become metric only once "
             "one distance is measured")
            if scale_mode == "assumed" else "lengths are metric",
        ],
    }
