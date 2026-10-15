#!/usr/bin/env python3
"""Put the reconstructed factory, and a Unitree robot standing in it, into MuJoCo.

Reads what `decompose`/`terrain` produced from the video (`scene_layout.json`, the
height map, the top-down colour texture) and writes an MJCF alongside the robot so
`<include>` resolves, in the same shape `robot_rl_scene.xml` already uses.

Four things here are decisions rather than mechanics, and each was forced by a
measurement rather than chosen for tidiness.

**The terrain carries the machines; the objects are not also bodies.** The height map
is the reconstructed surface, and 97% of its tall interior cells sit under a detected
object's box — the machines *are* in the terrain. The previous export added 43 object
boxes on top of that, which put every box inside solid ground: 63 overlapping pairs
and peak |qvel| 48.8, a scene that never settled. Representing each machine once
fixes that at the root. `--objects-as-bodies` switches to the other consistent
choice — carve the machines out of the terrain and add them as separate hull bodies —
for manipulation work where they must move.

**The walls are boxes, and they are cut out of the terrain first.** A heightfield
cannot have a hole in it, so a doorway is impossible while the wall is terrain. The
four wall bands are filled down to floor level and replaced by explicit slabs, which
can then have openings.

**Where the openings go comes from the data; how big they are does not.** The three
door instances project onto three different reconstructed walls — one within 0.002 of
its wall — so their *position* is well evidenced. Their extents are not: the same
lift3d pass that produced them under-segmented the floor badly enough to fail Stage J,
and one "door" is a 0.42 fragment. So the opening is placed where the door is and
sized to be robot-passable, and the file says so.

**Scale is a parameter whose default is DA3's own estimate.** DA3 is a metric depth
model, so 2.307 is its estimate of this ceiling, not an arbitrary number — but it is
unverified. Default 1.0 m/unit means "trust that estimate". One tape measurement of
anything in the footage sets `--metres-per-unit` and rescales every length, mass and
force here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def obb_footprint_mask(objects, ax, ay, u0, v0, cell, shape) -> np.ndarray:
    """Which grid cells each object's oriented box covers, seen from above.

    The box is oriented, so its shadow is a hexagon, not a rectangle. Rasterising the
    axis-aligned bound of the corners instead over-claims badly for a rotated machine
    — it read 79% of the room as occupied against a measured obstacle fraction of
    45%. This projects the eight corners, takes their convex hull, and tests cells
    against the hull's own half-planes.
    """
    from scipy.spatial import ConvexHull, QhullError

    nv, nu = shape
    mask = np.zeros(shape, bool)
    signs = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5)
                      for sz in (-.5, .5)])
    for o in objects:
        c = np.array(o["centre"], float)
        e = np.array(o["extent"], float)
        A = np.array(o["obb_axes"], float)
        pts = c + (signs * e) @ A
        uu = pts @ ax - u0
        vv = pts @ ay - v0
        poly = np.column_stack([uu, vv])
        i0 = max(int(np.floor(uu.min() / cell)), 0)
        i1 = min(int(np.ceil(uu.max() / cell)) + 1, nu)
        j0 = max(int(np.floor(vv.min() / cell)), 0)
        j1 = min(int(np.ceil(vv.max() / cell)) + 1, nv)
        if i1 <= i0 or j1 <= j0:
            continue
        try:
            hull = ConvexHull(poly)
        except (QhullError, ValueError):
            mask[j0:j1, i0:i1] = True
            continue
        gu = (np.arange(i0, i1) + 0.5) * cell
        gv = (np.arange(j0, j1) + 0.5) * cell
        GU, GV = np.meshgrid(gu, gv)
        P = np.column_stack([GU.ravel(), GV.ravel()])
        inside = np.all(P @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9,
                        axis=1)
        mask[j0:j1, i0:i1] |= inside.reshape(j1 - j0, i1 - i0)
    return mask


def fill_from_floor(h: np.ndarray, drop: np.ndarray, tol: float = 0.15):
    """Replace masked cells with the nearest *floor* height, not the nearest height.

    Nearest-neighbour fill is the obvious choice and it is wrong here: a wall band
    removed next to a machine gets filled with the machine's height, leaving a ledge
    running along the wall that the robot then has to climb. Filling from floor-like
    cells only gives a continuous floor.

    The floor under a wall or a machine was never seen by the camera, so this is
    interpolation. The caller reports what fraction of the map it covers.
    """
    from scipy.ndimage import distance_transform_edt

    if not drop.any():
        return h.copy(), 0.0
    keep = ~drop
    if not keep.any():
        return np.zeros_like(h), 0.0
    floor_level = float(np.percentile(h[keep], 10))
    src = keep & (h < floor_level + tol)
    if not src.any():
        out = h.copy()
        out[drop] = floor_level
        return out, floor_level
    _, idx = distance_transform_edt(~src, return_indices=True)
    out = h.copy()
    out[drop] = h[tuple(idx)][drop]
    return out, floor_level


def cluster_walls(planes, ax, ay, up, u0, v0, extent_u, extent_v, tol=0.35):
    """Collapse the detected wall planes into distinct room edges.

    Seven planes were found for four walls — RANSAC fits the same surface more than
    once when it is large. Each plane is `n.x + d = 0`; its position along its own
    normal, expressed in the grid frame, is what identifies it.
    """
    found = []
    for i, p in enumerate(planes):
        if not str(p.get("role", "")).startswith("wall"):
            continue
        n = np.array(p["normal"], float)
        d = float(p["offset"])
        nu_, nv_ = float(n @ ax), float(n @ ay)
        if abs(nu_) >= abs(nv_):
            if abs(nu_) < 1e-6:
                continue
            found.append(("u", -d / nu_ - u0, i, p.get("span_frac", 0.0)))
        else:
            if abs(nv_) < 1e-6:
                continue
            found.append(("v", -d / nv_ - v0, i, p.get("span_frac", 0.0)))

    clusters = []
    for axis, pos, idx, span in sorted(found, key=lambda t: (t[0], t[1])):
        for cl in clusters:
            if cl["axis"] == axis and abs(cl["pos"] - pos) <= tol:
                cl["members"].append(idx)
                cl["spans"].append(span)
                cl["pos"] = float(np.mean([cl["pos"], pos]))
                break
        else:
            clusters.append({"axis": axis, "pos": float(pos), "members": [idx],
                             "spans": [span]})
    # Only edges, not interior partitions the plane fitter happened to find.
    keep = []
    for cl in clusters:
        lim = extent_u if cl["axis"] == "u" else extent_v
        if -0.6 <= cl["pos"] <= lim + 0.6:
            keep.append(cl)
    return keep


def wall_band_mask(walls, h, cell, shape, thresh=0.3, cap=0.45):
    """How wide each wall actually is in the terrain, measured rather than assumed.

    A fixed +/-9 cell band left 0.24 units of wall standing in front of two of the
    three doorways, which made those openings decorative. This walks outward from the
    wall line along a *median* profile — median so a machine parked against the wall
    cannot drag the band across the room — and stops where the surface drops to floor.
    """
    nv, nu = shape
    m = np.zeros(shape, bool)
    cap_cells = int(round(cap / cell))
    spans = []
    for w in walls:
        c = int(round(w["pos"] / cell))
        prof = np.median(h, axis=0) if w["axis"] == "u" else np.median(h, axis=1)
        n = prof.size
        c = int(np.clip(c, 0, n - 1))
        lo = hi = c
        while lo - 1 >= 0 and c - (lo - 1) <= cap_cells and prof[lo - 1] > thresh:
            lo -= 1
        while hi + 1 < n and (hi + 1) - c <= cap_cells and prof[hi + 1] > thresh:
            hi += 1
        # A wall sitting at the very edge of the grid has nothing outside it to walk
        # through, so always clear to the boundary on that side.
        if c <= cap_cells:
            lo = 0
        if c >= n - 1 - cap_cells:
            hi = n - 1
        lo = max(lo - 2, 0)
        hi = min(hi + 2, n - 1)
        spans.append((w["axis"], w["pos"], (hi - lo + 1) * cell))
        if w["axis"] == "u":
            m[:, lo:hi + 1] = True
        else:
            m[lo:hi + 1, :] = True
    return m, spans


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_unitree = os.environ.get("UNITREE_DIR")
    if not default_unitree:
        for candidate in (
            Path.home() / "source_code" / "test reconstruct_3d" / "unitree_mujoco_si",
            Path("../test reconstruct_3d/unitree_mujoco_si"),
            Path("../unitree_mujoco_si"),
        ):
            if candidate.is_dir():
                default_unitree = str(candidate)
                break
    ap.add_argument("--unitree", type=Path,
                    default=Path(default_unitree or "../unitree_mujoco_si"))
    ap.add_argument("--robot", default=os.environ.get("ROBOT_NAME", "robot_model"))
    ap.add_argument("--metres-per-unit", type=float, default=1.0)
    ap.add_argument("--objects-as-bodies", action="store_true",
                    help="carve machines out of the terrain and add them as hull "
                         "bodies (for manipulation; heavier and less stable)")
    ap.add_argument("--no-partition", action="store_true",
                    help="omit the added interior dividing wall")
    ap.add_argument("--door-width", type=float, default=1.20)
    ap.add_argument("--door-height", type=float, default=2.05)
    ap.add_argument("--out-name", default="scene_factory.xml")
    args = ap.parse_args()

    A = args.assets
    S = float(args.metres_per_unit)
    lay = json.loads((A / "scene_layout.json").read_text())
    hm = np.load(A / "terrain" / "height_map.npy").astype(np.float64)

    b = lay["basis"]
    ax, ay, up = (np.array(b["u"], float), np.array(b["v"], float),
                  np.array(b["up"], float))
    u0, v0 = lay["origin_uv"]
    cell_u = float(lay["cell"])
    fo = float(lay["floor_offset"])
    room_h = float(lay["room_height"])
    nv_r, nu_r = hm.shape
    ext_u, ext_v = nu_r * cell_u, nv_r * cell_u

    robot_dir = args.unitree / "unitree_robots" / args.robot / "origin"
    if not robot_dir.is_dir():
        raise SystemExit(f"no robot directory at {robot_dir}. Specify --unitree or set UNITREE_DIR.")
    inc = [f for f in (f"{args.robot}_rl.xml", f"{args.robot}_sensors.xml",
                       f"{args.robot}_actuators.xml") if (robot_dir / f).exists()]
    if not inc:
        raise SystemExit(f"no robot xml found in {robot_dir}")

    print("=" * 74)
    print(f"reconstructed factory -> MuJoCo, with {args.robot}")
    print("=" * 74)
    print(f"grid {nu_r} x {nv_r} cells of {cell_u:.5f}  -> room {ext_u:.2f} x "
          f"{ext_v:.2f} x {room_h:.2f} (DA3 units)")

    # --- walls ---------------------------------------------------------------
    walls = cluster_walls(lay["shell_planes"], ax, ay, up, u0, v0, ext_u, ext_v)
    n_planes = sum(1 for p in lay["shell_planes"]
                   if str(p.get("role", "")).startswith("wall"))
    print(f"\nwalls: {n_planes} detected planes -> {len(walls)} distinct edges")
    for w in walls:
        print(f"  {w['axis']} = {w['pos']:.3f}   from planes {w['members']} "
              f"(span_frac {max(w['spans']):.2f})")

    u_walls = sorted([w for w in walls if w["axis"] == "u"], key=lambda w: w["pos"])
    v_walls = sorted([w for w in walls if w["axis"] == "v"], key=lambda w: w["pos"])
    u_lo = u_walls[0]["pos"] if u_walls else 0.0
    u_hi = u_walls[-1]["pos"] if len(u_walls) > 1 else ext_u
    v_lo = v_walls[0]["pos"] if v_walls else 0.0
    v_hi = v_walls[-1]["pos"] if len(v_walls) > 1 else ext_v
    print(f"  interior: u [{u_lo:.2f}, {u_hi:.2f}]  v [{v_lo:.2f}, {v_hi:.2f}]")

    # --- doors: position from the data, size chosen to be passable -----------
    doors = []
    for d in lay["doors"]:
        c = np.array(d["center"], float)
        du = float(c @ ax) - u0
        dv = float(c @ ay) - v0
        if max(d["size"]) < 0.9:
            print(f"  skip {d['id']}: largest extent {max(d['size']):.2f} — a fragment")
            continue
        best = min(walls, key=lambda w: abs((du if w["axis"] == "u" else dv) - w["pos"]))
        perp = abs((du if best["axis"] == "u" else dv) - best["pos"])
        along = dv if best["axis"] == "u" else du
        doors.append({"id": d["id"], "wall": id(best), "along": along, "perp": perp,
                      "wall_ref": best})
        print(f"  {d['id']}: u={du:+.2f} v={dv:+.2f} -> opening in "
              f"{best['axis']}={best['pos']:.2f} wall at {best['axis'] == 'u' and 'v' or 'u'}"
              f"={along:.2f} (perp. distance {perp:.3f})")
    # One opening per wall: the best-evidenced door wins.
    per_wall: dict[int, dict] = {}
    for d in doors:
        if d["wall"] not in per_wall or d["perp"] < per_wall[d["wall"]]["perp"]:
            per_wall[d["wall"]] = d

    # --- terrain: remove whatever is represented some other way --------------
    drop, wall_spans = wall_band_mask(walls, hm, cell_u, hm.shape)
    wall_frac = drop.mean()
    for axis, pos, width in wall_spans:
        print(f"  wall {axis}={pos:.2f}: terrain is above floor for {width:.2f} units "
              f"across — that whole band is removed")
    obj_mask = obb_footprint_mask(lay["objects"], ax, ay, u0, v0, cell_u, hm.shape)
    print(f"\nterrain: wall bands {wall_frac*100:.1f}% of cells removed so the walls "
          f"can have doorways")
    print(f"         object boxes cover {obj_mask.mean()*100:.1f}% of cells "
          f"(measured obstacle fraction was 45.5%)")
    if args.objects_as_bodies:
        drop |= obj_mask
        print(f"         machines carved out too -> {drop.mean()*100:.1f}% of the "
              f"terrain becomes interpolated floor")
    else:
        print("         machines LEFT IN the terrain (represented once, not twice)")
    hm_t, floor_level = fill_from_floor(hm, drop)
    print(f"         floor level {floor_level:+.3f}; {drop.mean()*100:.1f}% of the map is interpolated floor")

    # doorways must also be clear of terrain, or the opening is decorative
    for d in per_wall.values():
        w = d["wall_ref"]
        ci = int(round(w["pos"] / cell_u))
        aj = int(round(d["along"] / cell_u))
        hw = int(round(args.door_width / 2 / cell_u))
        if w["axis"] == "u":
            patch = hm_t[max(aj - hw, 0):aj + hw, max(ci - 12, 0):ci + 12]
        else:
            patch = hm_t[max(ci - 12, 0):ci + 12, max(aj - hw, 0):aj + hw]
        blocked = float((patch > 0.35).mean()) if patch.size else 1.0
        d["blocked_frac"] = blocked
        state = "clear" if blocked < 0.25 else "OBSTRUCTED by terrain"
        print(f"         doorway from {d['id']}: {state} "
              f"({blocked*100:.0f}% of its cells above 0.35)")

    lo, hi = float(hm_t.min()), float(hm_t.max())
    elev = max(hi - lo, 1e-6)

    # --- an added partition, because the request was to divide the space -----
    partition = None
    if not args.no_partition:
        obst = hm_t > 0.35
        half_w = max(int(round(args.door_width / 2 / cell_u)), 1)
        reach = max(int(round(0.85 / cell_u)), 1)      # corridor depth either side

        def corridor_cost(axis, c, a):
            """How obstructed a door-width corridor through (axis=c, along=a) is.

            Scoring only the wall line is not enough: the first partition this picked
            crossed 0% occupied cells and still had a machine 0.12 beyond the opening,
            so the doorway led straight into it. This looks through the opening, both
            sides, and reports the worse side -- a doorway is only passable if both
            approaches are.
            """
            a0, a1 = max(a - half_w, 0), a + half_w
            near = (obst[a0:a1, max(c - reach, 0):c] if axis == "u"
                    else obst[max(c - reach, 0):c, a0:a1])
            far = (obst[a0:a1, c:c + reach] if axis == "u"
                   else obst[c:c + reach, a0:a1])
            if near.size == 0 or far.size == 0:
                return 1.0
            return max(float(near.mean()), float(far.mean()))

        best = None
        for axis in ("u", "v"):
            n = nu_r if axis == "u" else nv_r
            a_lo, a_hi = ((v_lo, v_hi) if axis == "u" else (u_lo, u_hi))
            a0 = int(a_lo / cell_u) + half_w + 2
            a1 = int(a_hi / cell_u) - half_w - 2
            if a1 <= a0:
                continue
            for c in range(int(0.30 * n), int(0.70 * n), 2):
                line = obst[:, c] if axis == "u" else obst[c, :]
                seg = line[int(a_lo / cell_u):int(a_hi / cell_u)]
                if seg.size == 0:
                    continue
                line_cost = float(seg.mean())
                if best is not None and line_cost > best[0] + 0.25:
                    continue
                cands = range(a0, a1, 3)
                door = min(cands, key=lambda a: corridor_cost(axis, c, a))
                cc = corridor_cost(axis, c, door)
                cost = line_cost + 2.0 * cc
                if best is None or cost < best[0]:
                    best = (cost, axis, c, door, line_cost, cc)
        if best:
            cost, axis, c, door_a, line_cost, cc = best
            partition = {"axis": axis, "pos": c * cell_u,
                         "along": (door_a + 0.5) * cell_u,
                         "line_blocked": line_cost, "corridor_blocked": cc}
            other = "v" if axis == "u" else "u"
            print(f"\npartition (ADDED, not reconstructed): {axis} = "
                  f"{partition['pos']:.2f}, crosses {line_cost*100:.0f}% occupied "
                  f"cells; doorway at {other}={partition['along']:.2f} with "
                  f"{cc*100:.0f}% of its approach corridor obstructed")
            if cc > 0.2:
                print("  WARNING: that corridor is partly blocked — the robot may not "
                      "get through this doorway")

    # --- robot spawn ---------------------------------------------------------
    from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

    # Anything the robot cannot stand on: machines, and the ground behind a wall.
    # Removing the wall bands from the terrain made the strip *outside* each wall the
    # flattest, most open surface on the map, so without an interior bound the
    # best-scoring cell is behind a wall, outside the building.
    tall = hm_t > 0.25
    outside = np.ones(hm_t.shape, bool)
    j0, j1 = max(int(np.ceil(v_lo / cell_u)), 0), int(np.floor(v_hi / cell_u))
    i0, i1 = max(int(np.ceil(u_lo / cell_u)), 0), int(np.floor(u_hi / cell_u))
    outside[j0:j1, i0:i1] = False
    blocked = tall | outside
    if partition:
        c = int(round(partition["pos"] / cell_u))
        half = int(round(0.20 / cell_u))
        if partition["axis"] == "u":
            blocked[:, max(c - half, 0):c + half] = True
        else:
            blocked[max(c - half, 0):c + half, :] = True
    openness = distance_transform_edt(~blocked) * cell_u
    k = 25
    flat = (maximum_filter(hm_t, k) - minimum_filter(hm_t, k)) < 0.06
    score = np.where(flat & ~blocked, openness, -1.0)
    if score.max() <= 0:
        raise SystemExit("no flat open cell inside the room for the robot")
    jv, iu = np.unravel_index(int(np.argmax(score)), score.shape)
    sx, sy = (iu + 0.5) * cell_u, (jv + 0.5) * cell_u
    ground = float(hm_t[jv, iu])
    where = "n/a"
    if partition:
        coord = sx if partition["axis"] == "u" else sy
        where = "beyond" if coord > partition["pos"] else "before"
    print(f"\nspawn: cell ({iu},{jv}) -> u={sx:.3f} v={sy:.3f}, ground {ground:.3f}, "
          f"clear radius {openness[jv, iu]:.2f}, {where} the partition")

    # Put the world origin at that cell. The robot's own `home` keyframe stands it at
    # (0, 0, 0.841), so shifting the scene rather than rewriting the pose lands the
    # robot on the chosen floor whichever reset path the simulator takes -- including
    # a plain qpos0 reset, which a rewritten keyframe would not survive.
    ox, oy, oz = sx, sy, ground

    key_xml = ""
    canon = robot_dir / f"{args.robot}_rl_scene.xml"
    if canon.exists():
        try:
            import mujoco
            mk = mujoco.MjModel.from_xml_path(str(canon))
            kid = mujoco.mj_name2id(mk, mujoco.mjtObj.mjOBJ_KEY, "home")
            if kid >= 0:
                q = np.array(mk.key_qpos[kid], float).copy()
                q[0] = q[1] = 0.0
                q[2] = q[2] * S
                key_xml = ("  <keyframe>\n    <key name=\"home\" qpos=\""
                           + " ".join(f"{x:g}" for x in q) + "\"/>\n  </keyframe>\n")
                print(f"       reusing the robot's own home pose: {len(q)} values, "
                      f"pelvis {q[2]:.3f} above the floor")
        except Exception as exc:                            # noqa: BLE001
            print(f"       could not reuse a home pose: {type(exc).__name__} {exc}")

    # --- write the terrain assets -------------------------------------------
    from PIL import Image

    norm = np.clip((hm_t - lo) / elev, 0.0, 1.0)
    # MuJoCo reads image row 0 as the LAST hfield row, so flip once here. The
    # validator re-derives this from the compiled model rather than trusting it.
    png = Image.fromarray((norm[::-1] * 65535.0).round().astype(np.uint16))
    tex = Image.open(A / "terrain_texture.png").convert("RGB")
    if tex.size != (nu_r, nv_r):
        print(f"  note: texture {tex.size} != grid ({nu_r},{nv_r}); resampling")
        tex = tex.resize((nu_r, nv_r), Image.LANCZOS)
    tex = tex.transpose(Image.FLIP_TOP_BOTTOM)
    # The robot file sets `meshdir="../meshes/"`, which is where MuJoCo looks for a
    # heightfield; textures fall back to the model's own directory. Rather than depend
    # on which rule applies to which asset, write both files to both places.
    asset_dirs = [robot_dir]
    meshdir = robot_dir.parent / "meshes"
    if meshdir.is_dir():
        asset_dirs.append(meshdir)
    for dd in asset_dirs:
        png.save(dd / "factory_hfield.png")
        tex.save(dd / "factory_texture.png")
    print(f"assets: written to {', '.join(str(d.name) for d in asset_dirs)}/")

    # --- objects, only when they are bodies ---------------------------------
    obj_xml, mesh_xml, n_obj = [], [], 0
    if args.objects_as_bodies:
        import trimesh
        mdir = (robot_dir.parent / "meshes" if (robot_dir.parent / "meshes").is_dir()
                else robot_dir) / "factory_hulls"
        mdir.mkdir(exist_ok=True)
        for o in lay["objects"]:
            rel = o.get("collision_mesh")
            src = A / rel if rel else None
            if not src or not src.exists():
                continue
            try:
                mesh = trimesh.load(str(src), process=False)
                if mesh.is_empty or len(mesh.faces) == 0:
                    continue
                mesh.apply_scale(S)
                name = f"hull_{o['index']:04d}"
                mesh.export(str(mdir / f"{name}.stl"))
            except Exception as exc:                       # noqa: BLE001
                print(f"  object {o['index']}: {type(exc).__name__} {exc}")
                continue
            mesh_xml.append(f'<mesh name="{name}" file="factory_hulls/{name}.stl"/>')
            # The hull vertices are already in world coordinates, so the body sits at
            # the origin of the scene frame and the mesh carries the placement.
            obj_xml.append(
                f'<body name="obj_{o["index"]:04d}" '
                f'pos="{-ox*S:.4f} {-oy*S:.4f} {-oz*S:.4f}">\n'
                f'        <geom type="mesh" mesh="{name}" material="objmat" '
                f'friction="0.7 0.005 0.0001"/>\n      </body>')
            n_obj += 1
        print(f"objects: {n_obj} hull bodies written to {mdir}")

    # --- wall geometry -------------------------------------------------------
    T = 0.06
    geoms = []

    def slab(name, cu, cv, hu, hv, hz=None, cz=None, mat="wallmat"):
        hz = room_h / 2 if hz is None else hz
        cz = hz if cz is None else cz
        geoms.append(f'<geom name="{name}" type="box" '
                     f'pos="{(cu-ox)*S:.4f} {(cv-oy)*S:.4f} {(cz-oz)*S:.4f}" '
                     f'size="{max(hu,1e-3)*S:.4f} {max(hv,1e-3)*S:.4f} '
                     f'{max(hz,1e-3)*S:.4f}" material="{mat}"/>')

    def wall_with_optional_door(tag, axis, pos, lo_a, hi_a, door):
        """A wall slab, split into two segments and a lintel when it has a doorway."""
        if door is None:
            if axis == "u":
                slab(tag, pos, (lo_a + hi_a) / 2, T, (hi_a - lo_a) / 2)
            else:
                slab(tag, (lo_a + hi_a) / 2, pos, (hi_a - lo_a) / 2, T)
            return 0
        w2 = args.door_width / 2
        dh = min(args.door_height, room_h * 0.95)
        a = float(np.clip(door["along"], lo_a + w2 + 0.05, hi_a - w2 - 0.05))
        segs = [(lo_a, a - w2), (a + w2, hi_a)]
        for k, (s0, s1) in enumerate(segs):
            if s1 - s0 <= 0.02:
                continue
            if axis == "u":
                slab(f"{tag}_seg{k}", pos, (s0 + s1) / 2, T, (s1 - s0) / 2)
            else:
                slab(f"{tag}_seg{k}", (s0 + s1) / 2, pos, (s1 - s0) / 2, T)
        if room_h - dh > 0.04:
            hz = (room_h - dh) / 2
            if axis == "u":
                slab(f"{tag}_lintel", pos, a, T, w2, hz=hz, cz=dh + hz)
            else:
                slab(f"{tag}_lintel", a, pos, w2, T, hz=hz, cz=dh + hz)
        return 1

    n_doors = 0
    for i, w in enumerate(walls):
        door = per_wall.get(id(w))
        lo_a, hi_a = ((v_lo, v_hi) if w["axis"] == "u" else (u_lo, u_hi))
        n_doors += wall_with_optional_door(f"wall_{w['axis']}{i}", w["axis"],
                                           w["pos"], lo_a, hi_a, door)
    if partition:
        lo_a, hi_a = ((v_lo, v_hi) if partition["axis"] == "u" else (u_lo, u_hi))
        wall_with_optional_door("partition", partition["axis"], partition["pos"],
                                lo_a, hi_a, partition)
        n_doors += 1

    unit_note = (f"{args.metres_per_unit} m per DA3 unit"
                 if abs(S - 1.0) > 1e-9 else "DA3 units, taken as metres")
    body_note = (f"{n_obj} machines are separate hull bodies and the terrain was "
                 f"carved to match" if args.objects_as_bodies else
                 "machines are part of the terrain, so nothing is represented twice")

    xml = f"""<mujoco model="{args.robot}_factory">
  <!-- The small factory, reconstructed from one handheld video, with {args.robot}
       standing in it.

       UNITS: {unit_note}. DA3 is a metric depth model, so this is its own estimate of
       scale ({room_h:.3f} unit ceiling) and it is unverified. Re-run the generator with
       --metres-per-unit <measured_ceiling / {room_h:.3f}> to correct every length,
       mass and force at once.

       TERRAIN: heightfield from the reconstructed surface, {nu_r}x{nv_r} cells of
       {cell_u:.5f}, coloured by the scene's own top-down texture (not a height ramp).
       {body_note}.

       WALLS: {len(walls)} edges recovered from the mesh's RANSAC planes. {n_doors}
       doorway(s). Opening POSITIONS come from detected door instances; their widths
       are set to {args.door_width} x {args.door_height} because the semantic extents
       from that pass are not trustworthy (one 'door' was a 0.42 fragment).
       Geoms named partition* are ADDED to divide the space and are not reconstructed.
  -->
  {chr(10).join('  <include file="' + f + '"/>' for f in inc)}

  <option timestep="0.002" integrator="implicitfast"/>
  <statistic center="{(ext_u/2-ox)*S:.3f} {(ext_v/2-oy)*S:.3f} {(room_h/2-oz)*S:.3f}"
             extent="{max(ext_u, ext_v)*S:.3f}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.18 0.22 1"/>
    <global azimuth="140" elevation="-25" offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.22 0.28 0.36"
             rgb2="0.04 0.05 0.07" width="512" height="3072"/>
    <texture type="2d" name="factory_tex" file="factory_texture.png"/>
    <material name="terrainmat" texture="factory_tex" texrepeat="1 1"
              texuniform="false" reflectance="0.03" specular="0.15" shininess="0.1"/>
    <material name="wallmat" rgba="0.66 0.64 0.60 1" reflectance="0.02"/>
    <material name="objmat" rgba="0.52 0.56 0.62 1" reflectance="0.04"/>
    <hfield name="factory" file="factory_hfield.png"
            size="{ext_u/2*S:.6f} {ext_v/2*S:.6f} {elev*S:.6f} {max(0.05*S, 0.01):.6f}"/>
    {chr(10).join('    ' + m for m in mesh_xml)}
  </asset>

  <worldbody>
    <light pos="{(ext_u*0.5-ox)*S:.3f} {(ext_v*0.5-oy)*S:.3f} {(room_h+1.2-oz)*S:.3f}"
           dir="0 0 -1" directional="true" diffuse="0.45 0.45 0.45"/>
    <light pos="{(ext_u*0.25-ox)*S:.3f} {(ext_v*0.7-oy)*S:.3f} {(room_h*0.9-oz)*S:.3f}"
           dir="0.3 -0.3 -1" diffuse="0.25 0.25 0.25"/>

    <geom name="terrain" type="hfield" hfield="factory"
          pos="{(ext_u/2-ox)*S:.4f} {(ext_v/2-oy)*S:.4f} {(lo-oz)*S:.4f}"
          material="terrainmat" friction="0.9 0.005 0.0001"
          solref="0.01 1" condim="3"/>

      {chr(10).join('      ' + g for g in geoms)}

      {chr(10).join('      ' + o for o in obj_xml)}

    <site name="spawn" pos="0 0 {0.02*S:.4f}" size="0.05"
          rgba="0 1 0 0.5"/>
  </worldbody>
{key_xml}</mujoco>
"""
    out = robot_dir / args.out_name
    out.write_text(xml)

    report = {
        "scale_mode": "assumed", "metres_per_unit": args.metres_per_unit,
        "units": "DA3 units" if abs(S - 1) < 1e-9 else "metres (rescaled)",
        "grid": [nu_r, nv_r], "cell": cell_u, "room_height": room_h,
        "wall_planes_detected": n_planes, "wall_edges": len(walls),
        "walls": [{"axis": w["axis"], "pos": w["pos"], "planes": w["members"]}
                  for w in walls],
        "doorways": [{"id": d["id"], "along": d["along"],
                      "perp_distance_to_wall": d["perp"],
                      "terrain_blocked_fraction": d.get("blocked_frac")}
                     for d in per_wall.values()],
        "partition_added": partition,
        "terrain_interpolated_fraction": float(drop.mean()),
        "object_footprint_fraction": float(obj_mask.mean()),
        "objects_as_bodies": bool(args.objects_as_bodies), "object_bodies": n_obj,
        "world_origin_at_grid": {"u": sx, "v": sy, "z": ground},
        "spawn": {"u": sx, "v": sy, "ground": ground,
                  "clear_radius": float(openness[jv, iu])},
        "hfield": {"z_min": lo, "z_max": hi, "elevation": elev},
        "offset_applied": {"u": ox, "v": oy, "z": oz},
        "doorway_size": [args.door_width, args.door_height],
    }
    (robot_dir / "scene_factory_report.json").write_text(json.dumps(report, indent=2))
    # The terrain actually written, so a validator can check the heightfield's
    # orientation against it instead of trusting the flip applied above.
    np.save(robot_dir / "scene_factory_terrain.npy", hm_t)

    print(f"\nwrote {out}")
    print(f"      {robot_dir/'factory_hfield.png'}")
    print(f"      {robot_dir/'factory_texture.png'}")
    print(f"      {robot_dir/'scene_factory_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
