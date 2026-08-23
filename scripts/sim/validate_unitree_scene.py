#!/usr/bin/env python3
"""Check the generated factory scene against the terrain it claims to represent.

Written so each check can fail, and so the two that would otherwise have been
assumptions are measured: which way MuJoCo reads the heightfield image, and whether
the robot is standing on the floor rather than inside a machine.

Two notes on how the checks are posed, both learned by getting them wrong first:

A ray cast straight down from above the origin hits the *robot's own head*, not the
ground — it reported the terrain at 1.718 when the floor is at 0. So every terrain
probe here advances past non-terrain hits before believing an answer.

Comparing ray heights against the source array is only well-posed on locally flat
ground. On the vertical side of a machine column a ray clips the wall of the column
and lands over a metre off, which is a property of the probe and not of the terrain.
The orientation check therefore judges flat cells and reports the cliff tail
separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("validate: mujoco is not installed")

D = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/Users/traoanhuy/source_code/test reconstruct_3d/unitree_mujoco_si/"
         "unitree_robots/vr_h3_2/origin")
XML = D / "scene_factory.xml"
rep = json.loads((D / "scene_factory_report.json").read_text())
hm = np.load(D / "scene_factory_terrain.npy")
cell = rep["cell"]
ox, oy, oz = (rep["offset_applied"][k] for k in ("u", "v", "z"))
fails: list[str] = []

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
TERRAIN = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
print(f"compiled {XML.name}: nq={m.nq} nu={m.nu} ngeom={m.ngeom} nbody={m.nbody} "
      f"nhfield={m.nhfield} ntex={m.ntex} nkey={m.nkey}")


# Which geoms are the scene, as opposed to the robot standing in it. Probes must
# ignore the robot: a ray cast down from the origin otherwise reports the height of
# the robot's head, and a ray through the doorway reports the robot's own shoulder.
SCENE_GEOMS = {
    gid for gid in range(m.ngeom)
    if (lambda n: n is not None and (n == "terrain" or n.startswith("wall_")
                                     or n.startswith("partition")
                                     or n.startswith("obj_")))(
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid))
}
print(f"scene geoms (terrain, walls, objects): {len(SCENE_GEOMS)} of {m.ngeom}; "
      f"the remaining {m.ngeom - len(SCENE_GEOMS)} are the robot and are ignored by "
      f"the probes below")


def ray(pnt, vec, want=None, skips=80):
    """Cast a ray, advancing past every geom that is not scene geometry.

    `want` is a geom id, or None for "any scene geom". The robot is always skipped:
    it is standing in the scene, so it lies across most of the interesting sightlines.
    """
    pnt = np.array(pnt, float)
    vec = np.array(vec, float)
    travelled = 0.0
    for _ in range(skips):
        out = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, pnt, vec, None, 1, -1, out)
        if dist < 0:
            return None, None
        travelled += dist
        gid = int(out[0])
        if (gid == want) if want is not None else (gid in SCENE_GEOMS):
            return travelled, gid
        pnt = pnt + vec * (dist + 1e-4)
        travelled += 1e-4
    return None, None


# --- 1. is the heightfield the terrain, the right way round? -----------------
mujoco.mj_forward(m, d)
from scipy.ndimage import maximum_filter, minimum_filter

rough = maximum_filter(hm, 5) - minimum_filter(hm, 5)
rng = np.random.default_rng(0)
flat_err, all_err = [], []
for _ in range(3000):
    iu = int(rng.integers(30, hm.shape[1] - 30))
    jv = int(rng.integers(30, hm.shape[0] - 30))
    x, y = (iu + 0.5) * cell - ox, (jv + 0.5) * cell - oy
    dist, _ = ray([x, y, 6.0], [0, 0, -1], want=TERRAIN)
    if dist is None:
        continue
    e = (6.0 - dist) - (hm[jv, iu] - oz)
    all_err.append(e)
    if rough[jv, iu] < 0.02:
        flat_err.append(e)
fe, ae = np.abs(np.array(flat_err)), np.abs(np.array(all_err))
rms = float(np.sqrt((fe ** 2).mean())) if fe.size else 9e9
print(f"\n1. heightfield vs the source terrain, over {fe.size} locally flat cells:")
print(f"   RMS {rms*1000:.2f} mm, p99 {np.percentile(fe,99)*1000:.2f} mm, "
      f"worst {fe.max()*1000:.2f} mm")
print(f"   (over all {ae.size} cells the tail reaches {ae.max()*1000:.0f} mm — rays "
      f"clipping the vertical sides of machine columns, not a terrain error)")
if rms > 0.02:
    fails.append(f"the heightfield does not match the terrain on flat ground "
                 f"(RMS {rms*1000:.0f} mm) — the image orientation is wrong")

# --- 2. is the robot standing on the floor? ---------------------------------
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
if kid >= 0:
    mujoco.mj_resetDataKeyframe(m, d, kid)
else:
    mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
pz = float(d.xpos[pelvis][2])
dist, _ = ray([0, 0, 6.0], [0, 0, -1], want=TERRAIN)
ground = 6.0 - dist if dist is not None else float("nan")
print(f"\n2. robot at its home pose: pelvis z={pz:.3f}, terrain under it z={ground:.3f}, "
      f"clearance {pz-ground:.3f}")
if not np.isfinite(ground):
    fails.append("no terrain under the world origin — the robot has nothing to stand on")
elif abs(ground) > 0.05:
    fails.append(f"the floor under the spawn is at {ground:.3f}, not 0 — the world "
                 f"shift is wrong")
elif not 0.4 < pz - ground < 1.4:
    fails.append(f"pelvis clearance {pz-ground:.3f} is not a standing height")

deep = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
         mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2), float(c.dist))
        for c in d.contact[:d.ncon] if c.dist < -0.005]
print(f"   contacts at t=0: {d.ncon}; penetrating over 5 mm: {len(deep)}")
for a, b, dd in deep[:5]:
    print(f"     {a} / {b}  {dd*1000:.1f} mm")
if len(deep) > 2:
    fails.append(f"{len(deep)} contacts penetrate more than 5 mm at t=0")

# --- 3. does the scene stay finite, and does the robot stay on top of it? ---
peak = 0.0
for i in range(1000):
    mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qvel)):
        fails.append(f"qvel became non-finite at step {i}")
        break
    peak = max(peak, float(np.abs(d.qvel).max()))
end_z = float(d.xpos[pelvis][2])
print(f"\n3. 2 s uncontrolled: finite={np.all(np.isfinite(d.qvel))}, peak |qvel|={peak:.2f}, "
      f"pelvis {pz:.3f} -> {end_z:.3f}")
print("   (it collapses because nothing is controlling it; what matters is that it "
      "settles ON the terrain)")
if end_z < ground - 0.15:
    fails.append(f"the robot sank through the terrain (pelvis {end_z:.3f} below "
                 f"ground {ground:.3f})")

# --- 4. can the robot actually walk through the doorway? --------------------
# Back to the home pose: after 2 s uncontrolled the robot is sprawled across the
# floor, and a doorway probe run on that state measures the robot, not the doorway.
if kid >= 0:
    mujoco.mj_resetDataKeyframe(m, d, kid)
else:
    mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
print("\n4. doorways")
for dw in rep["doorways"]:
    print(f"   {dw['id']}: {dw['terrain_blocked_fraction']*100:3.0f}% of the opening's "
          f"cells blocked by terrain, {dw['perp_distance_to_wall']:.3f} from its wall")
part = rep.get("partition_added")
if part:
    axis, pos, along = part["axis"], part["pos"], part["along"]
    dw, dh = rep["doorway_size"]
    ok = True
    for h in (0.25, 0.9, 1.6):
        for off in (-0.35, 0.0, 0.35):
            if axis == "u":
                p0 = [pos - 0.9 - ox, along + off - oy, h]
                vec = [1.0, 0, 0]
            else:
                p0 = [along + off - ox, pos - 0.9 - oy, h]
                vec = [0, 1.0, 0]
            dist, gid = ray(p0, vec)
            name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
                    if gid is not None else None)
            clear = dist is None or dist > 1.8
            if not clear:
                ok = False
                print(f"   partition opening at height {h:.2f}, offset {off:+.2f}: "
                      f"blocked after {dist:.2f} by {name}")
    if ok:
        print(f"   partition opening ({dw:.2f} wide x {dh:.2f} high at "
              f"{'v' if axis=='u' else 'u'}={along:.2f}): clear at 3 heights x 3 offsets")
    else:
        fails.append("the partition doorway is not passable")

    # and the wall it is cut into must actually block on either side of the opening
    blocked_somewhere = False
    for off in (-1.2, 1.2):
        if axis == "u":
            p0 = [pos - 0.9 - ox, along + off - oy, 1.0]
            vec = [1.0, 0, 0]
        else:
            p0 = [along + off - ox, pos - 0.9 - oy, 1.0]
            vec = [0, 1.0, 0]
        dist, gid = ray(p0, vec)
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) if gid is not None else None
        if dist is not None and dist < 1.8 and nm and nm.startswith("partition"):
            blocked_somewhere = True
    print(f"   the partition itself blocks beside the opening: {blocked_somewhere}")
    if not blocked_somewhere:
        fails.append("the partition does not divide the space — it blocks nothing")

# --- 5. is the terrain actually textured? -----------------------------------
tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
mat = int(m.geom_matid[tid])
tex = m.mat_texid[mat]
tex_ids = [int(t) for t in np.atleast_1d(tex) if int(t) >= 0]
print(f"\n5. terrain material {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mat)}"
      f" -> texture ids {tex_ids}")
sizes = [(int(m.tex_width[t]), int(m.tex_height[t])) for t in tex_ids]
print(f"   texture size(s) {sizes}, grid is {rep['grid']}")
if not tex_ids:
    fails.append("the terrain has no texture — it would render as a flat colour")
elif not any(sz == tuple(rep["grid"]) for sz in sizes):
    fails.append(f"the terrain texture {sizes} does not match the height grid "
                 f"{rep['grid']}, so colour will not line up with elevation")

print("\n" + "=" * 72)
if fails:
    print("FAIL")
    for f in fails:
        print("  -", f)
else:
    print("PASS — the scene matches the reconstruction on every check above")
sys.exit(1 if fails else 0)
