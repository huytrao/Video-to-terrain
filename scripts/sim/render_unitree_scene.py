#!/usr/bin/env python3
"""Render the factory scene so the result can be seen rather than trusted.

Stills for the things that were asked for specifically — the coloured terrain, the
wall with its doorway, the robot standing in the room — and an orbit mp4.

Every frame is captioned with the unit, because these lengths are DA3's estimate and
a video is exactly where that caveat gets lost.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]

default_scene = os.environ.get("UNITREE_SCENE_DIR")
if not default_scene:
    for candidate in (
        REPO_ROOT / "outputs" / "robot_sim" / "mujoco",
        REPO_ROOT / "outputs-full-factory" / "robot_sim" / "mujoco",
    ):
        if candidate.is_dir():
            default_scene = str(candidate)
            break

parser = argparse.ArgumentParser(description="Render factory scene in MuJoCo")
parser.add_argument(
    "--scene",
    type=Path,
    default=Path(default_scene or REPO_ROOT / "outputs" / "robot_sim" / "mujoco"),
    help="Path to origin directory containing scene_factory.xml and scene_factory_report.json",
)
parser.add_argument(
    "--out",
    type=Path,
    default=REPO_ROOT / "outputs-smallroom" / "robot_sim" / "videos",
    help="Output directory for rendered stills and orbit video",
)
args = parser.parse_args()

D = args.scene
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
if not (D / "scene_factory_report.json").is_file():
    raise SystemExit(f"No scene found at {D}. Specify --scene or set UNITREE_SCENE_DIR.")
rep = json.loads((D / "scene_factory_report.json").read_text())
W, H = 1280, 720

m = mujoco.MjModel.from_xml_path(str(D / "scene_factory.xml"))
d = mujoco.MjData(m)
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
if kid >= 0:
    mujoco.mj_resetDataKeyframe(m, d, kid)
mujoco.mj_forward(m, d)

ox, oy = rep["offset_applied"]["u"], rep["offset_applied"]["v"]
ext_u = rep["grid"][0] * rep["cell"]
ext_v = rep["grid"][1] * rep["cell"]
cu, cv = ext_u / 2 - ox, ext_v / 2 - oy
part = rep.get("partition_added") or {}

# The first top-down render looked as though the texture had holes in it. It does
# not — 0.00% of its texels are black. Those were MuJoCo's hard shadows: the machine
# columns are over two units tall and threw the floor between them into black. Lifting
# the ambient term and dropping shadow casting is a change to the *picture*, not to the
# scene file the simulator loads, and it is what makes the reconstructed colour legible.
m.vis.headlight.ambient[:] = 0.75
m.vis.headlight.diffuse[:] = 0.45
m.light_castshadow[:] = 0

renderer = mujoco.Renderer(m, height=H, width=W)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)


SCENE_GEOMS = {
    gid for gid in range(m.ngeom)
    if (lambda n: n is not None and (n == "terrain" or n.startswith("wall_")
                                     or n.startswith("partition")
                                     or n.startswith("obj_")))(
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid))
}


def to_camera(azim, elev):
    """Unit vector from the subject towards where MuJoCo will put the camera.

    MuJoCo places the camera at `lookat - distance * forward`, so this is -forward.
    Getting that sign wrong is not a small error: it probes the opposite side of the
    room, reports it clear, and the camera is then placed inside whatever was actually
    behind the subject. Two of these stills were renders of the inside of a wall
    before this was fixed.
    """
    a, e = np.radians(azim), np.radians(elev)
    return np.array([-np.cos(e) * np.cos(a), -np.cos(e) * np.sin(a), -np.sin(e)])


def sightline(target, azim, elev):
    """How far a camera can back away along (azim, elev) before it is inside something.

    Guessing an angle put the first robot camera inside a machine column and rendered a
    close-up of nothing. This room is 66% occupied by machines seen from above, so a
    clear view has to be found rather than assumed: cast a ray from the subject out
    along the camera direction and see where the scene stops it.
    """
    vec = to_camera(azim, elev)
    pnt = np.array(target, float) + vec * 0.05
    travelled = 0.0
    for _ in range(40):
        out = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, pnt, vec, None, 1, -1, out)
        if dist < 0:
            return 1e3
        travelled += dist
        if int(out[0]) in SCENE_GEOMS:
            return travelled
        pnt = pnt + vec * (dist + 1e-4)
    return travelled


# The room, in world coordinates, so a camera can be kept inside it. A clear
# sightline is not enough on its own: the clearest bearing from the robot ran straight
# out through a doorway, and "nothing in the way" put the camera outside the building
# looking at the back of a wall.
U = sorted(w["pos"] for w in rep["walls"] if w["axis"] == "u")
V = sorted(w["pos"] for w in rep["walls"] if w["axis"] == "v")
ROOM = ((U[0] - ox, U[-1] - ox) if len(U) > 1 else (-1e3, 1e3),
        (V[0] - oy, V[-1] - oy) if len(V) > 1 else (-1e3, 1e3),
        (0.15, rep["room_height"] - 0.15))


def exit_distance(target, azim, elev):
    """How far along the camera bearing the room's own walls are."""
    vec = to_camera(azim, elev)
    t = 1e3
    for k in range(3):
        lo, hi = ROOM[k]
        if abs(vec[k]) < 1e-9:
            continue
        for bound in (lo, hi):
            tt = (bound - target[k]) / vec[k]
            if tt > 0.05:
                t = min(t, tt)
    return t


def best_view(target, elev, want, step=10):
    """The bearing with the most room, and a distance that stays inside the room."""
    best = (-1.0, 0.0, 0.0)
    for az in range(0, 360, step):
        clear = min(sightline(target, az, elev), exit_distance(target, az, elev))
        if clear > best[0]:
            best = (clear, float(az), clear)
    clear, az, _ = best
    return az, max(min(want, clear - 0.35), 0.8), clear


def shot(name, lookat, dist, azim, elev, caption):
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = azim
    cam.elevation = elev
    renderer.update_scene(d, cam)
    img = renderer.render()
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    bar = 30
    dr.rectangle([0, H - bar, W, H], fill=(12, 12, 14))
    dr.text((10, H - bar + 9), caption, fill=(235, 235, 235))
    p = OUT / f"{name}.png"
    im.save(p)
    print(f"  {p.name}  {p.stat().st_size/1e3:.0f} kB")
    return img


UNIT = ("lengths are DA3 units (its own metric estimate, unverified) — "
        "scale_mode=assumed")
print("stills:")
shot("factory_topdown", [cu, cv, 0.4], 8.2, 90, -89,
     f"reconstructed factory, top-down — terrain coloured by the scene's own "
     f"texture. {UNIT}")
az_r, dist_r, clear_r = best_view([0, 0, 0.9], -12, 3.2)
print(f"  (robot camera: azimuth {az_r:.0f} has {clear_r:.2f} clear, "
      f"standing off {dist_r:.2f})")
shot("factory_robot", [0, 0, 0.7], dist_r, az_r, -12,
     f"robot standing on the reconstructed floor at its home pose. {UNIT}")
if part:
    axis, pos, along = part["axis"], part["pos"], part["along"]
    lx = (pos - ox) if axis == "u" else (along - ox)
    ly = (along - oy) if axis == "u" else (pos - oy)
    # The camera has to stand on one side of the partition and look at the opening.
    # The first attempt put it *inside* the wall, which rendered a flat grey field --
    # azimuth 0 for a wall of constant u places it on the +u side, looking back along
    # -u through the doorway.
    dw, dh = rep["doorway_size"]
    for tag, az in (("factory_doorway", 0 if axis == "u" else 90),
                    ("factory_doorway_far", 180 if axis == "u" else 270)):
        shot(tag, [lx, ly, 0.9], 2.6, az, -8,
             f"the added partition dividing the factory — {dw:.1f} x {dh:.1f} "
             f"doorway, seen from the {'far' if 'far' in tag else 'near'} side. {UNIT}")
az_c, dist_c, _ = best_view([cu, cv, 1.2], -22, 7.5)
shot("factory_corner", [cu, cv, 1.0], max(dist_c, 4.0), az_c, -22,
     f"the room shell: 4 walls recovered from the mesh, {len(rep['doorways'])} "
     f"reconstructed door openings. {UNIT}")

# --- orbit, with the robot settling under gravity ---------------------------
frames = OUT / "_orbit"
frames.mkdir(exist_ok=True)
for f in frames.glob("*.png"):
    f.unlink()
print("orbit video:")
N = 240
from PIL import Image, ImageDraw
for i in range(N):
    for _ in range(4):
        mujoco.mj_step(m, d)
    cam.lookat[:] = [cu * 0.4, cv * 0.4, 0.6]
    # An orbit at a fixed radius spends part of every revolution inside a machine.
    # Pulling the camera in to whatever is clear on this bearing keeps the subject
    # visible the whole way round.
    _t = [cu * 0.4, cv * 0.4, 0.6]
    _az = 360.0 * i / N
    cam.distance = max(min(7.0, min(sightline(_t, _az, 20),
                                    exit_distance(_t, _az, 20)) - 0.3), 1.6)
    cam.azimuth = 360.0 * i / N
    cam.elevation = -20
    renderer.update_scene(d, cam)
    im = Image.fromarray(renderer.render())
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, H - 30, W, H], fill=(12, 12, 14))
    dr.text((10, H - 21), f"reconstructed factory + robot — {UNIT}",
            fill=(235, 235, 235))
    im.save(frames / f"f{i:04d}.png")
mp4 = OUT / "factory_scene_orbit.mp4"
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
       "-i", str(frames / "f%04d.png"),
       "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
       "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("  ffmpeg failed:", r.stderr[-500:])
else:
    print(f"  {mp4.name}  {mp4.stat().st_size/1e6:.2f} MB")
    for f in frames.glob("*.png"):
        f.unlink()
    frames.rmdir()
