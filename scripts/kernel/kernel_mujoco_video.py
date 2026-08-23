"""The MuJoCo rollout video, on its own.

Separate from `kernel_outputs.py` for one reason that is not organisational:
`MUJOCO_GL` has to be set **before mujoco is first imported**, and in the outputs
stage the terrain step imports mujoco to compile-check the MJCF long before the
rollout step runs. Setting it inside the rollout function was too late — the GL
backend had already been resolved without it, and the render died with
`FatalError: gladLoadGL error`.

Same class of mistake as a .pth file not reaching an already-running interpreter:
the environment has to precede the import, not follow it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Before any mujoco import. This is the whole point of the file.
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, "/kaggle/working/src")

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import mujoco  # noqa: E402

SCENE = os.environ.get("V3D_SCENE", "small_factory")
OUT = Path(os.environ.get("V3D_DELIVER", "/tmp/deliver"))
VID = OUT / "videos"
SCALE_MODE = os.environ.get("V3D_SCALE_MODE", "assumed")
SCALE_CAPTION = ("lengths in DA3 units, NOT metres (scale = assumed)"
                 if SCALE_MODE == "assumed" else "lengths in metres")


def stamp(img, title: str, extra: str = "") -> None:
    h, w = img.shape[:2]
    s = max(0.45, min(0.85, w / 1600))
    lines = [title] + ([extra] if extra else []) + [SCALE_CAPTION]
    box_h = int((len(lines) * 26 + 12) * s)
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    y = int(20 * s) + 6
    for line in lines:
        colour = (90, 200, 255) if line == SCALE_CAPTION else (255, 255, 255)
        cv2.putText(img, line, (int(10 * s) + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * s / 0.6, colour, 1, cv2.LINE_AA)
        y += int(24 * s)


def main() -> None:
    mjcf = OUT / "mujoco" / "scene.xml"
    if not mjcf.exists():
        raise SystemExit(f"no MJCF at {mjcf}")
    print(f"MUJOCO_GL={os.environ['MUJOCO_GL']}  model={mjcf}")

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    W, H = 1280, 720
    print(f"framebuffer declared in model: {model.vis.global_.offwidth}x"
          f"{model.vis.global_.offheight}; requesting {W}x{H}")
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)

    fd = VID / "_10_mujoco_rollout_frames"
    fd.mkdir(parents=True, exist_ok=True)
    for old in fd.glob("*.png"):
        old.unlink()

    n_frames, per_frame = 180, 10
    max_qvel = 0.0
    settling = []
    for i in range(n_frames):
        for _ in range(per_frame):
            mujoco.mj_step(model, data)
        if data.qvel.size:
            v = float(np.abs(data.qvel).max())
            max_qvel = max(max_qvel, v)
            settling.append(v)
        cam.azimuth = 360.0 * i / n_frames
        cam.elevation = -20.0
        renderer.update_scene(data, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        stamp(img, f"MuJoCo rollout  t = {data.time:.2f} s",
              f"{model.nbody - 1} bodies, {model.ngeom} geoms, "
              f"|qvel| now {settling[-1] if settling else 0:.2f}, peak {max_qvel:.2f}")
        cv2.imwrite(str(fd / f"{i:05d}.png"), img)
    renderer.close()

    pngs = sorted(fd.glob("*.png"))
    for n, p in enumerate(pngs):
        p.rename(fd / f"seq_{n:05d}.png")
    out = VID / "10_mujoco_rollout.mp4"
    import subprocess
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
         "-i", str(fd / "seq_%05d.png"),
         "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
        capture_output=True, text=True)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise SystemExit(f"ffmpeg failed: {r.stderr[-500:]}")

    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    # Is the scene settling? Compare the last fifth of the run to the first fifth.
    # A scene at rest ends far quieter than it starts; one that keeps moving does not,
    # and that is the signature of objects spawned inside each other.
    k = max(1, len(settling) // 5)
    early, late = float(np.mean(settling[:k])), float(np.mean(settling[-k:]))
    print(f"{n_frames * per_frame} steps, sim time {data.time:.2f}s")
    print(f"finite {finite}, peak |qvel| {max_qvel:.3f}")
    print(f"settling: mean |qvel| {early:.3f} (first fifth) -> {late:.3f} (last fifth)"
          f"  ratio {late / max(early, 1e-9):.3f}")
    if late > 0.5 * early:
        print("  NOTE: the scene is NOT settling. With overlapping objects at spawn "
              "this is interpenetration being resolved, not physics. Convex "
              "decomposition and a de-penetration pass are the fix.")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("\nMUJOCO_VIDEO_OK")


main()
