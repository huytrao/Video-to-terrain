"""Render the physics scene, before and after it settles.

A validator prints numbers; this shows what those numbers look like. The pair
matters more than either half: an object that drifts 1.3 m between the two
frames is one you can see leaving, and the settled frame is the arrangement a
robot would actually find.
"""
from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[2]


# One hue per free body. Rendered in the room's own grey, a body that moved is
# indistinguishable from the wall behind it, which is exactly what the first
# version of this render showed: four views of a grey box telling you nothing.
TINTS = [
    (0.95, 0.35, 0.25), (0.30, 0.75, 0.95), (0.95, 0.75, 0.20),
    (0.55, 0.85, 0.35), (0.85, 0.40, 0.90), (0.25, 0.55, 0.95),
    (0.95, 0.55, 0.15), (0.40, 0.90, 0.70),
]


def tint_bodies(model):
    """Give every free body's visual geom its own colour, in place."""
    coloured = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        rgb = TINTS[len(coloured) % len(TINTS)]
        start = model.body_geomadr[body_id]
        for geom in range(start, start + model.body_geomnum[body_id]):
            if model.geom_contype[geom] == 0:        # the visual one
                model.geom_rgba[geom] = (*rgb, 1.0)
        coloured.append((name, rgb))
    return coloured


def shoot(renderer, model, data, camera):
    renderer.update_scene(data, camera=camera)
    return Image.fromarray(renderer.render())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default="outputs-full-factory/delivery/OUTPUT_K/sim/factory_physics.xml")
    ap.add_argument("--out", default="outputs-full-factory/delivery/OUTPUT_K/review_physics.png")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=640)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(ROOT / args.xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    legend = tint_bodies(model)

    mujoco.mj_forward(model, data)
    # Aim at the bodies, not at the room: the interesting half-metre of this
    # scene is the bench, and framing the whole shell hides it.
    focus = np.array([data.body(name).xpos for name, _ in legend]).mean(axis=0)

    cameras = []
    for azimuth, distance, elevation, label in (
        (40, 7.0, -22, "wide 40deg"),
        (140, 7.0, -22, "wide 140deg"),
        (40, 3.2, -14, "close 40deg"),
        (140, 3.2, -14, "close 140deg"),
    ):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = focus
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        cameras.append((label, camera))

    before = [shoot(renderer, model, data, cam) for _, cam in cameras]

    for _ in range(int(args.settle / model.opt.timestep)):
        mujoco.mj_step(model, data)
    after = [shoot(renderer, model, data, cam) for _, cam in cameras]

    width, height = before[0].size
    sheet = Image.new("RGB", (width * len(cameras), height * 2), (24, 24, 26))
    draw = ImageDraw.Draw(sheet)
    for i, ((label, _), image) in enumerate(zip(cameras, before)):
        sheet.paste(image, (i * width, 0))
        draw.text((i * width + 10, 8), f"t = 0 s   {label}", fill=(255, 220, 60))
    for i, ((label, _), image) in enumerate(zip(cameras, after)):
        sheet.paste(image, (i * width, height))
        draw.text((i * width + 10, height + 8),
                  f"t = {args.settle:.0f} s settled   {label}", fill=(120, 255, 170))

    swatch = 16
    for i, (name, rgb) in enumerate(legend):
        y = height * 2 - (len(legend) - i) * (swatch + 4) - 8
        draw.rectangle([10, y, 10 + swatch, y + swatch],
                       fill=tuple(int(c * 255) for c in rgb))
        draw.text((10 + swatch + 8, y + 2), name, fill=(235, 235, 235))

    out = ROOT / args.out
    sheet.save(out)
    print(f"wrote {out.relative_to(ROOT)}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
