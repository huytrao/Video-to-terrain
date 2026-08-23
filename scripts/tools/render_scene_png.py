"""Offscreen preview renderer with no GPU and no OpenGL dependency.

Samples points over the surface in proportion to area, projects them through a
pinhole camera and resolves visibility with a z-buffer.  It is not pretty, but
it answers the only question a carve review needs: did the object leave, and
did the hole close?

World up is -Z in this scene, so the camera up vector is (0, 0, -1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
UP = np.array([0.0, 0.0, -1.0])


def look_at(eye, target):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, UP)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return np.vstack([right, true_up, forward])


def sample_cloud(mesh: trimesh.Trimesh, count: int, tint=None):
    points, face_index = trimesh.sample.sample_surface(mesh, count)
    colours = np.asarray(mesh.visual.face_colors)[face_index][:, :3].astype(float)
    if tint is not None:
        colours = 0.35 * colours + 0.65 * np.asarray(tint, dtype=float)
    shade = 0.45 + 0.55 * np.abs(mesh.face_normals[face_index] @ np.array([0.3, 0.4, -0.87]))
    return points, np.clip(colours * shade[:, None], 0, 255)


def render(clouds, eye, target, size=(1100, 780), fov_deg=55.0, background=22):
    width, height = size
    basis = look_at(eye, target)
    focal = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    depth = np.full((height, width), np.inf)
    image = np.full((height, width, 3), background, dtype=np.uint8)

    for points, colours in clouds:
        camera = (points - eye) @ basis.T
        ahead = camera[:, 2] > 0.05
        camera, colours = camera[ahead], colours[ahead]
        u = (focal * camera[:, 0] / camera[:, 2] + width / 2).astype(int)
        v = (focal * camera[:, 1] / camera[:, 2] + height / 2).astype(int)
        on = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z, colours = u[on], v[on], camera[on, 2], colours[on]
        order = np.argsort(-z)  # far first, so near overwrites
        u, v, z, colours = u[order], v[order], z[order], colours[order]
        keep = z < depth[v, u]
        depth[v[keep], u[keep]] = z[keep]
        image[v[keep], u[keep]] = colours[keep].astype(np.uint8)
    return Image.fromarray(image)


def orbit_eyes(centre, radius, height, count):
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + np.pi / 4
    return [
        centre + np.array([radius * np.cos(a), radius * np.sin(a), height]) for a in angles
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--room", required=True)
    ap.add_argument("--objects", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--samples", type=int, default=2_400_000)
    ap.add_argument("--highlight", action="store_true", help="tint objects magenta")
    args = ap.parse_args()

    def load(path):
        scene = trimesh.load(ROOT / path)
        return scene.dump(concatenate=True) if hasattr(scene, "geometry") else scene

    room = load(args.room)
    clouds = [sample_cloud(room, args.samples)]
    for path in args.objects:
        mesh = load(path)
        tint = (255, 40, 190) if args.highlight else None
        share = max(int(args.samples * mesh.area / room.area), 30_000)
        clouds.append(sample_cloud(mesh, share, tint))

    centre = room.bounds.mean(axis=0)
    span = float(np.linalg.norm(room.extents[:2]))
    eyes = orbit_eyes(centre, span * 0.62, -span * 0.30, args.views)

    tiles = [render(clouds, eye, centre) for eye in eyes]
    w, h = tiles[0].size
    cols = 2 if len(tiles) > 1 else 1
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (22, 22, 22))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}  ({sheet.size[0]}x{sheet.size[1]}, {len(tiles)} views)")


if __name__ == "__main__":
    main()
