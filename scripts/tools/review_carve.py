"""Before/after contact sheet for a carve: one row per object, shot from inside.

Left  = original room, object still welded in place.
Right = carved room, object removed and its hole patched.
The camera sits inside the room at the object's own height, backed off along
the direction with the most free space, because an exterior orbit only ever
shows the outside of the scan shell.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))
from render_scene_png import render, sample_cloud  # noqa: E402

TILE = (760, 560)


def best_eye(tree, target, distance):
    """Pick the horizontal direction with the most open space in front of it.

    trimesh's own ray caster needs `rtree`, which is not installed, so walk the
    ray in steps and stop at the first step that comes close to any scan vertex.
    """
    best, best_clearance = np.array([1.0, 0.0, 0.0]), -1.0
    steps = np.arange(0.6, 6.0, 0.12)
    for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        probes = target + steps[:, None] * direction
        blocked = np.where(tree.query(probes)[0] < 0.22)[0]
        clearance = float(steps[blocked[0]]) if len(blocked) else float(steps[-1])
        if clearance > best_clearance:
            best, best_clearance = direction, clearance
    reach = float(np.clip(distance, 1.0, max(best_clearance - 0.3, 1.0)))
    # Up is -Z, so subtracting from z raises the camera above the target.
    return target + best * reach + np.array([0.0, 0.0, -0.45])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=1_600_000)
    args = ap.parse_args()

    def load(path):
        scene = trimesh.load(ROOT / path)
        return scene.dump(concatenate=True) if hasattr(scene, "geometry") else scene

    before, after = load(args.before), load(args.after)
    manifest = json.loads((ROOT / args.manifest).read_text())
    tree = cKDTree(before.vertices)
    cloud_before = [sample_cloud(before, args.samples)]
    cloud_after = [sample_cloud(after, args.samples)]

    entries = manifest["objects"]
    sheet = Image.new("RGB", (TILE[0] * 2, TILE[1] * len(entries)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for row, entry in enumerate(entries):
        target = np.asarray(entry["centroid"])
        reach = float(np.linalg.norm(np.asarray(entry["bounds_max"]) - np.asarray(entry["bounds_min"])))
        eye = best_eye(tree, target, max(2.2, reach * 1.6))
        for col, clouds in ((0, cloud_before), (1, cloud_after)):
            sheet.paste(render(clouds, eye, target, size=TILE, fov_deg=50), (col * TILE[0], row * TILE[1]))
        label = f"{entry['name']}  {entry['faces']} faces  {entry['area_m2']} m2"
        draw.text((10, row * TILE[1] + 8), f"BEFORE   {label}", fill=(255, 220, 60))
        draw.text((TILE[0] + 10, row * TILE[1] + 8), "AFTER  (carved + patched)", fill=(90, 255, 160))
        draw.line([(0, row * TILE[1]), (sheet.width, row * TILE[1])], fill=(70, 70, 70))
    draw.line([(TILE[0], 0), (TILE[0], sheet.height)], fill=(70, 70, 70), width=2)

    out = ROOT / args.out
    sheet.save(out)
    print(f"wrote {out}  ({sheet.size[0]}x{sheet.size[1]}, {len(entries)} objects)")


if __name__ == "__main__":
    main()
