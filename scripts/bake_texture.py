"""Bake the reconstructed mesh's per-vertex colors into a real UV texture (PNG),
so Blender/glTF viewers show an actual image texture instead of flat vertex colors.

Uses xatlas to unwrap UVs, then rasterizes each triangle into texture space with
barycentric-interpolated vertex colors.

Output: outputs/run_mvp/blender/terrain_mesh_textured.glb (+ texture.png alongside)
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import trimesh
import xatlas
from PIL import Image

BG_COLOR = 40

RUN = Path("/Users/traoanhuy/source_code/Full-pipeline-scan-terrain/Videos_to_terrain/outputs/run_mvp")
OUT = RUN / "blender"
TEX_SIZE = 2048


def rasterize_triangle(tex: np.ndarray, uv0, uv1, uv2, c0, c1, c2, size: int):
    """Fill one triangle in texture space with barycentric-interpolated color."""
    p0 = uv0 * size
    p1 = uv1 * size
    p2 = uv2 * size

    min_x = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
    max_x = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), size - 1)
    min_y = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
    max_y = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), size - 1)
    if min_x > max_x or min_y > max_y:
        return

    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(denom) < 1e-9:
        return

    xs, ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
    xs = xs.astype(np.float64) + 0.5
    ys = ys.astype(np.float64) + 0.5

    w0 = ((p1[1] - p2[1]) * (xs - p2[0]) + (p2[0] - p1[0]) * (ys - p2[1])) / denom
    w1 = ((p2[1] - p0[1]) * (xs - p2[0]) + (p0[0] - p2[0]) * (ys - p2[1])) / denom
    w2 = 1 - w0 - w1

    inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
    if not inside.any():
        return

    color = (
        w0[..., None] * c0 + w1[..., None] * c1 + w2[..., None] * c2
    ).clip(0, 255).astype(np.uint8)

    px = (xs.astype(int))[inside]
    py = (ys.astype(int))[inside]
    tex[py, px] = color[inside]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(RUN / "geometry" / "scene_mesh.ply")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    vcolors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    print(f"Input mesh: {len(vertices)} verts, {len(faces)} faces")

    print("Unwrapping UVs with xatlas...")
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    new_vertices = vertices[vmapping]
    new_colors = vcolors[vmapping]
    print(f"Unwrapped: {len(new_vertices)} verts (after atlas split), {len(indices)} faces")

    tex = np.full((TEX_SIZE, TEX_SIZE, 3), BG_COLOR, dtype=np.uint8)
    for tri in indices:
        i0, i1, i2 = tri
        rasterize_triangle(
            tex,
            uvs[i0], uvs[i1], uvs[i2],
            new_colors[i0].astype(np.float64),
            new_colors[i1].astype(np.float64),
            new_colors[i2].astype(np.float64),
            TEX_SIZE,
        )
    # Fill seams/padding between UV chart islands by inpainting from
    # neighboring texel color, instead of leaving a "shattered glass" look.
    gap_mask = np.all(tex == BG_COLOR, axis=-1).astype(np.uint8) * 255
    tex = cv2.inpaint(tex, gap_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

    # glTF/image UV origin is bottom-left in our rasterizer; flip to match image top-left
    tex = np.flipud(tex)

    img = Image.fromarray(tex, mode="RGB")
    tex_path = OUT / "terrain_texture.png"
    img.save(tex_path)
    print(f"Texture saved: {tex_path}")

    uvs_flipped = uvs.copy()
    uvs_flipped[:, 1] = 1.0 - uvs_flipped[:, 1]

    textured_mesh = trimesh.Trimesh(vertices=new_vertices, faces=indices, process=False)
    textured_mesh.visual = trimesh.visual.TextureVisuals(
        uv=uvs_flipped, image=img
    )

    glb_path = OUT / "terrain_mesh_textured.glb"
    textured_mesh.export(glb_path)
    print(f"Textured GLB saved: {glb_path}")


if __name__ == "__main__":
    main()
