"""Bake the reconstructed mesh's per-vertex colors into a real UV texture (PNG),
so Blender/glTF viewers show an actual image texture instead of flat vertex colors.

Uses xatlas to unwrap UVs, then rasterizes each triangle into texture space with
barycentric-interpolated vertex colors.

Output: outputs/run_mvp/blender/terrain_mesh_textured.glb (+ texture.png alongside)
"""
from __future__ import annotations
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh
import xatlas
from PIL import Image

BG_COLOR = 40

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = Path(os.getenv(
    "V2T_RUN_DIR",
    str(REPO_ROOT / "outputs" / "run_mvp"),
))
OUT = RUN / "blender"
TEX_SIZE = int(os.getenv("V2T_TEX_SIZE", "2048"))


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
    # Match the frame Stage F/Blender use, so the baked GLB lines up with the boxes.
    aligned = RUN / "geometry" / "scene_mesh_aligned.ply"
    mesh = trimesh.load(aligned if aligned.exists() else RUN / "geometry" / "scene_mesh.ply")

    # xatlas unwrap time grows sharply with face count (800k faces runs for well over
    # half an hour), and a 4k atlas cannot resolve that many charts anyway. Decimate
    # first; the full-resolution mesh stays available as the vertex-coloured export.
    max_faces = int(os.getenv("V2T_BAKE_MAX_FACES", "150000"))
    if len(mesh.faces) > max_faces:
        print(f"Decimating {len(mesh.faces)} -> {max_faces} faces for a tractable unwrap...")
        # trimesh's own decimation needs the optional fast_simplification wheel;
        # open3d is already a pipeline dependency, so use it and keep vertex colours.
        import open3d as o3d
        om = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
        )
        om.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
        )
        om = om.simplify_quadric_decimation(max_faces)
        mesh = trimesh.Trimesh(
            vertices=np.asarray(om.vertices),
            faces=np.asarray(om.triangles),
            vertex_colors=(np.asarray(om.vertex_colors) * 255).astype(np.uint8),
            process=False,
        )
        print(f"  now {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    vcolors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    print(f"Input mesh: {len(vertices)} verts, {len(faces)} faces")

    print("Unwrapping UVs with xatlas (with chart padding to avoid seam bleed)...")
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    pack_options = xatlas.PackOptions()
    pack_options.padding = 8  # texels between charts, so the inpaint halo below can't bridge them
    pack_options.bilinear = True
    atlas.generate(pack_options=pack_options)
    vmapping, indices, uvs = atlas.get_mesh(0)
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
    # Fill only a thin halo immediately around each chart's own rasterized pixels
    # (anti-aliasing seams), not the whole empty canvas — inpainting the full
    # background lets cv2.inpaint bridge/blend colors from UNRELATED charts
    # that happen to be nearby in atlas space, producing a scrambled look.
    filled_mask = (~np.all(tex == BG_COLOR, axis=-1)).astype(np.uint8) * 255
    halo = cv2.dilate(filled_mask, np.ones((7, 7), np.uint8)) - filled_mask
    gap_mask = ((halo > 0) & np.all(tex == BG_COLOR, axis=-1)).astype(np.uint8) * 255
    tex = cv2.inpaint(tex, gap_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

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
