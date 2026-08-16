"""Render the reconstruction so it can be looked at, not just measured.

Stage J answers "is it the right size". These answer "is it the right room" —
which is the question a person actually asks first, and the one the previous
pipeline never let anyone answer.

Everything renders offscreen through Open3D's EGL backend, so it runs on a
headless kernel. Nothing here reads a dataset directly; it takes a mesh, a
camera file and a set of boxes, so a scene reconstructed from an arbitrary
video renders identically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

# Deliberately cool neutral ground: warm greys make reconstructed interiors
# look like clay renders and hide the real surface colour.
BG_LIGHT = [0.94, 0.945, 0.95, 1.0]
BOX_COLOR = [0.98, 0.42, 0.21]
SHELL_COLOR = [0.16, 0.55, 0.85]


@dataclass
class Camera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    position: np.ndarray
    rotation: np.ndarray  # world-from-camera


def load_cameras(path: str | Path) -> list[Camera]:
    """3DGS cameras.json — the poses PGSR actually optimised."""
    raw = json.loads(Path(path).read_text())
    return [
        Camera(
            name=c["img_name"],
            width=int(c["width"]),
            height=int(c["height"]),
            fx=float(c["fx"]),
            fy=float(c["fy"]),
            position=np.asarray(c["position"], dtype=float),
            rotation=np.asarray(c["rotation"], dtype=float),
        )
        for c in raw
    ]


def _renderer(width: int, height: int):
    import open3d as o3d

    r = o3d.visualization.rendering.OffscreenRenderer(width, height)
    r.scene.set_background(BG_LIGHT)
    r.scene.scene.set_sun_light([-0.3, -0.8, -0.5], [1.0, 1.0, 1.0], 90000)
    r.scene.scene.enable_sun_light(True)
    return r


def _lit_material():
    import open3d as o3d

    m = o3d.visualization.rendering.MaterialRecord()
    m.shader = "defaultLit"
    m.base_roughness = 0.85
    m.base_reflectance = 0.1
    return m


def _line_material(width: float = 3.0):
    import open3d as o3d

    m = o3d.visualization.rendering.MaterialRecord()
    m.shader = "unlitLine"
    m.line_width = width
    return m


def orbit_views(
    mesh_path: str | Path,
    out_dir: str | Path,
    n_views: int = 8,
    size: tuple[int, int] = (1600, 1100),
    elevation_deg: float = 28.0,
    stem: str = "orbit",
) -> list[Path]:
    """Circle the room at a fixed elevation, looking at its centre."""
    import open3d as o3d

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    centre = verts.mean(axis=0)
    radius = float(np.linalg.norm(verts - centre, axis=1).max()) * 1.35

    r = _renderer(*size)
    r.scene.add_geometry("mesh", mesh, _lit_material())

    written: list[Path] = []
    el = np.radians(elevation_deg)
    for i in range(n_views):
        az = 2 * np.pi * i / n_views
        eye = centre + radius * np.array(
            [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
        )
        r.scene.camera.look_at(centre.tolist(), eye.tolist(), [0, 0, 1])
        img = r.render_to_image()
        p = out / f"{stem}_{i:02d}.png"
        o3d.io.write_image(str(p), img)
        written.append(p)
    return written


def detect_up_sign(vertex_z: np.ndarray, camera_z: np.ndarray) -> int:
    """+1 if larger z is up, -1 if larger z is down.

    Reconstruction frames do not agree on this. In the ScanNet++ scene used
    here, +z points *down*: the ground-truth floor sits at z = -0.04 and the
    ceiling at z = -3.33. Assuming the usual convention produced a floor plan
    that had sliced off the floor, kept the ceiling, and viewed it from below —
    a picture that looked entirely reasonable and was upside down.

    The signal used is that a camera held by a person sits between floor and
    ceiling and nearer the floor: roughly 1.4 m up in a 3.3 m room. So the end
    of the vertical range closer to the median camera height is the floor.
    """
    lo, hi = float(vertex_z.min()), float(vertex_z.max())
    cam = float(np.median(camera_z))
    return -1 if abs(cam - lo) < abs(cam - hi) else 1


def topdown_view(
    mesh_path: str | Path,
    out_path: str | Path,
    size: tuple[int, int] = (1600, 1600),
    cut_fraction: float = 0.62,
    cameras_json: str | Path | None = None,
    up_sign: int | None = None,
) -> Path:
    """Floor plan: slice the ceiling off and look straight down.

    A room photographed from inside is hard to judge; the plan view is where a
    reader sees at once whether walls are straight and the layout is real.

    ``up_sign`` may be given explicitly; otherwise it is derived from the
    camera heights, and if neither is available the usual +z-is-up convention
    is assumed and the caller should check the result.
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts = np.asarray(mesh.vertices)

    if up_sign is None and cameras_json is not None:
        cams = load_cameras(cameras_json)
        up_sign = detect_up_sign(verts[:, 2], np.array([c.position[2] for c in cams]))
    if up_sign is None:
        up_sign = 1

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    height = hi[2] - lo[2]
    # Keep the part of the room below the cut, whichever direction "below" is.
    if up_sign > 0:
        cut_z = lo[2] + height * cut_fraction
        keep = verts[:, 2] < cut_z
    else:
        cut_z = hi[2] - height * cut_fraction
        keep = verts[:, 2] > cut_z

    tris = np.asarray(mesh.triangles)
    mesh.remove_triangles_by_mask(~keep[tris].all(axis=1))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    kept = np.asarray(mesh.vertices)
    klo, khi = kept.min(axis=0), kept.max(axis=0)
    centre = (klo + khi) / 2.0
    span = float(max(khi[0] - klo[0], khi[1] - klo[1])) * 0.62

    r = _renderer(*size)
    r.scene.add_geometry("mesh", mesh, _lit_material())
    eye_z = (khi[2] + span) if up_sign > 0 else (klo[2] - span)
    r.scene.camera.look_at(centre.tolist(), [centre[0], centre[1], eye_z], [0, 1, 0])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(out), r.render_to_image())
    return out


def _box_lineset(center: np.ndarray, size: np.ndarray, axes: np.ndarray, color: list[float]):
    import open3d as o3d

    half = (axes.T * (size / 2.0)).T
    corners = np.array(
        [center + sx * half[0] + sy * half[1] + sz * half[2]
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls


def boxes_over_mesh(
    mesh_path: str | Path,
    boxes,
    out_dir: str | Path,
    n_views: int = 4,
    size: tuple[int, int] = (1600, 1100),
    stem: str = "boxes",
    min_size_m: float = 0.0,
) -> list[Path]:
    """The room with its object boxes drawn on it.

    Shell boxes are drawn in a second colour: separating structure from
    furniture is the distinction the whole scene graph rests on.
    """
    import open3d as o3d

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    centre = verts.mean(axis=0)

    r = _renderer(*size)
    r.scene.add_geometry("mesh", mesh, _lit_material())

    drawn = 0
    line_mat = _line_material()
    extremes = [verts]
    for i, b in enumerate(boxes):
        if float(np.max(b.size)) < min_size_m:
            continue
        colour = SHELL_COLOR if getattr(b, "is_shell", False) else BOX_COLOR
        r.scene.add_geometry(f"box_{i}", _box_lineset(b.center, b.size, b.axes, colour), line_mat)
        extremes.append(np.array([b.center - b.size / 2, b.center + b.size / 2]))
        drawn += 1

    # Boxes reach past the mesh; framing on the mesh alone crops them off.
    span = np.vstack(extremes)
    radius = float(np.linalg.norm(span - centre, axis=1).max()) * 1.6

    written: list[Path] = []
    el = np.radians(30.0)
    for i in range(n_views):
        az = 2 * np.pi * i / n_views
        eye = centre + radius * np.array(
            [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
        )
        r.scene.camera.look_at(centre.tolist(), eye.tolist(), [0, 0, 1])
        p = out / f"{stem}_{i:02d}.png"
        o3d.io.write_image(str(p), r.render_to_image())
        written.append(p)
    return written


def depth_sheet(
    depth_dir: str | Path,
    out_path: str | Path,
    n: int = 12,
    cols: int = 4,
) -> Path:
    """A contact sheet of DA3 depth maps, colourised on a shared scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(Path(depth_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no depth maps in {depth_dir}")
    step = max(1, len(files) // n)
    picked = files[::step][:n]

    arrays = [np.load(f) for f in picked]
    vmin = float(min(np.nanmin(a) for a in arrays))
    vmax = float(max(np.nanpercentile(a, 99) for a in arrays))

    rows = (len(arrays) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.4), facecolor="#0e1116")
    for ax, arr, f in zip(np.ravel(axes), arrays, picked):
        im = ax.imshow(arr, cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(f.stem[:14], color="#c9d1d9", fontsize=8)
        ax.axis("off")
    for ax in np.ravel(axes)[len(arrays):]:
        ax.axis("off")

    cb = fig.colorbar(im, ax=list(np.ravel(axes)), fraction=0.02, pad=0.01)
    cb.set_label("metres", color="#c9d1d9")
    cb.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#c9d1d9")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="#0e1116", bbox_inches="tight")
    plt.close(fig)
    return out


def render_vs_photo(
    renders_dir: str | Path,
    photos_dir: str | Path,
    out_path: str | Path,
    n: int = 4,
) -> Path:
    """Rows of trained render beside the real photograph, same frame.

    This is what a PSNR number looks like; a reader who distrusts the metric
    can judge the pair directly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    renders = sorted(Path(renders_dir).glob("*.jpg"))
    if not renders:
        raise FileNotFoundError(f"no renders in {renders_dir}")
    step = max(1, len(renders) // n)
    picked = renders[::step][:n]

    def photo_for(stem: str) -> Path | None:
        """PGSR writes .jpg; ScanNet++ ships .JPG. Case and extension both vary."""
        for ext in (".JPG", ".jpg", ".png", ".PNG", ".jpeg"):
            cand = Path(photos_dir) / f"{stem}{ext}"
            if cand.exists():
                return cand
        return None

    missing = [p.stem for p in picked if photo_for(p.stem) is None]
    if missing:
        raise FileNotFoundError(
            f"no photograph found in {photos_dir} for {missing[:3]} — checked .JPG/.jpg/.png"
        )

    fig, axes = plt.subplots(len(picked), 2, figsize=(13, 4.2 * len(picked)), facecolor="#0e1116")
    axes = np.atleast_2d(axes)
    for row, rp in enumerate(picked):
        gp = photo_for(rp.stem)
        for col, (path, label) in enumerate(((rp, "3DGS render"), (gp, "photograph"))):
            ax = axes[row, col]
            ax.imshow(Image.open(path))
            ax.set_title(f"{label} — {rp.stem}", color="#c9d1d9", fontsize=10)
            ax.axis("off")
    fig.subplots_adjust(wspace=0.02, hspace=0.12)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor="#0e1116", bbox_inches="tight")
    plt.close(fig)
    return out
