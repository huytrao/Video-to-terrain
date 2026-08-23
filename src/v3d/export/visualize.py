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

import base64
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

# Deliberately cool neutral ground: warm greys make reconstructed interiors
# look like clay renders and hide the real surface colour.
BG_LIGHT = [0.94, 0.945, 0.95, 1.0]
BOX_COLOR = [0.98, 0.42, 0.21]
SHELL_COLOR = [0.16, 0.55, 0.85]

# glTF is defined Y-up, so an exporter bakes +90° about X into the vertices on
# the way out and the reconstruction frame is lost. This undoes it:
# (x, y, z) -> (x, z, -y). Same correction as v3d.export.blender.import_mesh,
# which had to learn it the hard way — twice.
GLTF_UNDO = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


# --- glTF reading -----------------------------------------------------------
#
# Open3D 0.19 cannot read back the .glb it writes. Its exporter puts the whole
# geometry in a base64 `data:` URI inside the JSON chunk and emits no BIN chunk;
# its ASSIMP-based importer expects the BIN chunk, finds nothing, and returns an
# **empty mesh** having printed only a warning:
#
#     [Open3D WARNING] Unable to load file room_300k.glb with ASSIMP:
#     GLTF: Buffer view with offset/length (0/3600000) is out of range.
#
# So the delivered mesh reads as zero vertices unless the caller checks. Blender
# and every spec-compliant loader read the same file correctly — the data is
# intact. Decoding the buffers directly is a dozen lines and removes the
# dependency on which loader happens to be installed.

_GLTF_DTYPE = {
    5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4",
}
_GLTF_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
# Integer component types, when an accessor is `normalized`, map onto [0,1].
_GLTF_NORMALISE = {5121: 255.0, 5123: 65535.0, 5120: 127.0, 5122: 32767.0}


def _gltf_document(path: Path) -> tuple[dict, bytes | None]:
    """Return (json document, BIN chunk) for a .glb, or (document, None) for .gltf."""
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        return json.loads(raw), None
    _, version, total = struct.unpack_from("<4sII", raw, 0)
    if total != len(raw):
        raise ValueError(
            f"{path.name} declares {total:,} bytes but holds {len(raw):,} — "
            "the file is truncated, not merely unreadable"
        )
    doc: dict | None = None
    blob: bytes | None = None
    off = 12
    while off + 8 <= len(raw):
        clen, ctype = struct.unpack_from("<I4s", raw, off)
        body = raw[off + 8: off + 8 + clen]
        if len(body) < clen:
            raise ValueError(
                f"{path.name}: chunk {ctype.decode(errors='replace')} claims "
                f"{clen:,} bytes but only {len(body):,} remain"
            )
        if ctype == b"JSON":
            doc = json.loads(body)
        elif ctype == b"BIN\x00":
            blob = bytes(body)
        off += 8 + clen
    if doc is None:
        raise ValueError(f"{path.name} is a .glb with no JSON chunk (version {version})")
    return doc, blob


def _gltf_buffers(doc: dict, blob: bytes | None, path: Path) -> list[bytes]:
    out: list[bytes] = []
    for i, spec in enumerate(doc.get("buffers", [])):
        uri = spec.get("uri")
        if uri is None:
            if blob is None:
                raise ValueError(f"{path.name}: buffer {i} needs a BIN chunk and there is none")
            data = blob
        elif uri.startswith("data:"):
            data = base64.b64decode(uri.split(",", 1)[1])
        else:
            data = (path.parent / unquote(uri)).read_bytes()
        want = spec.get("byteLength", len(data))
        if len(data) < want:
            raise ValueError(
                f"{path.name}: buffer {i} resolved to {len(data):,} bytes, needs {want:,}"
            )
        # Trim to the declared length: bytes past byteLength are not part of the
        # buffer, so an accessor reaching into them is a fault to report, not
        # data to use.
        out.append(data[:want])
    return out


def _gltf_accessor(doc: dict, buffers: list[bytes], index: int, path: Path) -> np.ndarray:
    acc = doc["accessors"][index]
    if "sparse" in acc:
        raise ValueError(f"{path.name}: accessor {index} is sparse, which is not handled")
    if "bufferView" not in acc:
        # Legal glTF: means "all zeros". Never what a mesh wants.
        raise ValueError(f"{path.name}: accessor {index} has no bufferView")
    n, ncomp = acc["count"], _GLTF_NCOMP[acc["type"]]
    dtype = np.dtype(_GLTF_DTYPE[acc["componentType"]])
    view = doc["bufferViews"][acc["bufferView"]]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    packed = ncomp * dtype.itemsize
    stride = view.get("byteStride") or packed
    buf = buffers[view["buffer"]]
    need = base + (n - 1) * stride + packed
    if need > len(buf):
        raise ValueError(
            f"{path.name}: accessor {index} reads to byte {need:,} of a "
            f"{len(buf):,}-byte buffer"
        )
    if stride == packed:
        values = np.frombuffer(buf, dtype=dtype, count=n * ncomp, offset=base)
        values = values.reshape(n, ncomp)
    else:  # interleaved attributes — gather each element out of its stride slot
        rows = np.frombuffer(buf, dtype=np.uint8, count=need - base, offset=base)
        pick = np.arange(n)[:, None] * stride + np.arange(packed)[None, :]
        values = rows[pick].copy().view(dtype).reshape(n, ncomp)
    if acc.get("normalized") and acc["componentType"] in _GLTF_NORMALISE:
        values = values.astype(np.float64) / _GLTF_NORMALISE[acc["componentType"]]
    return values


def _gltf_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        # glTF stores matrices column-major, so the flat 16 transpose into row-major.
        return np.asarray(node["matrix"], float).reshape(4, 4).T
    m = np.eye(4)
    if "rotation" in node:
        # glTF quaternion order is (x, y, z, w) — real part LAST. The opposite of
        # the (w, x, y, z) used elsewhere in this project, which has already cost
        # it one bug. Do not "tidy" this into the other convention.
        x, y, z, w = (float(v) for v in node["rotation"])
        m[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    if "scale" in node:
        m[:3, :3] = m[:3, :3] @ np.diag([float(v) for v in node["scale"]])
    if "translation" in node:
        m[:3, 3] = [float(v) for v in node["translation"]]
    return m


def _gltf_instances(doc: dict) -> list[tuple[int, np.ndarray]]:
    """Every (mesh index, world matrix) the default scene draws."""
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes")
    roots = (scenes[doc.get("scene", 0)].get("nodes", [])
             if scenes else list(range(len(nodes))))
    out: list[tuple[int, np.ndarray]] = []
    stack = [(int(i), np.eye(4)) for i in roots]
    seen: set[int] = set()
    while stack:
        i, parent = stack.pop()
        if i in seen:      # glTF forbids cycles; a malformed file should not hang
            continue
        seen.add(i)
        node = nodes[i]
        world = parent @ _gltf_local_matrix(node)
        if "mesh" in node:
            out.append((int(node["mesh"]), world))
        for child in node.get("children", []):
            stack.append((int(child), world))
    return out


def _mesh_from_gltf(path: Path):
    """Build a mesh straight from the glTF buffers, ignoring any installed loader."""
    import open3d as o3d

    doc, blob = _gltf_document(path)
    buffers = _gltf_buffers(doc, blob, path)
    meshes = doc.get("meshes", [])
    instances = _gltf_instances(doc) or [(i, np.eye(4)) for i in range(len(meshes))]

    verts: list[np.ndarray] = []
    tris: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    offset = 0
    for mesh_index, world in instances:
        for prim in meshes[mesh_index].get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue                      # not triangles; nothing to render here
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            p = _gltf_accessor(doc, buffers, attrs["POSITION"], path).astype(np.float64)
            p = p @ world[:3, :3].T + world[:3, 3]
            if "indices" in prim:
                idx = _gltf_accessor(doc, buffers, prim["indices"], path).reshape(-1)
            else:
                idx = np.arange(len(p), dtype=np.uint32)
            if len(idx) % 3:
                raise ValueError(f"{path.name}: {len(idx)} indices is not a whole number "
                                 "of triangles")
            verts.append(p)
            tris.append(idx.astype(np.int64).reshape(-1, 3) + offset)
            if "NORMAL" in attrs:
                nrm = _gltf_accessor(doc, buffers, attrs["NORMAL"], path).astype(np.float64)
                normals.append(nrm @ world[:3, :3].T)
            if "COLOR_0" in attrs:
                c = _gltf_accessor(doc, buffers, attrs["COLOR_0"], path).astype(np.float64)
                colors.append(c[:, :3])       # drop alpha; Open3D colours are RGB
            offset += len(p)

    if not verts:
        raise ValueError(f"{path.name} contains no triangle geometry")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.vstack(verts))
    mesh.triangles = o3d.utility.Vector3iVector(np.vstack(tris))
    # Only keep per-vertex arrays that cover every vertex — a partial array is
    # worse than none, because Open3D indexes it by vertex id.
    total = len(mesh.vertices)
    if normals and sum(len(a) for a in normals) == total:
        mesh.vertex_normals = o3d.utility.Vector3dVector(np.vstack(normals))
    if colors and sum(len(a) for a in colors) == total:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.vstack(colors), 0.0, 1.0))
    return mesh


def read_triangle_mesh(path: str | Path, quiet: bool = False):
    """Read a mesh, falling back to a direct glTF decode when the loader gives up.

    Open3D returning an empty mesh is not an error in its API — it prints a
    warning and hands back zero vertices — so the empty case is checked here and
    turned into either a working mesh or a raise. Never a silent empty scene.
    """
    import open3d as o3d

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) and len(mesh.triangles):
        return mesh
    if path.suffix.lower() not in {".glb", ".gltf"}:
        raise ValueError(
            f"no geometry in {path} ({path.stat().st_size:,} bytes) — the loader read "
            f"{len(mesh.vertices):,} vertices and {len(mesh.triangles):,} triangles"
        )
    mesh = _mesh_from_gltf(path)
    if not quiet:
        print(f"{path.name}: Open3D read no geometry, decoded the glTF buffers directly "
              f"-> {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    return mesh


def load_mesh_in_box_frame(mesh_path: str | Path, boxes=None, tol_m: float = 1.2):
    """Read a mesh and return it in the same frame as the annotations.

    A ``.glb`` written from the reconstruction comes back rotated, because glTF
    is Y-up and the conversion is baked into the vertex data rather than left in
    a node transform — so nothing about the file looks wrong. Boxes annotated in
    the reconstruction frame then sit at right angles to the geometry they
    describe. That mismatch has already shipped twice in this project.

    Rather than assume either frame, both candidates are tried and the one whose
    bounds actually agree with the boxes wins. With no boxes to check against,
    the fallback is that a room's smallest extent is its height, so it belongs on
    z. If neither candidate agrees, this raises: a silently misaligned scene is
    the failure mode being guarded against.
    """
    mesh = read_triangle_mesh(mesh_path)
    verts = np.asarray(mesh.vertices)

    candidates = {"as_loaded": np.eye(3), "gltf_undo": GLTF_UNDO}

    if boxes:
        signs = np.array([[sx, sy, sz]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        corners = np.vstack([
            b.center + signs @ (b.axes.T * (b.size / 2.0)).T for b in boxes
        ])
        blo, bhi = _aabb(corners)
        scored = {}
        for name, R in candidates.items():
            mlo, mhi = _aabb(verts @ R.T)
            inside = int(sum(
                blo[i] - tol_m <= mlo[i] and mhi[i] <= bhi[i] + tol_m for i in range(3)
            ))
            overhang = float(np.sum(np.maximum(0.0, blo - mlo) + np.maximum(0.0, mhi - bhi)))
            scored[name] = (inside, -overhang, R)
        best = max(scored, key=lambda k: scored[k][:2])
        inside, neg_overhang, R = scored[best]
        if inside < 3:
            mlo, mhi = _aabb(verts @ R.T)
            raise ValueError(
                "mesh does not fit the annotation bounds in either frame — best was "
                f"{best} with {inside}/3 axes inside and {-neg_overhang:.2f} m of "
                f"overhang. boxes {blo.round(2)}..{bhi.round(2)}, "
                f"mesh {mlo.round(2)}..{mhi.round(2)}"
            )
    else:
        lo, hi = _aabb(verts)
        best = "as_loaded" if int(np.argmin(hi - lo)) == 2 else "gltf_undo"
        R = candidates[best]

    if best != "as_loaded":
        T = np.eye(4)
        T[:3, :3] = R
        mesh.transform(T)
        print(f"mesh frame: applied {best} (glTF Y-up bake undone)")
    mesh.compute_vertex_normals()
    return mesh



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
    of the vertical range closer to the median camera height is the floor — and
    if that end is the *low* end, then up is +z.

    The first version of this function mapped that comparison to the opposite
    sign, so it returned +1 for exactly the Z-down scene the docstring describes
    (floor -0.04, ceiling -3.33, cameras -1.44) and left bug B2 alive behind a
    helper that looked like its fix. Both conventions are now pinned by
    ``selftest()``.
    """
    lo, hi = float(vertex_z.min()), float(vertex_z.max())
    cam = float(np.median(camera_z))
    return 1 if abs(cam - lo) < abs(cam - hi) else -1


def _read_image_any(path: str | Path) -> np.ndarray:
    """Decode an image to an HxWx3+ uint8 array using any available library.

    Tried in order of weight: Pillow, imageio, OpenCV, Open3D. The kernel has
    Open3D and the laptop does not; the laptop has Pillow and the delivery is
    reviewed there. Nothing here needs a GPU or a render context.
    """
    tried = []
    try:
        from PIL import Image

        with Image.open(str(path)) as im:
            return np.asarray(im.convert("RGB"))
    except ImportError:
        tried.append("PIL")
    try:
        import imageio.v3 as iio

        return np.asarray(iio.imread(str(path)))
    except ImportError:
        tried.append("imageio")
    try:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"{path}: OpenCV could not decode it")
        return np.asarray(bgr[:, :, ::-1])
    except ImportError:
        tried.append("cv2")
    try:
        import open3d as o3d

        return np.asarray(o3d.io.read_image(str(path)))
    except ImportError:
        tried.append("open3d")
    raise ImportError(
        f"no image decoder available to read {path}; tried {', '.join(tried)}. "
        f"Install any one of them — `pip install pillow` is the smallest."
    )


def image_content_extent(path: str | Path, tol_levels: float = 6.0,
                         patch: int = 12) -> dict:
    """How much of a rendered frame the subject actually occupies.

    The automated half of "look at the picture before shipping it". Framing is
    computed from geometry, but whether the renderer honoured the projection it
    was given is only observable in the pixels — and a render that puts its
    subject in the middle third of the frame passes every other check in this
    module, which is how the first cutaway stills shipped.

    The background is **sampled from the corners**, not taken from ``BG_LIGHT``.
    Open3D tone-maps before writing, so a background set to 0.94 arrives as 228
    rather than 240, and it dithers by a level or two: assuming the constant
    classified every pixel in the frame as subject and reported perfect framing
    for the exact images that motivated this function.

    Returns the non-background bounding box as fractions of the frame, so a
    caller can require the subject to fill it.

    Reads the PNG with whatever decoder is present, preferring the lightest.
    This is the one function here that judges a *delivered* file rather than
    building one, so it has to run where the delivery is reviewed — a laptop
    with Pillow and no CUDA — not only on the kernel that rendered it. Binding
    it to ``open3d.io.read_image`` made checking a shipped video impossible off
    the kernel, which is the same gap that let the loose framing ship.
    """
    img = _read_image_any(path)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"{path}: expected a colour image, got shape {img.shape}")
    rgb = img[:, :, :3].astype(float)
    h, w = rgb.shape[:2]

    corners = np.array([
        np.median(p.reshape(-1, 3), axis=0) for p in (
            rgb[:patch, :patch], rgb[:patch, -patch:],
            rgb[-patch:, :patch], rgb[-patch:, -patch:],
        )
    ])
    bg = np.median(corners, axis=0)
    spread = float(np.abs(corners - bg).max())
    if spread > tol_levels:
        raise ValueError(
            f"{path}: the four corners disagree by {spread:.1f} levels, so there is "
            f"no reliable background sample — the subject reaches a corner, or the "
            f"background is not flat. Corners: {corners.round(1).tolist()}"
        )

    content = np.abs(rgb - bg).max(axis=2) > tol_levels
    rows, cols = np.any(content, axis=1), np.any(content, axis=0)
    if not rows.any():
        raise ValueError(f"{path}: nothing but background — the render is empty")
    r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return {
        "width_fraction": round((c1 - c0) / w, 3),
        "height_fraction": round((r1 - r0) / h, 3),
        "coverage_fraction": round(float(content.mean()), 3),
        "background_rgb": bg.round(1).tolist(),
        "touches_edge": bool(c0 == 0 or r0 == 0 or c1 == w or r1 == h),
    }


def selftest() -> dict:
    """Check the two sign conventions, the section cut, and the framing fit.

    Every bug this guards against produced an image that looked entirely
    reasonable — an upside-down floor plan, a section that cut away the far half
    so the camera stared at a wall, and a subject occupying a third of the frame
    with the boxes running off the edges. None is detectable from an exit code.
    """
    # A Z-up room: floor 0, ceiling +3.3, camera 1.4 m up.
    up = detect_up_sign(np.array([0.0, 3.3]), np.array([1.4]))
    # The real ScanNet++ scene: floor -0.04, ceiling -3.33, camera -1.44.
    down = detect_up_sign(np.array([-3.327, -0.036]), np.array([-1.44]))
    if (up, down) != (1, -1):
        raise AssertionError(f"detect_up_sign inverted: z-up gave {up}, z-down gave {down}")

    rng = np.random.default_rng(0)
    pts = rng.uniform([-4.0, -4.5, -3.3], [4.0, 4.5, 0.0], size=(40000, 3))
    centre = np.array([0.0, 0.0, -1.65])
    fractions = []
    for az in np.linspace(0.0, 2 * np.pi, 8, endpoint=False):
        keep = cutaway_keep_mask(pts, centre, float(az))
        n = np.array([np.cos(az), np.sin(az), 0.0])
        # Every surviving point must be on the far side of the plane, or it
        # would stand between the camera and the room it is meant to reveal.
        if float(((pts[keep] - centre) @ n).max()) > 1e-9:
            raise AssertionError(f"cut kept geometry on the camera side at az={az:.2f}")
        fractions.append(float(keep.mean()))
    if not all(abs(f - 0.5) < 0.02 for f in fractions):
        raise AssertionError(f"cut is not halving the room: {fractions}")
    if cutaway_keep_mask(pts, centre, 0.0, cut_offset_m=1.0).mean() <= 0.5:
        raise AssertionError("positive cut_offset_m must keep more than half")

    # Framing. A unit cube viewed along +x, where the distance is analytic:
    # the nearest corner sits at depth d-1 and its 1 m offset must land on the
    # frame edge, so d = 1/tan(fov/2) + 1 = 2.7320508 at fov 60 and aspect 1.
    cube = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                    dtype=float)
    origin, fwd, upv = np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    d1 = _fit_eye_distance(cube, origin, fwd, upv, 60.0, 1.0, margin=1.0)
    if abs(d1 - (1.0 / np.tan(np.radians(30.0)) + 1.0)) > 1e-9:
        raise AssertionError(f"framing fit is wrong on the analytic cube: {d1}")

    # And on the real shape of the problem — a wide, low room seen obliquely,
    # where a bounding-radius fit was out by a factor of nearly three. Measured
    # by projecting, not by re-deriving the same algebra, and measured as the
    # **span** across the frame rather than the largest screen coordinate: the
    # two differ, and only the span is what the pixels show. Aiming at the centre
    # of the 3D bounding box put the content off-centre on screen — perspective
    # magnifies its near half — and the first sections rendered that way spanned
    # 0.81 of the width while satisfying a fit that promised 0.94.
    room = np.array([[sx, sy, sz] for sx in (-4.0, 4.0)
                     for sy in (-4.5, 4.5) for sz in (-1.6, 1.6)])
    margin, aspect, fov = 1.06, 1900 / 1300, 60.0
    el, az = np.radians(22.0), 0.9
    direction = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    tgt, d2 = _frame_content(room, direction, [0, 0, 1], fov, aspect, margin)
    sw, sh = _screen_span(room, tgt, -direction, [0, 0, 1], fov, aspect, d2)
    fill = max(sw, sh)
    if abs(fill - 1.0 / margin) > 1e-4:
        raise AssertionError(
            f"framing is not tight: content spans {sw:.4f} x {sh:.4f} of the frame, "
            f"expected {1.0 / margin:.4f} on the binding axis"
        )
    # Two-sided, for the same reason detect_up_sign needed two cases: a fit that
    # merely zoomed far enough out would pass the check above just as well.
    closer = max(_screen_span(room, tgt, -direction, [0, 0, 1], fov, aspect, d2 * 0.95))
    if closer <= 1.0 / margin:
        raise AssertionError("framing is loose: pulling 5% closer did not fill more frame")
    # And centred, which is the part the span check cannot see on its own.
    x0, x1, y0, y1 = _screen_bounds(room, tgt, -direction, [0, 0, 1], fov, aspect, d2)
    if max(abs(x0 + x1), abs(y0 + y1)) / 2.0 > 1e-3:
        raise AssertionError(
            f"framing is off-centre: x [{x0:.4f}, {x1:.4f}], y [{y0:.4f}, {y1:.4f}]"
        )

    # The section's drawn geometry and its framing points must be the same set.
    # When they were not, the camera was fitted to a percentile-clipped subset
    # while every kept triangle was rendered, so the subject overflowed the frame
    # it had been fitted to and three delivered stills were cropped at the border.
    # A fit that is exact on the analytic room above cannot detect that: the
    # arithmetic was right and the input was wrong.
    #
    # Two cases, because the guard has two branches and a test that only takes one
    # of them proves half the function. A grid plus one far spike: the clip should
    # engage and the spike should leave the drawn set. A bare grid whose extremes
    # are dense walls: the clip would cut real surface, so it must be abandoned
    # and *everything* must stay drawn.
    gx, gy = np.meshgrid(np.linspace(-3, 3, 12), np.linspace(-3, 3, 12))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    quads = [[r * 12 + c, r * 12 + c + 1, (r + 1) * 12 + c] for r in range(11) for c in range(11)]
    clip_report = {}
    for case, extra in (("with_spike", np.array([[40.0, 40.0, 40.0]])), ("dense_only", None)):
        v = grid if extra is None else np.vstack([grid, extra])
        t = np.array(quads if extra is None
                     else quads + [[len(grid), 0, 1]], dtype=int)
        keep = np.ones(len(t), dtype=bool)
        draw, framed = _section_and_clip(v, t, keep, pct=0.5)

        drawn_pts = v[np.unique(t[draw])]
        if framed.shape != drawn_pts.shape or not np.array_equal(
                np.sort(framed, axis=0), np.sort(drawn_pts, axis=0)):
            raise AssertionError(
                f"{case}: framing set is not the drawn set — "
                f"{len(framed)} framed vs {len(drawn_pts)} drawn"
            )
        reach = float(np.abs(drawn_pts).max())
        clip_report[case] = {"triangles": int(draw.sum()), "reach_m": round(reach, 2)}

    if clip_report["with_spike"]["reach_m"] > 10.0:
        raise AssertionError(
            "the clip did not remove an isolated spike 40 m out: "
            f"drawn geometry still reaches {clip_report['with_spike']['reach_m']} m"
        )
    if clip_report["dense_only"]["triangles"] != len(quads):
        raise AssertionError(
            "the clip ate real surface on a mesh with no outliers: kept "
            f"{clip_report['dense_only']['triangles']} of {len(quads)} triangles"
        )

    return {
        "detect_up_sign": [up, down],
        "kept_fraction_per_azimuth": fractions,
        "frame_span_fraction": [round(sw, 4), round(sh, 4)],
        "fit_distance_m": round(d2, 3),
        "section_clip": clip_report,
    }


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


def _box_corners(center: np.ndarray, size: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """The eight corners of an oriented box, along the box's own axes.

    Shared with ``_box_lineset`` so that what gets framed and what gets drawn can
    never disagree about where a box is. B4 was exactly that disagreement.
    """
    center = np.asarray(center, dtype=float)
    half = (np.asarray(axes, dtype=float).T * (np.asarray(size, dtype=float) / 2.0)).T
    return np.array(
        [center + sx * half[0] + sy * half[1] + sz * half[2]
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )


def _box_lineset(center: np.ndarray, size: np.ndarray, axes: np.ndarray, color: list[float]):
    import open3d as o3d

    corners = _box_corners(center, size, axes)
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


def encode_video(
    frame_dir: str | Path,
    out_path: str | Path,
    pattern: str = "%04d.png",
    fps: int = 30,
    crf: int = 18,
) -> Path:
    """Encode a numbered frame sequence to H.264.

    crf 18 is visually lossless for this material; the reconstruction's own
    artefacts should be what a viewer sees, not the codec's.
    """
    import subprocess

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(Path(frame_dir) / pattern),
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        # H.264 needs even dimensions; renders are not guaranteed to have them
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-800:]}")
    return out


def turntable_video(
    mesh_path: str | Path,
    out_path: str | Path,
    n_frames: int = 180,
    size: tuple[int, int] = (1280, 900),
    elevation_deg: float = 25.0,
    fps: int = 30,
    boxes=None,
    min_size_m: float = 0.0,
) -> Path:
    """One full orbit of the reconstruction, optionally with object boxes.

    A still frame cannot show whether a surface is flat or whether a box sits
    where the object is; a rotation can. This is the artefact worth showing
    someone first.
    """
    import tempfile

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    centre = verts.mean(axis=0)

    r = _renderer(*size)
    r.scene.add_geometry("mesh", mesh, _lit_material())

    extremes = [verts]
    if boxes:
        line_mat = _line_material(2.5)
        for i, b in enumerate(boxes):
            if float(np.max(b.size)) < min_size_m:
                continue
            colour = SHELL_COLOR if getattr(b, "is_shell", False) else BOX_COLOR
            r.scene.add_geometry(f"box_{i}", _box_lineset(b.center, b.size, b.axes, colour), line_mat)
            extremes.append(np.array([b.center - b.size / 2, b.center + b.size / 2]))

    radius = float(np.linalg.norm(np.vstack(extremes) - centre, axis=1).max()) * (1.6 if boxes else 1.35)
    el = np.radians(elevation_deg)

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n_frames):
            az = 2 * np.pi * i / n_frames
            eye = centre + radius * np.array(
                [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
            )
            r.scene.camera.look_at(centre.tolist(), eye.tolist(), [0, 0, 1])
            o3d.io.write_image(f"{tmp}/{i:04d}.png", r.render_to_image())
        return encode_video(tmp, out_path, fps=fps)


def cutaway_keep_mask(
    centroids: np.ndarray,
    centre: np.ndarray,
    azimuth_rad: float,
    cut_offset_m: float = 0.0,
) -> np.ndarray:
    """Triangles to keep for a vertical section facing a camera at ``azimuth_rad``.

    The plane normal is horizontal and points *at* the camera, so the half of
    the room between the viewer and the centre is discarded and the viewer looks
    into the section. Rotating the normal with the azimuth is what makes an
    orbit useful: the removed half is always the one that would be in the way.

    ``cut_offset_m`` slides the plane along its normal — positive keeps more
    than half, which helps when the interesting furniture straddles the centre.

    Separated from the rendering so it can be checked without a GPU.
    """
    n = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0])
    return (centroids - centre) @ n < cut_offset_m


@dataclass
class _CutawayScene:
    mesh: object
    centroids: np.ndarray
    centre: np.ndarray          # xy centre of the footprint, z at mid-height
    floor_z: float
    height: float
    up_sign: int
    radius: float
    below_ceiling: np.ndarray   # triangle mask, ceiling slab already removed
    cut_z: float                # height of the ceiling cut, in scene units
    n_triangles: int
    lost_colour: bool


def _prepare_cutaway(
    mesh_or_path,
    cameras_json: str | Path | None,
    up_sign: int | None,
    ceiling_keep: float,
    zoom: float,
    max_triangles: int | None,
) -> _CutawayScene:
    """Load, decimate, and work out which way is up and where the ceiling is.

    Accepts an already-loaded mesh so a caller can hand over the output of
    ``load_mesh_in_box_frame`` — a ``.glb`` read straight from disk is in the
    wrong frame and would section the room along the wrong axis.

    Decimation is not cosmetic. The full TSDF mesh is 18.5 M triangles and a
    plain 180-frame turntable of it took 1358 s; a cutaway re-uploads geometry
    every frame because the cut changes, so the full mesh would be far worse.
    """
    import open3d as o3d

    if hasattr(mesh_or_path, "triangles"):
        mesh = mesh_or_path
    else:
        mesh = o3d.io.read_triangle_mesh(str(mesh_or_path))
    had_colour = mesh.has_vertex_colors()
    if max_triangles and len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(int(max_triangles))
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    centroids = verts[tris].mean(axis=1)

    if up_sign is None and cameras_json is not None:
        cams = load_cameras(cameras_json)
        up_sign = detect_up_sign(verts[:, 2], np.array([c.position[2] for c in cams]))
    if up_sign is None:
        up_sign = 1

    # Percentiles rather than min/max: isolated fusion spikes stick several
    # metres out of the walls and would put the ceiling cut in mid-air.
    z_lo, z_hi = (float(v) for v in np.percentile(verts[:, 2], [1.0, 99.0]))
    floor_z, ceil_z = (z_lo, z_hi) if up_sign > 0 else (z_hi, z_lo)
    height = abs(ceil_z - floor_z)

    # Take the ceiling off as well as the near wall. A raised camera looking
    # into a section still has the far ceiling between it and the far floor.
    cut_z = floor_z + up_sign * height * ceiling_keep
    below_ceiling = (centroids[:, 2] < cut_z) if up_sign > 0 else (centroids[:, 2] > cut_z)

    xy_lo = np.percentile(verts[:, :2], 1.0, axis=0)
    xy_hi = np.percentile(verts[:, :2], 99.0, axis=0)
    centre = np.array([
        (xy_lo[0] + xy_hi[0]) / 2.0,
        (xy_lo[1] + xy_hi[1]) / 2.0,
        floor_z + up_sign * height / 2.0,
    ])
    radius = float(np.linalg.norm(xy_hi - xy_lo)) / 2.0 * zoom

    return _CutawayScene(
        mesh=mesh,
        centroids=centroids,
        centre=centre,
        floor_z=floor_z,
        height=height,
        up_sign=int(up_sign),
        radius=radius,
        below_ceiling=below_ceiling,
        cut_z=float(cut_z),
        n_triangles=len(tris),
        lost_colour=had_colour and not mesh.has_vertex_colors(),
    )


def _sectioned_mesh(mesh, keep: np.ndarray):
    """A mesh sharing the source vertices, holding only the kept triangles.

    Cheaper than copying and calling ``remove_triangles_by_mask``, which
    rebuilds the vertex array every frame. Unreferenced vertices are left in
    place; the renderer simply does not draw them.
    """
    import open3d as o3d

    out = o3d.geometry.TriangleMesh()
    out.vertices = mesh.vertices
    out.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[keep])
    if mesh.has_vertex_colors():
        out.vertex_colors = mesh.vertex_colors
    if mesh.has_vertex_normals():
        out.vertex_normals = mesh.vertex_normals
    return out


def _set_projection(r, fov_deg: float, aspect: float,
                    near: float = 0.05, far: float = 400.0) -> None:
    """Pin the projection rather than relying on the renderer's default.

    The framing solver has to know the field of view exactly. Leaving it implicit
    would make the fit an assumption about Open3D's default instead of a
    computation, and this project has paid for enough implicit conventions.
    """
    import open3d as o3d

    r.scene.camera.set_projection(
        float(fov_deg), float(aspect), float(near), float(far),
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )


def _camera_basis(forward, up):
    """Orthonormal (forward, up, right), with ``up`` reduced to the part across the view."""
    forward = np.asarray(forward, dtype=float)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=float)
    up = up - forward * float(up @ forward)
    n = float(np.linalg.norm(up))
    if n < 1e-9:
        raise ValueError("up is parallel to the view direction; framing is undefined")
    up = up / n
    return forward, up, np.cross(forward, up)


def _project_screen(
    points: np.ndarray,
    target: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    aspect: float,
    distance: float,
):
    """Screen coordinates and depths for a symmetric perspective frame.

    Screen units put the frame edge at 1.0. The eye is at
    ``target - distance * forward``. Returns ``(sx, sy, depth)``, with ``depth``
    negative for anything behind the eye so callers can reject it.
    """
    forward, up, right = _camera_basis(forward, up)
    rel = np.asarray(points, dtype=float).reshape(-1, 3) - np.asarray(target, dtype=float)
    th = np.tan(np.radians(fov_deg) / 2.0)
    depth = distance + rel @ forward
    if float(depth.min()) <= 1e-9:
        return None, None, depth
    return (rel @ right) / (depth * th * aspect), (rel @ up) / (depth * th), depth


def _screen_bounds(
    points: np.ndarray,
    target: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    aspect: float,
    distance: float,
) -> np.ndarray:
    """Screen-space bounding box ``[x_min, x_max, y_min, y_max]`` in frame units.

    1.0 is the frame edge. An independent projection of the same geometry the fit
    solves for, so framing can be checked by measurement rather than by
    re-deriving its own algebra. Returns infinities when anything is at or behind
    the eye, which is an overflow by any definition.
    """
    sx, sy, _ = _project_screen(points, target, forward, up, fov_deg, aspect, distance)
    if sx is None:
        return np.array([-np.inf, np.inf, -np.inf, np.inf])
    return np.array([sx.min(), sx.max(), sy.min(), sy.max()])


def _screen_span(*args, **kwargs) -> tuple[float, float]:
    """Fraction of the frame width and height the content spans.

    This — not the largest ``|screen coordinate|`` — is what the rendered pixels
    show, so it is what the fit has to maximise and what the checks assert.
    """
    x0, x1, y0, y1 = _screen_bounds(*args, **kwargs)
    return float((x1 - x0) / 2.0), float((y1 - y0) / 2.0)


def _fit_eye_distance(
    points: np.ndarray,
    target: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    aspect: float,
    margin: float = 1.06,
) -> float:
    """The smallest camera distance that keeps every point inside the frustum.

    Framing by a bounding radius — what the cutaway did before — sets the
    distance from the whole room's footprint and then aims the camera at content
    that is half a room further away. The subject came out about a third of the
    frame high with the rest empty, and boxes ran off the edges. This solves for
    the distance instead.

    With the eye at ``target - d * forward``, a point at camera-space
    ``(x, y, z)`` — z measured along ``forward``, so larger z is further from the
    camera — is inside the frustum when ``|x| <= (d + z) * tan(fov_h / 2)`` and
    ``|y| <= (d + z) * tan(fov_v / 2)``. Each point therefore imposes a lower
    bound on ``d``, and the answer is the largest of those bounds.

    ``margin`` shrinks the usable frustum, so the content reaches at most
    ``1 / margin`` of the way to an edge and nothing sits flush against one.
    """
    forward, up, right = _camera_basis(forward, up)
    rel = np.asarray(points, dtype=float).reshape(-1, 3) - np.asarray(target, dtype=float)
    if not len(rel):
        raise ValueError("nothing to frame: no points given")

    th = np.tan(np.radians(fov_deg) / 2.0) / margin
    tw = th * aspect
    z = rel @ forward
    d = max(
        float(np.max(np.abs(rel @ right) / tw - z)),
        float(np.max(np.abs(rel @ up) / th - z)),
    )
    # A target sitting inside the content can put every bound below zero; the
    # camera still has to be outside it.
    return max(d, 1e-3)


def _frame_content(
    points: np.ndarray,
    direction: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    aspect: float,
    margin: float = 1.06,
    iters: int = 24,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Look-at target and eye distance that centre ``points`` and fill the frame.

    Solving the distance alone is not enough. With the target at the centre of the
    content's *bounding box*, perspective projects the near half larger than the
    far half, so the content lands off-centre on screen and the frame it actually
    spans is smaller than the frustum fit implies — 0.81 of the width where the
    fit promised 0.94, on the first sections rendered this way.

    So this alternates: solve the distance for the current target, then slide the
    target across the view to bring the content back to the middle of the frame.

    The slide has to be scaled by the depth of the points that *bind* the frame,
    not by the depth of the target. Moving the camera by ``δ`` moves a point's
    screen position by ``δ / depth``, so scaling by the target's depth
    over-corrects for everything nearer than the target — which is precisely the
    geometry touching the frame edge. That version oscillated with a growing
    amplitude (vertical centre −0.28, −0.01, +0.03, −0.06, +0.10, −0.13, +0.17)
    and left the content at 0.85 of the frame. Solving the two binding points
    exactly gives ``δ = tan · (s_hi + s_lo) / (1/depth_hi + 1/depth_lo)``, a
    Newton step that converges in a handful of rounds.

    ``direction`` points from the target toward the eye. Returns the best iterate
    rather than the last, so a scene whose binding points keep swapping still gets
    the tightest framing found instead of wherever the loop happened to stop.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    forward = -np.asarray(direction, dtype=float)
    lo, hi = _aabb(points)
    target = (lo + hi) / 2.0
    d = _fit_eye_distance(points, target, forward, up, fov_deg, aspect, margin)

    _f, u, r = _camera_basis(forward, up)
    th = np.tan(np.radians(fov_deg) / 2.0)
    best = (target, d, -np.inf)
    for _ in range(iters):
        sx, sy, depth = _project_screen(points, target, forward, up, fov_deg, aspect, d)
        if sx is None:
            break                     # content reached the eye; keep the last good one
        span = max(float(sx.max() - sx.min()), float(sy.max() - sy.min())) / 2.0
        if span > best[2]:
            best = (target, d, span)

        shift = np.zeros(3)
        centred = True
        for s, axis, tan_half in ((sx, r, th * aspect), (sy, u, th)):
            i, j = int(np.argmax(s)), int(np.argmin(s))
            offset = float(s[i] + s[j])                  # twice the screen centre
            if abs(offset) <= 2.0 * tol:
                continue
            centred = False
            shift = shift + axis * (
                tan_half * offset / (1.0 / depth[i] + 1.0 / depth[j])
            )
        if centred:
            break
        target = target + shift
        d = _fit_eye_distance(points, target, forward, up, fov_deg, aspect, margin)

    return best[0], best[1]


def _section_and_clip(
    verts: np.ndarray,
    tris: np.ndarray,
    keep: np.ndarray,
    pct: float,
    max_clipped_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangles to draw, and the points to frame on — the same set, by construction.

    The bug this exists to close: the section was drawn from every kept triangle
    while the camera was fitted to a ``_percentile_clip`` *subset* of those
    vertices. Whatever the clip discarded was still rendered, so the subject
    overflowed the frame it had been fitted to — measured at 0.973–0.978 of the
    width against a 1/1.06 = 0.943 target, with ``touches_edge`` true on three of
    the delivered ``cutaway_nobox`` stills. The framing arithmetic was never
    wrong; it was being asked about different geometry than the renderer drew.

    So framing points are *derived from* the returned triangle mask rather than
    computed alongside it. Every return path below goes through the same final
    expression, which is what makes "drawn == framed" hold rather than be
    something a future edit has to remember.

    Clipping now removes triangles as well, and that has a failure mode the
    framing-only version did not: a percentile is a rank, not a distance, and a
    room's extreme coordinate is a *wall* carrying thousands of near-coplanar
    vertices. Cutting at the 0.5th percentile there can slice a visible slab off
    the wall instead of trimming isolated TSDF spikes. When the clip would drop
    more than ``max_clipped_fraction`` of the section, it is abandoned for this
    frame: the fit then contains every drawn vertex, spikes included, which
    frames wider than ideal but still crops nothing. A loose frame is a cosmetic
    problem; a cropped one is the bug.

    What the clip is for, when it does apply: TSDF fusion leaves isolated
    vertices standing metres outside the walls, and framing on one pushes the
    camera back and undoes the fit. ``_prepare_cutaway`` takes percentiles rather
    than min/max for the same reason.

    Mesh vertices only. Box corners are annotations, not noisy reconstruction
    output, and they are unioned in by the caller *after* this returns —
    deliberately outside the clip, because clipping a box corner is what put one
    off the edge of the frame in bug B4.
    """
    if not keep.any():
        return keep, verts[:0]

    tri_keep = keep
    used = np.unique(tris[keep])
    pts = verts[used]
    if pct > 0.0 and len(pts):
        lo = np.percentile(pts, pct, axis=0)
        hi = np.percentile(pts, 100.0 - pct, axis=0)
        inside = np.zeros(len(verts), dtype=bool)
        inside[used] = np.all((pts >= lo) & (pts <= hi), axis=1)
        clipped = keep & inside[tris].all(axis=1)
        dropped = 1.0 - clipped.sum() / keep.sum()
        if clipped.any() and dropped <= max_clipped_fraction:
            tri_keep = clipped

    return tri_keep, verts[np.unique(tris[tri_keep])]


def _cutaway_eye(
    scene: _CutawayScene,
    azimuth_rad: float,
    elevation_deg: float,
    content: np.ndarray | None = None,
    aspect: float = 1.0,
    fov_deg: float = 60.0,
    margin: float = 1.06,
):
    """Camera position and look-at target for one section view.

    Given ``content`` — the points actually drawn this frame — the target and the
    distance are solved together so the content is centred in the frame and fills
    it. Without it this falls back to the footprint radius, which frames the whole
    room rather than the section that survived the cut.
    """
    el = np.radians(elevation_deg)
    direction = np.array([
        np.cos(azimuth_rad) * np.cos(el),
        np.sin(azimuth_rad) * np.cos(el),
        scene.up_sign * np.sin(el),
    ])
    if content is None or not len(content):
        target = np.array([
            scene.centre[0],
            scene.centre[1],
            scene.floor_z + scene.up_sign * scene.height * 0.35,
        ])
        return target, target + scene.radius * direction

    target, d = _frame_content(
        content, direction, [0.0, 0.0, float(scene.up_sign)],
        fov_deg, aspect, margin,
    )
    return target, target + d * direction


def _below_ceiling_cut(z: float, scene: _CutawayScene) -> bool:
    """Whether a point at height ``z`` survived the ceiling cut.

    The mesh loses its top slab but box visibility used to test only the vertical
    section plane, so boxes near the ceiling were drawn over empty space above
    the cut wall tops — clearly visible in the first delivered stills.
    """
    return z < scene.cut_z if scene.up_sign > 0 else z > scene.cut_z


def _cutaway_boxes(r, boxes, min_size_m: float, show_shell: bool = False):
    """Add every box once; visibility is toggled per frame instead of re-adding.

    Returns geometry name, centre, shell flag and corners per box, so the caller
    can hide the ones the section removed rather than leaving them floating in
    the void, and can frame on the ones it kept.

    Shell boxes are off by default. In this dataset they are oriented slabs whose
    corner wrap spans 11.4 m across a 6.4 m floor (bug B1), one "ceiling" is a
    24 cm sliver, and the section has already removed the near wall and ceiling
    they describe — so drawing them clutters the interior and drags the camera
    back. ``show_shell=True`` restores them for callers that want structure.
    """
    if not boxes:
        return []
    line_mat = _line_material(2.5)
    registered = []
    for i, b in enumerate(boxes):
        is_shell = bool(getattr(b, "is_shell", False))
        if is_shell and not show_shell:
            continue
        if float(np.max(b.size)) < min_size_m:
            continue
        colour = SHELL_COLOR if is_shell else BOX_COLOR
        name = f"box_{i}"
        centre = np.asarray(b.center, dtype=float)
        size = np.asarray(b.size, dtype=float)
        axes = np.asarray(b.axes, dtype=float)
        r.scene.add_geometry(name, _box_lineset(centre, size, axes, colour), line_mat)
        registered.append((name, centre, is_shell, _box_corners(centre, size, axes)))
    return registered


def cutaway_turntable(
    mesh_path: str | Path,
    out_path: str | Path,
    n_frames: int = 72,
    size: tuple[int, int] = (1280, 900),
    elevation_deg: float = 22.0,
    fps: int = 24,
    boxes=None,
    min_size_m: float = 0.0,
    cameras_json: str | Path | None = None,
    up_sign: int | None = None,
    ceiling_keep: float = 0.82,
    cut_offset_m: float = 0.0,
    zoom: float = 1.25,
    max_triangles: int | None = 400_000,
    frame_dir: str | Path | None = None,
    fov_deg: float = 60.0,
    margin: float = 1.06,
    show_shell: bool = False,
    content_percentile: float = 0.5,
) -> Path:
    """Orbit a dollhouse section: the near half of the room is always removed.

    The plain turntable places the camera outside the object radius, so every
    frame of it shows the outside of the ceiling and walls and none of the
    furniture — which is why the interior was invisible in every video delivered
    so far. Here the cutting plane rotates with the camera, so the viewer is
    always looking into an open half of the room and the objects inside it are
    the subject.

    ``frame_dir`` writes frames to a real directory instead of a temporary one
    and skips any that already exist, so a dropped connection costs only the
    frames still to be rendered. That has already happened once: these functions
    run in-process, unlike the pipeline stages, which are subprocesses and
    survive.

    Raises rather than returning a misleading video when the section is
    degenerate — a cut that removed nothing looks like a normal turntable, and a
    cut that removed everything still encodes cleanly.
    """
    import tempfile

    import open3d as o3d

    scene = _prepare_cutaway(mesh_path, cameras_json, up_sign, ceiling_keep, zoom, max_triangles)

    r = _renderer(*size)
    aspect = size[0] / size[1]
    _set_projection(r, fov_deg, aspect)
    mat = _lit_material()
    registered = _cutaway_boxes(r, boxes, min_size_m, show_shell)

    verts = np.asarray(scene.mesh.vertices)
    tris = np.asarray(scene.mesh.triangles)
    kept_fractions: list[float] = []
    interior_counts: list[int] = []
    eye_distances: list[float] = []

    def render(target_dir: str) -> None:
        for i in range(n_frames):
            path = Path(target_dir) / f"{i:04d}.png"
            az = 2 * np.pi * i / n_frames
            normal = np.array([np.cos(az), np.sin(az), 0.0])

            keep = cutaway_keep_mask(scene.centroids, scene.centre, az, cut_offset_m)
            keep &= scene.below_ceiling
            kept_fractions.append(float(keep.mean()))

            visible_interior = 0
            visible_corners = []
            for name, centre, is_shell, corners in registered:
                shown = bool(
                    (centre - scene.centre) @ normal < cut_offset_m
                    and _below_ceiling_cut(float(centre[2]), scene)
                )
                r.scene.show_geometry(name, shown)
                if shown:
                    visible_corners.append(corners)
                    if not is_shell:
                        visible_interior += 1
            interior_counts.append(visible_interior)

            if path.exists():
                continue

            r.scene.remove_geometry("mesh")
            draw, framed = _section_and_clip(verts, tris, keep, content_percentile)
            r.scene.add_geometry("mesh", _sectioned_mesh(scene.mesh, draw), mat)
            content = np.vstack([framed, *visible_corners])
            target, eye = _cutaway_eye(scene, az, elevation_deg, content, aspect, fov_deg, margin)
            eye_distances.append(float(np.linalg.norm(eye - target)))
            r.scene.camera.look_at(target.tolist(), eye.tolist(), [0, 0, scene.up_sign])
            o3d.io.write_image(str(path), r.render_to_image())

    tmp = None
    if frame_dir is not None:
        d = Path(frame_dir)
        d.mkdir(parents=True, exist_ok=True)
        render(str(d))
        frames = str(d)
    else:
        tmp = tempfile.TemporaryDirectory()
        render(tmp.name)
        frames = tmp.name

    mean_kept = float(np.mean(kept_fractions))
    summary = {
        "triangles_after_decimation": scene.n_triangles,
        "up_sign": scene.up_sign,
        "floor_z": round(scene.floor_z, 3),
        "room_height_m": round(scene.height, 3),
        "mean_kept_triangle_fraction": round(mean_kept, 3),
        "boxes_drawn": len(registered),
        "mean_visible_object_boxes":
            round(float(np.mean(interior_counts)), 1) if registered else None,
        "eye_distance_m":
            [round(min(eye_distances), 2), round(max(eye_distances), 2)]
            if eye_distances else None,
        "vertex_colour_lost_in_decimation": scene.lost_colour,
    }
    print("cutaway:", summary)

    if not 0.1 <= mean_kept <= 0.9:
        raise RuntimeError(
            f"cutaway is degenerate — the section kept {mean_kept:.1%} of triangles, "
            f"expected roughly half. The cutting plane is probably not inside the "
            f"geometry: {summary}"
        )
    if registered and max(interior_counts) == 0:
        raise RuntimeError(
            "cutaway shows no object boxes in any frame, which is the failure it "
            f"exists to fix: {summary}"
        )

    out = encode_video(frames, out_path, fps=fps)
    if tmp is not None:
        tmp.cleanup()
    return out


def cutaway_views(
    mesh_path: str | Path,
    out_dir: str | Path,
    n_views: int = 4,
    size: tuple[int, int] = (1900, 1300),
    elevation_deg: float = 22.0,
    boxes=None,
    min_size_m: float = 0.0,
    cameras_json: str | Path | None = None,
    up_sign: int | None = None,
    ceiling_keep: float = 0.82,
    cut_offset_m: float = 0.0,
    zoom: float = 1.25,
    max_triangles: int | None = 400_000,
    stem: str = "cutaway",
    fov_deg: float = 60.0,
    margin: float = 1.06,
    show_shell: bool = False,
    content_percentile: float = 0.5,
) -> list[Path]:
    """Section stills at higher resolution, for a document rather than a video."""
    import open3d as o3d

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scene = _prepare_cutaway(mesh_path, cameras_json, up_sign, ceiling_keep, zoom, max_triangles)
    r = _renderer(*size)
    aspect = size[0] / size[1]
    _set_projection(r, fov_deg, aspect)
    mat = _lit_material()
    registered = _cutaway_boxes(r, boxes, min_size_m, show_shell)

    verts = np.asarray(scene.mesh.vertices)
    tris = np.asarray(scene.mesh.triangles)

    written: list[Path] = []
    for i in range(n_views):
        az = 2 * np.pi * i / n_views
        normal = np.array([np.cos(az), np.sin(az), 0.0])
        keep = cutaway_keep_mask(scene.centroids, scene.centre, az, cut_offset_m)
        keep &= scene.below_ceiling

        visible_corners = []
        for name, centre, _is_shell, corners in registered:
            shown = bool(
                (centre - scene.centre) @ normal < cut_offset_m
                and _below_ceiling_cut(float(centre[2]), scene)
            )
            r.scene.show_geometry(name, shown)
            if shown:
                visible_corners.append(corners)

        r.scene.remove_geometry("mesh")
        draw, framed = _section_and_clip(verts, tris, keep, content_percentile)
        r.scene.add_geometry("mesh", _sectioned_mesh(scene.mesh, draw), mat)
        content = np.vstack([framed, *visible_corners])
        target, eye = _cutaway_eye(scene, az, elevation_deg, content, aspect, fov_deg, margin)
        r.scene.camera.look_at(target.tolist(), eye.tolist(), [0, 0, scene.up_sign])

        p = out / f"{stem}_{i:02d}.png"
        o3d.io.write_image(str(p), r.render_to_image())
        written.append(p)
    return written


def frames_to_video(
    frame_dir: str | Path,
    out_path: str | Path,
    fps: int = 12,
    limit: int | None = None,
) -> Path:
    """Turn an arbitrarily-named image directory into a video, in sorted order.

    Used for the trained 3DGS renders, whose filenames follow the capture
    (DSC00850.jpg …) rather than a frame index.
    """
    import shutil
    import tempfile

    files = sorted(
        p for p in Path(frame_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise FileNotFoundError(f"no images in {frame_dir}")
    if limit:
        files = files[:limit]

    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(files):
            shutil.copy(f, f"{tmp}/{i:04d}{f.suffix.lower()}")
        return encode_video(tmp, out_path, pattern=f"%04d{files[0].suffix.lower()}", fps=fps)


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
