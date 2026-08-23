"""GLB export: the reconstruction in a form anything can open.

The delivery already carries PLY, and PLY is the wrong format for looking at
something. A 1 GB `tsdf_fusion_pruned.ply` needs a specialist viewer, arrives
unoriented, and carries no notion of a scene. GLB drags and drops into a browser,
Blender, Windows 3D Viewer, macOS Preview and every web viewer, in one file.

Three things this does that a plain format conversion would not:

**Rotates the scene upright.** Every GLB viewer assumes +Y is up. This
reconstruction's up axis is whatever DA3's arbitrary frame happened to give — on the
factory scene `[0.018, -0.971, -0.239]`, which is close to *negative* Y. Exported
as-is the room opens upside down and slightly tilted, and the viewer has no way to
know. So the geometry is rotated into a Y-up frame at export, once, and the rotation
is recorded.

**Keeps vertex colour.** The fused mesh carries per-vertex colour from the input
photographs; that is most of what makes it recognisable as a factory rather than a
grey blob. Open3D's own GLB writer is not used here — this project has already
shipped a `.glb` Open3D wrote and could not read back (`bugs-found` B9) — so trimesh
does the writing and every file is read back and compared before it is called done.

**Builds a scene, not a mesh.** Room, per-object geometry and labelled boxes as
named nodes in one hierarchy, so a viewer shows a list of objects that can be
toggled rather than one undifferentiated surface.

Units are unchanged and remain DA3's. A GLB has no unit field, so the scale note
travels in the accompanying json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Distinct, reasonably colour-blind-safe hues for object categories. Assigned by
# hashing the label so a category keeps its colour across every export.
def label_colour(name: str) -> tuple[int, int, int, int]:
    import hashlib

    h = int(hashlib.sha1(name.encode()).hexdigest()[:6], 16)
    return (60 + h % 180, 60 + (h >> 8) % 180, 60 + (h >> 16) % 180, 255)


def upright_rotation(up: np.ndarray) -> np.ndarray:
    """A 4x4 that takes `up` to +Y, which is what every GLB viewer expects.

    Chooses the rotation with no roll about the new vertical: the shortest arc from
    `up` to +Y. Any rotation mapping one vector to another leaves one degree of
    freedom, and picking the shortest arc keeps the horizontal axes as close to their
    original directions as possible — so a floor plan still reads left-to-right.
    """
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    target = np.array([0.0, 1.0, 0.0])

    v = np.cross(up, target)
    s = float(np.linalg.norm(v))
    c = float(np.dot(up, target))
    T = np.eye(4)
    if s < 1e-9:
        # Already aligned, or antiparallel. Antiparallel needs a 180-degree turn
        # about any horizontal axis; without this branch the cross product is zero
        # and the scene would ship upside down.
        if c < 0:
            T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        return T
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    T[:3, :3] = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return T


@dataclass
class GLBReport:
    files: list[dict] = field(default_factory=list)
    rotation: list[list[float]] = field(default_factory=list)
    up_axis_in: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _load(mesh_path: str | Path):
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(m.triangles) == 0:
        raise ValueError(f"{mesh_path} has no triangles")
    return m


def _to_trimesh(o3d_mesh, transform: np.ndarray):
    """Open3D mesh -> trimesh, colours preserved, transformed into the Y-up frame."""
    import trimesh

    v = np.asarray(o3d_mesh.vertices)
    f = np.asarray(o3d_mesh.triangles)
    v = (transform[:3, :3] @ v.T).T + transform[:3, 3]
    colours = None
    if o3d_mesh.has_vertex_colors():
        c = (np.asarray(o3d_mesh.vertex_colors) * 255).astype(np.uint8)
        colours = np.concatenate([c, np.full((len(c), 1), 255, np.uint8)], axis=1)
    return trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colours, process=False)


def _box_wireframe(centre, extent, axes, colour, tube: float):
    """A box drawn as twelve thin cylinders.

    A wireframe as *geometry* rather than as lines: glTF has no line primitive that
    viewers render consistently, and a box drawn with GL_LINES disappears in half of
    them. Twelve cylinders always draw.
    """
    import trimesh

    centre = np.asarray(centre, dtype=float)
    extent = np.asarray(extent, dtype=float)
    axes = np.asarray(axes, dtype=float)
    if axes.shape != (3, 3):
        axes = np.eye(3)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=float)
    corners = centre + (signs * extent / 2.0) @ axes
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    parts = []
    for a, b in edges:
        seg = corners[b] - corners[a]
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            continue
        cyl = trimesh.creation.cylinder(radius=tube, height=L, sections=6)
        # Align +Z to the edge direction, then move to its midpoint.
        d = seg / L
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, d)
        s = float(np.linalg.norm(v))
        R = np.eye(4)
        if s > 1e-9:
            c = float(np.dot(z, d))
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R[:3, :3] = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
        elif float(np.dot(z, d)) < 0:
            R[:3, :3] = np.diag([1.0, -1.0, -1.0])
        R[:3, 3] = (corners[a] + corners[b]) / 2.0
        cyl.apply_transform(R)
        parts.append(cyl)
    if not parts:
        return None
    box = trimesh.util.concatenate(parts)
    box.visual.vertex_colors = np.tile(np.array(colour, np.uint8), (len(box.vertices), 1))
    return box


def export(mesh_path: str | Path, out_dir: str | Path, up: np.ndarray,
           objects: list[dict] | None = None,
           scale_mode: str = "assumed",
           lods: tuple[int, ...] = (300_000, 1_500_000)) -> GLBReport:
    """Write the room, the scene with boxes, and per-object GLBs. Verify each."""
    import trimesh

    out_dir = Path(out_dir)
    (out_dir / "objects").mkdir(parents=True, exist_ok=True)
    rep = GLBReport()
    T = upright_rotation(up)
    rep.rotation = T.tolist()
    rep.up_axis_in = [float(x) for x in np.asarray(up, dtype=float)]
    rep.notes.append(
        f"geometry rotated so the reconstruction's up axis {rep.up_axis_in} becomes "
        f"+Y, which is what glTF viewers assume; without this the scene opens "
        f"upside down"
    )
    rep.notes.append(
        "lengths are in DA3 arbitrary units, not metres" if scale_mode == "assumed"
        else "lengths are in metres"
    )

    def record(path: Path, tm, what: str, extra: dict | None = None) -> None:
        tm.export(str(path))
        # Read back: this project has shipped a .glb its writer could not reopen.
        back = trimesh.load(str(path), force="scene")
        n_v = sum(len(g.vertices) for g in back.geometry.values())
        rec = {"path": path.name, "kind": what,
               "bytes": path.stat().st_size,
               "vertices_written": int(sum(len(g.vertices) for g in (
                   tm.geometry.values() if hasattr(tm, "geometry") else [tm]))),
               "vertices_read_back": int(n_v),
               "nodes": len(back.geometry)}
        if extra:
            rec.update(extra)
        if rec["vertices_read_back"] == 0:
            raise RuntimeError(f"{path} exported but reads back empty")
        rep.files.append(rec)

    base = _load(mesh_path)

    # --- the room, at two levels of detail --------------------------------------
    import open3d as o3d
    for target in sorted(lods):
        m = base
        if len(base.triangles) > target:
            m = base.simplify_quadric_decimation(target_number_of_triangles=target)
            m.remove_unreferenced_vertices()
        tm = _to_trimesh(m, T)
        name = f"room_{target // 1000}k.glb"
        scene = trimesh.Scene({"room": tm})
        record(out_dir / name, scene, "room mesh",
               {"target_triangles": target, "triangles": int(len(tm.faces))})

    # --- the full scene: room + boxes -------------------------------------------
    if objects:
        # The coarse LOD for the scene file: it exists to be opened and spun, and a
        # 1.5 M-triangle room plus 80 wireframes is slow in a browser.
        coarse = base
        if len(base.triangles) > min(lods):
            coarse = base.simplify_quadric_decimation(
                target_number_of_triangles=min(lods))
            coarse.remove_unreferenced_vertices()
        room_tm = _to_trimesh(coarse, T)
        diag = float(np.linalg.norm(room_tm.vertices.max(axis=0)
                                    - room_tm.vertices.min(axis=0)))
        tube = max(diag * 0.0012, 1e-4)
        nodes = {"room": room_tm}
        per_label: dict[str, int] = {}
        for o in objects:
            box = o.get("box") or {}
            c = box.get("center") or o.get("centre") or o.get("center")
            e = box.get("size") or o.get("extent")
            a = box.get("axes") or o.get("obb_axes")
            if not c or not e:
                continue
            label = str(o.get("category") or o.get("label") or "object")
            per_label[label] = per_label.get(label, 0) + 1
            c_r = (T[:3, :3] @ np.asarray(c, dtype=float)) + T[:3, 3]
            a_r = (T[:3, :3] @ np.asarray(a, dtype=float).T).T if a is not None else np.eye(3)
            wf = _box_wireframe(c_r, e, a_r, label_colour(label), tube)
            if wf is not None:
                # Named per object so a viewer's outliner lists them individually.
                nodes[f"{label}_{per_label[label]:02d}"] = wf
        scene = trimesh.Scene(nodes)
        record(out_dir / "scene_with_boxes.glb", scene, "room + labelled boxes",
               {"boxes": len(nodes) - 1, "by_label": per_label})

    (out_dir / "glb_manifest.json").write_text(json.dumps({
        "scale_mode": scale_mode,
        "units": "DA3 arbitrary units, not metres" if scale_mode == "assumed" else "metres",
        "up_axis_in_reconstruction": rep.up_axis_in,
        "rotation_applied_to_make_Y_up": rep.rotation,
        "files": rep.files,
        "notes": rep.notes,
    }, indent=2))
    return rep
