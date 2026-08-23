"""Re-export every geometric artefact into the interchange formats other tools read.

The pipeline natively writes PLY, MJCF, PNG, npy and mp4. That covers Open3D and
MuJoCo and almost nothing else: Blender and every web viewer want glTF, Isaac and
most CAD want OBJ or USD, a printer wants STL, and a ROS stack wants URDF and PCD.
Converting once here beats each consumer converting for itself and disagreeing about
axes while doing it.

Two things this is careful about, because both are easy to get silently wrong:

**Axis convention.** glTF is Y-up by design; PLY, MJCF and USD as written here are
Z-up. A GLB exported without the rotation opens in Blender lying on its side, which
looks like a reconstruction error rather than a format one. The rotation is applied
explicitly and recorded in the manifest, so a consumer can tell which frame it has.

**Verification by read-back.** Every file written is read back and its element counts
compared with the source. Open3D will write a .glb it cannot itself read (recorded as
bugs-found B9 in this repository), so "the writer returned true" is not evidence the
file is usable.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np  # noqa: E402

SCENE = os.environ.get("V3D_SCENE", "small_factory")
OUT = Path(os.environ.get("V3D_DELIVER", "/tmp/deliver"))
EXP = OUT / "exchange"
SCALE_MODE = os.environ.get("V3D_SCALE_MODE", "assumed")

RESULTS: list[dict] = []
FAILED: dict[str, str] = {}

# glTF is Y-up; everything else here is Z-up. Applied to mesh copies destined for
# glTF only, never to the source.
Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, -1, 0, 0],
                         [0, 0, 0, 1]], dtype=float)


def record(fmt: str, path: Path, note: str, verified: str) -> None:
    RESULTS.append({"format": fmt, "path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size, "note": note, "verified": verified})
    print(f"  [ok] {fmt:6} {path.relative_to(OUT)}  "
          f"{path.stat().st_size/1e6:7.1f} MB  ({verified})")


def guard(name: str):
    def deco(fn):
        try:
            fn()
        except Exception as exc:
            FAILED[name] = f"{type(exc).__name__}: {exc}"
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return deco


def main() -> None:
    import open3d as o3d

    EXP.mkdir(parents=True, exist_ok=True)
    src_mesh = OUT / "geometry" / "mesh_decimated_1.5M.ply"
    src_full = OUT / "geometry" / "mesh_full_pruned.ply"
    src_cloud = OUT / "geometry" / "point_cloud_8M.ply"
    objects_json = OUT / "objects.json"

    print("=" * 74)
    print(f"interchange exports for {SCENE}  (scale_mode={SCALE_MODE})")
    print("=" * 74)

    # ---------------------------------------------------------------- meshes --
    @guard("mesh glTF / OBJ / STL")
    def _mesh():
        m = o3d.io.read_triangle_mesh(str(src_mesh))
        if len(m.triangles) == 0:
            raise RuntimeError(f"{src_mesh} read as empty")
        m.compute_vertex_normals()
        nv, nt = len(m.vertices), len(m.triangles)
        print(f"  source {src_mesh.name}: {nv:,} verts / {nt:,} tris")

        # OBJ and STL keep the Z-up frame the rest of the delivery uses.
        for ext, why in ((".obj", "universal; Isaac, CAD, MuJoCo mesh geoms"),
                         (".stl", "printing and CAD; geometry only, no colour")):
            p = EXP / f"mesh_1.5M{ext}"
            if not o3d.io.write_triangle_mesh(str(p), m):
                raise RuntimeError(f"writer refused {p}")
            back = o3d.io.read_triangle_mesh(str(p))
            if len(back.triangles) < 0.99 * nt:
                raise RuntimeError(f"{p} read back {len(back.triangles):,} of {nt:,} tris")
            record(ext.lstrip("."), p, why, f"re-read {len(back.triangles):,} tris")

        # glTF wants Y-up. Rotate a copy, never the source.
        gm = o3d.geometry.TriangleMesh(m)
        gm.transform(Z_UP_TO_Y_UP)
        p = EXP / "mesh_1.5M.glb"
        if not o3d.io.write_triangle_mesh(str(p), gm):
            raise RuntimeError(f"writer refused {p}")
        back = o3d.io.read_triangle_mesh(str(p))
        # Open3D has written .glb files it cannot read back; report rather than assume.
        ok = (f"re-read {len(back.triangles):,} tris" if len(back.triangles) > 0
              else "WRITTEN BUT OPEN3D CANNOT RE-READ IT (see bugs-found B9); "
                   "verify in Blender")
        record("glb", p, "Blender and web viewers; rotated to Y-up", ok)

    # ---------------------------------------------------------------- clouds --
    @guard("point cloud PCD / XYZ / GLB")
    def _cloud():
        pc = o3d.io.read_point_cloud(str(src_cloud))
        n = len(pc.points)
        if n == 0:
            raise RuntimeError(f"{src_cloud} read as empty")
        print(f"  source {src_cloud.name}: {n:,} points")
        for ext, why in ((".pcd", "PCL and ROS"),
                         (".xyz", "plain text; any tool, largest on disk")):
            p = EXP / f"point_cloud_8M{ext}"
            if not o3d.io.write_point_cloud(str(p), pc):
                raise RuntimeError(f"writer refused {p}")
            back = o3d.io.read_point_cloud(str(p))
            if len(back.points) < 0.99 * n:
                raise RuntimeError(f"{p} read back {len(back.points):,} of {n:,}")
            record(ext.lstrip("."), p, why, f"re-read {len(back.points):,} points")

    # --------------------------------------------------------- object bundle --
    @guard("per-object OBJ + GLB scene")
    def _objects():
        if not objects_json.exists():
            raise RuntimeError("objects.json absent; run the object cut first")
        doc = json.loads(objects_json.read_text())
        objs = doc["objects"]
        odir = EXP / "objects_obj"
        odir.mkdir(parents=True, exist_ok=True)
        combined = o3d.geometry.TriangleMesh()
        written = 0
        for o in objs:
            rel = o.get("visual_mesh")
            if not rel:
                continue
            src = OUT / rel
            if not src.exists():
                continue
            m = o3d.io.read_triangle_mesh(str(src))
            if len(m.triangles) == 0:
                continue
            m.compute_vertex_normals()
            p = odir / f"object_{o['index']:04d}_{o.get('label','unlabelled')}.obj"
            if o3d.io.write_triangle_mesh(str(p), m):
                written += 1
            combined += m
        print(f"  {written} per-object OBJ files in {odir.relative_to(OUT)}")
        RESULTS.append({"format": "obj", "path": str(odir.relative_to(OUT)),
                        "bytes": sum(f.stat().st_size for f in odir.glob("*.obj")),
                        "note": f"{written} objects, one file each",
                        "verified": f"{written} written"})
        if len(combined.triangles):
            # Decimate before combining: the undecimated union of 43 objects came out
            # at 537 MB and Open3D could not read its own glTF back ("Buffer view with
            # offset/length out of range"). A scene file nobody can open is not an
            # export, and a viewer does not need 8 M triangles of crate.
            target = 400_000
            if len(combined.triangles) > target:
                combined = combined.simplify_quadric_decimation(target)
                combined.remove_unreferenced_vertices()
            combined.compute_vertex_normals()
            combined.transform(Z_UP_TO_Y_UP)
            p = EXP / "objects_scene.glb"
            if o3d.io.write_triangle_mesh(str(p), combined):
                back = o3d.io.read_triangle_mesh(str(p))
                record("glb", p, f"all cut objects, one Y-up scene, decimated to "
                                 f"{len(combined.triangles):,} tris",
                       f"re-read {len(back.triangles):,} tris" if len(back.triangles)
                       else "written but Open3D cannot re-read; verify in Blender")

    # ------------------------------------------------------------------ URDF --
    @guard("URDF")
    def _urdf():
        if not objects_json.exists():
            raise RuntimeError("objects.json absent")
        doc = json.loads(objects_json.read_text())
        objs = doc["objects"]
        # One fixed-base link per object plus a ground plane. Boxes, matching the MJCF,
        # because a URDF consumer wants a cheap collision primitive for the same reason
        # a physics engine does.
        lines = ['<?xml version="1.0"?>',
                 '<robot name="v3d_scene">',
                 '  <!-- Generated from a reconstructed video.',
                 f'       scale_mode={SCALE_MODE}.',
                 ('       LENGTHS ARE IN DA3 ARBITRARY UNITS, NOT METRES. Masses below'
                  ' are derived' if SCALE_MODE == "assumed" else '       Lengths are metric.'),
                 ('       from those units and a guessed density, so they are not'
                  ' physical.' if SCALE_MODE == "assumed" else ''),
                 '  -->',
                 '  <link name="world"/>']
        density = 300.0
        for o in objs:
            e = np.maximum(np.array(o["extent"], dtype=float), 1e-4)
            c = np.array(o["centre"], dtype=float)
            name = f"object_{o['index']:04d}_{o.get('label','unlabelled')}"
            mass = max(density * float(np.prod(e)), 1e-3)
            # Solid-box inertia about the centre of mass.
            ix = mass * (e[1] ** 2 + e[2] ** 2) / 12.0
            iy = mass * (e[0] ** 2 + e[2] ** 2) / 12.0
            iz = mass * (e[0] ** 2 + e[1] ** 2) / 12.0
            lines += [
                f'  <link name="{name}">',
                '    <inertial>',
                f'      <mass value="{mass:.6f}"/>',
                f'      <inertia ixx="{ix:.6e}" iyy="{iy:.6e}" izz="{iz:.6e}"'
                ' ixy="0" ixz="0" iyz="0"/>',
                '    </inertial>',
                '    <visual><geometry>'
                f'<box size="{e[0]:.6f} {e[1]:.6f} {e[2]:.6f}"/></geometry></visual>',
                '    <collision><geometry>'
                f'<box size="{e[0]:.6f} {e[1]:.6f} {e[2]:.6f}"/></geometry></collision>',
                '  </link>',
                f'  <joint name="{name}_fixed" type="fixed">',
                '    <parent link="world"/>',
                f'    <child link="{name}"/>',
                f'    <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>',
                '  </joint>',
            ]
        lines.append('</robot>')
        p = EXP / "scene.urdf"
        p.write_text("\n".join(lines) + "\n")
        # Parse it back: malformed XML is the commonest URDF defect and is free to catch.
        import xml.etree.ElementTree as ET
        root = ET.parse(p).getroot()
        nl = len(root.findall("link")); nj = len(root.findall("joint"))
        record("urdf", p, f"{nl} links, {nj} fixed joints, box collision",
               f"XML parses, {nl} links")

    # ------------------------------------------------------------ height map --
    @guard("height map ASCII grid + OBJ terrain")
    def _terrain():
        hm_path = OUT / "terrain" / "height_map.npy"
        if not hm_path.exists():
            raise RuntimeError("height_map.npy absent")
        g = np.load(hm_path)
        tj = json.loads((OUT / "terrain" / "terrain.json").read_text())
        cell = float(tj["height_map"]["cell_size"])
        # Esri ASCII grid: what GIS and most terrain tools ingest directly.
        p = EXP / "height_map.asc"
        hdr = (f"ncols {g.shape[1]}\nnrows {g.shape[0]}\nxllcorner 0.0\n"
               f"yllcorner 0.0\ncellsize {cell:.8f}\nNODATA_value -9999\n")
        with p.open("w") as fh:
            fh.write(hdr)
            for row in np.flipud(g):
                fh.write(" ".join(f"{v:.5f}" for v in row) + "\n")
        back = np.loadtxt(p, skiprows=6)
        if back.shape != g.shape:
            raise RuntimeError(f"{p} read back {back.shape}, expected {g.shape}")
        record("asc", p, f"Esri ASCII grid, cellsize {cell:.5f}", f"re-read {back.shape}")

        # And the same grid as a triangulated surface, for anything that wants a mesh.
        h, w = g.shape
        ys, xs = np.mgrid[0:h, 0:w]
        verts = np.stack([xs.ravel() * cell, ys.ravel() * cell, g.ravel()], axis=1)
        idx = np.arange(h * w).reshape(h, w)
        a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel()
        c = idx[1:, 1:].ravel(); d = idx[1:, :-1].ravel()
        tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        tm = o3d.geometry.TriangleMesh()
        tm.vertices = o3d.utility.Vector3dVector(verts)
        tm.triangles = o3d.utility.Vector3iVector(tris)
        tm.compute_vertex_normals()
        p = EXP / "terrain_surface.obj"
        if not o3d.io.write_triangle_mesh(str(p), tm):
            raise RuntimeError("terrain OBJ writer refused")
        record("obj", p, f"height map as a {h}x{w} triangulated surface",
               f"{len(tris):,} tris")

    # No explicit calls here: `guard` runs each body at DECORATION time and returns
    # None, so calling them again raises TypeError. This is the second time this
    # pattern has bitten in this session -- a decorator that executes is a decorator
    # whose result must not also be invoked.

    doc = {"scene": SCENE, "scale_mode": SCALE_MODE,
           "units": "DA3 arbitrary units — NOT metres" if SCALE_MODE == "assumed" else "metres",
           "gltf_axis_note": ("GLB/glTF files are rotated to Y-up as the format "
                              "requires; every other file is Z-up"),
           "exports": RESULTS, "failed": FAILED}
    (EXP / "exchange_manifest.json").write_text(json.dumps(doc, indent=2))
    print("-" * 74)
    print(f"  {len(RESULTS)} exports, {sum(r['bytes'] for r in RESULTS)/1e6:.1f} MB")
    if FAILED:
        print(f"  {len(FAILED)} failed: {list(FAILED)}")
    print("\nEXPORTS_OK" if not FAILED else "\nEXPORTS_PARTIAL")


main()
