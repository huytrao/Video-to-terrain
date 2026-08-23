"""Build a MuJoCo scene: the carved room as terrain, the carved objects as bodies.

The room is static geometry; each carved object is a free body with real mass,
friction and contact parameters, so a robot can push it, knock it over, or lift
it off the bench it stands on.

Two facts this file exists to respect, both of which have bitten this project:

1. **Z points down in the reconstruction.** The scan's floor is at z = -0.155
   and its ceiling at z = -3.18. MuJoCo wants +Z up, so the geometry is turned
   over by a 180 deg rotation about X -- `FLIP_X180`, shared with
   make_room_scene.py. Negating Z instead would mirror the scene and quietly
   flip every rotation in it.

2. **MuJoCo collides a mesh geom as its convex hull.** The hull of a room is a
   solid block; the hull of a workbench is a wedge with no space under it. Both
   the room and every object therefore go through CoACD, and the visual mesh is
   carried separately at full detail with collision switched off.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

ROOT = pathlib.Path(__file__).resolve().parents[2]

FLIP_X180 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# Per material: sliding / torsional / rolling friction, then contact softness.
# Torsional and rolling friction only do anything at condim 4 and 6, which is
# why the free bodies below ask for 6 -- at the default condim 3 a crate spins
# on the spot forever and the last two numbers are decoration.
MATERIALS = {
    "machine":    dict(friction=(0.70, 0.030, 0.0010), solref=(0.008, 1.00), mass=10.0),
    "workbench":  dict(friction=(0.60, 0.020, 0.0008), solref=(0.010, 1.00), mass=10.0),
    "table":      dict(friction=(0.60, 0.020, 0.0008), solref=(0.010, 1.00), mass=10.0),
    "box":        dict(friction=(0.55, 0.010, 0.0005), solref=(0.012, 0.95), mass=10.0),
    "crate":      dict(friction=(0.55, 0.010, 0.0005), solref=(0.012, 0.95), mass=10.0),
    "trash_can":  dict(friction=(0.35, 0.005, 0.0002), solref=(0.018, 0.70), mass=10.0),
    "blade":      dict(friction=(0.45, 0.010, 0.0004), solref=(0.006, 1.00), mass=10.0),
}
DEFAULT_MATERIAL = dict(friction=(0.60, 0.020, 0.0008), solref=(0.010, 1.00), mass=10.0)


def material_for(semantic_class):
    """Look up a material, reporting a miss instead of silently defaulting.

    The VLM writes classes with spaces -- `trash can`, not `trash_can` -- so a
    plain dict lookup fell through to the default and gave the bin a workbench's
    friction without saying so.
    """
    key = str(semantic_class).strip().lower().replace(" ", "_").replace("-", "_")
    if key in MATERIALS:
        return MATERIALS[key], key
    return DEFAULT_MATERIAL, None

ROOM_FRICTION = (1.00, 0.020, 0.0010)   # concrete, as in make_room_scene.py


def fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.asarray(values).ravel())


def load_mesh(path):
    scene = trimesh.load(path)
    return scene.dump(concatenate=True) if hasattr(scene, "geometry") else scene


def convex_parts(mesh, threshold, max_hulls, decimate_to):
    """CoACD parts as trimesh hulls, largest first, tiny slivers dropped."""
    import coacd

    coacd.set_log_level("error")
    reduced = mesh
    if len(mesh.faces) > decimate_to:
        reduced = mesh.simplify_quadric_decimation(face_count=decimate_to)
    raw = coacd.run_coacd(
        coacd.Mesh(reduced.vertices, reduced.faces),
        threshold=threshold, max_convex_hull=max_hulls, preprocess_mode="off",
    )
    hulls = []
    for vertices, faces in raw:
        hull = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces),
                               process=False)
        if hull.volume > 1e-7:
            hulls.append(hull)
    hulls.sort(key=lambda h: -h.volume)
    return hulls


def box_planes(centre, extent, rotation):
    """The six outward half-space planes of an oriented box."""
    planes = []
    for axis in range(3):
        direction = rotation[:, axis]
        half = extent[axis] / 2
        planes.append((direction, centre + direction * half))
        planes.append((-direction, centre - direction * half))
    return planes


def hull_intrudes(hull, centre, extent, rotation, samples=6):
    """True if any point inside the box also lies inside the convex hull.

    Testing the hull's VERTICES against the box -- which is what the first
    version did -- misses the case that matters. A large collider can span the
    void an object was carved from with every one of its vertices far outside
    the object's box, and that is exactly how the machine ended up 77.8 cm
    inside the room.
    """
    grid = np.linspace(-0.5, 0.5, samples)
    local = np.stack(np.meshgrid(grid, grid, grid), axis=-1).reshape(-1, 3) * extent
    probes = local @ rotation.T + centre
    # Convex: inside means behind every face plane.
    offsets = np.einsum("ij,ij->i", hull.facets_normal if False else hull.face_normals,
                        hull.triangles[:, 0])
    behind = (probes @ hull.face_normals.T) <= (offsets + 1e-9)
    return bool(behind.all(axis=1).any())


def clip_convex(hull, normal, origin, eps=1e-9):
    """The part of a convex hull on the positive side of a plane, or None.

    trimesh's own `slice_mesh_plane` needs shapely, which is not installed here,
    and wrapping it in a bare `except` hid that: every slice failed and the
    carve silently became delete-only. For a CONVEX body the clip is exact
    without any library -- keep the vertices in front of the plane, add where
    the crossing edges meet it, re-hull.
    """
    vertices = hull.vertices
    signed = vertices @ normal - float(normal @ origin)
    front = signed > eps
    if front.all():
        return hull
    if not front.any():
        return None

    points = [vertices[front]]
    edges = hull.edges_unique
    crossing = front[edges[:, 0]] != front[edges[:, 1]]
    if crossing.any():
        a, b = edges[crossing, 0], edges[crossing, 1]
        t = signed[a] / (signed[a] - signed[b])
        points.append(vertices[a] + t[:, None] * (vertices[b] - vertices[a]))
    cloud = np.vstack(points)
    if len(cloud) < 4:
        return None
    try:
        return trimesh.Trimesh(vertices=cloud, process=False).convex_hull
    except Exception as error:
        raise RuntimeError(f"convex clip failed on {len(cloud)} points: {error}") from error


def carve_hulls(hulls, boxes, min_volume=2e-4):
    """Cut the carved objects' volume out of the room colliders.

    Carving an object leaves a concave void, and a convex hull cannot express a
    void -- CoACD spans one happily, so the room collider ends up solid exactly
    where the object stands.

    `hull minus box` is not convex, but it is covered by the six convex pieces
    `hull intersected with the outside of each box face`. Slicing produces those
    six, and MuJoCo is content with the overlap between them.
    """
    kept = []
    dropped = split = 0
    for hull in hulls:
        pieces = [hull]
        for centre, extent, rotation in boxes:
            nxt = []
            for piece in pieces:
                if not hull_intrudes(piece, centre, extent, rotation):
                    nxt.append(piece)
                    continue
                split += 1
                for normal, origin in box_planes(centre, extent, rotation):
                    cut = clip_convex(piece, normal, origin)
                    if cut is not None and cut.volume > min_volume:
                        nxt.append(cut)
            pieces = nxt
        if not pieces:
            dropped += 1
        kept.extend(pieces)
    return kept, dropped, split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carved", default="outputs-full-factory/delivery/OUTPUT_K")
    ap.add_argument("--out", default="outputs-full-factory/delivery/OUTPUT_K/sim")
    ap.add_argument("--room-threshold", type=float, default=0.06)
    ap.add_argument("--room-hulls", type=int, default=96)
    ap.add_argument("--object-threshold", type=float, default=0.05)
    ap.add_argument("--object-hulls", type=int, default=12)
    ap.add_argument("--timestep", type=float, default=0.002)
    ap.add_argument("--reuse-assets", action="store_true",
                    help="reuse colliders already on disk; CoACD on the room takes minutes")
    args = ap.parse_args()

    carved = ROOT / args.carved
    out_dir = ROOT / args.out
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    if not args.reuse_assets:
        for stale in assets.glob("*.obj"):
            stale.unlink()

    manifest = json.loads((carved / "carve_manifest.json").read_text())

    # The oriented OBB of every carved object, in the MuJoCo frame. The room's
    # colliders must stay out of these; the objects themselves live there.
    stage_g = ROOT / "outputs-full-factory/delivery/OUTPUT_G"
    object_boxes = []
    for entry in manifest["objects"]:
        meta_path = stage_g / entry["name"] / "alignment_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            centre = np.asarray(meta["obb_center_world"], dtype=float)
            extent = np.asarray(meta["obb_extent_xyz"], dtype=float)
            rotation = np.asarray(meta["transform_matrix_4x4"], dtype=float)[:3, :3]
        else:  # hand-specified --extra region: axis aligned, from the carve bounds
            lo = np.asarray(entry["bounds_min"], dtype=float)
            hi = np.asarray(entry["bounds_max"], dtype=float)
            centre, extent, rotation = (lo + hi) / 2, hi - lo, np.eye(3)
        object_boxes.append((centre, extent, rotation))

    room = load_mesh(carved / "room_carved.glb")
    room.apply_transform(FLIP_X180)
    floor_z = float(np.percentile(room.vertices[:, 2], 1.0))
    shift = np.array([0.0, 0.0, -floor_z])
    room.apply_translation(shift)
    if room.bounds[1][2] <= 0.5:
        raise SystemExit(
            f"room top is z={room.bounds[1][2]:.3f} after orienting; it is upside down"
        )
    print(f"room  : oriented, floor dropped to 0, top at z={room.bounds[1][2]:.2f} m")

    def to_world(mesh):
        mesh = mesh.copy()
        mesh.apply_transform(FLIP_X180)
        mesh.apply_translation(shift)
        return mesh

    # CoACD's own output is cached as `room_raw_*` and never modified. The
    # colliders written below are derived from it every run: reusing an already
    # carved collider as input compounds the carve on each rebuild, which shrank
    # the collider set instead of growing it.
    cached_room = sorted(assets.glob("room_raw_*.obj"))
    if args.reuse_assets and cached_room:
        room_hulls = [trimesh.load(str(f), process=False) for f in cached_room]
        print(f"room  : reusing {len(room_hulls)} cached CoACD parts")
    else:
        room.export(assets / "room_visual.obj")
        room_hulls = convex_parts(room, args.room_threshold, args.room_hulls, 40_000)
        for stale in assets.glob("room_raw_*.obj"):
            stale.unlink()
        for i, hull in enumerate(room_hulls):
            hull.export(assets / f"room_raw_{i:03d}.obj")
    flip = FLIP_X180[:3, :3]
    boxes_in_sim = [
        (flip @ centre + shift, extent, flip @ rotation)
        for centre, extent, rotation in object_boxes
    ]
    before = len(room_hulls)
    room_hulls, dropped, shrunk = carve_hulls(room_hulls, boxes_in_sim)
    for stale in assets.glob("room_collider_*.obj"):
        stale.unlink()
    for i, hull in enumerate(room_hulls):
        hull.export(assets / f"room_collider_{i:03d}.obj")
    print(f"room  : {before} convex colliders -> {len(room_hulls)} "
          f"({dropped} dropped entirely, {shrunk} sliced clear of a carved object)")

    bodies = []
    for entry in manifest["objects"]:
        name = entry["name"]
        mesh = to_world(load_mesh(carved / entry["file"]))
        pivot = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-pivot)          # geoms are written in body frame

        material, matched = material_for(entry["semantic_class"])
        if matched is None:
            print(f"  note  {name:24} class {entry['semantic_class']!r} has no material; "
                  f"using the default")
        cached = sorted(assets.glob(f"{name}_hull_*.obj"))
        if args.reuse_assets and cached:
            hulls = [trimesh.load(str(f), process=False) for f in cached]
        else:
            hulls = convex_parts(mesh, args.object_threshold, args.object_hulls, 20_000)
        if not hulls:
            print(f"  skip  {name:24} no convex part survived")
            continue

        if not (args.reuse_assets and (assets / f"{name}_visual.obj").exists()):
            mesh.export(assets / f"{name}_visual.obj")
        volumes = np.array([h.volume for h in hulls])
        share = volumes / volumes.sum()          # mass follows volume, not count
        if not (args.reuse_assets and cached):
            for i, hull in enumerate(hulls):
                hull.export(assets / f"{name}_hull_{i:02d}.obj")

        bodies.append(dict(name=name, pos=pivot, hulls=len(hulls),
                           mass=material["mass"] * share, material=material,
                           semantic=entry["semantic_class"]))
        print(f"  body  {name:24} {len(hulls):3d} hulls   "
              f"{material['mass']:.1f} kg   friction {material['friction'][0]:.2f}")

    # ---- MJCF -------------------------------------------------------------
    mj = ET.Element("mujoco", model="factory_carved_physics")
    ET.SubElement(mj, "compiler", angle="radian", assetdir="assets",
                  autolimits="true", inertiafromgeom="true")
    ET.SubElement(mj, "option", gravity="0 0 -9.81", timestep=str(args.timestep),
                  integrator="implicitfast")
    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080",
                  azimuth="135", elevation="-20")
    ET.SubElement(visual, "headlight", diffuse="0.65 0.65 0.65",
                  ambient="0.35 0.35 0.35", specular="0.1 0.1 0.1")

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", name="skybox", type="skybox", builtin="gradient",
                  rgb1="0.53 0.6 0.7", rgb2="0.1 0.12 0.16", width="256", height="256")
    ET.SubElement(asset, "mesh", name="room_visual", file="room_visual.obj")
    for i in range(len(room_hulls)):
        ET.SubElement(asset, "mesh", name=f"room_collider_{i:03d}",
                      file=f"room_collider_{i:03d}.obj")
    for body in bodies:
        ET.SubElement(asset, "mesh", name=f"{body['name']}_visual",
                      file=f"{body['name']}_visual.obj")
        for i in range(body["hulls"]):
            ET.SubElement(asset, "mesh", name=f"{body['name']}_hull_{i:02d}",
                          file=f"{body['name']}_hull_{i:02d}.obj")

    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="4 4 6", dir="-0.4 -0.4 -1", directional="true")
    # Visual only: never referenced by a collider, so its 287k faces cost nothing.
    ET.SubElement(world, "geom", type="mesh", mesh="room_visual", contype="0",
                  conaffinity="0", group="2", rgba="0.82 0.80 0.77 1")
    for i in range(len(room_hulls)):
        ET.SubElement(world, "geom", type="mesh", mesh=f"room_collider_{i:03d}",
                      group="0", friction=fmt(ROOM_FRICTION), condim="6",
                      rgba="0.4 0.5 0.65 0")

    for body in bodies:
        node = ET.SubElement(world, "body", name=body["name"], pos=fmt(body["pos"]))
        ET.SubElement(node, "freejoint", name=f"{body['name']}_free")
        ET.SubElement(node, "geom", type="mesh", mesh=f"{body['name']}_visual",
                      contype="0", conaffinity="0", group="2", mass="0",
                      rgba="0.85 0.83 0.80 1")
        for i in range(body["hulls"]):
            ET.SubElement(
                node, "geom", type="mesh", mesh=f"{body['name']}_hull_{i:02d}",
                group="0", condim="6",
                friction=fmt(body["material"]["friction"]),
                solref=fmt(body["material"]["solref"]),
                solimp="0.9 0.95 0.001",
                mass=f"{body['mass'][i]:.6g}",
                rgba="0.9 0.4 0.3 0",
            )

    ET.indent(mj, space="  ")
    xml_path = out_dir / "factory_physics.xml"
    xml_path.write_text(ET.tostring(mj, encoding="unicode"))
    print(f"\nwrote {xml_path.relative_to(ROOT)}  "
          f"({len(bodies)} free bodies, {len(room_hulls)} room colliders)")


if __name__ == "__main__":
    main()
