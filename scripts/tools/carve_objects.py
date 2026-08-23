"""Carve named objects out of the fused room scan and patch the holes they leave.

The room scan is a single fused shell: an object is not a separate connected
component, it is welded to the floor it stands on.  So the cut is geometric --
oriented box from stage G, minus anything lying in the floor plane -- and the
hole is closed by fanning each new boundary loop to its own centroid.

Up in this scene is -Z (floor at z ~ -0.21, ceiling at z ~ -3.18).  Do not
"fix" that to +Z; see docs/reference for how it was established.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import trimesh
from matplotlib.path import Path
from scipy.spatial import Delaunay, cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[2]
UP_SIGN = -1.0  # world up is -Z


def find_floor_z(mesh: trimesh.Trimesh) -> float:
    """Return the z of the dominant horizontal plane nearest the floor (max z)."""
    normals = mesh.face_normals
    horizontal = np.abs(normals[:, 2]) > 0.9
    if not horizontal.any():
        raise RuntimeError("room mesh has no horizontal faces; check the up axis")
    centres = mesh.triangles_center[horizontal, 2]
    weights = mesh.area_faces[horizontal]
    hist, edges = np.histogram(centres, bins=200, weights=weights)
    # Up is -Z, so the floor is the dominant plane with the LARGEST z.
    strong = np.where(hist > 0.25 * hist.max())[0]
    return float(edges[strong.max()])


def parse_extra(spec):
    """`name:xmin,ymin,zmin,xmax,ymax,zmax` -> the same tuple load_obb returns.

    An escape hatch for objects stage G detected badly.  obj_0181_box, for one,
    came back as a 5x3x18 cm sliver while the real crate is half a metre across,
    so its own box is useless and the region has to be given by hand.
    """
    name, _, numbers = spec.partition(":")
    values = [float(v) for v in numbers.split(",")]
    if len(values) != 6:
        raise ValueError(f"--extra {spec!r} needs six comma-separated bounds")
    lo, hi = np.asarray(values[:3]), np.asarray(values[3:])
    if not (hi > lo).all():
        raise ValueError(f"--extra {spec!r}: max bound must exceed min bound")
    meta = {"semantic_class": name.rsplit("_", 1)[-1], "source": "manual --extra"}
    return name, ((lo + hi) / 2, hi - lo, np.eye(3), meta)


def load_obb(meta_path):
    meta = json.loads(meta_path.read_text())
    centre = np.asarray(meta["obb_center_world"], dtype=float)
    extent = np.asarray(meta["obb_extent_xyz"], dtype=float)
    rotation = np.asarray(meta["transform_matrix_4x4"], dtype=float)[:3, :3]
    return centre, extent, rotation, meta


def select_faces(mesh, centre, extent, rotation, floor_z, pad, floor_eps, escape=0.30):
    """Face indices inside the padded OBB, minus the floor the object stands on.

    Height alone cannot tell the floor from an object's own base -- a bench's
    bottom shelf lies in the same 3 cm band as the concrete under it, and
    protecting it by height leaves a table-shaped footprint behind after the
    carve. A face counts as floor only if the surface it belongs to carries on
    at least `escape` metres past the box; the object's own base does not.
    """
    local = (mesh.triangles_center - centre) @ rotation
    inside = (np.abs(local) <= (extent / 2 + pad)).all(axis=1)
    flat_and_low = (np.abs(mesh.triangles_center[:, 2] - floor_z) < floor_eps) & (
        np.abs(mesh.face_normals[:, 2]) > 0.85
    )

    suspect = np.where(inside & flat_and_low)[0]
    is_real_floor = np.zeros(len(mesh.faces), dtype=bool)
    if len(suspect):
        member = np.zeros(len(mesh.faces), dtype=bool)
        member[flat_and_low] = True
        pairs = mesh.face_adjacency
        internal = pairs[member[pairs].all(axis=1)]
        for part in trimesh.graph.connected_components(
            internal, nodes=np.where(flat_and_low)[0], min_len=1, engine="scipy"
        ):
            reach = np.abs((mesh.triangles_center[part] - centre) @ rotation)
            outside = (reach > (extent / 2 + pad + escape)).any(axis=1)
            if outside.any():
                is_real_floor[part] = True
    return np.where(inside & ~is_real_floor)[0]


def keep_significant(mesh, faces, min_area_frac):
    """Drop scan speckle: keep only connected pieces holding real area.

    Components are found on the original mesh's face-adjacency graph restricted
    to `faces`.  Building a submesh first and splitting it would re-weld its
    vertices, and the face indices could no longer be mapped back.
    """
    member = np.zeros(len(mesh.faces), dtype=bool)
    member[faces] = True
    pairs = mesh.face_adjacency
    internal = pairs[member[pairs].all(axis=1)]
    components = trimesh.graph.connected_components(
        internal, nodes=faces, min_len=1, engine="scipy"
    )
    if len(components) <= 1:
        return faces, max(len(components), 1)
    areas = np.asarray([mesh.area_faces[c].sum() for c in components])
    keep = np.concatenate(
        [c for c, a in zip(components, areas) if a >= min_area_frac * areas.sum()]
    )
    return np.asarray(sorted(keep)), len(components)


def boundary_loops(mesh) -> list[list[int]]:
    """Closed vertex loops around every open boundary of `mesh`.

    Traversal consumes *edges*, not vertices.  A vertex where two holes touch
    belongs to both loops, and marking vertices globally would chop both into
    open chains that cannot be capped.
    """
    edges = mesh.edges_sorted
    _, index, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    border = edges[index[counts == 1]]
    if len(border) == 0:
        return []

    neighbours: dict[int, list[int]] = {}
    for a, b in border:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))

    unused = {frozenset((int(a), int(b))) for a, b in border}
    loops = []
    while unused:
        first, second = tuple(next(iter(unused)))
        unused.discard(frozenset((first, second)))
        loop, current, previous = [first, second], second, first
        while True:
            step = next(
                (n for n in neighbours[current] if frozenset((current, n)) in unused),
                None,
            )
            if step is None:
                break
            unused.discard(frozenset((current, step)))
            if step == loop[0]:
                break  # closed
            loop.append(step)
            previous, current = current, step
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def cap_loop(vertices, colours, loop, tree, room_xyz, room_rgb, edge, max_lift,
             force_rgb=None, fan_only=False, rim_mask=None):
    """Tessellate one boundary loop into a patch stitched to its own rim.

    Four things this has to get right:

    * The rim reuses the loop's ORIGINAL vertex indices.  Emitting fresh rim
      vertices leaves the cap floating beside the hole instead of closing it.
    * Delaunay runs on the convex hull, so on a concave loop -- and a boundary
      torn out of a scan is nothing but concave -- some rim edges end up
      covered by no kept triangle at all.  Every such edge is fanned to a hub
      afterwards, which is what actually guarantees the loop closes.
    * Interior points sit at the room's own triangle spacing, so the patch does
      not read as a few giant facets among fine scan geometry.
    * Colour is interpolated per interior point from the surrounding scan,
      unless the caller forces one (an object's unseen back gets flat grey).
    """
    ring = vertices[loop]
    n = len(loop)
    centroid = ring.mean(axis=0)
    _, _, basis = np.linalg.svd(ring - centroid, full_matrices=False)
    e1, e2, normal = basis[0], basis[1], basis[2]

    if fan_only:
        # One hub, one triangle per rim edge: every edge is used exactly once,
        # so the loop is provably closed. A grid interior cannot promise that --
        # Delaunay's containment filter drops triangles and exposes the interior
        # edges of whatever it dropped, which is how a cap reopens a hole.
        hub_colour = (
            np.asarray(force_rgb, float) if force_rgb is not None
            else room_rgb[tree.query(centroid[None], k=8)[1][0]].mean(axis=0)
        )
        fan = np.asarray([[i, (i + 1) % n, n] for i in range(n)], dtype=np.int64)
        return (
            centroid.reshape(1, 3),
            np.append(hub_colour.round().clip(0, 255).astype(np.uint8), 255).reshape(1, 4),
            fan,
            np.asarray(loop),
        )

    uv = np.column_stack([(ring - centroid) @ e1, (ring - centroid) @ e2])
    polygon = Path(uv)

    lo, hi = uv.min(axis=0), uv.max(axis=0)
    gu = np.arange(lo[0] + edge, hi[0], edge)
    gv = np.arange(lo[1] + edge, hi[1], edge)
    if len(gu) and len(gv):
        grid = np.stack(np.meshgrid(gu, gv), axis=-1).reshape(-1, 2)
        grid = grid[polygon.contains_points(grid, radius=-edge * 0.5)]
    else:
        grid = np.empty((0, 2))

    points = np.vstack([uv, grid])
    triangles = np.empty((0, 3), dtype=np.int64)
    if len(points) >= 3:
        try:
            candidate = Delaunay(points).simplices
            inside = polygon.contains_points(points[candidate].mean(axis=1))
            triangles = candidate[inside]
        except Exception:
            triangles = np.empty((0, 3), dtype=np.int64)

    # Which rim edges did that actually cover?
    covered = np.zeros(n, dtype=bool)
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if a < n and b < n:
                if (a + 1) % n == b:
                    covered[a] = True
                elif (b + 1) % n == a:
                    covered[b] = True

    flat = centroid + np.outer(grid[:, 0], e1) + np.outer(grid[:, 1], e2)
    if len(flat):
        distance, index = tree.query(flat, k=12)
        weight = 1.0 / np.maximum(distance, 1e-4) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        rgb = (room_rgb[index] * weight[:, :, None]).sum(axis=1)

        # The surface behind a removed object is what belongs in its hole. Torn
        # rim vertices sit ON the object's own silhouette, so letting them set
        # the depth rebuilds the object as a lid -- the "ghost table". `tree`
        # therefore holds background vertices only, and the estimate blends in
        # with distance from the rim so the patch still meets the rim cleanly.
        background = (room_xyz[index] * weight[:, :, None]).sum(axis=1)
        rim_gap = np.maximum(cKDTree(ring).query(flat)[0], 0.0)
        pull = np.clip(rim_gap / max(edge * 6.0, 1e-6), 0.0, 1.0)[:, None]
        flat = flat * (1.0 - pull) + background * pull
    else:
        rgb = np.empty((0, 3))

    extra_points, extra_rgb = list(flat), list(rgb)
    if not covered.all():
        open_edges = np.where(~covered)[0]
        # Stitch each missed rim edge to the interior vertex nearest to it. A
        # single shared hub would work too, but every triangle would then reach
        # the loop's centre, and the patch ends up covered in long radial spokes.
        if len(flat):
            midpoints = 0.5 * (ring[open_edges] + ring[(open_edges + 1) % n])
            nearest = cKDTree(flat).query(midpoints)[1] + n
            fan = np.column_stack([open_edges, (open_edges + 1) % n, nearest])
        else:
            hub_index = n + len(extra_points)
            extra_points.append(centroid)
            extra_rgb.append(room_rgb[tree.query(centroid[None], k=8)[1][0]].mean(axis=0))
            fan = np.column_stack(
                [open_edges, (open_edges + 1) % n, np.full(len(open_edges), hub_index)]
            )
        triangles = np.vstack([triangles, fan.astype(np.int64)])

    if len(triangles) == 0:
        return None

    interior = np.asarray(extra_points).reshape(-1, 3)
    shade = np.asarray(extra_rgb).reshape(-1, 3)
    if force_rgb is not None and len(shade):
        shade = np.tile(np.asarray(force_rgb, float), (len(shade), 1))
    interior_colours = np.column_stack(
        [shade.round().clip(0, 255).astype(np.uint8), np.full(len(shade), 255, np.uint8)]
    ) if len(shade) else np.empty((0, 4), np.uint8)

    return interior, interior_colours, triangles, np.asarray(loop)


def patch_holes(mesh, removed_centres, max_span, reach, edge, max_lift,
                force_rgb=None, fan_only=False):
    """Close the boundary loops this carve opened. Returns (mesh, capped, skipped)."""
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    colours = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8).copy()
    removed_tree = cKDTree(removed_centres) if removed_centres is not None else None
    # Vertices on an open edge are the torn rim of a hole; a patch must not take
    # its depth from them, so they are kept out of the background tree entirely.
    loops = boundary_loops(mesh)
    rim = np.zeros(len(mesh.vertices), dtype=bool)
    for loop in loops:
        rim[np.asarray(loop)] = True
    background = np.where(~rim)[0]
    room_tree = cKDTree(mesh.vertices[background])
    room_xyz = mesh.vertices[background]
    room_rgb = np.asarray(mesh.visual.vertex_colors)[background][:, :3].astype(float)

    capped = skipped = 0
    for loop in loops:
        ring = vertices[loop]
        centroid = ring.mean(axis=0)
        if removed_tree is not None and float(removed_tree.query(centroid)[0]) > reach:
            continue  # a gap the scan already had, not one this carve opened
        if float(np.linalg.norm(ring - centroid, axis=1).max()) > max_span:
            skipped += 1
            continue
        built = cap_loop(
            vertices, colours, loop, room_tree, room_xyz, room_rgb, edge,
            max_lift, force_rgb, fan_only,
        )
        if built is None:
            skipped += 1
            continue
        interior, interior_colours, triangles, rim = built

        # Delaunay indexed rim points 0..len(rim)-1 and interior after them.
        # Remap: rim -> its original vertex id, interior -> freshly appended ids.
        remap = np.concatenate([rim, len(vertices) + np.arange(len(interior))])
        vertices = np.vstack([vertices, interior])
        colours = np.vstack([colours, interior_colours])
        faces = np.vstack([faces, remap[triangles]])
        capped += 1

    out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=colours)
    return out, capped, skipped


def grey_untextured(mesh, threshold, grey, feather):
    """Paint the scan's unobserved patches a flat grey.

    Where no camera ever saw the surface, fusion leaves the vertex colour near
    black, which reads as a hole rather than as geometry.  Luminance alone
    decides: a genuinely dark object still carries some, and `threshold` is
    exposed so it can be pulled down if a real black machine gets caught.
    """
    colours = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8).copy()
    rgb = colours[:, :3].astype(float)
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    # Blend in over a band so the repaint has no hard rim against real texture.
    blend = np.clip((threshold + feather - luminance) / max(feather, 1e-6), 0, 1)
    touched = blend > 0
    colours[:, :3] = (
        rgb * (1 - blend[:, None]) + np.asarray(grey, float) * blend[:, None]
    ).round().clip(0, 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colours)
    return int(touched.sum()), int((blend > 0.5).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--room", default="outputs-full-factory/delivery/OUTPUT_B1/mesh/room_300k.glb")
    ap.add_argument("--stage-g", default="outputs-full-factory/delivery/OUTPUT_G")
    ap.add_argument("--out", default="outputs-full-factory/delivery/OUTPUT_K")
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--extra", nargs="*", default=[], metavar="NAME:X0,Y0,Z0,X1,Y1,Z1",
                    help="carve an axis-aligned region stage G got wrong")
    ap.add_argument("--pad", type=float, default=0.02, help="metres to grow the OBB by")
    ap.add_argument("--floor-eps", type=float, default=0.035, help="floor band to protect")
    ap.add_argument("--min-area-frac", type=float, default=0.02)
    ap.add_argument("--max-patch-span", type=float, default=2.5)
    ap.add_argument("--close-objects", action="store_true", default=True,
                    help="cap each carved object's own holes so it is not see-through")
    ap.add_argument("--no-close-objects", dest="close_objects", action="store_false")
    ap.add_argument("--back-colour", default="150,150,154",
                    help="flat colour for the surfaces the scan never saw")
    ap.add_argument("--object-span", type=float, default=3.0,
                    help="largest object hole to cap, in metres")
    ap.add_argument("--grey-below", type=float, default=26.0,
                    help="luminance under which a vertex counts as unobserved; 0 disables")
    ap.add_argument("--grey-feather", type=float, default=14.0,
                    help="luminance band over which the grey fades into real texture")
    ap.add_argument("--grey-colour", default="132,132,136")
    ap.add_argument("--patch-edge", type=float, default=0.032,
                    help="target patch triangle edge; room median is ~0.032 m")
    ap.add_argument("--max-lift", type=float, default=0.12,
                    help="how far a patch point may leave its own plane")
    ap.add_argument("--patch-reach", type=float, default=0.45,
                    help="how near a removed face a loop must be to count as ours")
    args = ap.parse_args()

    back_grey = [float(v) for v in args.back_colour.split(",")]
    if len(back_grey) != 3:
        raise ValueError("--back-colour needs three comma-separated channels")
    grey = [float(v) for v in args.grey_colour.split(",")]
    if len(grey) != 3:
        raise ValueError("--grey-colour needs three comma-separated channels")

    room_path = ROOT / args.room
    scene = trimesh.load(room_path)
    room = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
    if room.visual.kind != "vertex":
        raise RuntimeError(f"expected vertex colours on {room_path}, got {room.visual.kind}")

    floor_z = find_floor_z(room)
    print(f"room   : {len(room.vertices)} verts, {len(room.faces)} faces, floor z = {floor_z:.3f}")

    out_dir = ROOT / args.out
    (out_dir / "objects").mkdir(parents=True, exist_ok=True)
    # `objects_raw` is the carve's own output and is never written to again.
    # backfill_objects.py reads from it and writes to `objects`, so running the
    # backfill twice re-fuses from the carve rather than onto its own result.
    (out_dir / "objects_raw").mkdir(parents=True, exist_ok=True)

    carved_faces: list[int] = []
    manifest = []
    claimed = np.zeros(len(room.faces), dtype=bool)

    # Boxes stacked on a bench have boxes whose OBB overlaps the bench's own.
    # Settle each face once, smallest box first, so the thing sitting on top
    # keeps its own geometry instead of being swallowed by what holds it up.
    boxes = {
        name: load_obb(ROOT / args.stage_g / name / "alignment_meta.json")
        for name in args.objects
    }
    for spec in args.extra:
        name, box = parse_extra(spec)
        boxes[name] = box
    order = sorted(boxes, key=lambda n: float(np.prod(boxes[n][1])))
    if order != list(args.objects):
        print("claim order (smallest first): " + ", ".join(order))

    for name in order:
        centre, extent, rotation, meta = boxes[name]
        faces = select_faces(room, centre, extent, rotation, floor_z, args.pad, args.floor_eps)
        contested = int(claimed[faces].sum())
        faces = faces[~claimed[faces]]
        if contested:
            print(f"  note  {name:24} yielded {contested} faces already claimed")
        if len(faces) < 20:
            raise RuntimeError(f"{name}: only {len(faces)} faces inside its box; nothing to carve")
        faces, n_parts = keep_significant(room, faces, args.min_area_frac)
        piece = room.submesh([faces], append=True)
        if args.grey_below > 0:
            grey_untextured(piece, args.grey_below, grey, args.grey_feather)

        backed = 0
        if args.close_objects:
            # A carved object is a shell: the scan only ever saw its front, so
            # the back is absent rather than dark. Cap every one of its own
            # boundary loops in flat grey so it reads as a solid thing.
            piece, backed, dropped = patch_holes(
                piece, None, args.object_span, None,
                args.patch_edge, args.max_lift, force_rgb=back_grey, fan_only=True,
            )
            if dropped:
                print(f"  note  {name:24} left {dropped} loops open (span > {args.object_span} m)")
        piece.export(out_dir / "objects_raw" / f"{name}.glb")
        piece.export(out_dir / "objects" / f"{name}.glb")

        carved_faces.extend(faces.tolist())
        claimed[faces] = True
        rest = float(piece.bounds[1][2])  # bottom is max z, because up is -Z
        manifest.append(
            dict(
                name=name,
                semantic_class=meta["semantic_class"],
                faces=int(len(faces)),
                vertices=int(len(piece.vertices)),
                area_m2=round(float(piece.area), 3),
                back_caps=int(backed),
                parts_found=int(n_parts),
                centroid=[round(v, 4) for v in piece.centroid.tolist()],
                bounds_min=[round(v, 4) for v in piece.bounds[0].tolist()],
                bounds_max=[round(v, 4) for v in piece.bounds[1].tolist()],
                clearance_to_floor_m=round(floor_z - rest, 3),
                file=f"objects/{name}.glb",
            )
        )
        print(
            f"  carve {name:24} {len(faces):6d} faces  {piece.area:6.2f} m2  "
            f"{n_parts:3d} parts -> {len(piece.vertices):6d} verts"
        )

    mask = np.ones(len(room.faces), dtype=bool)
    mask[np.asarray(carved_faces)] = False
    carved = room.submesh([np.where(mask)[0]], append=True)
    removed_centres = room.triangles_center[np.asarray(carved_faces)]
    patched, n_patched, n_skipped = patch_holes(
        carved, removed_centres, args.max_patch_span, args.patch_reach,
        args.patch_edge, args.max_lift,
    )
    patched.export(out_dir / "room_carved.glb")

    (out_dir / "carve_manifest.json").write_text(
        json.dumps(
            dict(
                room_source=args.room,
                up_axis="-Z",
                floor_z=round(floor_z, 4),
                gravity_world=[0.0, 0.0, 9.81 * -UP_SIGN],
                room_faces_before=int(len(room.faces)),
                room_faces_after=int(len(patched.faces)),
                holes_patched=int(n_patched),
                holes_too_large_to_patch=int(n_skipped),
                objects=manifest,
            ),
            indent=2,
        )
    )
    if args.grey_below > 0:
        touched, solid = grey_untextured(patched, args.grey_below, grey, args.grey_feather)
        print(f"grey   : repainted {solid} unobserved verts ({touched} touched by the blend)")

    colours = np.asarray(patched.visual.vertex_colors)[:, :3]
    if colours.std() < 5:
        raise RuntimeError("carved room lost its vertex colours; check the uint8 cast")
    print(
        f"room   : carved {len(carved_faces)} faces, capped {n_patched} loops "
        f"({n_skipped} too large) -> {len(patched.faces)} faces, "
        f"colour mean {colours.mean(axis=0).round(1)}"
    )


if __name__ == "__main__":
    main()
