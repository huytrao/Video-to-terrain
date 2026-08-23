"""Cut a reconstructed room into individual objects, from geometry alone.

The obvious route to per-object assets runs through semantics: label the classes,
segment them, lift the masks. That route is real and this project builds it, but
it inherits three weaknesses. It can only ever produce objects whose class a VLM
happened to name — nothing outside the vocabulary can become an object at all. It
depends on a gated checkpoint. And text-prompted masks are one mask per *class*
per frame, so five pallets in a row become one `pallet` region and one box around
all five.

This module takes the other route, which needs no model. A room is mostly a few
large planes; almost everything that is *not* one of those planes is a thing.
So: find the shell, remove it, and what falls out in separate pieces are the
objects. The output is per-object geometry that semantics can later *label*,
rather than semantics being what creates it.

Three parts, each of which can be wrong in its own way, so each reports the
evidence for what it did:

1. **The up axis is derived, never assumed.** This project has been bitten by
   coordinate conventions three times, including a floor plan that came out
   upside down and looked entirely reasonable. Here the axis comes from the
   cameras: the floor is the large plane most of them look down on, and its normal
   is oriented to point from the floor toward the cameras. That is a statement the
   data can contradict, and it is checked.

2. **Shell planes are recognised by extent, not by flatness.** A machine housing
   and a factory wall are both planar; what distinguishes them is that the wall
   spans the room. So a plane is only shell if it is both well-populated *and*
   covers a large fraction of the scene's footprint in its own plane. Removing
   every planar patch would delete the flat sides of every crate.

3. **Components are filtered by size relative to the largest, not by an absolute
   count.** An absolute triangle threshold is a length in disguise and does not
   survive a change of scene scale.

Units. When scale mode is `assumed` these are DA3's units, not metres, and every
reported dimension says so. A ratio — an object's size against the room's — is
the only dimensionless thing here and therefore the only thing directly
comparable across scenes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


# --- configuration -----------------------------------------------------------


@dataclass
class CutPolicy:
    """Every threshold is a ratio or a count of samples. None is a length."""

    # RANSAC, in units of the scene's own diagonal rather than metres, so the same
    # number means the same thing on a garage and on a factory bay.
    plane_tol_frac_diagonal: float = 0.0025
    plane_ransac_n: int = 3
    plane_iterations: int = 2000
    # A plane is a candidate only if it holds this fraction of the remaining
    # points, which keeps RANSAC from reporting a lucky sliver.
    min_plane_point_frac: float = 0.02
    # ...and it is *shell* only if it also spans this fraction of the scene
    # footprint within its own plane. This is the test that separates a wall from
    # the flat side of a crate.
    min_shell_span_frac: float = 0.35
    max_shell_planes: int = 12
    # A single shell plane may not claim more than this fraction of the mesh.
    # Half-space removal is powerful, and one mis-fit plane through the middle of
    # the scene would otherwise delete the room: on a synthetic test it took
    # 99.6% of the triangles and left one object out of three.
    max_halfspace_claim: float = 0.45
    # ...and the accumulated removal may not exceed this. Individually-reasonable
    # planes can still sum to the whole room, which is how a synthetic test lost
    # every object while no single plane looked at fault.
    max_total_claim: float = 0.90
    # If half-space removal leaves less than this fraction of the mesh, it is
    # rejected wholesale and inlier-only removal is used instead. Keeping phantom
    # wall faces is a recoverable error; deleting every object is not.
    min_survival_frac: float = 0.02
    # A shell plane must sit at the scene's BOUNDARY along its own normal, within
    # this fraction of the total spread from one extreme. This is the criterion
    # that separates a wall from a plane through the middle of the room: both are
    # flat, both can span the full cross-section, and only their position differs.
    # Without it, a plane fitted across the interior was classified as a wall and
    # its half-space deleted a crate that was standing behind it.
    #
    # Raised from 0.06 to 0.10 on evidence from a real scene. The synthetic room put
    # its six shell planes at 0.012-0.031 and a thin panel at 0.13, which suggested
    # 0.06; but on the factory reconstruction the genuine CEILING sits at edge 0.08
    # and was being rejected as an interior plane, which is why the derived room
    # height came out at 2.31 with no ceiling in the shell. 0.10 keeps the real
    # ceiling and still sits clear of the 0.13 panel case.
    #
    # The lesson is about the method, not the number: a threshold fitted on synthetic
    # geometry has to be re-checked against a real reconstruction, because a TSDF
    # ceiling is nowhere near as flush with the bounding box as a modelled one.
    max_shell_edge_frac: float = 0.10
    # How near vertical a normal must be to count as floor or ceiling, and how
    # near horizontal to count as a wall. Cosine, so 0.9 is about 26 degrees.
    floor_normal_cos: float = 0.85
    wall_normal_cos: float = 0.30
    # After the shell is removed, keep components at least this fraction of the
    # largest remaining one. Relative, so it transfers between scenes.
    min_component_frac: float = 0.0015
    max_objects: int = 400
    # A component thinner than this fraction of the scene diagonal in every
    # direction is reconstruction speckle rather than an object.
    min_object_extent_frac: float = 0.004
    # Points sampled per object for the oriented box fit. More is slower and does
    # not improve a box that is already determined by its extremes.
    obb_samples: int = 20000


@dataclass
class Plane:
    normal: list[float]
    offset: float
    points: int
    span: list[float]
    span_frac: float
    role: str            # floor | ceiling | wall | large-planar
    verticality: float
    # How many vertices this plane's half-space actually claimed, and how many were
    # still unclaimed when it ran. Without these a shell that eats the room is
    # indistinguishable from one that eats a wall, which is exactly the confusion
    # that cost a debugging round here.
    removed: int = 0
    removed_frac: float = 0.0
    interior_signed: float = 0.0
    # Distance from the plane to the nearer extreme of the scene along its own
    # normal, as a fraction of the total spread. Near zero for a boundary surface.
    edge_frac: float = 0.0


@dataclass
class CutObject:
    index: int
    triangles: int
    vertices: int
    centre: list[float]
    extent: list[float]           # OBB edge lengths, longest first
    obb_axes: list[list[float]]
    aabb_min: list[float]
    aabb_max: list[float]
    height_above_floor: float
    footprint_area: float
    volume: float
    extent_over_room_diagonal: float
    rests_on_floor: bool
    label: str = "unlabelled"     # semantics fills this in later
    visual_mesh: str | None = None
    collision_mesh: str | None = None


@dataclass
class CutResult:
    scale_mode: str
    scene_diagonal: float
    up_axis: list[float]
    up_axis_evidence: dict
    floor_offset: float
    room_extent: list[float]
    # Extent is a SIZE and says nothing about position. A consumer building a floor
    # polygon from extent alone has to assume the scene is centred on the origin,
    # which no reconstruction is — that assumption put the footprint in the wrong
    # place and made eleven objects look like they had escaped the room.
    room_aabb_min: list[float] = field(default_factory=list)
    room_aabb_max: list[float] = field(default_factory=list)
    shell_planes: list[Plane] = field(default_factory=list)
    objects: list[CutObject] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


# --- the up axis, from the cameras -------------------------------------------


def derive_up_axis(vertices: np.ndarray, camera_centres: np.ndarray | None,
                   policy: CutPolicy) -> tuple[np.ndarray, float, dict]:
    """Find the floor plane and orient its normal to point up.

    The floor is taken to be the largest near-planar population of vertices whose
    normal is close to the direction of least vertex spread — in a room, the
    dimension you have least of is height. The *sign* then comes from the cameras:
    a camera stands above the floor. Without cameras it comes from the geometry
    itself, since most of a room's volume is above its floor, and which rule was
    used is recorded because they are not equally trustworthy.
    """
    import open3d as o3d

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    # The thinnest world axis is the best first guess at the vertical, and it is
    # only a guess: a long corridor is thinner across than it is tall.
    guess = np.zeros(3)
    guess[int(np.argmin(extent))] = 1.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    model, inliers = pcd.segment_plane(
        distance_threshold=policy.plane_tol_frac_diagonal * diagonal,
        ransac_n=policy.plane_ransac_n,
        num_iterations=policy.plane_iterations,
    )
    a, b, c, d = model
    normal = np.array([a, b, c], dtype=float)
    normal /= np.linalg.norm(normal)

    # RANSAC's largest plane is usually the floor, but in a room with a big wall in
    # view it can be that wall. Prefer it only if it is roughly perpendicular to
    # the thin axis; otherwise fall back to the thin axis itself.
    if abs(float(normal @ guess)) < policy.floor_normal_cos:
        used = "thinnest world axis (RANSAC's largest plane was not near-horizontal)"
        normal = guess
        # Offset from the vertex distribution: the floor is at the low end.
        d = -float(np.percentile(vertices @ normal, 1.0))
    else:
        used = "largest RANSAC plane"

    # Sign. Positive `normal` should point from the floor into the room.
    if camera_centres is not None and len(camera_centres):
        signed = float(np.median(camera_centres @ normal + d))
        rule = "cameras lie on the positive side"
        if signed < 0:
            normal, d = -normal, -d
            signed = -signed
        evidence = {"rule": rule, "median_signed_camera_height": signed,
                    "plane_source": used, "cameras": int(len(camera_centres))}
    else:
        signed = float(np.median(vertices @ normal + d))
        if signed < 0:
            normal, d = -normal, -d
            signed = -signed
        evidence = {"rule": "most vertices lie on the positive side",
                    "median_signed_vertex_height": signed, "plane_source": used,
                    "cameras": 0}

    height = float(np.percentile(vertices @ normal + d, 99.0))
    evidence["room_height_estimate"] = height
    evidence["scene_diagonal"] = diagonal
    if height <= 0:
        raise ValueError(
            f"after orienting the up axis the room has non-positive height "
            f"({height:.4f}); the floor plane or its sign is wrong, and every "
            f"height reported downstream would be too"
        )
    return normal, float(d), evidence


# --- shell removal -----------------------------------------------------------


def _plane_span(points: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, float]:
    """Extent of a point set measured *within* its own plane."""
    # Two directions spanning the plane, built from the normal without a
    # degenerate cross product.
    helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    proj = np.stack([points @ u, points @ v], axis=1)
    span = proj.max(axis=0) - proj.min(axis=0)
    return span, float(span[0] * span[1])


def find_shell(vertices: np.ndarray, up: np.ndarray, floor_d: float,
               policy: CutPolicy,
               interior: np.ndarray | None = None,
               halfspace: bool = True) -> tuple[list[Plane], np.ndarray]:
    """Peel off the large planes that make up the room, returning a keep mask.

    Iterative RANSAC: fit, decide whether the plane is shell, remove its inliers
    from the *fitting pool* either way. Dropping a large non-shell plane from the
    pool while keeping its vertices is deliberate — otherwise one dominant machine
    face is re-found every iteration and the loop never reaches the walls.

    Two removal modes, and `cut` tries them in this order:

    `halfspace=True` deletes everything on the exterior side of each shell plane.
    That is the physically meaningful statement — a room is the intersection of the
    half-spaces its surfaces bound — and it is what handles wall *thickness*: a
    reconstructed wall has a far surface a few centimetres behind its near one,
    outside any sane inlier tolerance, and deleting only the near face leaves the
    far one to survive as a large flat phantom "object".

    `halfspace=False` deletes only the RANSAC inliers. Conservative: it keeps
    phantoms, but it cannot delete the room.

    Half-space removal is powerful enough to need two guards, both learned from a
    synthetic room it emptied completely: no single plane may claim more than
    `max_halfspace_claim` of the mesh, and the accumulated removal may not exceed
    `max_total_claim`. Which side is exterior comes from `interior` — a point known
    to be inside — never from RANSAC's arbitrary choice of normal sign.
    """
    import open3d as o3d

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    footprint_span, footprint_area = _plane_span(vertices, up)
    tol = policy.plane_tol_frac_diagonal * diagonal
    if interior is None:
        interior = np.median(vertices, axis=0)

    remaining = np.arange(len(vertices))
    is_shell = np.zeros(len(vertices), dtype=bool)
    planes: list[Plane] = []
    ceiling_thresh = 0.5 * float(np.percentile(vertices @ up + floor_d, 99.0))

    for _ in range(policy.max_shell_planes):
        if len(remaining) < max(100, policy.min_plane_point_frac * len(vertices)):
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices[remaining])
        model, inliers = pcd.segment_plane(
            distance_threshold=tol,
            ransac_n=policy.plane_ransac_n,
            num_iterations=policy.plane_iterations,
        )
        if len(inliers) < policy.min_plane_point_frac * len(vertices):
            break
        idx = remaining[np.asarray(inliers)]
        normal = np.array(model[:3], dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            break
        normal /= norm
        offset = float(model[3]) / norm
        verticality = abs(float(normal @ up))
        span, area = _plane_span(vertices[idx], normal)
        span_frac = float(np.sqrt(area / max(footprint_area, 1e-12)))

        height = float(np.median(vertices[idx] @ up + floor_d))
        if verticality >= policy.floor_normal_cos:
            role = "floor" if height < ceiling_thresh else "ceiling"
        elif verticality <= policy.wall_normal_cos:
            role = "wall"
        else:
            role = "large-planar"

        # Where the plane sits along its own normal, relative to the scene's spread
        # in that direction. A boundary surface is near one extreme; a plane through
        # the interior is not, however flat and however wide it is.
        proj = vertices @ normal
        lo, hi = float(proj.min()), float(proj.max())
        spread = max(hi - lo, 1e-12)
        at = -offset
        edge_frac = min(abs(at - lo), abs(hi - at)) / spread

        shell = (role in {"floor", "ceiling", "wall"}
                 and span_frac >= policy.min_shell_span_frac
                 and edge_frac <= policy.max_shell_edge_frac)
        if (role in {"floor", "ceiling", "wall"}
                and span_frac >= policy.min_shell_span_frac and not shell):
            role = f"interior({role},edge {edge_frac:.2f})"
        plane = Plane(
            normal=[float(x) for x in normal], offset=offset,
            points=int(len(idx)), span=[float(x) for x in span],
            span_frac=span_frac, role=role if shell else f"kept:{role}",
            verticality=verticality, edge_frac=edge_frac,
        )
        planes.append(plane)
        if shell:
            signed = vertices @ normal + offset
            inside_ref = float(interior @ normal + offset)
            plane.interior_signed = inside_ref
            if halfspace:
                doomed = (signed <= tol) if inside_ref >= 0 else (signed >= -tol)
            else:
                doomed = np.zeros(len(vertices), dtype=bool)
                doomed[idx] = True
            fresh = doomed & ~is_shell
            claim = float(fresh.sum()) / max(len(vertices), 1)
            total_after = float((is_shell | doomed).sum()) / max(len(vertices), 1)
            # An interior reference sitting on the plane makes the sign meaningless,
            # so the direction of removal would be a coin toss.
            if (claim > policy.max_halfspace_claim
                    or total_after > policy.max_total_claim
                    or (halfspace and abs(inside_ref) <= tol)):
                plane.role = (f"rejected:{role}(claim {claim:.0%}, "
                              f"total would be {total_after:.0%})")
            else:
                plane.removed = int(fresh.sum())
                plane.removed_frac = claim
                is_shell |= doomed
        remaining = np.setdiff1d(remaining, idx, assume_unique=False)

    return planes, ~is_shell


# --- the cut ------------------------------------------------------------------


def cut(mesh_path: str | Path, out_dir: str | Path,
        camera_centres: np.ndarray | None = None,
        policy: CutPolicy | None = None,
        scale_mode: str = "assumed",
        write_meshes: bool = True) -> CutResult:
    """Split a room mesh into per-object assets. Returns what it found and why."""
    import open3d as o3d

    policy = policy or CutPolicy()
    out_dir = Path(out_dir)
    (out_dir / "objects").mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"{mesh_path} has no triangles")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    diagonal = float(np.linalg.norm(extent))

    up, floor_d, up_evidence = derive_up_axis(vertices, camera_centres, policy)
    # The interior reference for the half-space test. The camera path is the better
    # witness — it is by construction inside the room — and the vertex centroid is
    # the fallback when poses were not supplied.
    interior = (np.median(camera_centres, axis=0)
                if camera_centres is not None and len(camera_centres)
                else np.median(vertices, axis=0))

    # Half-space removal first, inlier-only as a fallback. The fallback exists
    # because the two errors are not symmetric: half-space removal that goes wrong
    # deletes the objects and leaves nothing to inspect, while inlier-only removal
    # that goes wrong leaves extra flat phantoms which the audit then flags. Prefer
    # the recoverable failure.
    removal_mode = "halfspace"
    planes, keep_vertex = find_shell(vertices, up, floor_d, policy,
                                     interior=interior, halfspace=True)
    survival = float(keep_vertex.mean())
    if survival < policy.min_survival_frac:
        removal_mode = (f"inliers (half-space removal left only {survival:.2%} "
                        f"of the mesh)")
        planes, keep_vertex = find_shell(vertices, up, floor_d, policy,
                                         interior=interior, halfspace=False)
        survival = float(keep_vertex.mean())

    # A triangle survives only if all three of its vertices did: a triangle
    # straddling the boundary belongs to the wall it is attached to, and keeping it
    # would weld every object to the shell and defeat the whole clustering step.
    keep_tri = keep_vertex[triangles].all(axis=1)
    n_shell_tris = int((~keep_tri).sum())

    objects_mesh = o3d.geometry.TriangleMesh()
    objects_mesh.vertices = mesh.vertices
    objects_mesh.triangles = o3d.utility.Vector3iVector(triangles[keep_tri])
    if mesh.has_vertex_colors():
        objects_mesh.vertex_colors = mesh.vertex_colors
    objects_mesh.remove_unreferenced_vertices()

    labels, counts, _ = objects_mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    room_height = float(np.percentile(vertices @ up + floor_d, 99.0))

    result = CutResult(
        scale_mode=scale_mode,
        scene_diagonal=diagonal,
        up_axis=[float(x) for x in up],
        up_axis_evidence=up_evidence,
        floor_offset=floor_d,
        room_extent=[float(x) for x in extent],
        room_aabb_min=[float(x) for x in vertices.min(axis=0)],
        room_aabb_max=[float(x) for x in vertices.max(axis=0)],
        shell_planes=planes,
    )

    # Diagnostics are filled in on every path, including this one: a consumer that
    # KeyErrors on the empty case reports a crash where the real finding is
    # "shell removal took everything", which is a different and more useful thing
    # to be told.
    result.diagnostics = {
        "mesh": str(mesh_path),
        "mesh_triangles": int(len(triangles)),
        "removal_mode": removal_mode,
        "vertex_survival_fraction": survival,
        "shell_triangles_removed": n_shell_tris,
        "shell_fraction": float(n_shell_tris / max(len(triangles), 1)),
        "components_after_shell_removal": int(counts.size),
        "objects_kept": 0,
        "dropped_below_size_fraction": 0,
        "dropped_too_thin": 0,
        "largest_component_triangles": int(counts.max()) if counts.size else 0,
        "room_height": float(np.percentile(vertices @ up + floor_d, 99.0)),
        "policy": asdict(policy),
        "units": ("DA3 units, not metres" if scale_mode == "assumed" else "metres"),
    }
    if counts.size == 0:
        result.diagnostics["note"] = (
            "nothing remained after shell removal, in either mode; the plane "
            "detection is consuming the scene rather than its boundary"
        )
        return result

    floor_tol = policy.plane_tol_frac_diagonal * diagonal * 4.0
    order = np.argsort(counts)[::-1][: policy.max_objects]
    floor_of_largest = counts[order[0]]
    kept, dropped_small, dropped_thin = 0, 0, 0

    obj_tris = np.asarray(objects_mesh.triangles)
    obj_verts = np.asarray(objects_mesh.vertices)

    for cid in order:
        if counts[cid] < policy.min_component_frac * floor_of_largest:
            dropped_small += 1
            continue
        sel = labels == cid
        tris = obj_tris[sel]
        vids = np.unique(tris)
        pts = obj_verts[vids]
        comp_extent = pts.max(axis=0) - pts.min(axis=0)
        if float(comp_extent.max()) < policy.min_object_extent_frac * diagonal:
            dropped_thin += 1
            continue

        sub = o3d.geometry.TriangleMesh()
        remap = -np.ones(len(obj_verts), dtype=np.int64)
        remap[vids] = np.arange(len(vids))
        sub.vertices = o3d.utility.Vector3dVector(pts)
        sub.triangles = o3d.utility.Vector3iVector(remap[tris])
        if objects_mesh.has_vertex_colors():
            sub.vertex_colors = o3d.utility.Vector3dVector(
                np.asarray(objects_mesh.vertex_colors)[vids])

        # Oriented box rather than axis-aligned: ground-truth annotations in this
        # domain are oriented and never axis-aligned, and comparing a world-axis
        # extent against them inflated the pilot room from 6.42 to 11.49 m.
        try:
            obb = sub.get_oriented_bounding_box(robust=True)
            box_extent = np.asarray(obb.extent, dtype=float)
            box_centre = np.asarray(obb.center, dtype=float)
            box_axes = np.asarray(obb.R, dtype=float).T
        except Exception:
            aabb = sub.get_axis_aligned_bounding_box()
            box_extent = np.asarray(aabb.get_extent(), dtype=float)
            box_centre = np.asarray(aabb.get_center(), dtype=float)
            box_axes = np.eye(3)

        srt = np.argsort(box_extent)[::-1]
        heights = pts @ up + floor_d
        low = float(heights.min())

        # Convex hull as the collision proxy: cheap for a physics engine, always
        # closed, and never larger than the object. A concave object is
        # over-approximated, which is the safe direction for a robot's clearance
        # and the wrong one for reaching into a bin — recorded, not hidden.
        collision_rel = None
        hull_ok = False
        try:
            hull, _ = sub.compute_convex_hull()
            hull.remove_unreferenced_vertices()
            hull_ok = len(hull.triangles) > 0
        except Exception:
            hull = None

        name = f"object_{kept:04d}"
        visual_rel = None
        if write_meshes:
            vp = out_dir / "objects" / f"{name}_visual.ply"
            o3d.io.write_triangle_mesh(str(vp), sub, write_ascii=False)
            visual_rel = str(vp.relative_to(out_dir))
            if hull_ok:
                cp = out_dir / "objects" / f"{name}_collision.ply"
                o3d.io.write_triangle_mesh(str(cp), hull, write_ascii=False)
                collision_rel = str(cp.relative_to(out_dir))

        result.objects.append(CutObject(
            index=kept,
            triangles=int(sel.sum()),
            vertices=int(len(vids)),
            centre=[float(x) for x in box_centre],
            extent=[float(box_extent[i]) for i in srt],
            obb_axes=[[float(x) for x in box_axes[i]] for i in srt],
            aabb_min=[float(x) for x in pts.min(axis=0)],
            aabb_max=[float(x) for x in pts.max(axis=0)],
            height_above_floor=low,
            footprint_area=float(box_extent[srt[0]] * box_extent[srt[1]]),
            volume=float(np.prod(box_extent)),
            extent_over_room_diagonal=float(box_extent.max() / max(diagonal, 1e-9)),
            rests_on_floor=bool(low <= floor_tol),
            visual_mesh=visual_rel,
            collision_mesh=collision_rel,
        ))
        kept += 1

    result.diagnostics.update({
        "objects_kept": kept,
        "dropped_below_size_fraction": dropped_small,
        "dropped_too_thin": dropped_thin,
        "room_height": room_height,
    })
    return result


# --- sanity checks that do not need ground truth ------------------------------


def audit(result: CutResult, max_object_frac: float = 0.6,
          max_same_size_overlap: float = 0.5) -> dict:
    """Checks available for a scene with no annotation — which is every real video.

    None of these proves an object is right. Each one catches a specific way the
    cut goes wrong, and they are the only falsifiable statements available without
    a ground truth.
    """
    objs = result.objects
    findings = []

    oversize = [o.index for o in objs
                if o.extent_over_room_diagonal > max_object_frac]
    if oversize:
        findings.append({
            "check": "object_not_larger_than_room",
            "status": "FAIL",
            "detail": f"{len(oversize)} object(s) span more than "
                      f"{max_object_frac:.0%} of the room diagonal: {oversize[:8]}. "
                      f"Most likely a shell plane was not recognised, so a wall is "
                      f"being reported as an object.",
        })
    else:
        findings.append({"check": "object_not_larger_than_room", "status": "PASS",
                         "detail": f"largest object is "
                                   f"{max((o.extent_over_room_diagonal for o in objs), default=0):.2%} "
                                   f"of the room diagonal"})

    below = [o.index for o in objs if o.height_above_floor < -1e-6]
    findings.append({
        "check": "nothing_below_the_floor",
        "status": "FAIL" if below else "PASS",
        "detail": (f"{len(below)} object(s) extend below the floor plane: {below[:8]}"
                   if below else "every object sits on or above the floor"),
    })

    resting = sum(1 for o in objs if o.rests_on_floor)
    findings.append({
        "check": "some_objects_rest_on_the_floor",
        "status": "PASS" if resting else "WARN",
        "detail": f"{resting} of {len(objs)} objects touch the floor plane"
                  + ("" if resting else "; a room where nothing touches the floor "
                                        "suggests the floor plane is misplaced"),
    })

    # Duplicate instances: two boxes of near-identical size at near-identical
    # centres are one object found twice, which is the failure mode a
    # component-based cut produces when a thin gap splits something in half.
    dupes = []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            ea, eb = np.array(a.extent), np.array(b.extent)
            if np.allclose(ea, eb, rtol=0.15):
                sep = np.linalg.norm(np.array(a.centre) - np.array(b.centre))
                if sep < max_same_size_overlap * float(ea.max()):
                    dupes.append((a.index, b.index))
    findings.append({
        "check": "no_duplicate_instances",
        "status": "WARN" if dupes else "PASS",
        "detail": (f"{len(dupes)} pair(s) of same-size objects sit closer than "
                   f"{max_same_size_overlap:.0%} of their own size: {dupes[:6]}"
                   if dupes else "no same-size near-coincident pairs"),
    })

    shell = result.diagnostics.get("shell_fraction", 0.0)
    findings.append({
        "check": "shell_is_a_plausible_fraction",
        "status": "PASS" if 0.05 <= shell <= 0.95 else "FAIL",
        "detail": f"the shell took {shell:.1%} of the mesh's triangles"
                  + ("" if 0.05 <= shell <= 0.95 else
                     "; outside 5-95% means plane detection either found nothing "
                     "or consumed the room"),
    })

    statuses = [f["status"] for f in findings]
    return {
        "overall": "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS"),
        "checks": findings,
        "objects": len(objs),
        "note": "no ground truth is involved; these are self-consistency checks",
    }


def write(result: CutResult, audit_report: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "scale_mode": result.scale_mode,
        "units": "DA3 units, not metres" if result.scale_mode == "assumed" else "metres",
        "scene_diagonal": result.scene_diagonal,
        "up_axis": result.up_axis,
        "up_axis_evidence": result.up_axis_evidence,
        "floor_offset": result.floor_offset,
        "room_extent": result.room_extent,
        "room_aabb_min": result.room_aabb_min,
        "room_aabb_max": result.room_aabb_max,
        "shell_planes": [asdict(p) for p in result.shell_planes],
        "objects": [asdict(o) for o in result.objects],
        "diagnostics": result.diagnostics,
        "audit": audit_report,
    }
    path = out_dir / "objects.json"
    path.write_text(json.dumps(doc, indent=2))
    return path
