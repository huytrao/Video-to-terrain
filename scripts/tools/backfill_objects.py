"""Fill the unseen side of a carved object with the generated mesh from stage J.

A carved object is only ever a front: the scan saw one side of it and nothing
else, so capping its boundary with flat triangles invents a shape rather than
recovering one.  Stage J already holds a full 360-degree mesh for each object,
produced by Hunyuan3D from the same views, so the missing half is not missing
at all -- it just is not registered tightly enough to drop straight in.

This registers it properly and keeps only the parts the scan does not cover:

  scan surface (real colour, correct where it exists)
  + generated surface, but only where the scan has nothing within `--cover`

Stage G's own alignment is scored on mask IoU, which fixes silhouette but not
depth, so it lands 12-14 cm out on the large objects.  Plain ICP cannot fix
that here: the target is a partial surface, and every generated back-face would
drag towards the front.  Trimmed ICP -- fit on the best-matching fraction only
-- is what makes a partial target safe.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[2]


def cluster_decimate(mesh, cell):
    """Vertex-cluster decimation. No decimation library is installed, and a
    generated mesh at 500k-1M faces is far too heavy to carry into a browser."""
    keys = np.floor(mesh.vertices / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(float)

    position = np.zeros((len(count), 3))
    np.add.at(position, inverse, mesh.vertices)
    position /= count[:, None]

    source = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float)
    colour = np.zeros((len(count), 3))
    np.add.at(colour, inverse, source)
    colour /= count[:, None]

    faces = inverse[mesh.faces]
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[keep]
    # Deduplicate by sorted key but re-index the ORIGINAL rows, so winding lives.
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(first)]

    out = trimesh.Trimesh(vertices=position, faces=faces, process=False)
    out.visual = trimesh.visual.ColorVisuals(
        mesh=out,
        vertex_colors=np.column_stack(
            [colour.round().clip(0, 255).astype(np.uint8), np.full(len(colour), 255, np.uint8)]
        ),
    )
    return out


def trimmed_icp(source, target, iterations, keep, sample, allow_scale=False):
    """Rigid+scale fit of `source` onto `target`, ignoring the worst matches.

    `keep` is the fraction of correspondences trusted each round.  Everything
    beyond it is assumed to be surface the target simply does not have -- which
    for a one-sided scan is most of the generated mesh.
    """
    step = max(1, len(source) // sample)
    probe = source[::step]
    tree = cKDTree(target)
    total = np.eye(4)
    current = probe.copy()
    for _ in range(iterations):
        distance, index = tree.query(current)
        order = np.argsort(distance)[: max(12, int(len(current) * keep))]
        # Rigid only by default. With scale free, trimmed ICP has a degenerate
        # escape: shrinking the source collapses it into the covered patch and
        # drives the trusted residual down while wrecking the fit -- it pulled
        # the workbench to 0.56x. Stage G already fixed size from the OBB.
        matrix, _, _ = trimesh.registration.procrustes(
            current[order], target[index[order]],
            reflection=False, translation=True, scale=allow_scale,
        )
        current = trimesh.transform_points(current, matrix)
        total = matrix @ total
    distance, _ = tree.query(current)
    return total, float(np.median(distance))


def hull_backing(scan, cover, inset, grey):
    """A neutral grey shell behind the scan, for objects with no usable generation.

    Used only when registration is rejected. It invents no detail -- it is the
    object's own convex hull pulled slightly inward so it stays behind the real
    surface -- but it stops the object being see-through, which capping the torn
    boundary could not do without throwing out spikes.
    """
    hull = scan.convex_hull
    hull.vertices = hull.vertices - hull.vertex_normals * inset
    keep = cKDTree(scan.vertices).query(hull.triangles_center)[0] > cover
    if not keep.any():
        return None
    backing = hull.submesh([np.where(keep)[0]], append=True)
    backing.visual = trimesh.visual.ColorVisuals(
        mesh=backing,
        vertex_colors=np.tile(
            np.append(np.asarray(grey).round().astype(np.uint8), 255), (len(backing.vertices), 1)
        ),
    )
    return backing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carved", default="outputs-full-factory/delivery/OUTPUT_K")
    ap.add_argument("--generated", default="outputs-full-factory/delivery/OUTPUT_J")
    ap.add_argument("--generated-file", default="mesh_textured.glb")
    ap.add_argument("--cell", type=float, default=0.018, help="decimation cell, metres")
    ap.add_argument("--cover", type=float, default=0.05,
                    help="scan is considered to cover a generated face within this radius")
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--keep", type=float, default=0.55)
    ap.add_argument("--sample", type=int, default=6000)
    ap.add_argument("--grey", default="150,150,154",
                    help="colour for generated surface that carries none")
    ap.add_argument("--fallback", choices=("hull", "none"), default="none",
                    help="what to do for objects whose generated mesh is rejected. "
                         "`hull` is kept only for experiments: a convex hull sits "
                         "OUTSIDE every concavity, so on a scan shell it covers the "
                         "real front surface instead of backing it.")
    ap.add_argument("--inset", type=float, default=0.02,
                    help="metres to pull the hull backing behind the scan surface")
    ap.add_argument("--allow-scale", action="store_true",
                    help="let registration resize the generated mesh (degenerate; off)")
    ap.add_argument("--max-residual", type=float, default=0.055,
                    help="reject a generated mesh that will not register closer than this")
    ap.add_argument("--objects", nargs="+", required=True)
    args = ap.parse_args()

    grey = np.asarray([float(v) for v in args.grey.split(",")])
    carved_dir = ROOT / args.carved
    manifest = json.loads((carved_dir / "carve_manifest.json").read_text())
    by_name = {entry["name"]: entry for entry in manifest["objects"]}

    def load(path):
        scene = trimesh.load(path)
        return scene.dump(concatenate=True) if hasattr(scene, "geometry") else scene

    report = []
    for name in args.objects:
        source = ROOT / args.generated / name / args.generated_file
        if not source.exists():
            print(f"  skip  {name:24} no generated mesh at {source.relative_to(ROOT)}")
            continue

        # Always the carve's untouched output, never the file this writes.
        scan = load(carved_dir / "objects_raw" / f"{name}.glb")
        generated = cluster_decimate(load(source), args.cell)

        # Stage G scores yaw on mask IoU, and for a slab-shaped object 0 and
        # 180 degrees score almost the same -- obj_0179 came back 0.360 vs
        # 0.343. ICP only refines locally, so a yaw picked wrong stays wrong.
        # Start it once per quarter turn and keep whichever lands closest.
        pivot = generated.bounds.mean(axis=0)
        best = None
        for yaw in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
            spin = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1], pivot)
            start = trimesh.transform_points(generated.vertices, spin)
            matrix, residual = trimmed_icp(
                start, scan.vertices, args.iterations, args.keep, args.sample,
                allow_scale=args.allow_scale,
            )
            if best is None or residual < best[1]:
                best = (matrix @ spin, residual, np.degrees(yaw))
        matrix, residual, yaw_deg = best

        if residual > args.max_residual:
            print(
                f"  REJECT {name:23} best fit {residual*100:5.1f} cm at yaw {yaw_deg:3.0f}deg"
                f" > limit {args.max_residual*100:.1f} cm; leaving the scan shell alone"
            )
            entry = by_name[name]
            entry.update(
                backfill_rejected_residual_cm=round(residual * 100, 2),
                backfilled_from=None,
            )
            if args.fallback == "hull":
                backing = hull_backing(scan, args.cover, args.inset, grey)
                if backing is not None:
                    trimesh.util.concatenate([scan, backing]).export(
                        carved_dir / "objects" / f"{name}.glb"
                    )
                    entry["hull_backing_faces"] = int(len(backing.faces))
                    print(
                        f"         {name:23} -> grey hull backing, {len(backing.faces)} faces"
                    )
            continue

        generated.apply_transform(matrix)

        # Keep only what the scan has no answer for.
        covered = cKDTree(scan.vertices).query(generated.triangles_center)[0] <= args.cover
        missing = np.where(~covered)[0]
        if len(missing) == 0:
            print(f"  skip  {name:24} scan already covers the generated surface")
            continue
        backing = generated.submesh([missing], append=True)

        colours = np.asarray(backing.visual.vertex_colors, dtype=np.uint8)
        luminance = colours[:, :3].astype(float) @ np.array([0.2126, 0.7152, 0.0722])
        blank = luminance < 24
        colours[blank, :3] = grey.round().astype(np.uint8)
        backing.visual = trimesh.visual.ColorVisuals(mesh=backing, vertex_colors=colours)

        fused = trimesh.util.concatenate([scan, backing])
        fused.export(carved_dir / "objects" / f"{name}.glb")

        scale = float(np.cbrt(abs(np.linalg.det(matrix[:3, :3]))))
        entry = by_name[name]
        entry.update(
            backfilled_from=f"{args.generated}/{name}/{args.generated_file}",
            backfill_faces=int(len(backing.faces)),
            backfill_residual_cm=round(residual * 100, 2),
            backfill_scale=round(scale, 4),
            backfill_yaw_deg=round(float(yaw_deg), 1),
        )
        report.append((name, len(scan.faces), len(backing.faces), residual * 100, scale))
        print(
            f"  fuse  {name:24} scan {len(scan.faces):6d} + gen {len(backing.faces):6d} faces"
            f"   residual {residual*100:5.1f} cm   yaw {yaw_deg:3.0f}deg   scale x{scale:.3f}"
        )

    (carved_dir / "carve_manifest.json").write_text(json.dumps(manifest, indent=2))
    if report:
        worst = max(report, key=lambda r: r[3])
        print(f"\nworst fit: {worst[0]} at {worst[3]:.1f} cm")


if __name__ == "__main__":
    main()
