"""TSDF-fuse the gaussians into a mesh, then render the mesh-guided masks.

This is where the four inherited `mesh.sh` constants would do their damage, and
where the derived ones earn their keep. On this scene, measured:

    render_max_depth  4.0 (paper) erases 16.5% of believable depth
                      5.086 (derived) erases 2.0%

Sixteen percent of the depth deleted is not a subtle quality loss — it is the far
half of a factory bay reconstructed only from the few frames that stood close to
it, which is exactly where `output-quality.md` reports ragged edges and holes.

`num_cluster` moves from 1 to 200 and `use_depth_filter` from off to on, and those
two have to move together. With the filter off, multi-view-inconsistent depth gets
fused as if it were real — the pilot measured 140,334 connected components — and
keeping 200 of those would keep 199 pieces of noise. Keeping only the largest, on
the other hand, deletes every freestanding machine and crate the fusion did not
weld to the floor: precisely the objects the semantic-box and collision
deliverables are about. Filter on plus a size-relative cull afterwards is the
combination that removes speckle *and* keeps objects, and `prune_components`
reports what it dropped, because a cleanup that does not report its own losses is
indistinguishable from one that removed the subject.

The mesh is also decimated for delivery, separately from the mesh used for
computation. At the derived voxel size this mesh will be far larger than the
pilot's 9.5 M vertices, and detail should be paid for in computation and given up
only at the last step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")
DA3_SRC = "/tmp/da3src/Depth-Anything-3/src"
if DA3_SRC not in sys.path:
    sys.path.insert(0, DA3_SRC)
os.environ.setdefault("V3D_IMPORT_PATHS", f"/kaggle/working/src{os.pathsep}{DA3_SRC}")

SCENE = os.environ.get("V3D_SCENE", "small_factory")
WORK = Path(os.environ.get("V3D_WORK_ROOT", "/tmp/run"))
DELIVER_TRIS = int(os.environ.get("V3D_DELIVER_TRIANGLES", "1500000"))

import numpy as np  # noqa: E402


def mesh_stats(path: Path) -> dict:
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(path))
    v = np.asarray(m.vertices)
    return {
        "vertices": int(len(m.vertices)),
        "triangles": int(len(m.triangles)),
        "mb": round(path.stat().st_size / 1e6, 1),
        "extent": [float(x) for x in (v.max(axis=0) - v.min(axis=0))] if len(v) else None,
        "watertight": bool(m.is_watertight()),
    }


def main() -> None:
    from v3d.config import PipelineConfig, detect_layout
    from v3d import paper_runner as pr
    from v3d.geometry import quality as q

    cfg = PipelineConfig()
    layout = detect_layout(SCENE, cfg.data_root)
    model_dir = cfg.gs_root / layout.scene_id

    print("=" * 78)
    print("mesh extraction and mesh-guided masks")
    print("=" * 78)
    for p in pr.apply_offline_patches(cfg):
        print(f"  {p.describe()}")

    print("\n--- knobs, derived against the inherited constants ---")
    knobs, schedule = pr.derive_knobs(layout, cfg)

    mesh_path = model_dir / pr.MESH_PATH_REL
    if mesh_path.exists():
        print(f"\n--- mesh already present: {mesh_path} ---")
    else:
        print(f"\n--- render.py: TSDF fusion at voxel {knobs.voxel_size:.6f}, "
              f"max_depth {knobs.render_max_depth:.3f}, "
              f"num_cluster {knobs.num_cluster}, depth_filter {knobs.use_depth_filter} ---")
        t = time.time()
        rc = pr.stage_mesh(layout, cfg, knobs=knobs)
        print(f"    exit {rc} after {(time.time() - t) / 60:.1f} min")
        if rc != 0:
            for name in (f"{layout.scene_id}_render.log", f"{layout.scene_id}_mesh2mask.log"):
                log = cfg.log_root / name
                if log.exists():
                    print(f"--- {name} ---")
                    print("\n".join(log.read_text().splitlines()[-30:]))
            raise SystemExit("stage_mesh failed")

    if not mesh_path.exists():
        raise SystemExit(f"render.py reported success but {mesh_path} is absent")

    print("\n--- what fusion produced ---")
    stats = mesh_stats(mesh_path)
    for k, v in stats.items():
        print(f"  {k:12} {v}")

    # --- size-relative component cull ----------------------------------------
    import open3d as o3d

    print("\n--- component cull (relative to the largest, not absolute) ---")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    pruned_path = model_dir / "mesh" / "tsdf_fusion_pruned.ply"
    if pruned_path.exists():
        print(f"  already pruned: {pruned_path}")
        prune = json.loads((WORK / f"prune_{SCENE}.json").read_text()) \
            if (WORK / f"prune_{SCENE}.json").exists() else {}
    else:
        prune = q.prune_components(mesh, q.KnobPolicy())
        for k, v in prune.items():
            print(f"  {k:34} {v}")
        beyond = prune.get("triangles_beyond_largest_component", 0)
        total = prune.get("triangles_kept", 1) + prune.get("triangles_removed", 0)
        print(f"  -> num_cluster=1 would have deleted {beyond:,} triangles "
              f"({beyond / max(total, 1) * 100:.1f}% of the mesh) that are not the "
              f"largest component")
        o3d.io.write_triangle_mesh(str(pruned_path), mesh, write_ascii=False)
        (WORK / f"prune_{SCENE}.json").write_text(json.dumps(prune, indent=2))
        print(f"  wrote {pruned_path} ({pruned_path.stat().st_size / 1e6:.1f} MB)")

    pruned_stats = mesh_stats(pruned_path)
    print(f"  pruned mesh: {pruned_stats['vertices']:,} verts, "
          f"{pruned_stats['triangles']:,} tris")

    # --- a delivery mesh, decimated only at the last step ---------------------
    deliver = WORK / "mesh" / f"{SCENE}_deliver.ply"
    if not deliver.exists():
        print(f"\n--- decimating for delivery to ~{DELIVER_TRIS:,} triangles ---")
        m = o3d.io.read_triangle_mesh(str(pruned_path))
        if len(m.triangles) > DELIVER_TRIS:
            m = m.simplify_quadric_decimation(target_number_of_triangles=DELIVER_TRIS)
        m.remove_unreferenced_vertices()
        deliver.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(deliver), m, write_ascii=False)
        print(f"  {deliver} ({deliver.stat().st_size / 1e6:.1f} MB, "
              f"{len(m.triangles):,} tris)")

    # --- masks ---------------------------------------------------------------
    n_masks = len(list(layout.mask_dir.glob("*"))) if layout.mask_dir.is_dir() else 0
    n_frames = len(list(layout.images_dir.glob("*.JPG")))
    print(f"\n--- mesh-guided masks: {n_masks} for {n_frames} frames ---")
    if n_masks < n_frames:
        print(f"  WARNING: {n_frames - n_masks} frames have no mask; sam3 will skip them")

    out = WORK / f"mesh_summary_{SCENE}.json"
    out.write_text(json.dumps({
        "scene": SCENE,
        "knobs": {k: v for k, v in vars(knobs).items() if k != "evidence"},
        "knob_evidence": knobs.evidence,
        "fused": stats,
        "pruned": pruned_stats,
        "prune": prune,
        "deliver_mesh": str(deliver),
        "masks": n_masks,
        "frames": n_frames,
    }, indent=2, default=str))
    print(f"\n  -> {out}")
    print("\nMESH_OK")


main()
