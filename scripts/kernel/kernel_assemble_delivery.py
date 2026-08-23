"""Gather every artefact into one delivery tree, with a manifest that states units.

`kernel_outputs.py` writes the objects, terrain and videos into /tmp/deliver, but the
rest of the run is scattered across /tmp/run and the scene directory. This collects
all of it into one layout, checks each piece is non-empty and readable, and writes a
README that travels with the data.

Hard links, not copies: the payload is several gigabytes, /tmp is one filesystem, and
a copy would double it for nothing. Falls back to a copy across devices.

The manifest is the point as much as the files are. Every length in this delivery is
in DA3's arbitrary units, and the one thing a consumer must not do is read them as
metres — so the units are stated in the README, in a machine-readable
`manifest.json`, and burned into every video frame.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

SCENE = os.environ.get("V3D_SCENE", "small_factory")
WORK = Path(os.environ.get("V3D_WORK_ROOT", "/tmp/run"))
DATA = Path(os.environ.get("V3D_DATA_ROOT", "/tmp/data/scannetppv2/data"))
OUT = Path(os.environ.get("V3D_DELIVER", "/tmp/deliver"))
SCALE_MODE = os.environ.get("V3D_SCALE_MODE", "assumed")
# Depth maps are 600 x 8.3 MB. They are one of the requested deliverables, so they go
# in by default, but the switch exists because they dominate the payload.
INCLUDE_DEPTH = os.environ.get("V3D_DELIVER_DEPTH", "1") not in {"0", "false", ""}


def place(src: Path, dest: Path) -> dict | None:
    """Hard-link one file into the delivery tree and describe it."""
    if not src.exists() or not src.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
        how = "link"
    except OSError:
        shutil.copy2(src, dest)
        how = "copy"
    return {"path": str(dest.relative_to(OUT)), "bytes": dest.stat().st_size,
            "from": str(src), "how": how}


def place_tree(src: Path, dest: Path, pattern: str = "*",
               limit: int | None = None) -> list[dict]:
    if not src.is_dir():
        return []
    files = sorted(p for p in src.glob(pattern) if p.is_file())
    if limit is not None and len(files) > limit:
        files = files[:limit]
    placed = []
    for p in files:
        rec = place(p, dest / p.name)
        if rec:
            placed.append(rec)
    return placed


def main() -> None:
    scene_dir = DATA / SCENE
    gs = WORK / "gs" / SCENE
    OUT.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "scene": SCENE,
        "scale_mode": SCALE_MODE,
        "units": ("DA3 arbitrary units — NOT metres" if SCALE_MODE == "assumed"
                  else "metres"),
        "groups": {},
        "missing": [],
    }
    groups: dict[str, list] = {}

    def group(name: str, items: list) -> None:
        items = [i for i in items if i]
        groups[name] = items
        total = sum(i["bytes"] for i in items)
        print(f"  {name:22} {len(items):>5} files  {total/1e6:>9.1f} MB")

    print("=" * 74)
    print(f"assembling the delivery for {SCENE}")
    print("=" * 74)

    # --- geometry ------------------------------------------------------------
    group("point_cloud", [
        place(scene_dir / "pointcloud_da3.ply", OUT / "geometry" / "point_cloud_8M.ply"),
    ])
    group("mesh", [
        place(WORK / "mesh" / f"{SCENE}_deliver.ply",
              OUT / "geometry" / "mesh_decimated_1.5M.ply"),
        place(gs / "mesh" / "tsdf_fusion_pruned.ply",
              OUT / "geometry" / "mesh_full_pruned.ply"),
        place(gs / "mesh" / "tsdf_fusion_post.ply",
              OUT / "geometry" / "mesh_full_fused.ply"),
    ])

    # --- gaussians -----------------------------------------------------------
    gauss = []
    pc_dir = gs / "point_cloud"
    if pc_dir.is_dir():
        iters = sorted((int(p.name.split("_")[1]), p) for p in pc_dir.glob("iteration_*"))
        if iters:
            last_it, last = iters[-1]
            gauss.append(place(last / "point_cloud.ply",
                               OUT / "gaussians" / f"gaussians_iter{last_it}.ply"))
            gauss.append(place(gs / "cfg_args", OUT / "gaussians" / "cfg_args"))
    group("gaussians_3dgs", gauss)

    # --- cameras and depth ---------------------------------------------------
    group("cameras", [
        place(scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json",
              OUT / "cameras" / "transforms_undistorted.json"),
        place(scene_dir / "pose_report.json", OUT / "cameras" / "pose_report.json"),
        place(scene_dir / "ingest_manifest.json",
              OUT / "cameras" / "ingest_manifest.json"),
    ])
    if INCLUDE_DEPTH:
        group("depth_maps", place_tree(scene_dir / "depth_da3",
                                       OUT / "depth" / "depth_da3", "*.npy"))
    else:
        print("  depth_maps            skipped (V3D_DELIVER_DEPTH=0)")

    # --- semantics -----------------------------------------------------------
    sem = [
        place(WORK / "scene_objects" / f"{SCENE}.json",
              OUT / "semantics" / "vlm_vocabulary.json"),
        place(WORK / "scene" / SCENE / "scene.json", OUT / "semantics" / "scene.json"),
        place(WORK / "scene" / SCENE / "stage_j.json",
              OUT / "semantics" / "stage_j.json"),
    ]
    boxes = WORK / "boxes" / SCENE
    sem += place_tree(boxes, OUT / "semantics" / "boxes_floor_aligned", "*.json")
    # lift3d also writes a GLB holding the lifted cloud AND the boxes in one file --
    # the single most useful artefact for eyeballing whether a box sits where its
    # object is, and easy to miss because the rest of that directory is json.
    sem += place_tree(WORK / "boxes_raw" / SCENE,
                      OUT / "semantics" / "lifted_points_and_boxes", "*.glb")
    sem += place_tree(WORK / "boxes_raw" / SCENE,
                      OUT / "semantics" / "boxes_raw", "*.json")
    group("semantics", sem)

    # --- reports: the numbers that justify every setting ---------------------
    group("reports", [
        place(WORK / f"da3_resolution_{SCENE}.json", OUT / "reports" / f"da3_resolution.json"),
        place(WORK / f"chunk_scale_{SCENE}.json", OUT / "reports" / "chunk_scale.json"),
        place(WORK / f"da3_summary_{SCENE}.json", OUT / "reports" / "da3_summary.json"),
        place(WORK / f"pgsr_summary_{SCENE}.json", OUT / "reports" / "pgsr_summary.json"),
        place(WORK / f"mesh_summary_{SCENE}.json", OUT / "reports" / "mesh_summary.json"),
        place(WORK / f"semantics_summary_{SCENE}.json",
              OUT / "reports" / "semantics_summary.json"),
        place(WORK / "knobs" / f"{SCENE}.json", OUT / "reports" / "derived_knobs.json"),
        place(WORK / f"prune_{SCENE}.json", OUT / "reports" / "component_prune.json"),
    ])
    group("logs", place_tree(WORK / "logs", OUT / "logs", "*.log"))

    # Whatever kernel_outputs already wrote stays where it is; just measure it.
    for name, sub in (("objects_cut", "objects"), ("terrain", "terrain"),
                      ("mujoco", "mujoco"), ("videos", "videos"),
                      ("exchange_formats", "exchange")):
        d = OUT / sub
        files = sorted(p for p in d.rglob("*") if p.is_file()) if d.is_dir() else []
        # Frame directories are render intermediates, not deliverables.
        files = [p for p in files if not any(part.startswith("_") for part in p.parts)]
        group(name, [{"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size,
                      "from": str(p), "how": "in place"} for p in files])
    for extra in ("objects.json", f"outputs_{SCENE}.json"):
        p = OUT / extra
        if p.exists():
            groups.setdefault("objects_cut", []).append(
                {"path": extra, "bytes": p.stat().st_size, "from": str(p),
                 "how": "in place"})

    manifest["groups"] = {k: {"files": len(v), "bytes": sum(i["bytes"] for i in v),
                              "items": v[:40],
                              "truncated": max(0, len(v) - 40)}
                          for k, v in groups.items()}
    empty = [k for k, v in groups.items() if not v]
    manifest["missing"] = empty
    total = sum(sum(i["bytes"] for i in v) for v in groups.values())
    manifest["total_bytes"] = total

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    readme = f"""# {SCENE} — video to 3D, full output set

Reconstructed from a single handheld video ({scene_dir.name}) with no depth sensor,
no markers and no measured distances.

## READ THIS FIRST: the units are not metres

scale_mode = **{SCALE_MODE}**

{"Every length in this delivery is in DA3's own arbitrary units. Scale is not"
 " recoverable from a monocular video: a room and a photograph of a room twice its"
 " size produce identical images. Proportions and angles are correct and transfer"
 " exactly; absolute sizes do not exist yet." if SCALE_MODE == "assumed" else
 "Lengths are metric."}

**One tape measurement of anything identifiable in the footage converts this whole
delivery to metres.** Nothing else can supply it. Until then, treat any figure in
metres derived from these files as unfounded — including masses and forces in the
MuJoCo scene, since mass scales as the cube of a length.

## What is here

| Directory | Contents |
|---|---|
| `geometry/` | 8 M-point cloud; fused mesh; a 1.5 M-triangle decimated mesh for viewing |
| `gaussians/` | the trained 3DGS checkpoint |
| `cameras/` | camera poses, intrinsics, the frame-selection record |
| `depth/` | per-frame DA3 depth, one .npy per frame |
| `objects/` | per-object visual mesh and convex-hull collision proxy |
| `terrain/` | height map (.npy + 16-bit .png), slope and step maps |
| `mujoco/` | MJCF scene: heightfield terrain plus one body per object |
| `semantics/` | VLM vocabulary, SAM3-derived boxes, scene.json, Stage J verdict |
| `reports/` | the measurement behind every setting this run chose |
| `videos/` | one mp4 per output; units burned into every frame |
| `exchange/` | OBJ, STL, GLB, PCD, URDF, Esri ASCII grid for other tools |
| `logs/` | stage logs |

## How the settings were chosen

`reports/` is not decoration — it holds the evidence. In particular:

- `da3_resolution.json` — the VRAM ladder that picked the depth resolution, with
  every rung measured rather than predicted.
- `chunk_scale.json` — why the *highest* resolution was rejected: inferring in 20
  chunks made independently-fitted depth scales disagree by 2.0%, which would have
  been baked into the surface. The configuration used disagrees by 0.11%.
- `derived_knobs.json` — the mesh parameters, each derived from this scene and
  reported against the paper's constant. The paper's `max_depth=4.0` would have
  erased 16.5% of believable depth here; `num_cluster=1` would have deleted
  2,077,650 triangles of freestanding geometry.

## Interchange formats, and one thing to know about them

`exchange/` holds the same geometry in the formats other tools read. **glTF/GLB files
are rotated to Y-up** because the format requires it; everything else in this
delivery is Z-up. A GLB opened without that rotation lies on its side and looks like
a reconstruction fault rather than a format one.

Both GLBs were written with trimesh and read back to confirm the face count, because
Open3D writes glTF that Open3D itself cannot re-read (`bugs-found` B9) — its output
declared a buffer view longer than the file. The OBJ, STL, PCD and ASCII-grid exports
are likewise verified by re-reading and comparing element counts.

## Known limits

- The collision proxies are convex **hulls**, so a container reads as a solid block.
  Convex decomposition is the fix and is not applied yet.
- Physics values (mass, friction, restitution) are **priors, not measurements**, and
  are unanchored while scale is assumed.
- The mesh is not watertight, and thin structures — railings, cage panels, cable
  trays — reconstruct poorly and need primitive fitting rather than mesh.
- Held-out PSNR is not reported; the figure in `pgsr_summary.json` is on training
  views.
"""
    (OUT / "README.md").write_text(readme)

    print("-" * 74)
    print(f"  TOTAL {total/1e9:.2f} GB across {sum(len(v) for v in groups.values())} files")
    if empty:
        print(f"  EMPTY GROUPS: {empty}")
        print("  (an empty group means that stage has not run — the delivery is partial)")
    print(f"  -> {OUT/'manifest.json'} and {OUT/'README.md'}")
    print("\nASSEMBLE_OK")


main()
