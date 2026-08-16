"""Run the Holi-Spatial pipeline unmodified, on a kernel with no network.

The paper's code is 4,300 lines that went through review; reimplementing it
would trade a known-good method for fresh bugs. So it runs as published, and
this module supplies only what an offline machine cannot: local paths in place
of HuggingFace repo ids, and the stage ordering.

Nothing here knows that today's input is ScanNet++. Locations come from
``PipelineConfig`` (environment-overridable) and the shape of a scene comes
from ``SceneLayout``, so a scene reconstructed from an arbitrary video runs the
same stages by presenting the same two things: a directory of frames and a
camera transforms file.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from v3d.config import PipelineConfig, SceneLayout, detect_layout

# --- substitutions the published code needs in order to run offline ---------

# inference_da3_scannetppv2.py hardcodes a HuggingFace repo id.
DA3_REPO_ID = "depth-anything/DA3NESTED-GIANT-LARGE"

# PGSR's ScanNet++ reader falls back to an absolute path from the authors'
# machine when no --ply_path is given. render.py has no such flag, so this must
# be fixed at the source. The repository already does the right thing in
# readScannetV2SceneInfo — derive the cloud from the scene — and this makes the
# ScanNet++ reader agree with it. Its `path` is the transforms directory, so
# for the ScanNet++ layout the DA3 cloud sits two levels up.
STALE_PLY_FALLBACK = '    ply_path = "output_scannet/scannetppv2/0b031f3119/input.ply"'
DERIVED_PLY_FALLBACK = (
    '    ply_path = os.path.join(path, "..", "..", "pointcloud_da3.ply")'
    "  # v3d: was an absolute path from the authors' machine"
)

# mesh.sh defaults, kept identical so the surface matches the paper's.
RENDER_MAX_DEPTH = "4.0"
MASK_MAX_DEPTH = "6.0"
VOXEL_SIZE = "0.006"
NUM_CLUSTER = "1"
MESH_PATH_REL = "mesh/tsdf_fusion_post.ply"


@dataclass
class Patch:
    path: Path
    before: str
    after: str
    applied: bool


def apply_offline_patches(cfg: PipelineConfig | None = None) -> list[Patch]:
    """Replace the two paths that only existed on the authors' machine. Idempotent."""
    cfg = cfg or PipelineConfig()
    patches: list[Patch] = []

    def swap(rel: str, before: str, after: str) -> None:
        target = cfg.paper_root / rel
        text = target.read_text()
        if before in text:
            target.write_text(text.replace(before, after))
            patches.append(Patch(target, before, after, applied=True))
        else:
            patches.append(Patch(target, before, after, applied=False))

    swap("inference_da3_scannetppv2.py", f'"{DA3_REPO_ID}"', f'"{cfg.da3_model}"')
    swap("PGSR/scene/dataset_readers.py", STALE_PLY_FALLBACK, DERIVED_PLY_FALLBACK)
    return patches


def _run(cmd: list[str], cfg: PipelineConfig, log_name: str) -> int:
    """Run a stage, streaming to a log so a long GPU job can be watched."""
    log = cfg.log_root / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run(
            cmd,
            cwd=str(cfg.paper_root),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=cfg.subprocess_env(),
        )
    return proc.returncode


def _layout(scene: str | SceneLayout, cfg: PipelineConfig) -> SceneLayout:
    return scene if isinstance(scene, SceneLayout) else detect_layout(scene, cfg.data_root)


def stage_da3(scene: str | SceneLayout, cfg: PipelineConfig | None = None) -> int:
    """DA3 depth and point cloud, written beside the scene."""
    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    missing = layout.missing()
    if missing:
        raise FileNotFoundError("; ".join(missing))

    return _run(
        [sys.executable, "inference_da3_scannetppv2.py",
         "--scannetppv2_dir", str(layout.scene_dir)],
        cfg,
        f"{layout.scene_id}_da3.log",
    )


def stage_pgsr(scene: str | SceneLayout, cfg: PipelineConfig | None = None) -> int:
    """PGSR / 3DGS training — the stage that produces surface quality.

    Arguments mirror 3dgs_train.sh, with --ply_path supplied so the run does
    not reach the stale fallback described above.
    """
    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    missing = layout.missing()
    if missing:
        raise FileNotFoundError("; ".join(missing))
    if not layout.point_cloud.exists():
        raise FileNotFoundError(f"run stage_da3 first: {layout.point_cloud} is missing")

    cfg.gs_root.mkdir(parents=True, exist_ok=True)
    return _run(
        [sys.executable, "PGSR/train.py",
         "-s", str(layout.source_dir),
         "-i", str(layout.images_dir),
         "-m", str(cfg.gs_root / layout.scene_id),
         "--ply_path", str(layout.point_cloud),
         "--resolution", "1"],
        cfg,
        f"{layout.scene_id}_pgsr.log",
    )


def stage_mesh(scene: str | SceneLayout, cfg: PipelineConfig | None = None) -> int:
    """TSDF-fuse the trained gaussians into a mesh, then render mesh-guided masks."""
    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    model_dir = cfg.gs_root / layout.scene_id
    if not (model_dir / "point_cloud").is_dir():
        raise FileNotFoundError(f"run stage_pgsr first: no gaussians under {model_dir}")

    rc = _run(
        [sys.executable, "PGSR/render.py",
         "-s", str(layout.source_dir), "-m", str(model_dir), "-i", str(layout.images_dir),
         "--skip_test",
         "--max_depth", RENDER_MAX_DEPTH,
         "--voxel_size", VOXEL_SIZE,
         "--num_cluster", NUM_CLUSTER],
        cfg,
        f"{layout.scene_id}_render.log",
    )
    if rc != 0:
        return rc

    mesh = model_dir / MESH_PATH_REL
    if not mesh.exists():
        raise FileNotFoundError(f"render.py reported success but {mesh} is absent")

    return _run(
        [sys.executable, "PGSR/mesh2mask.py",
         "-m", str(model_dir), "-s", str(layout.source_dir), "-i", str(layout.images_dir),
         "--mesh_path", MESH_PATH_REL,
         "--max_depth", MASK_MAX_DEPTH,
         "--voxel_size", VOXEL_SIZE],
        cfg,
        f"{layout.scene_id}_mesh2mask.log",
    )


def stage_vlm(
    scene: str | SceneLayout,
    cfg: PipelineConfig | None = None,
    max_images: int | None = None,
) -> int:
    """Discover the class vocabulary, in-process instead of against a vLLM server.

    Writes the `{scene, per_image}` document sam3.py's ``load_scene_json``
    expects, so the paper's next stage consumes it unchanged.
    """
    import json

    from v3d.semantic.vlm import VLMLabeler

    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    images = sorted(p for p in layout.images_dir.iterdir() if p.suffix.lower() in
                    {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"no frames in {layout.images_dir}")

    if max_images and len(images) > max_images:
        # even sampling, as classic_vllm.collect_images does
        idx = sorted({round(i * (len(images) - 1) / (max_images - 1)) for i in range(max_images)})
        images = [images[i] for i in idx]

    labeler = VLMLabeler(cfg.vlm_model)
    results, vocabulary, merges = labeler.label_scene(images)

    out = cfg.work_root / "scene_objects" / f"{layout.scene_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {
            "scene": layout.scene_id,
            "per_image": {Path(r.image).name: r.labels for r in results},
            "vocabulary": vocabulary,
            "merged_specialisations": merges,
        },
        indent=2,
    ))
    print(f"  {len(results)} frames labelled, {len(vocabulary)} distinct classes -> {out}")
    if merges:
        print(f"  folded {len(merges)} specialisations into their base class:")
        for k, v in sorted(merges.items())[:15]:
            print(f"    {k!r} -> {v!r}")
    return 0


def vlm_output(scene_id: str, cfg: PipelineConfig) -> Path:
    return cfg.work_root / "scene_objects" / f"{scene_id}.json"


def stage_sam3(scene: str | SceneLayout, cfg: PipelineConfig | None = None) -> int:
    """Text-prompted masks for the discovered classes."""
    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    scene_json = vlm_output(layout.scene_id, cfg)
    if not scene_json.exists():
        raise FileNotFoundError(f"run stage_vlm first: {scene_json} is missing")

    return _run(
        [sys.executable, "sam3.py",
         "--data-root", str(cfg.data_root),
         "--scene-json", str(scene_json),
         "--output-dir", str(cfg.work_root / "sam_masks" / layout.scene_id),
         "--sam3-checkpoint", str(cfg.sam3_checkpoint)],
        cfg,
        f"{layout.scene_id}_sam3.log",
    )


def stage_report(scene: str | SceneLayout, cfg: PipelineConfig | None = None) -> dict[str, object]:
    """What exists so far, so a stage can be resumed rather than repeated."""
    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    mesh = cfg.gs_root / layout.scene_id / MESH_PATH_REL

    def mb(p: Path) -> float | None:
        return round(p.stat().st_size / 1e6, 1) if p.exists() else None

    return {
        "scene": layout.scene_id,
        "layout": str(layout.source_dir),
        "frames": len(list(layout.images_dir.glob("*"))) if layout.images_dir.is_dir() else 0,
        "depth_maps": len(list(layout.depth_dir.glob("*.npy"))) if layout.depth_dir.is_dir() else 0,
        "point_cloud_mb": mb(layout.point_cloud),
        "mesh_mb": mb(mesh),
        "masks": len(list(layout.mask_dir.glob("*"))) if layout.mask_dir.is_dir() else 0,
        "vlm_labels": vlm_output(layout.scene_id, cfg).exists(),
    }


STAGES = {"da3": stage_da3, "pgsr": stage_pgsr, "mesh": stage_mesh,
          "vlm": stage_vlm, "sam3": stage_sam3}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene")
    ap.add_argument("--stage", default="report",
                    choices=["patch", "da3", "pgsr", "mesh", "vlm", "sam3", "all", "report"])
    ap.add_argument("--paper-root", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--work-root", type=Path)
    args = ap.parse_args(argv)

    cfg = PipelineConfig()
    for name in ("paper_root", "data_root", "work_root"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))

    if args.stage != "report":
        for p in apply_offline_patches(cfg):
            print(f"  {p.path.name}: {'patched' if p.applied else 'already local'}")

    order = ["da3", "pgsr", "mesh", "vlm", "sam3"] if args.stage == "all" else [args.stage]
    for name in order:
        if name not in STAGES:
            continue
        rc = STAGES[name](args.scene, cfg)
        print(f"  {name} exit code {rc}")
        if rc != 0:
            return rc

    print(f"  {stage_report(args.scene, cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
