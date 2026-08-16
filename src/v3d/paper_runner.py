"""Run the Holi-Spatial pipeline unmodified, on a kernel with no network.

The paper's code is 4,300 lines that went through review; reimplementing it
would trade a known-good method for fresh bugs. So it runs as published, and
this module supplies only what an offline machine cannot: local paths in place
of HuggingFace repo ids, and the stage ordering.

Almost nothing needs patching. `sam3.py` and the lifting script already take
their checkpoints and roots as command-line flags. Exactly one file hardcodes
a repo id, and `apply_offline_patches` rewrites that single line.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Where things live on the offline Blackwell kernel.
PAPER_ROOT = Path("/tmp/paper/holi-spatial")
DATA_ROOT = Path("/tmp/data/scannetppv2/data")
DA3_MODEL = Path("/tmp/DA3NESTED-GIANT-LARGE")
SAM3_CHECKPOINT = Path("/kaggle/working/sam3")
WORK_ROOT = Path("/tmp/run")

# The paper's scripts import depth_anything_3 and the PGSR packages, which are
# unpacked rather than installed. A subprocess inherits none of the sys.path
# edits made inside a notebook kernel, so they have to be handed over
# explicitly — otherwise the stage dies on ModuleNotFoundError two seconds in.
IMPORT_PATHS = (Path("/kaggle/working"), Path("/kaggle/working/depth_anything_3"))

# The literal in inference_da3_scannetppv2.py that reaches for the network.
DA3_REPO_ID = "depth-anything/DA3NESTED-GIANT-LARGE"

# PGSR's ScanNet++ reader falls back to an absolute path from the authors'
# machine when no --ply_path is given. render.py has no such flag, so this one
# must be fixed at the source. The repository already does the right thing in
# readScannetV2SceneInfo — derive the cloud from the scene — and this makes the
# ScanNet++ reader agree with it. `path` there is <scene>/dslr/nerfstudio, so
# the DA3 cloud sits two levels up.
STALE_PLY_FALLBACK = '    ply_path = "output_scannet/scannetppv2/0b031f3119/input.ply"'
DERIVED_PLY_FALLBACK = (
    '    ply_path = os.path.join(path, "..", "..", "pointcloud_da3.ply")  # v3d: was an absolute path from the authors\' machine'
)


@dataclass
class Patch:
    path: Path
    before: str
    after: str
    applied: bool


def apply_offline_patches(paper_root: Path = PAPER_ROOT, da3_model: Path = DA3_MODEL) -> list[Patch]:
    """Replace the two paths that only existed on the authors' machine. Idempotent."""
    patches: list[Patch] = []

    def swap(rel: str, before: str, after: str) -> None:
        target = paper_root / rel
        text = target.read_text()
        if before in text:
            target.write_text(text.replace(before, after))
            patches.append(Patch(target, before, after, applied=True))
        else:
            patches.append(Patch(target, before, after, applied=False))

    swap("inference_da3_scannetppv2.py", f'"{DA3_REPO_ID}"', f'"{da3_model}"')
    swap("PGSR/scene/dataset_readers.py", STALE_PLY_FALLBACK, DERIVED_PLY_FALLBACK)
    return patches


def stage_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a stage subprocess, with the unpacked packages on the path."""
    import os

    paths = [str(p) for p in IMPORT_PATHS if p.exists()]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    return dict(os.environ, PYTHONPATH=os.pathsep.join(paths), **(extra or {}))


def _run(cmd: list[str], cwd: Path, log: Path | None = None, env: dict | None = None) -> int:
    """Run a stage, streaming to a log so a long GPU job can be watched."""
    log = log or (WORK_ROOT / "logs" / (cmd[1].replace("/", "_") + ".log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run(
            cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, env=env or stage_env()
        )
    return proc.returncode


def stage_da3(scene_id: str, paper_root: Path = PAPER_ROOT, data_root: Path = DATA_ROOT) -> int:
    """DA3 depth and point cloud. Writes depth_da3/*.npy and pointcloud_da3.ply."""
    return _run(
        [sys.executable, "inference_da3_scannetppv2.py",
         "--scannetppv2_dir", str(data_root / scene_id)],
        cwd=paper_root,
        log=WORK_ROOT / "logs" / f"{scene_id}_da3.log",
    )


def stage_pgsr(
    scene_id: str,
    paper_root: Path = PAPER_ROOT,
    data_root: Path = DATA_ROOT,
    output_root: Path = WORK_ROOT / "gs",
) -> int:
    """PGSR / 3DGS training — the stage that produces surface quality.

    Arguments mirror 3dgs_train.sh's ScanNet++ branch exactly: the nerfstudio
    directory as the source, the undistorted images alongside it, resolution 1.
    """
    scene_dir = data_root / scene_id
    source = scene_dir / "dslr" / "nerfstudio"
    images = scene_dir / "dslr" / "resized_undistorted_images"
    if not (source / "transforms_undistorted.json").exists():
        raise FileNotFoundError(f"no transforms_undistorted.json under {source}")

    # 3dgs_train.sh omits --ply_path, and dataset_readers.readNerfSyntheticInfo
    # then falls back to a path left over from the authors' machine
    # ("output_scannet/scannetppv2/0b031f3119/input.ply"), which does not exist
    # and takes the run down. The paper initialises 3DGS from the DA3 cloud, so
    # point it there — that is what the fallback was standing in for.
    init_cloud = da3_outputs(scene_id, data_root)["point_cloud"]
    if not init_cloud.exists():
        raise FileNotFoundError(f"run stage_da3 first: {init_cloud} is missing")

    output_root.mkdir(parents=True, exist_ok=True)
    return _run(
        [sys.executable, "PGSR/train.py",
         "-s", str(source),
         "-i", str(images),
         "-m", str(output_root / scene_id),
         "--ply_path", str(init_cloud),
         "--resolution", "1"],
        cwd=paper_root,
        log=WORK_ROOT / "logs" / f"{scene_id}_pgsr.log",
    )


# mesh.sh defaults, kept identical so the surface matches the paper's.
RENDER_MAX_DEPTH = "4.0"
MASK_MAX_DEPTH = "6.0"
VOXEL_SIZE = "0.006"
NUM_CLUSTER = "1"
MESH_PATH_REL = "mesh/tsdf_fusion_post.ply"


def stage_mesh(
    scene_id: str,
    paper_root: Path = PAPER_ROOT,
    data_root: Path = DATA_ROOT,
    output_root: Path = WORK_ROOT / "gs",
) -> int:
    """TSDF-fuse the trained gaussians into a mesh, then render mesh-guided masks.

    Two calls, matching mesh.sh: render.py writes mesh/tsdf_fusion_post.ply,
    and mesh2mask.py turns it into the per-frame masks the lifting stage
    consumes.
    """
    scene_dir = data_root / scene_id
    source = scene_dir / "dslr" / "nerfstudio"
    images = scene_dir / "dslr" / "resized_undistorted_images"
    model_dir = output_root / scene_id

    if not (model_dir / "point_cloud").is_dir():
        raise FileNotFoundError(f"run stage_pgsr first: no gaussians under {model_dir}")

    rc = _run(
        [sys.executable, "PGSR/render.py",
         "-s", str(source), "-m", str(model_dir), "-i", str(images),
         "--skip_test",
         "--max_depth", RENDER_MAX_DEPTH,
         "--voxel_size", VOXEL_SIZE,
         "--num_cluster", NUM_CLUSTER],
        cwd=paper_root,
        log=WORK_ROOT / "logs" / f"{scene_id}_render.log",
    )
    if rc != 0:
        return rc

    mesh = model_dir / MESH_PATH_REL
    if not mesh.exists():
        raise FileNotFoundError(f"render.py reported success but {mesh} is absent")

    return _run(
        [sys.executable, "PGSR/mesh2mask.py",
         "-m", str(model_dir), "-s", str(source), "-i", str(images),
         "--mesh_path", MESH_PATH_REL,
         "--max_depth", MASK_MAX_DEPTH,
         "--voxel_size", VOXEL_SIZE],
        cwd=paper_root,
        log=WORK_ROOT / "logs" / f"{scene_id}_mesh2mask.log",
    )


def da3_outputs(scene_id: str, data_root: Path = DATA_ROOT) -> dict[str, Path]:
    scene = data_root / scene_id
    return {
        "depth_dir": scene / "depth_da3",
        "point_cloud": scene / "pointcloud_da3.ply",
    }


def stage_report(scene_id: str, data_root: Path = DATA_ROOT) -> dict[str, object]:
    """What exists so far, so a stage can be resumed rather than repeated."""
    out = da3_outputs(scene_id, data_root)
    depth_dir = out["depth_dir"]
    return {
        "scene": scene_id,
        "depth_maps": len(list(depth_dir.glob("*.npy"))) if depth_dir.is_dir() else 0,
        "point_cloud_mb": round(out["point_cloud"].stat().st_size / 1e6, 1)
        if out["point_cloud"].exists() else None,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene")
    ap.add_argument("--stage", default="da3", choices=["patch", "da3", "pgsr", "mesh", "report"])
    ap.add_argument("--paper-root", type=Path, default=PAPER_ROOT)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--da3-model", type=Path, default=DA3_MODEL)
    args = ap.parse_args(argv)

    if args.stage in ("patch", "da3"):
        for p in apply_offline_patches(args.paper_root, args.da3_model):
            state = "patched" if p.applied else "already local"
            print(f"  {p.path.name}: {state} ({p.before} -> {p.after})")

    if args.stage == "da3":
        rc = stage_da3(args.scene, args.paper_root, args.data_root)
        print(f"  DA3 exit code {rc}")
        print(f"  {stage_report(args.scene, args.data_root)}")
        return rc

    if args.stage == "pgsr":
        rc = stage_pgsr(args.scene, args.paper_root, args.data_root)
        print(f"  PGSR exit code {rc}")
        return rc

    if args.stage == "mesh":
        rc = stage_mesh(args.scene, args.paper_root, args.data_root)
        print(f"  mesh exit code {rc}")
        return rc

    if args.stage == "report":
        print(stage_report(args.scene, args.data_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
