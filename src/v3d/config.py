"""Where things live, and what a scene looks like.

Two abstractions, both aimed at the same goal: nothing in the stage code should
know that today's input happens to be ScanNet++ on a Kaggle kernel.

``PipelineConfig`` holds the machine-specific paths. Every value has a default
that suits the offline Blackwell kernel and an environment variable that
overrides it, so the same code runs on a laptop, a rented box, or a different
kernel without edits.

``SceneLayout`` describes where one scene's posed images live. ScanNet++ ships
a nerfstudio transforms file next to an image directory; a scene reconstructed
from an arbitrary video will produce the same two things from COLMAP or from
DA3 poses. The stages consume the layout, never the dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@dataclass
class PipelineConfig:
    """Machine-specific locations. Override any of them through the environment."""

    paper_root: Path = field(default_factory=lambda: _env_path("V3D_PAPER_ROOT", "/tmp/paper/holi-spatial"))
    data_root: Path = field(default_factory=lambda: _env_path("V3D_DATA_ROOT", "/tmp/data/scannetppv2/data"))
    work_root: Path = field(default_factory=lambda: _env_path("V3D_WORK_ROOT", "/tmp/run"))
    da3_model: Path = field(default_factory=lambda: _env_path("V3D_DA3_MODEL", "/tmp/DA3NESTED-GIANT-LARGE"))
    vlm_model: Path = field(default_factory=lambda: _env_path("V3D_VLM_MODEL", "/tmp/qwen3vl8b"))
    sam3_checkpoint: Path = field(default_factory=lambda: _env_path("V3D_SAM3", "/kaggle/working/sam3"))

    # Packages unpacked rather than installed; handed to stage subprocesses,
    # which inherit no sys.path edits made inside a notebook kernel.
    import_paths: tuple[Path, ...] = field(
        default_factory=lambda: tuple(
            Path(p) for p in os.environ.get(
                "V3D_IMPORT_PATHS",
                "/kaggle/working:/kaggle/working/depth_anything_3",
            ).split(os.pathsep) if p
        )
    )

    @property
    def gs_root(self) -> Path:
        return self.work_root / "gs"

    @property
    def log_root(self) -> Path:
        return self.work_root / "logs"

    def subprocess_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        paths = [str(p) for p in self.import_paths if p.exists()]
        existing = os.environ.get("PYTHONPATH", "")
        if existing:
            paths.append(existing)
        return dict(os.environ, PYTHONPATH=os.pathsep.join(paths), **(extra or {}))


@dataclass
class SceneLayout:
    """One scene's posed images, however they were produced.

    ``source_dir`` is what PGSR is pointed at — the directory holding the
    transforms file. ``images_dir`` holds the frames. ``scene_dir`` is where
    per-scene artefacts (DA3 depth, point cloud, masks) are written.

    A scene derived from an arbitrary video satisfies this by writing a
    nerfstudio-style transforms file next to its extracted frames; no stage
    needs to know which route produced it.
    """

    scene_id: str
    scene_dir: Path
    source_dir: Path
    images_dir: Path
    transforms: Path

    @property
    def point_cloud(self) -> Path:
        return self.scene_dir / "pointcloud_da3.ply"

    @property
    def depth_dir(self) -> Path:
        return self.scene_dir / "depth_da3"

    @property
    def mask_dir(self) -> Path:
        return self.scene_dir / "mask"

    def missing(self) -> list[str]:
        problems = []
        if not self.images_dir.is_dir():
            problems.append(f"no image directory at {self.images_dir}")
        elif not any(self.images_dir.iterdir()):
            problems.append(f"image directory {self.images_dir} is empty")
        if not self.transforms.exists():
            problems.append(f"no camera transforms at {self.transforms}")
        return problems


def scannetpp_layout(scene_id: str, data_root: Path) -> SceneLayout:
    """The layout ScanNet++ ships, as run_da3.sh and 3dgs_train.sh expect it."""
    scene_dir = data_root / scene_id
    source = scene_dir / "dslr" / "nerfstudio"
    return SceneLayout(
        scene_id=scene_id,
        scene_dir=scene_dir,
        source_dir=source,
        images_dir=scene_dir / "dslr" / "resized_undistorted_images",
        transforms=source / "transforms_undistorted.json",
    )


def flat_layout(scene_id: str, data_root: Path) -> SceneLayout:
    """A scene reconstructed from an arbitrary video.

    Frames and a nerfstudio-style transforms file sit directly in the scene
    directory. This is the shape the video ingest stage should produce, so that
    everything downstream stays identical to the benchmark path.
    """
    scene_dir = data_root / scene_id
    return SceneLayout(
        scene_id=scene_id,
        scene_dir=scene_dir,
        source_dir=scene_dir,
        images_dir=scene_dir / "images",
        transforms=scene_dir / "transforms_undistorted.json",
    )


LAYOUTS = {"scannetpp": scannetpp_layout, "flat": flat_layout}


def detect_layout(scene_id: str, data_root: Path) -> SceneLayout:
    """Pick the layout that actually exists on disk, preferring ScanNet++."""
    for name in ("scannetpp", "flat"):
        layout = LAYOUTS[name](scene_id, data_root)
        if layout.transforms.exists():
            return layout
    # Nothing matched; return the ScanNet++ shape so the error names a real path.
    return scannetpp_layout(scene_id, data_root)
