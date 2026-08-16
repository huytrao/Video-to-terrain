"""Assemble one ScanNet++ scene from the Holi-Spatial release.

Three HuggingFace repos hold the pieces, none of them arranged the way the
paper's scripts expect:

  HoliSpatial-3D-Grounding/images-NNNN.tar   RGB + COLMAP, ~3 scenes per shard
  ScanNetPP/GS_model_w_mesh_pp/N.tar         cameras.json and the released mesh
  ScanNetPP/3d_bounding_box_pp/…tar          ground-truth instance boxes

This module pulls a single scene out of them by byte range and writes the
layout `run_da3.sh` documents:

    <root>/<scene_id>/dslr/resized_undistorted_images/*.JPG
    <root>/<scene_id>/dslr/colmap/{cameras,images,points3D}.txt

Shards are ordered by scene id, so the shard holding a scene is found by
binary search over first-member names — about ten range requests instead of
walking a thousand archives.
"""

from __future__ import annotations

import json
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from v3d.data.hf_tar import RemoteTar, TarMember, hf_url

IMAGES_REPO = "Holi-Spatial/HoliSpatial-3D-Grounding"
SCANNETPP_REPO = "Holi-Spatial/ScanNetPP"
BBOX_MEMBER_TAR = "3d_bounding_box_pp/3d_bounding_box.tar"

IMAGE_SHARDS = 1003


@dataclass
class SceneAssets:
    scene_id: str
    root: Path
    image_count: int
    colmap_files: list[str]
    gt_boxes: Path | None
    cameras_json: Path | None


def _shard_url(index: int) -> str:
    return hf_url(IMAGES_REPO, f"images-{index:04d}.tar")


def _first_scene_id(shard: int) -> str | None:
    """Scene id of the first real member in a shard, without walking it."""
    tar = RemoteTar(_shard_url(shard))
    try:
        for m in tar.members(stop_after=8):
            parts = m.name.split("/")
            # images/scannetpp/<scene_id>/...
            if len(parts) > 2 and parts[0] == "images" and not parts[2].startswith("."):
                return parts[2]
    except urllib.error.HTTPError:
        return None
    return None


def find_image_shard(scene_id: str, shards: int = IMAGE_SHARDS) -> int | None:
    """Binary search the ordered shards for the one containing ``scene_id``."""
    lo, hi = 0, shards - 1
    best: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        first = _first_scene_id(mid)
        if first is None:
            hi = mid - 1
            continue
        if first <= scene_id:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        return None

    # The scene starts in `best` unless it is the very first entry of best+1.
    for candidate in (best, best + 1):
        if candidate >= shards:
            continue
        tar = RemoteTar(_shard_url(candidate))
        for m in tar.members():
            if f"/{scene_id}/" in m.name:
                return candidate
    return None


def fetch_scene(
    scene_id: str,
    root: str | Path,
    shard: int | None = None,
    limit_images: int | None = None,
) -> SceneAssets:
    """Write ``scene_id`` into the layout the paper's scripts expect."""
    root = Path(root)
    dest = root / scene_id
    dest.mkdir(parents=True, exist_ok=True)

    shard = shard if shard is not None else find_image_shard(scene_id)
    if shard is None:
        raise LookupError(f"scene {scene_id!r} not found in any image shard")

    tar = RemoteTar(_shard_url(shard))
    members = [m for m in tar.members() if f"/{scene_id}/" in m.name]
    if not members:
        raise LookupError(f"scene {scene_id!r} absent from shard {shard}")

    images = [m for m in members if m.name.endswith(".JPG")]
    others = [m for m in members if not m.name.endswith(".JPG")]
    if limit_images is not None:
        images = images[:limit_images]

    colmap: list[str] = []
    for m in others:
        rel = m.name.split(f"/{scene_id}/", 1)[1]
        tar.extract(m, dest / rel)
        colmap.append(rel)

    for m in images:
        rel = m.name.split(f"/{scene_id}/", 1)[1]
        tar.extract(m, dest / rel)

    return SceneAssets(
        scene_id=scene_id,
        root=dest,
        image_count=len(images),
        colmap_files=sorted(colmap),
        gt_boxes=None,
        cameras_json=None,
    )


def fetch_gt_boxes(scene_id: str, dest: str | Path) -> Path:
    """The released instance annotation for one scene."""
    dest = Path(dest)
    tar = RemoteTar(hf_url(SCANNETPP_REPO, BBOX_MEMBER_TAR))
    for m in tar.members():
        if Path(m.name).stem == scene_id:
            dest.parent.mkdir(parents=True, exist_ok=True)
            return tar.extract(m, dest)
    raise LookupError(f"no ground-truth boxes for scene {scene_id!r}")


def cache_index(url: str, cache_path: str | Path) -> list[TarMember]:
    """Walk a tar once and keep the member table on disk."""
    return RemoteTar(url).index(cache=cache_path)


def load_cameras(path: str | Path) -> list[dict]:
    """3DGS cameras.json: img_name, width, height, fx, fy, position, rotation."""
    return json.loads(Path(path).read_text())
