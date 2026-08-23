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

import os
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

# `readCamerasFromTransforms` composites every frame over the background and then
# hands the result to PIL as `np.byte` — which is *signed* int8, so 0-255 pixel
# values overflow into negatives and PIL refuses the buffer outright:
#
#     TypeError: Cannot handle this data type: (1, 1, 3), |i1
#
# It is unconditional in the "Blender" reader, which is the branch a
# `transforms_undistorted.json` scene takes (`scene/__init__.py:57`), so any run
# on this layout dies before iteration 1. `np.uint8` is what PIL's "RGB" mode
# actually wants, and the surrounding arithmetic already clamps to [0, 255].
# `Mesh2DepthHelper` asks nvdiffrast for a **GL** rasteriser, which compiles
# `glutil.h` and so needs `EGL/egl.h`. The Kaggle image has no EGL headers and no
# internet to apt-get them, so the plugin build dies with
# `fatal error: EGL/egl.h: No such file or directory` and mesh2mask never runs --
# taking the whole mesh stage with it (rc=1), after TSDF fusion has already
# succeeded.
#
# nvdiffrast's CUDA rasteriser needs no EGL, no display, and no GL headers, and
# produces the same rasterisation for this use (depth rendering of a mesh from
# known cameras; `output_db=False` here, so the image-space derivatives that are
# the GL context's real speciality are not requested). It builds cleanly on this
# machine -- verified: `RasterizeCudaContext()` returns CUDA_CTX_OK.
#
# `RasterizeCudaContext` takes only `device`, so the GL-only keywords go away
# with the swap.
GL_CONTEXT = """            self.glctx = dr.RasterizeGLContext(
            output_db=False,     # 不需要 image-space 导数就关掉
            mode='automatic',    # 默认即可；也可用 'manual' 自己 set/release
            device=self.device   # 放在同一块 GPU 上
        )"""
CUDA_CONTEXT = """            self.glctx = dr.RasterizeCudaContext(
            device=self.device   # v3d: was RasterizeGLContext, which needs EGL
        )                        # headers this image lacks; CUDA needs none"""

NP_BYTE_OVERFLOW = 'image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")'
NP_UINT8_FIX = (
    'image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")'
    "  # v3d: np.byte is signed int8; 0-255 overflows and PIL rejects it"
)
DERIVED_PLY_FALLBACK = (
    '    ply_path = os.path.join(path, "..", "..", "pointcloud_da3.ply")'
    "  # v3d: was an absolute path from the authors' machine"
)

# SAM3 ships float32 weights — all 1102 parameters, and its config says
# dtype: float32 — but is meant to run under bf16 autocast: Meta's own
# measure_speed.py, qualitative_test.py and bench_compile.py all open
# `torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()` before
# touching the model. Holi-Spatial's sam3.py does not, and something inside the
# vision backbone casts to bf16 regardless, so inference dies on
# "mat1 and mat2 must have the same dtype, but got BFloat16 and Float".
# Entering the same context Meta uses is the fix, and it belongs at import time
# because the script builds and runs the model inside main().
SAM3_IMPORT_ANCHOR = "from sam3.model.sam3_image_processor import Sam3Processor"
SAM3_AUTOCAST = (
    "from sam3.model.sam3_image_processor import Sam3Processor\n"
    "import torch as _torch  # v3d: SAM3 expects a bf16 autocast context, as\n"
    "_torch.autocast(device_type='cuda', dtype=_torch.bfloat16).__enter__()  # Meta's own scripts do"
)

# Enabling that autocast makes the model's outputs bf16, and numpy has no bf16
# dtype, so sam3.py then dies on "Got unsupported ScalarType BFloat16" the
# moment it converts scores. All three conversions in normalize_masks_boxes
# need the same guard. Masks may legitimately be bool, so the cast is applied
# only to floating-point tensors rather than blanket .float().
SAM3_NUMPY_BEFORE = """    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if boxes is not None and isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()"""

SAM3_NUMPY_AFTER = """    def _np(t):  # v3d: bf16 from autocast has no numpy equivalent
        t = t.detach()
        return (t.float() if t.is_floating_point() else t).cpu().numpy()

    if isinstance(masks, torch.Tensor):
        masks = _np(masks)
    if boxes is not None and isinstance(boxes, torch.Tensor):
        boxes = _np(boxes)
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = _np(scores)"""

# mesh.sh defaults. Kept as the *control* — `v3d.geometry.quality` derives what
# actually runs, and every derived value is reported against these so a change
# can be judged rather than trusted. Three of the four destroy geometry on a room
# that is not a ScanNet++ bedroom; see that module's docstring for which and why.
RENDER_MAX_DEPTH = "4.0"
MASK_MAX_DEPTH = "6.0"
VOXEL_SIZE = "0.006"
NUM_CLUSTER = "1"
MESH_PATH_REL = "mesh/tsdf_fusion_post.ply"


# --- the run parameters the published script hardcodes -----------------------
#
# These are patched in as *environment reads*, never as new literals, so one
# checkout serves every run and the DA3 log records what was actually used. The
# defaults reproduce the paper's behaviour exactly, so an unset environment
# changes nothing.
#
# The two that decide output quality, and the evidence for each:
#
# `process_res` — DA3's inference resolution. `api.py:145` defaults it to 504 and
#   `upper_bound_resize` scales the *longest* side to it
#   (`input_processor.py:317-334`), so a 1920x1080 frame is predicted at 504x280
#   — 283.5 from the resize, rounded to the nearest multiple of PATCH_SIZE=14 by
#   `_make_divisible_by_resize` — and then bilinearly upsampled back to
#   1920x1080 by `inference_da3_scannetppv2.py:598-603`. That is 141,120 real
#   depth samples presented as 2,073,600: a 14.7x areal interpolation. Nothing
#   downstream recovers detail the model never predicted, so this is the ceiling
#   on the point cloud, the mesh, the height map and the boxes alike.
#
#   Note which preset this is. `model_inference.py:128` maps
#   `{"high_res": "lower_bound_resize", "low_res": "upper_bound_resize"}` — the
#   default is DA3's own *low* resolution mode. At the same process_res,
#   `lower_bound_resize` scales the shortest side instead and yields 896x504 on
#   this footage, 3.2x the samples.
#
#   The authors left it alone with the note "this can lead to OOM issue"
#   (line 557), which is a memory reason and not an accuracy one. Raising it does
#   not trip the intrinsics assertion two lines later: `api.py:352`
#   (`_align_to_input_extrinsics_intrinsics`) overwrites `prediction.intrinsics`
#   with the *input* matrix before anything compares them, so that difference is
#   exactly zero at any resolution.
#
# `target_num_points` — the size of the delivered cloud. The function's own
#   default is 20 million and its docstring says so (line 469, 476); both call
#   sites pass 4 million (lines 753, 762). The cut is not trimming interpolation:
#   at 600 frames of 141,120 real samples there are ~85 M measured points to draw
#   from, so 4 M keeps under 5% of them. Raising it recovers measurements.
#
# `conf_threshold` — `depth_to_pointcloud` keeps a pixel only where DA3's own
#   confidence reaches this (line 626, and 430 where it is applied). It trades
#   coverage against noise and belongs with the other knobs rather than buried.
#
# `MAX_CHUNK_SIZE` — how many views go through the model at once (line 724).
#   Views attend to each other, so this is the memory dial that has to move when
#   process_res goes up.
V3D_INJECT_ANCHOR = "def downsample_pointcloud(points, colors, target_num_points=20000000):"

V3D_INJECT = '''# --- v3d: run parameters, read from the environment --------------------------
# Injected by v3d.paper_runner.apply_offline_patches. Each name below replaces a
# literal this script hardcoded. The defaults are the literals, so an unset
# environment reproduces the paper exactly.
import os as _v3d_os


def _v3d_num(name, default, cast):
    """Read a number from the environment, or refuse — never fall back silently.

    A typo in an environment variable would otherwise run the paper's settings
    and let the log report them as this run's, which is the failure mode this
    whole project keeps paying for: a result that looks like the one requested.
    """
    raw = _v3d_os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        raise SystemExit(
            f"[v3d] {name}={raw!r} is not a valid {cast.__name__}. Refusing to "
            f"fall back to the default {default!r}, because the run would then "
            f"silently use the paper's setting and be reported as yours."
        )


_V3D_PROCESS_RES = _v3d_num("V3D_PROCESS_RES", 504, int)
_V3D_PROCESS_RES_METHOD = _v3d_os.environ.get("V3D_PROCESS_RES_METHOD") or "upper_bound_resize"
_V3D_TARGET_POINTS = _v3d_num("V3D_TARGET_POINTS", 4000000, int)
_V3D_CONF_THRESHOLD = _v3d_num("V3D_CONF_THRESHOLD", 0.3, float)
_V3D_MAX_CHUNK = _v3d_num("V3D_MAX_CHUNK_SIZE", 800, int)

if _V3D_PROCESS_RES_METHOD not in (
    "upper_bound_resize", "lower_bound_resize", "upper_bound_crop", "lower_bound_crop"
):
    raise SystemExit(
        f"[v3d] V3D_PROCESS_RES_METHOD={_V3D_PROCESS_RES_METHOD!r} is not one of the "
        f"four InputProcessor._resize_image accepts; it would raise inside the model "
        f"after the weights have loaded."
    )

print(
    f"[v3d] process_res={_V3D_PROCESS_RES} method={_V3D_PROCESS_RES_METHOD} "
    f"target_num_points={_V3D_TARGET_POINTS} conf_threshold={_V3D_CONF_THRESHOLD} "
    f"max_chunk={_V3D_MAX_CHUNK}",
    flush=True,
)


''' + V3D_INJECT_ANCHOR

# The line the authors commented out, and what replaces it. Passing the method as
# well as the resolution matters: the two together decide the pixel count, and
# leaving the method implicit is how "I raised the resolution" and "the longest
# side is still what got scaled" coexist.
PROCESS_RES_BEFORE = "            # process_res=upper_bound_res,  # this can lead to OOM issue."
PROCESS_RES_AFTER = (
    "            process_res=_V3D_PROCESS_RES,  # v3d: was the commented-out default of 504\n"
    "            process_res_method=_V3D_PROCESS_RES_METHOD,"
)

TARGET_POINTS_BEFORE = "target_num_points=4000000"
TARGET_POINTS_AFTER = "target_num_points=_V3D_TARGET_POINTS"

CONF_THRESHOLD_BEFORE = "conf_threshold=0.3  # 可以根据需要调整置信度阈值"
CONF_THRESHOLD_AFTER = "conf_threshold=_V3D_CONF_THRESHOLD  # v3d: was 0.3"

MAX_CHUNK_BEFORE = "MAX_CHUNK_SIZE=800  # tune this to avoid OOM"
MAX_CHUNK_AFTER = "MAX_CHUNK_SIZE=_V3D_MAX_CHUNK  # v3d: was 800"

# A latent bug in the paper's chunked branch, which its own defaults never reach.
# Chunking only engages at >= MAX_CHUNK_SIZE views, so at 800 it has effectively
# never run. `chunk_inference` returns CUDA tensors (line 465 is a `torch.cat` on
# device), and line 756 then hands a *list of CUDA tensors* to `np.concatenate`,
# which cannot convert them and raises "can't convert cuda:0 device type tensor to
# numpy".
#
# Forcing chunking is exactly what makes native-resolution inference fit in
# memory, so this path becomes live and has to be correct. Landing each chunk on
# the host also releases its GPU memory before the next chunk's forward pass,
# which is the entire point of chunking in the first place.
CHUNK_TO_HOST_BEFORE = """                points.append(point)
                colors.append(color)"""
CHUNK_TO_HOST_AFTER = """                # v3d: np.concatenate below cannot consume CUDA tensors, and moving
                # each chunk to the host frees its VRAM before the next forward pass.
                point = point.detach().cpu().numpy() if hasattr(point, "detach") else point
                color = color.detach().cpu().numpy() if hasattr(color, "detach") else color
                points.append(point)
                colors.append(color)"""

# --- one depth scale for the whole scene, instead of one per chunk -----------
#
# `api.py:353-362` fits a Umeyama scale per `model.inference` call and applies
# `prediction.depth /= scale`. The paper's `balanced_chunks` (line 776) hands
# each chunk a *disjoint* slice of an unseeded `torch.randperm`, so every chunk
# carries its own scalar, no two chunks share a frame that could reveal the
# difference, and nothing downstream ever compares them. Both outputs inherit
# it: `depth_da3/*.npy` (line 666), which PGSR reads as training supervision
# through `scene/cameras.py:89`, and the point cloud (line 671).
#
# Measured on scannetpp_00777c41d4, two 200-view chunks at native resolution:
#
#     depth ratio on 66 shared frames   median 1.00894, range 0.99959..1.01707
#     after removing one global scalar  worst 0.93%, p95 0.74%, rms 0.35%
#
# So 86% of the disagreement is a single number per chunk. That part is exactly
# recoverable; the rest is the model predicting a frame differently depending on
# which frames it saw it with, and no rescaling touches it. At 0.89% uncorrected
# this is roughly 4 cm across a room -- on its own the largest error in the
# pipeline, and invisible, because a wrongly-sized region still looks like a room.
#
# The correction needs frames that appear in more than one chunk, which the
# paper's chunker never produces. Every chunk therefore carries the same anchor
# set, and each is measured directly against the first chunk rather than chained
# to its neighbour, so five joins do not accumulate five errors. Anchors are
# spread evenly over frame order because a scale fitted in one corner of a room
# does not describe the rest of it. The cost is the anchors being inferred once
# per chunk: about 11% more compute at 24 anchors in 200-view chunks.
#
# Replacing `randperm` with a deterministic stride also removes an unseeded
# random number from the middle of the pipeline, which made a run impossible to
# reproduce exactly.
ANCHOR_BEFORE = "def chunk_inference(image_files, image_names, intrinsics, extrinsics):"

ANCHOR_AFTER = '''# --- v3d: keep every chunk on one depth scale --------------------------------
_V3D_ANCHOR_N = _v3d_num("V3D_ANCHOR_FRAMES", 24, int)
_v3d_anchor_set = set()      # frame names present in every chunk
_v3d_anchor_ref = {}         # frame name -> chunk-0 depth: the scale everything joins


def _v3d_anchored_chunks(n_frames, max_chunk, n_anchor):
    """Chunks that share an anchor set, so their per-call scales are comparable.

    Returns `(anchor_idx, [owned_idx, ...])`. Each chunk is inferred on its own
    frames *plus* the anchors, but only writes and unprojects the frames it owns,
    or the anchors would contribute duplicate points and a depth map written
    several times.

    Owned frames are taken with a stride rather than a contiguous slice so each
    chunk still spans the whole capture -- the property the paper wanted from
    `randperm`, without the unseeded randomness.
    """
    import math

    import numpy as _np

    if n_frames <= max_chunk:
        return [], [list(range(n_frames))]
    n_anchor = max(8, min(n_anchor, max_chunk // 4))
    anchor = sorted(set(_np.linspace(0, n_frames - 1, n_anchor).round().astype(int).tolist()))
    a_set = set(anchor)
    rest = [i for i in range(n_frames) if i not in a_set]
    room = max_chunk - len(anchor)
    if room < 1:
        raise SystemExit(
            f"[v3d] V3D_ANCHOR_FRAMES={n_anchor} leaves no room in a {max_chunk}-view chunk"
        )
    n_chunks = math.ceil(len(rest) / room)
    return anchor, [rest[k::n_chunks] for k in range(n_chunks)]


def _v3d_anchor_rescale(depth, conf, image_names):
    """Put this chunk on the first chunk's depth scale. Returns `(depth, scale)`.

    The first chunk defines the scale and is returned untouched. Every later
    chunk is divided by the median, over the anchor frames, of its own depth
    against the reference -- a ratio because a scale error is what corrupts a
    ratio, and a median over confident pixels because DA3's low-confidence
    regions carry noise that would swamp a mean.
    """
    import numpy as _np

    if not _v3d_anchor_set:
        return depth, 1.0                      # single-chunk run: nothing to join
    if not _v3d_anchor_ref:
        for i, name in enumerate(image_names):
            if name in _v3d_anchor_set:
                _v3d_anchor_ref[name] = depth[i].copy()
        print(f"[v3d] scale reference set from {len(_v3d_anchor_ref)} anchor frames",
              flush=True)
        return depth, 1.0

    ratios = []
    for i, name in enumerate(image_names):
        ref = _v3d_anchor_ref.get(name)
        if ref is None:
            continue
        cur = depth[i]
        ok = (_np.isfinite(cur) & _np.isfinite(ref) & (cur > 0) & (ref > 0)
              & (conf[i] > 0.5))
        if ok.sum() >= 1000:
            ratios.append(float(_np.median(cur[ok] / ref[ok])))
    if len(ratios) < 3:
        raise SystemExit(
            "[v3d] fewer than 3 anchor frames had enough confident pixels to measure "
            "this chunk's scale. Continuing would write depth on an unknown scale, "
            "which is the failure this patch exists to prevent."
        )
    arr = _np.array(ratios)
    scale = float(_np.median(arr))
    resid = float(_np.abs(arr / scale - 1.0).max() * 100)
    print(f"[v3d] chunk scale {scale:.5f} ({(scale - 1) * 100:+.2f}%) from "
          f"{len(ratios)} anchors; residual after correction {resid:.2f}%", flush=True)
    # Recorded, not just printed: the residual is the one number that says how
    # much scale inconsistency the depth still carries, and the manifest has to
    # be able to state it rather than have it live in a log nobody kept.
    import json as _json
    _rec = os.path.join(dl3dv_dir, "da3_chunk_scales.json")
    _hist = _json.load(open(_rec)) if os.path.exists(_rec) else []
    _hist.append({"chunk": len(_hist) + 1, "scale": scale,
                  "offset_pct": (scale - 1) * 100, "residual_pct": resid,
                  "anchors_used": len(ratios)})
    with open(_rec, "w") as _fh:
        _json.dump(_hist, _fh, indent=2)
    return depth / scale, scale


def chunk_inference(image_files, image_names, intrinsics, extrinsics, owned=None):'''

# `owned` is None on the single-chunk path, which leaves that path behaving
# exactly as the paper's does.
SAVE_BEFORE = """    # 保存深度图到 scannetv2/scans/scene0000_00/depth_da3
    depth_save_dir = os.path.join(dl3dv_dir, 'depth_da3')
    os.makedirs(depth_save_dir, exist_ok=True)
    for id, d in tqdm(enumerate(prediction.depth, start=0)):
        np.save(f'{depth_save_dir}/{image_names[id]}.npy', d)
    print(f'成功保存深度图到路径{depth_save_dir}')"""

SAVE_AFTER = """    # v3d: join this chunk to the scene's scale BEFORE anything reads the depth.
    # Below it is written to disk and unprojected; both would otherwise bake in
    # this call's own Umeyama scalar.
    prediction.depth, _v3d_scale = _v3d_anchor_rescale(
        prediction.depth, prediction.conf, image_names)
    _v3d_keep = [i for i, n in enumerate(image_names) if owned is None or n in owned]
    _v3d_all = len(_v3d_keep) == len(image_names)

    def _v3d_sel(a):
        # Fancy indexing copies, and processed_images is ~1.5 GB per chunk, so a
        # chunk that owns all of its frames passes the arrays through untouched.
        return a if _v3d_all else a[_v3d_keep]

    # 保存深度图到 scannetv2/scans/scene0000_00/depth_da3
    depth_save_dir = os.path.join(dl3dv_dir, 'depth_da3')
    os.makedirs(depth_save_dir, exist_ok=True)
    for id in tqdm(_v3d_keep):
        np.save(f'{depth_save_dir}/{image_names[id]}.npy', prediction.depth[id])
    print(f'成功保存深度图到路径{depth_save_dir} '
          f'({len(_v3d_keep)}/{len(image_names)} owned, scale {_v3d_scale:.5f})')"""

UNPROJECT_BEFORE = """    points, colors = depth_to_pointcloud(
        prediction.depth,
        intrinsics,
        prediction_extrinsics,
        prediction.processed_images,
        conf=prediction.conf,"""

UNPROJECT_AFTER = """    # v3d: anchors ride along in every chunk to measure scale, but only the chunk
    # that owns them contributes their points -- otherwise they are duplicated
    # once per chunk and over-weighted in the cloud.
    points, colors = depth_to_pointcloud(
        _v3d_sel(prediction.depth),
        _v3d_sel(intrinsics),
        _v3d_sel(prediction_extrinsics),
        _v3d_sel(prediction.processed_images),
        conf=_v3d_sel(prediction.conf),"""

CHUNK_LOOP_BEFORE = """            rand_idx = torch.randperm(len(image_files))
            points = []
            colors = []
            ranges = balanced_chunks(len(image_files), max_chunk=MAX_CHUNK_SIZE)
            for start_idx, end_idx in ranges:
                indices = rand_idx[start_idx : end_idx].to(dtype=torch.long, device=device).tolist()
                point, color = chunk_inference([image_files[i] for i in indices], [image_names[i] for i in indices], intrinsics[indices], extrinsics[indices])"""

CHUNK_LOOP_AFTER = """            points = []
            colors = []
            # v3d: was `balanced_chunks` over an unseeded randperm, which gave the
            # chunks no shared frame and so no way to detect -- or correct -- the
            # per-call depth scale each one was assigned.
            _anchor, _owned = _v3d_anchored_chunks(
                len(image_files), MAX_CHUNK_SIZE, _V3D_ANCHOR_N)
            _v3d_anchor_set.update(image_names[i] for i in _anchor)
            print(f"[v3d] {len(_owned)} chunks of <= {MAX_CHUNK_SIZE} views, "
                  f"sharing {len(_anchor)} anchor frames", flush=True)
            for _ci, _own in enumerate(_owned):
                indices = _anchor + _own
                own_names = {image_names[i] for i in _own}
                if _ci == 0:
                    own_names |= _v3d_anchor_set   # chunk 0 owns the anchors
                point, color = chunk_inference([image_files[i] for i in indices], [image_names[i] for i in indices], intrinsics[indices], extrinsics[indices], owned=own_names)"""


# The paper's DA3 writer and its PGSR reader disagree about where depth lives, so
# the depth supervision term silently never fires.
#
# `inference_da3_scannetppv2.py:612` writes depth to `<scene_dir>/depth_da3/`.
# `PGSR/scene/cameras.py:70-90` looks for it under a *sibling* tree,
# `<parent-of-scannetppv2>/DptV3/data/<scene>/depth_da3/`, which only existed on
# the authors' machine. It also finds the wrong `scene`: `parts.index("data")`
# returns the first "data" in the path, so for `/tmp/data/scannetppv2/data/<scene>/…`
# the scene becomes "scannetppv2".
#
# The consequence is not a crash. `cameras.py:163-164` prints
# "[Warning] depth not found" per camera and leaves `depth_map = None`, and
# `train.py:226-236` then skips its L1 depth term — so training runs to completion,
# reports a falling loss, and never uses the depth this pipeline spent its
# resolution budget on. Six hundred warnings scroll past during camera loading and
# nothing downstream reports that a supervision term was disabled.
#
# The fix derives the path the way the writer builds it: the scene directory is the
# parent of `dslr`, and depth sits directly inside it.
DEPTH_PATH_BEFORE = """    try:
        idx_data = parts.index("data")
        scene = parts[idx_data + 1]
    except Exception:
        return None
    try:
        idx_scan = parts.index("scannetppv2")
        scannet_root = Path(*parts[: idx_scan + 1])  # .../scannetppv2
    except Exception:
        return None
    depth_root = scannet_root.parent / "DptV3" / "data"
    depth_path = depth_root / scene / "depth_da3" / f"{p.stem}.npy"
    return str(depth_path)"""
DEPTH_PATH_AFTER = """    # v3d: derive the path the way inference_da3_scannetppv2.py:612 writes it --
    # depth lives in the scene directory, which is the parent of `dslr`. The
    # original searched for a sibling DptV3 tree that only existed on the authors'
    # machine, and took the first "data" in the path as the dataset root, so a root
    # like /tmp/data/scannetppv2/data/<scene> resolved the scene to "scannetppv2".
    # It failed by printing a warning per camera and disabling the depth loss.
    if "dslr" in parts:
        scene_dir = Path(*parts[: parts.index("dslr")])
        return str(scene_dir / "depth_da3" / f"{p.stem}.npy")
    return None"""


@dataclass
class Patch:
    path: Path
    before: str
    after: str
    status: str          # "patched" | "already applied"
    occurrences: int = 1

    @property
    def applied(self) -> bool:
        return self.status == "patched"

    def describe(self) -> str:
        n = f" x{self.occurrences}" if self.occurrences != 1 else ""
        return f"{self.path.name}: {self.status}{n}"


class PatchError(RuntimeError):
    """A patch matched neither its target nor its own output."""


def apply_offline_patches(cfg: PipelineConfig | None = None) -> list[Patch]:
    """Rewrite the published code for this machine and this run. Idempotent.

    Two kinds of substitution: paths that only existed on the authors' machine,
    and hardcoded run parameters that become environment reads.

    The important behaviour is the failure mode. The previous version appended
    ``applied=False`` both when a patch was already in place and when its anchor
    text could not be found at all, so an upstream edit of one character would
    silently stop a patch applying and report the same word for it. Every patch
    here is verified against its *own output* instead: absent both ways is a
    ``PatchError``, before the weights load and before a GPU hour is spent
    producing a result that would look exactly like the one asked for.
    """
    cfg = cfg or PipelineConfig()
    patches: list[Patch] = []

    def swap(rel: str, before: str, after: str, expect: int = 1) -> None:
        target = cfg.paper_root / rel
        if not target.exists():
            raise PatchError(f"no {target} to patch; is V3D_PAPER_ROOT right?")
        text = target.read_text()
        found = text.count(before)
        if found:
            if found != expect:
                raise PatchError(
                    f"{target}: expected {expect} occurrence(s) of {before[:60]!r}, "
                    f"found {found}. Upstream has changed shape; check the patch by "
                    f"hand rather than letting it apply somewhere it was not meant to."
                )
            target.write_text(text.replace(before, after))
            patches.append(Patch(target, before, after, "patched", found))
            return
        already = text.count(after)
        if already:
            patches.append(Patch(target, before, after, "already applied", already))
            return
        raise PatchError(
            f"{target}: neither {before[:60]!r} nor its replacement is present. "
            f"The patch cannot apply and the stage would run unpatched — which for "
            f"the point-cloud and resolution patches means silently running the "
            f"paper's settings and reporting them as this run's."
        )

    inference = "inference_da3_scannetppv2.py"
    # Paths that existed only on the authors' machine.
    swap(inference, f'"{DA3_REPO_ID}"', f'"{cfg.da3_model}"')
    swap("PGSR/scene/dataset_readers.py", STALE_PLY_FALLBACK, DERIVED_PLY_FALLBACK)
    swap("PGSR/scene/dataset_readers.py", NP_BYTE_OVERFLOW, NP_UINT8_FIX)
    swap("PGSR/Mesh2DepthHelper/Mesh2DepthHelper.py", GL_CONTEXT, CUDA_CONTEXT)
    swap("sam3.py", SAM3_IMPORT_ANCHOR, SAM3_AUTOCAST)
    swap("sam3.py", SAM3_NUMPY_BEFORE, SAM3_NUMPY_AFTER)
    # Run parameters. The injection has to land before anything reads the names,
    # and `downsample_pointcloud` at line 469 is above both the model call and
    # `chunk_inference`, so module-level execution defines them first.
    swap(inference, V3D_INJECT_ANCHOR, V3D_INJECT)
    swap(inference, PROCESS_RES_BEFORE, PROCESS_RES_AFTER)
    swap(inference, TARGET_POINTS_BEFORE, TARGET_POINTS_AFTER, expect=2)
    swap(inference, CONF_THRESHOLD_BEFORE, CONF_THRESHOLD_AFTER)
    swap(inference, MAX_CHUNK_BEFORE, MAX_CHUNK_AFTER)
    swap(inference, CHUNK_TO_HOST_BEFORE, CHUNK_TO_HOST_AFTER)
    # These four must follow the swaps above: they overlap the same region and
    # match against the text those leave behind.
    swap(inference, ANCHOR_BEFORE, ANCHOR_AFTER)
    swap(inference, SAVE_BEFORE, SAVE_AFTER)
    swap(inference, UNPROJECT_BEFORE, UNPROJECT_AFTER)
    swap(inference, CHUNK_LOOP_BEFORE, CHUNK_LOOP_AFTER)
    swap("PGSR/scene/cameras.py", DEPTH_PATH_BEFORE, DEPTH_PATH_AFTER)
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


# --- knobs derived from the scene, rather than inherited from a bedroom -------


def _read_transforms(layout: SceneLayout) -> tuple["np.ndarray", "np.ndarray", int]:
    """World-to-camera poses and the scene intrinsic, as the pipeline's reader sees them.

    Read back through `v3d.ingest.poses`, using the convention parsed from the
    paper's own reader, so these are the poses the stages will actually compute
    from the file on disk — not a second interpretation of it.
    """
    import json

    import numpy as np

    from v3d.ingest import poses as P

    doc = json.loads(layout.transforms.read_text())
    entries = doc.get("frames", []) + doc.get("test_frames", [])
    if not entries:
        raise ValueError(f"{layout.transforms} lists no frames")

    paper_file = None
    for candidate in (layout.transforms.parent, Path("/tmp/paper/holi-spatial")):
        guess = candidate / "inference_da3_scannetppv2.py"
        if guess.exists():
            paper_file = guess
            break
    conv = P.parse_reader_convention(paper_file) if paper_file else P.DEFAULT_CONVENTION

    w2c = np.stack([P.transform_matrix_to_w2c(np.array(e["transform_matrix"]), conv)
                    for e in entries])
    K = np.array([[doc["fl_x"], 0.0, doc["cx"]],
                  [0.0, doc["fl_y"], doc["cy"]],
                  [0.0, 0.0, 1.0]], dtype=float)
    return w2c, K, len(entries)


def derive_knobs(
    scene: str | SceneLayout,
    cfg: PipelineConfig | None = None,
    policy: "object | None" = None,
    sample_frames: int = 40,
) -> tuple["object", dict]:
    """Measure this scene's depth and camera path; return `(Knobs, pgsr_schedule)`.

    The depth maps are sampled rather than all loaded: at 1920x1080 float32 each
    is 8 MB, so 600 of them is 5 GB to compute a set of percentiles that 40 evenly
    spaced frames already determine. Evenly spaced, not the first 40 — the first
    40 frames of a walk are one corner of the room, and the fusion distance
    derived from them would be the distance to that corner's walls.

    The result is cached beside the run and re-read, because these numbers are
    reported as the justification for the stage arguments and must be the same
    ones the stage actually got.
    """
    import json

    import numpy as np

    from v3d.geometry.quality import KnobPolicy, derive, pgsr_schedule

    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    policy = policy or KnobPolicy()

    depth_files = sorted(layout.depth_dir.glob("*.npy"))
    if not depth_files:
        raise FileNotFoundError(
            f"run stage_da3 first: no depth maps in {layout.depth_dir}. The knobs "
            f"are derived from measured depth, so there is nothing to derive from."
        )
    if len(depth_files) > sample_frames:
        idx = np.linspace(0, len(depth_files) - 1, sample_frames).round().astype(int)
        depth_files = [depth_files[i] for i in sorted(set(idx.tolist()))]

    depth = np.stack([np.load(p) for p in depth_files])
    w2c, K, n_images = _read_transforms(layout)

    knobs = derive(depth, w2c, K, conf=None, policy=policy,
                   scale_mode="assumed")
    schedule = pgsr_schedule(n_images, w2c, policy)

    out = cfg.work_root / "knobs" / f"{layout.scene_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"knobs": {k: v for k, v in vars(knobs).items()}, "schedule": schedule,
         "depth_frames_sampled": len(depth_files), "images": n_images},
        indent=2, default=str,
    ))

    ev = knobs.evidence
    vr = ev["voxel_reasoning"]
    ag = ev["against_the_inherited_constants"]
    print(f"  derived from {len(depth_files)} depth maps and {n_images} poses "
          f"-> {out}")
    print(f"    render_max_depth {knobs.render_max_depth:.3f}  "
          f"(paper 4.0 would erase {ag['fraction_of_valid_depth_erased_at_4.0'] * 100:.1f}% "
          f"of believable depth; derived erases "
          f"{ag['fraction_of_valid_depth_erased_at_derived'] * 100:.1f}%)")
    print(f"    voxel_size       {knobs.voxel_size:.6f}  "
          f"({knobs.voxel_size / 0.006:.2f}x the paper's, bound by {vr['bound_by']}, "
          f"{vr['voxels_across_diagonal']:.0f} voxels across the scene)")
    print(f"    num_cluster      {knobs.num_cluster}  "
          f"use_depth_filter {knobs.use_depth_filter}")
    print(f"    pgsr iterations  {schedule['iterations']:,} "
          f"({schedule['evidence']['views_per_image']:.0f} views/image vs the "
          f"paper's {schedule['evidence']['reference_views_per_image']:.0f})")
    print(f"    view pairing     min {schedule['multi_view_min_dis']:.4f} "
          f"max {schedule['multi_view_max_dis']:.4f} "
          f"(paper 0.01 / 1.5, in units that are not metres)")
    return knobs, schedule


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


def stage_pgsr(
    scene: str | SceneLayout,
    cfg: PipelineConfig | None = None,
    schedule: dict | None = None,
    hold_out_test: bool = False,
) -> int:
    """PGSR / 3DGS training — the stage that produces surface quality.

    Arguments mirror 3dgs_train.sh, with --ply_path supplied so the run does
    not reach the stale fallback described above.

    `schedule` comes from `v3d.geometry.quality.pgsr_schedule` and replaces four
    families of default that carry a hidden assumption about the input. Passing
    none reproduces the paper's settings, which is the right control but not the
    right run: PGSR samples one image per iteration, so its 30,000 iterations are
    115 views of each image at the paper's 261 frames and only 50 at 600. Holding
    the schedule fixed while the frame count grows quietly *reduces* per-image
    supervision, which is the opposite of what more input should buy.

    `multi_view_min_dis` and `multi_view_max_dis` are the other half, and they are
    **lengths in scene units** (`arguments/__init__.py:61-62`). They decide which
    neighbouring views a frame must be geometrically consistent with, and
    multi-view consistency is the mechanism the paper credits for a clean surface —
    so a maximum smaller than the scene's own baselines silently switches the
    regulariser off, and one larger than the overlap enforces consistency between
    views that see different things. Neither failure raises.
    """
    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    missing = layout.missing()
    if missing:
        raise FileNotFoundError("; ".join(missing))
    if not layout.point_cloud.exists():
        raise FileNotFoundError(f"run stage_da3 first: {layout.point_cloud} is missing")

    cfg.gs_root.mkdir(parents=True, exist_ok=True)

    # torch 2.10's Adam constructor reaches `torch/_compile.py:47`, which imports
    # `torch._dynamo`, which dlopens `libtriton.so`. Doing that *after* 4M
    # gaussians and their optimiser state are resident segfaults the process
    # (rc=-11) inside `gaussian_model.training_setup`, before iteration 1 --
    # every time, and not reproducible by importing the same modules on an empty
    # GPU. `TORCHDYNAMO_DISABLE` does not help because the import is
    # unconditional. Loading it first, while the GPU is still empty, does.
    preload = cfg.work_root / "_train_preload.py"
    preload.parent.mkdir(parents=True, exist_ok=True)
    preload.write_text(
        "import sys\n"
        "import torch\n"
        "import torch._dynamo  # v3d: dlopen libtriton before the GPU fills\n"
        "import runpy\n"
        "sys.argv[0] = 'PGSR/train.py'\n"
        "runpy.run_path('PGSR/train.py', run_name='__main__')\n"
    )
    cmd = [sys.executable, str(preload),
           "-s", str(layout.source_dir),
           "-i", str(layout.images_dir),
           "-m", str(cfg.gs_root / layout.scene_id),
           "--ply_path", str(layout.point_cloud),
           "--resolution", "1"]

    # `eval=False` merges the transforms file's `test_frames` into the training
    # set, so every reported PSNR is measured on images the model was fitted to.
    # The 98k run did exactly that and reported 24.28 / 25.08, numbers that say
    # how well it memorised rather than how well it reconstructs. With
    # `--eval` the 106 `test_frames` are held out and never seen, which is the
    # only way this pipeline can produce a generalisation number.
    #
    # It is a real trade, not a free improvement: 106 of 852 views (12.4%) stop
    # supervising the geometry. Default off so the geometry run keeps every view;
    # turn it on when the question is quality *measurement*.
    if hold_out_test:
        cmd.append("--eval")

    if schedule:
        it = int(schedule["iterations"])
        cmd += [
            "--iterations", str(it),
            # Tied to `iterations`, not left at its own 30,000 default: the
            # position learning rate decays to its floor at
            # position_lr_max_steps, and a longer run past that point trains on
            # with positions effectively frozen — colour and opacity keep
            # improving while the geometry cannot move.
            "--position_lr_max_steps", str(int(schedule["position_lr_max_steps"])),
            "--densify_from_iter", str(int(schedule["densify_from_iter"])),
            "--densify_until_iter", str(int(schedule["densify_until_iter"])),
            "--single_view_weight_from_iter", str(int(schedule["single_view_weight_from_iter"])),
            "--multi_view_weight_from_iter", str(int(schedule["multi_view_weight_from_iter"])),
            "--multi_view_min_dis", f"{float(schedule['multi_view_min_dis']):.6f}",
            "--multi_view_max_dis", f"{float(schedule['multi_view_max_dis']):.6f}",
        ]
        # Densification thresholds, if the schedule supplies them. Absent, PGSR keeps
        # its object-scale defaults, which under-populate a room: the pilot got
        # 1.25 M gaussians for a smaller space than this run's 636 k.
        for key, flag in (("densify_grad_threshold", "--densify_grad_threshold"),
                          ("densify_abs_grad_threshold", "--densify_abs_grad_threshold")):
            if key in schedule:
                cmd += [flag, f"{float(schedule[key]):.8f}"]
        # The two cadences that decide whether a stretched run keeps densifying.
        # Left at their 30,000-iteration defaults on the 98k run, the primitive
        # count moved +0.1% between 23k and 98k: `train.py:412` densifies every
        # `densification_interval` while `train.py:418` trims every 1,000, and at
        # absolute spacing over a 3.3x longer run the two reach equilibrium. See
        # `quality.pgsr_schedule` for the derivation.
        if "densification_interval" in schedule:
            cmd += ["--densification_interval", str(int(schedule["densification_interval"]))]
        # train.py:532 already appends `iterations` to save_iterations, so the
        # final gaussians are saved whatever is passed. test_iterations is the one
        # that matters: its default [7000, 30000] would report no PSNR at all at
        # the end of a run that is not 30,000 long.
        cmd += ["--test_iterations", *[str(int(i)) for i in schedule["test_iterations"]]]

    return _run(cmd, cfg, f"{layout.scene_id}_pgsr.log")


def stage_mesh(
    scene: str | SceneLayout,
    cfg: PipelineConfig | None = None,
    knobs: "object | None" = None,
) -> int:
    """TSDF-fuse the trained gaussians into a mesh, then render mesh-guided masks.

    `knobs` comes from `v3d.geometry.quality.derive`. Passing none uses the
    paper's `mesh.sh` constants, which is the control rather than the run — three
    of the four silently destroy geometry on a room that is not a ScanNet++
    bedroom:

    - `--max_depth 4.0`: `render.py:122` does `ref_depth[ref_depth > max_depth] = 0`
      and passes it again as Open3D's `depth_trunc`. Surface further than this from
      every camera is not downweighted, it is **erased**. This scene's DA3 depth
      reaches 6.1 units, so 4.0 would delete real far geometry.
    - `--voxel_size 0.006`: the mesh's resolution ceiling, in units that have no
      fixed relationship to a metre from one video to the next.
    - `--num_cluster 1`: `render.py:166` keeps only the largest connected
      component, so any freestanding machine or crate the fusion did not weld to
      the floor is deleted — exactly the geometry the semantic-box and collision
      deliverables are about.
    - `--use_depth_filter` off: multi-view-inconsistent depth gets fused as if it
      were real. The interaction is the part that matters: `num_cluster=1` is only
      survivable *because* the filter is off, so the cluster cull is doing the
      filter's job bluntly and paying for it in real objects. Turning the filter
      on is what makes keeping more clusters safe.

    mesh2mask deliberately keeps its own, larger depth limit: it compares two
    renders rather than integrating a surface, so it tolerates more range. The
    paper's own ratio is 6.0 against 4.0, and `KnobPolicy.mask_depth_multiple`
    keeps that 1.5.
    """
    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    model_dir = cfg.gs_root / layout.scene_id
    if not (model_dir / "point_cloud").is_dir():
        raise FileNotFoundError(f"run stage_pgsr first: no gaussians under {model_dir}")

    if knobs is None:
        render_max, mask_max = RENDER_MAX_DEPTH, MASK_MAX_DEPTH
        voxel, clusters, depth_filter = VOXEL_SIZE, NUM_CLUSTER, False
    else:
        cli = knobs.as_cli()
        render_max, mask_max = cli["render_max_depth"], cli["mask_max_depth"]
        voxel, clusters = cli["voxel_size"], cli["num_cluster"]
        depth_filter = knobs.use_depth_filter

    render_cmd = [sys.executable, "PGSR/render.py",
                  "-s", str(layout.source_dir), "-m", str(model_dir),
                  "-i", str(layout.images_dir),
                  "--skip_test",
                  "--max_depth", render_max,
                  "--voxel_size", voxel,
                  "--num_cluster", clusters]
    if depth_filter:
        render_cmd.append("--use_depth_filter")

    rc = _run(render_cmd, cfg, f"{layout.scene_id}_render.log")
    if rc != 0:
        return rc

    mesh = model_dir / MESH_PATH_REL
    if not mesh.exists():
        raise FileNotFoundError(f"render.py reported success but {mesh} is absent")

    return _run(
        [sys.executable, "PGSR/mesh2mask.py",
         "-m", str(model_dir), "-s", str(layout.source_dir), "-i", str(layout.images_dir),
         "--mesh_path", MESH_PATH_REL,
         "--max_depth", mask_max,
         "--voxel_size", voxel],
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


def stage_boxes(
    scene: str | SceneLayout,
    cfg: PipelineConfig | None = None,
    vlm_api: str = "http://127.0.0.1:8000/v1/chat/completions",
) -> int:
    """Lift the 2D masks into 3D instance boxes, against the trained gaussians.

    This stage had no wrapper, which meant `boxes_raw` was never produced and
    `stage_postprocess` could only ever raise. It is the step that turns masks
    into the semantic boxes P1 is asked for.

    `vlm_api` is the piece that decides how many instances survive.
    `3d_bounding_instance_gs_region.py` accepts an instance outright at
    score >= 0.9, drops it below 0.75, and asks a VLM about everything between --
    and the paper points that at a vLLM server. With no server the request fails
    and every borderline instance is silently rejected, so the box set is biased
    toward whatever SAM3 happened to be confident about. Measured on this scene,
    the 0.70-0.90 band held 755 of 1,507 masks: half the detections.
    `scripts/kernel/qwen_openai_shim.py` serves the same OpenAI-shaped endpoint
    from the local Qwen3-VL, so the verification path actually runs offline.
    """
    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    model_dir = cfg.gs_root / layout.scene_id
    if not (model_dir / "point_cloud").is_dir():
        raise FileNotFoundError(f"run stage_pgsr first: no gaussians under {model_dir}")

    return _run(
        [sys.executable, "3d_bounding_instance_gs_region.py",
         "--scene", layout.scene_id,
         "--data-root", str(cfg.data_root),
         "--mask-root", str(cfg.work_root / "sam_masks" / layout.scene_id),
         "--model-path", str(model_dir),
         "--output-dir", str(cfg.work_root / "boxes_raw" / layout.scene_id),
         "--vllm-api-url", vlm_api],
        cfg,
        f"{layout.scene_id}_boxes.log",
    )


def stage_postprocess(scene: str | SceneLayout, cfg: PipelineConfig | None = None) -> int:
    """Floor-align the boxes, which is what the released annotations are."""
    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)

    raw = cfg.work_root / "boxes_raw" / layout.scene_id
    if not raw.is_dir() or not any(raw.glob("*.json")):
        raise FileNotFoundError(f"run stage_postprocess first: nothing in {raw}")

    return _run(
        [sys.executable, "postprocess_3d_bbox_aabb.py",
         "--input_dir", str(raw),
         "--output_dir", str(cfg.work_root / "boxes" / layout.scene_id)],
        cfg,
        f"{layout.scene_id}_postprocess.log",
    )


def boxes_output(scene_id: str, cfg: PipelineConfig) -> Path:
    return cfg.work_root / "boxes" / scene_id


def scene_output(scene_id: str, cfg: PipelineConfig) -> Path:
    return cfg.work_root / "scene" / scene_id / "scene.json"


def gate_output(scene_id: str, cfg: PipelineConfig) -> Path:
    return cfg.work_root / "scene" / scene_id / "stage_j.json"


def _sole_json(directory: Path, what: str) -> Path:
    """The one instance-annotation json in `directory`, or a refusal to guess.

    postprocess_3d_bbox_aabb.py writes to a directory, while build_scene reads a
    file. Picking the newest or the alphabetically-first would silently grade a
    stale run, and grading the wrong file is indistinguishable from a good
    result, so an ambiguous directory is an error.
    """
    found = sorted(p for p in directory.glob("*.json") if not p.name.startswith("."))
    if not found:
        raise FileNotFoundError(f"no {what} json in {directory}")
    if len(found) > 1:
        raise ValueError(
            f"{directory} holds {len(found)} json files "
            f"({', '.join(p.name for p in found)}); pass one explicitly rather "
            f"than letting the gate pick, since grading the wrong file looks "
            f"exactly like grading the right one"
        )
    return found[0]


def stage_decompose(
    scene: str | SceneLayout,
    cfg: PipelineConfig | None = None,
    out_dir: Path | None = None,
) -> int:
    """Cut the fused mesh into a room shell plus per-object assets.

    This had no stage either, and its absence is what makes the accuracy gate
    fail. `stage_scene` prefers a mesh-derived room shell and falls back to one
    lifted from the floor/wall *instances* when `objects.json` is missing --
    and that fallback is badly undersized. Measured on this scene: the lifted
    shell gave a 42.0 m^3 room in which **89 of 89 objects** lay outside the
    footprint, the worst by 6.55 m, tripping both `no_stray_objects` and
    `ceiling_plausible`. RANSAC over the mesh itself does not depend on whether
    a floor happened to be segmented as an instance.
    """
    from v3d.scene import decompose as dc

    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    mesh = cfg.gs_root / layout.scene_id / MESH_PATH_REL
    if not mesh.exists():
        raise FileNotFoundError(f"run stage_mesh first: no mesh at {mesh}")

    out = Path(out_dir or os.environ.get("V3D_DELIVER", "/tmp/deliver"))
    w2c, _K, _n = _read_transforms(layout)
    # camera centres orient the up-axis search: -R^T t per view
    import numpy as np
    centres = np.stack([-w[:3, :3].T @ w[:3, 3] for w in w2c])

    result = dc.cut(mesh, out, camera_centres=centres, scale_mode="assumed")
    report = dc.audit(result)
    path = dc.write(result, report, out)
    print(f"  room extent {result.room_extent}  objects {len(result.objects)} -> {path}")
    return 0


def stage_scene(
    scene: str | SceneLayout,
    cfg: PipelineConfig | None = None,
    instances: Path | None = None,
) -> int:
    """Turn the floor-aligned boxes into the `scene.json` the gate reads.

    The simulator-facing document, and the last stage that is pure geometry
    bookkeeping — everything above it needed a GPU.
    """
    from v3d.scene.from_holi import build_scene

    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)

    src = instances or _sole_json(boxes_output(layout.scene_id, cfg), "floor-aligned box")
    # Prefer the mesh-derived room shell when the geometric cut has produced one. The
    # lifted floor instances gave a polygon spanning 5.90 x 1.43 on this scene while
    # the mesh's floor plane covered 96.3% of the footprint, and Stage J's stray-object
    # and oversized-object failures were both consequences of that undersized shell.
    cut = Path(os.environ.get("V3D_DELIVER", "/tmp/deliver")) / "objects.json"
    doc = build_scene(layout.scene_id, src,
                      objects_json=cut if cut.exists() else None)
    if cut.exists():
        print(f"  room shell from the mesh ({cut})")
    else:
        print(f"  room shell from lifted instances; no {cut} to prefer")
    out = doc.write(scene_output(layout.scene_id, cfg))

    length, width = doc.room.footprint_extent_m()
    # Unit suffix from the document, not hardcoded. This printed "m" for what are
    # DA3's arbitrary units, which is precisely the mislabelling that let a room
    # 8.6 m tall be reported as measured. The reader of a log has no other cue.
    scale_mode = getattr(getattr(doc, "scale", None), "mode", None) or "assumed"
    unit = "m" if scale_mode not in {"assumed", None} else "DA3 units (scale assumed)"
    print(f"  {len(doc.objects)} objects, room {length:.2f} x {width:.2f} x "
          f"{doc.room.height_m():.2f} {unit} -> {out}")
    return 0


def stage_gate(
    scene: str | SceneLayout,
    cfg: PipelineConfig | None = None,
    gt: Path | None = None,
    tol_m: float | None = None,
) -> int:
    """Stage J — grade `scene.json` against the released annotation.

    Returns 0 only when the gate passes, so `--stage all` fails loudly on a
    scene that reconstructed but did not reconstruct *accurately*. Without a
    ground truth it runs the structural half and says so, which is the only
    thing available for a scene that came from a phone video.
    """
    import json

    from v3d.validate import stage_j

    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    scene_path = scene_output(layout.scene_id, cfg)
    if not scene_path.exists():
        raise FileNotFoundError(f"run stage_scene first: {scene_path} is missing")

    report = stage_j.run(scene_path, gt, tol_m if tol_m is not None else stage_j.DEFAULT_TOL_M)
    out = gate_output(layout.scene_id, cfg)
    out.write_text(json.dumps(report.to_dict(), indent=2))

    for c in report.checks:
        print(f"  [{c.status}] {c.id}  {c.detail}")
    if report.recall is not None:
        print(f"  recall {report.recall * 100:.1f}%  "
              f"structural {report.structural_verdict}  metric {report.metric_verdict}")
    print(f"  OVERALL {report.overall} -> {out}")
    return 0 if report.overall == "PASS" else 1


def stage_report(scene: str | SceneLayout, cfg: PipelineConfig | None = None) -> dict[str, object]:
    """What exists so far, so a stage can be resumed rather than repeated."""
    cfg = cfg or PipelineConfig()
    layout = _layout(scene, cfg)
    mesh = cfg.gs_root / layout.scene_id / MESH_PATH_REL

    def mb(p: Path) -> float | None:
        return round(p.stat().st_size / 1e6, 1) if p.exists() else None

    raw_boxes = cfg.work_root / "boxes_raw" / layout.scene_id
    boxes = boxes_output(layout.scene_id, cfg)
    return {
        "scene": layout.scene_id,
        "layout": str(layout.source_dir),
        "frames": len(list(layout.images_dir.glob("*"))) if layout.images_dir.is_dir() else 0,
        "depth_maps": len(list(layout.depth_dir.glob("*.npy"))) if layout.depth_dir.is_dir() else 0,
        "point_cloud_mb": mb(layout.point_cloud),
        "mesh_mb": mb(mesh),
        "masks": len(list(layout.mask_dir.glob("*"))) if layout.mask_dir.is_dir() else 0,
        "vlm_labels": vlm_output(layout.scene_id, cfg).exists(),
        "boxes_raw": len(list(raw_boxes.glob("*.json"))) if raw_boxes.is_dir() else 0,
        "boxes_aligned": len(list(boxes.glob("*.json"))) if boxes.is_dir() else 0,
        "scene_json_mb": mb(scene_output(layout.scene_id, cfg)),
        "gate": mb(gate_output(layout.scene_id, cfg)),
    }


# Every stage the CLI can dispatch, in the order they depend on each other.
# postprocess existed as a function but was absent from this table,
# from the --stage choices, and from the "all" ordering, so the boxes the whole
# benchmark is measured on could not be produced from the command line at all;
# scene and gate had no stage wrapper. The tail is what produces the number.
STAGES = {"da3": stage_da3, "pgsr": stage_pgsr, "mesh": stage_mesh,
          "boxes": stage_boxes, "decompose": stage_decompose,
          "vlm": stage_vlm, "sam3": stage_sam3,
          "postprocess": stage_postprocess, "scene": stage_scene, "gate": stage_gate}

ALL_ORDER = ["da3", "pgsr", "mesh", "vlm", "sam3", "postprocess", "scene", "gate"]

# The stages that shell into the paper's checkout, and so need its two
# machine-specific paths rewritten first. The rest run in this process: vlm
# calls v3d.semantic.vlm, and scene/gate/report are geometry bookkeeping over
# json. Patching unconditionally made `--stage scene` fail for want of a paper
# repo it never reads, on a laptop that will never have one.
PAPER_STAGES = frozenset({"da3", "pgsr", "mesh", "sam3", "postprocess"})


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene")
    ap.add_argument("--stage", default="report",
                    choices=["patch", *ALL_ORDER, "all", "tail", "report"],
                    help="'tail' is postprocess..gate: everything after the GPU stages")
    ap.add_argument("--paper-root", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--work-root", type=Path)
    ap.add_argument("--gt", type=Path, help="ground-truth boxes, for --stage gate")
    ap.add_argument("--tol", type=float, help="gate tolerance in metres")
    args = ap.parse_args(argv)

    cfg = PipelineConfig()
    for name in ("paper_root", "data_root", "work_root"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))

    if args.stage == "all":
        order = ALL_ORDER
    elif args.stage == "tail":
        order = ALL_ORDER[ALL_ORDER.index("postprocess"):]
    else:
        order = [args.stage]

    if args.stage == "patch" or PAPER_STAGES.intersection(order):
        for p in apply_offline_patches(cfg):
            print(f"  {p.describe()}")

    for name in order:
        if name not in STAGES:
            continue
        print(f"--- {name} ---")
        if name == "gate":
            rc = stage_gate(args.scene, cfg, gt=args.gt, tol_m=args.tol)
        else:
            rc = STAGES[name](args.scene, cfg)
        print(f"  {name} exit code {rc}")
        if rc != 0:
            return rc

    print(f"  {stage_report(args.scene, cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
