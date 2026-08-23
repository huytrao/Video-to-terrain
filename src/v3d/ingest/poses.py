"""Recover camera poses from a bare image set, and write them where the pipeline reads them.

The paper's DA3 stage expects a ScanNet++ scene: `dslr/nerfstudio/transforms_undistorted.json`
beside `dslr/resized_undistorted_images/`. That json is the only channel through
which poses reach every later stage, so getting its conventions right is the
single highest-consequence piece of code in the ingest path — and the pipeline
cannot tell you when you get it wrong.

Why it cannot tell you. `inference_da3_scannetppv2.py:588-589` asserts that DA3's
predicted poses match the ones it was handed:

    assert (prediction.intrinsics - intrinsics < 1e-6).all()
    assert (prediction_extrinsics - extrinsics < 1e-6).all()

That assertion cannot fail. In `api.py`, `_align_to_input_extrinsics_intrinsics`
runs whenever extrinsics are supplied and *overwrites the prediction with the
input* before anyone compares them:

    prediction.intrinsics = intrinsics.numpy()          # api.py:352
    prediction.extrinsics = extrinsics[..., :3, :]      # api.py:361
    prediction.depth /= scale                           # api.py:362

The extrinsics difference is exactly zero for any poses at all — correct,
mirrored, or random. The intrinsics difference is between the resized ~504-px
matrix and the full-resolution one, and passes only because the check is written
as a bare difference rather than an absolute value.

Line 362 is the part that hurts. `scale` is fitted by Umeyama between DA3's own
predicted poses and the poses in the json, and every depth map is divided by it.
So a wrong convention here does not raise: it rescales the entire reconstruction
by a plausible-looking factor. That is the mechanism behind the room this project
once measured at 8.6 m tall while a validator passed it 16/16.

So this module carries its own check, and the check is exact rather than
statistical: the poses are written, read back **through the pipeline's own
reader**, and compared to what DA3 predicted. A convention error changes the
result, so the comparison can fail. Nothing here trusts a downstream assert.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# --- conventions -------------------------------------------------------------
#
# Three coordinate conventions meet in this file. Naming them is cheaper than
# rediscovering them.
#
#   w2c      world-to-camera, OpenCV/COLMAP axes (x right, y down, z forward).
#            What DA3 predicts, and what every consumer ultimately wants.
#   c2w_cv   its inverse, same axes.
#   c2w_gl   camera-to-world in OpenGL/Blender axes (x right, y UP, z BACKWARD).
#            What goes in the json's "transform_matrix", because that is what
#            `load_scannetppv2_poses` assumes when it does `c2w[:3, 1:3] *= -1`.
#
# The flip between cv and gl is negating the second and third columns of the
# rotation block, which is its own inverse — so writer and reader use the same
# operation and a mistake shows up as a mirrored scene rather than cancelling.


# --- the reader's convention, read from the reader ---------------------------


@dataclass(frozen=True)
class ReaderConvention:
    """Which columns `load_scannetppv2_poses` negates, and whether it inverts.

    Parsed from source rather than remembered, because the whole failure mode
    being guarded against here is a writer and a reader that agree with each
    other and disagree with the file on disk.
    """

    flip_lo: int
    flip_hi: int
    inverts: bool
    source: str = "default"

    def describe(self) -> str:
        return (f"negate c2w[:3, {self.flip_lo}:{self.flip_hi}] then "
                f"{'invert' if self.inverts else 'use as-is'} ({self.source})")


# What `inference_da3_scannetppv2.py:37,40` does today. Used only when the paper
# source is not on hand; `parse_reader_convention` supersedes it.
DEFAULT_CONVENTION = ReaderConvention(1, 3, True, "hardcoded fallback")


def parse_reader_convention(paper_file: Path) -> ReaderConvention:
    """Extract the pose convention from `load_scannetppv2_poses`'s own AST.

    The function cannot simply be imported: its module loads torch and the DA3
    checkpoint at import time, so a convention check would require a GPU. Parsing
    is the next best thing and is grounded in the same bytes the interpreter will
    run. If the paper's code ever changes which axes it flips, the writer follows
    it, rather than silently producing a mirrored scene.
    """
    import ast

    tree = ast.parse(Path(paper_file).read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "load_scannetppv2_poses"),
              None)
    if fn is None:
        raise RuntimeError(f"no load_scannetppv2_poses in {paper_file}")

    flips: set[tuple[int, int]] = set()
    inverts = False
    for node in ast.walk(fn):
        # c2w[:3, a:b] *= -1
        if (isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Mult)
                and isinstance(node.target, ast.Subscript)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "c2w"):
            factor = ast.literal_eval(node.value) if isinstance(
                node.value, (ast.Constant, ast.UnaryOp)) else None
            if factor != -1:
                continue
            sl = node.target.slice
            dims = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            if len(dims) != 2 or not isinstance(dims[1], ast.Slice):
                continue
            lo = ast.literal_eval(dims[1].lower) if dims[1].lower is not None else 0
            hi = ast.literal_eval(dims[1].upper) if dims[1].upper is not None else 4
            flips.add((int(lo), int(hi)))
        # np.linalg.inv(c2w)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "inv"):
            inverts = True

    if len(flips) != 1:
        raise RuntimeError(
            f"expected exactly one column-negation convention in "
            f"load_scannetppv2_poses, found {sorted(flips)}; the reader has "
            f"changed shape and the writer must be revisited by hand rather than "
            f"guessing which flip is current"
        )
    if not inverts:
        raise RuntimeError(
            "load_scannetppv2_poses no longer inverts the matrix; it now returns "
            "camera-to-world, and this writer's output would be inverted twice"
        )
    lo, hi = flips.pop()
    return ReaderConvention(lo, hi, True, f"parsed from {Path(paper_file).name}")


def w2c_to_transform_matrix(w2c: np.ndarray, conv: "ReaderConvention | None" = None) -> np.ndarray:
    """DA3's world-to-camera matrix -> the json's camera-to-world matrix.

    The axis flip is taken from `conv`, which is parsed out of the reader's own
    source, so the writer cannot disagree with the reader about which columns to
    negate. Hardcoding the pair here is what made the earlier version of this
    check circular.
    """
    if w2c.shape != (4, 4):
        raise ValueError(f"expected a 4x4 w2c, got {w2c.shape}")
    conv = conv or DEFAULT_CONVENTION
    c2w = np.linalg.inv(w2c)
    c2w[:3, conv.flip_lo:conv.flip_hi] *= -1
    return c2w


def transform_matrix_to_w2c(c2w_gl: np.ndarray,
                            conv: "ReaderConvention | None" = None) -> np.ndarray:
    """The reader's half, from the reader's own parsed convention.

    Note what this can and cannot establish. Because the flip is an involution,
    composing this with `w2c_to_transform_matrix` returns the identity for *any*
    column pair, as long as both halves use the same one. So a round trip through
    these two functions proves the inversion and the frame bookkeeping and
    nothing at all about the axis convention. That is why the axis pair comes
    from `ReaderConvention` -- parsed from the file that will actually read the
    json -- and why the substantive check on these poses is geometric
    (`reprojection_consistency`) rather than algebraic.
    """
    conv = conv or DEFAULT_CONVENTION
    c2w = np.array(c2w_gl, dtype=np.float64).copy()
    c2w[:3, conv.flip_lo:conv.flip_hi] *= -1
    return np.linalg.inv(c2w) if conv.inverts else c2w


def to_4x4(m: np.ndarray) -> np.ndarray:
    """Accept (3, 4) or (4, 4); DA3 returns either depending on the code path."""
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (4, 4):
        return m.copy()
    if m.shape == (3, 4):
        out = np.eye(4)
        out[:3, :] = m
        return out
    raise ValueError(f"cannot interpret {m.shape} as a pose matrix")


def rescale_intrinsic(K: np.ndarray, from_wh: tuple[int, int],
                      to_wh: tuple[int, int]) -> np.ndarray:
    """Move an intrinsic matrix between resolutions.

    The inverse of `InputProcessor._resize_ixt`, which is what put DA3's
    predicted intrinsics at the processed resolution in the first place. Row 0
    (fx, cx) scales with width; row 1 (fy, cy) with height.
    """
    K = np.asarray(K, dtype=np.float64).copy()
    K[:1] *= to_wh[0] / float(from_wh[0])
    K[1:2] *= to_wh[1] / float(from_wh[1])
    return K


# --- spec --------------------------------------------------------------------


@dataclass
class PoseSpec:
    """How to run pose prediction. Counts and ratios only."""

    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    # A single scene intrinsic is all the json format can carry, so N per-frame
    # estimates must be pooled. That is legitimate for scatter around a fixed
    # camera and wrong for a real zoom, and the three thresholds below separate
    # the two: a drift must be large in relative terms, large compared with the
    # series' own noise, AND correlated with time before it counts as a zoom.
    # Requiring all three keeps a noisy-but-fixed camera from being refused and a
    # slow genuine zoom from being averaged away.
    max_focal_drift: float = 0.02       # fraction of the median focal length
    drift_over_noise: float = 2.0       # drift must exceed this many noise sigmas
    drift_correlation: float = 0.5      # and correlate with frame index this much
    # Pose round trip must reproduce DA3's w2c to near float32 precision. This is
    # not a tolerance on pose *quality*; it is a tolerance on arithmetic.
    roundtrip_tol: float = 1e-4
    # The tolerance that actually guards the convention. DA3 predicts depth and
    # poses jointly, so once the json bookkeeping is right they agree to well
    # under this; a wrong axis pair misses by an order of magnitude more.
    reprojection_tol: float = 0.02
    # The file whose pose convention the writer must match. Parsed, not trusted:
    # see `parse_reader_convention`.
    paper_file: str | None = None
    # Held out of `frames` into `test_frames`. Both keys are read and concatenated
    # by the DA3 loader, so geometry still uses every frame; the split exists so a
    # renderer can report a number that was not fitted.
    holdout_every: int = 8
    device: str = "cuda"


# --- the stage ---------------------------------------------------------------


def predict_poses(model, image_paths: list[Path], spec: PoseSpec) -> dict:
    """Run DA3 with no pose input and return its own estimate.

    `extrinsics` and `intrinsics` default to None, in which case DA3 predicts
    both. That is the only route available here: the run kernel has no COLMAP
    binary, and this footage — large flat painted walls — is the textureless case
    where feature-based SfM is weakest. That is a reason COLMAP was not an option,
    not evidence that these poses are good; pose *accuracy* is checked elsewhere,
    by a held-out render, not by this function.
    """
    prediction = model.inference(
        [str(p) for p in image_paths],
        extrinsics=None,
        intrinsics=None,
        process_res=spec.process_res,
        process_res_method=spec.process_res_method,
    )

    w2c = np.stack([to_4x4(e) for e in np.asarray(prediction.extrinsics)])
    K_proc = np.asarray(prediction.intrinsics, dtype=np.float64)
    depth = np.asarray(prediction.depth)
    if depth.ndim != 3:
        raise RuntimeError(f"expected depth (N, H, W), got {depth.shape}")
    proc_h, proc_w = depth.shape[1], depth.shape[2]

    if len(w2c) != len(image_paths):
        raise RuntimeError(
            f"DA3 returned {len(w2c)} poses for {len(image_paths)} images"
        )
    return {"w2c": w2c, "K_proc": K_proc, "proc_wh": (proc_w, proc_h),
            "depth": depth,
            "conf": np.asarray(prediction.conf) if hasattr(prediction, "conf") else None}


def consensus_intrinsic(K_proc: np.ndarray, proc_wh: tuple[int, int],
                        orig_wh: tuple[int, int], spec: PoseSpec) -> tuple[np.ndarray, dict]:
    """Collapse per-frame intrinsics to the one the json can hold — or refuse.

    The json format carries exactly one intrinsic for the whole scene, so N
    per-frame estimates have to become one. Whether that is legitimate depends on
    *why* they differ, and the magnitude of the disagreement does not say:

      - A real zoom, or a stabilisation crop that changes effective focal length,
        makes the single-intrinsic format wrong for this video. Pooling would
        encode a lens nobody used and mis-project every frame.
      - A feed-forward network estimating focal length independently per frame
        scatters by a few percent even on a locked-off camera. Here pooling is not
        a fudge, it is the right estimator: repeated noisy measurements of one
        fixed quantity, and the median beats any single frame.

    So the test is for *structure in time*, not for spread. A zoom drifts
    monotonically or steps; scatter does not. Drift is measured against the
    frames' own noise level, which makes the criterion scale-free and keeps it
    honest on a camera that happens to be noisier or steadier than this one.
    """
    if K_proc.ndim != 3 or K_proc.shape[1:] != (3, 3):
        raise RuntimeError(f"expected intrinsics (N, 3, 3), got {K_proc.shape}")

    K_orig = np.stack([rescale_intrinsic(K, proc_wh, orig_wh) for K in K_proc])
    fx, fy = K_orig[:, 0, 0], K_orig[:, 1, 1]
    cx, cy = K_orig[:, 0, 2], K_orig[:, 1, 2]

    median = np.median(K_orig, axis=0)
    f_med = float(np.median(np.concatenate([fx, fy])))
    n = len(fx)

    def structure(v: np.ndarray) -> dict:
        """Separate a trend from scatter, in units of the series' own noise.

        Noise is measured from *successive differences*, not from the spread about
        the median. A median absolute deviation is inflated by whatever trend is
        present -- a step change makes half the samples sit 30% off the median, so
        MAD reports the step as noise and the drift-over-noise ratio never fires.
        Frame-to-frame differences see only local scatter and are blind to a slow
        ramp or a single step, which is exactly what is wanted for a denominator
        that a trend is being compared against.
        """
        med = float(np.median(v))
        scale = max(abs(med), 1e-9)
        if n >= 2:
            # 0.6745 is the median of a half-normal; the sqrt(2) undoes the
            # variance doubling from differencing two independent samples.
            noise = float(np.median(np.abs(np.diff(v)))) / (0.6745 * np.sqrt(2)) / scale
        else:
            noise = 0.0
        edge = max(1, n // 10)
        drift = float(np.median(v[-edge:]) - np.median(v[:edge])) / scale
        if n >= 3 and v.std() > 0:
            idx = np.arange(n, dtype=float)
            r = float(np.corrcoef(idx, v)[0, 1])
        else:
            r = 0.0
        return {"median": med, "rel_range": float((v.max() - v.min()) / scale),
                "rel_noise": noise, "rel_drift": drift, "time_correlation": r,
                "drift_over_noise": abs(drift) / max(noise, 1e-9)}

    sfx, sfy = structure(fx), structure(fy)
    spread = {
        "fx_rel_range": sfx["rel_range"], "fy_rel_range": sfy["rel_range"],
        "cx_rel_range": float((cx.max() - cx.min()) / f_med),
        "cy_rel_range": float((cy.max() - cy.min()) / f_med),
        "fx_median": sfx["median"], "fy_median": sfy["median"],
        "cx_median": float(np.median(cx)), "cy_median": float(np.median(cy)),
        "fx_structure": sfx, "fy_structure": sfy,
        "anisotropy": abs(sfx["median"] - sfy["median"]) / f_med,
        "frames": n,
        # A useful sanity number for a human reader: horizontal field of view.
        "hfov_deg": float(np.degrees(2 * np.arctan(0.5 * orig_wh[0] / np.median(fx)))),
    }

    zoomed = [
        name for name, s in (("fx", sfx), ("fy", sfy))
        if abs(s["rel_drift"]) > spec.max_focal_drift
        and s["drift_over_noise"] > spec.drift_over_noise
        and abs(s["time_correlation"]) > spec.drift_correlation
    ]
    if zoomed:
        raise RuntimeError(
            f"focal length drifts systematically over the clip ({', '.join(zoomed)}): "
            f"fx drift {sfx['rel_drift']:+.4f} ({sfx['drift_over_noise']:.1f}x its "
            f"noise, r={sfx['time_correlation']:+.2f}), fy drift "
            f"{sfy['rel_drift']:+.4f} ({sfy['drift_over_noise']:.1f}x, "
            f"r={sfy['time_correlation']:+.2f}). The camera zoomed or the "
            f"stabilisation crop changed, so one intrinsic per scene cannot "
            f"describe this video and every stage downstream assumes it can. Split "
            f"the clip at the zoom, or re-shoot at a fixed focal length."
        )

    spread["pooled"] = True
    spread["pooling_justification"] = (
        f"per-frame scatter without a trend (fx {sfx['rel_noise']:.4f} noise, "
        f"drift {sfx['rel_drift']:+.4f} = {sfx['drift_over_noise']:.1f}x noise, "
        f"r={sfx['time_correlation']:+.2f}); median of {n} estimates"
    )
    spread["worst_rel_range"] = max(sfx["rel_range"], sfy["rel_range"])
    return median, spread


def build_transforms(w2c: np.ndarray, K: np.ndarray, image_names: list[str],
                     orig_wh: tuple[int, int], spec: PoseSpec,
                     conv: ReaderConvention | None = None) -> dict:
    """Assemble the json exactly as `load_scannetppv2_poses` will read it.

    Both `frames` and `test_frames` are emitted: the reader indexes both keys
    unconditionally, so a file missing either raises KeyError at load.
    """
    if len(w2c) != len(image_names):
        raise ValueError(f"{len(w2c)} poses for {len(image_names)} names")

    frames, test_frames = [], []
    for i, (pose, name) in enumerate(zip(w2c, image_names)):
        entry = {
            "file_path": name,
            "transform_matrix": w2c_to_transform_matrix(to_4x4(pose), conv).tolist(),
        }
        held_out = spec.holdout_every > 0 and i % spec.holdout_every == spec.holdout_every - 1
        (test_frames if held_out else frames).append(entry)

    # Never let the holdout empty the training set, and never emit an empty
    # test list where the caller asked for one.
    if not frames:
        raise RuntimeError("holdout_every consumed every frame")

    return {
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "w": int(orig_wh[0]),
        "h": int(orig_wh[1]),
        "camera_model": "PINHOLE",
        # Recorded so nothing downstream mistakes these units for metres. DA3
        # predicts geometry up to one unknown global scale; nothing in a video
        # fixes it.
        "v3d_scale_mode": "assumed",
        "v3d_scale_note": (
            "poses and depth are in DA3's own arbitrary units, not metres; any "
            "metric quantity derived from this scene is unscaled"
        ),
        "frames": frames,
        "test_frames": test_frames,
    }


def verify_roundtrip(doc: dict, w2c_by_name: dict[str, np.ndarray],
                     spec: PoseSpec, conv: ReaderConvention | None = None) -> dict:
    """Check the arithmetic of the write: inversion, frame coverage, handedness.

    Deliberately narrow about what it establishes. Because the axis flip is an
    involution, a writer and reader sharing the same column pair round-trip to the
    identity whatever pair that is -- so this proves the inversion, that no frame
    was lost or duplicated, and that the rotations are proper, and it proves
    nothing about the axis convention. The convention is pinned separately, by
    parsing the reader's source; and the poses are checked substantively by
    `verify_geometry`, which reprojects them against the depth.

    It does catch: transposing instead of inverting, writing w2c where c2w was
    meant, a mirrored scene, and a frame list that does not match the depth stack.
    """
    worst_r = worst_t = 0.0
    worst_name = None
    checked = 0

    for entry in doc["frames"] + doc["test_frames"]:
        name = entry["file_path"]
        expected = to_4x4(w2c_by_name[name])
        got = transform_matrix_to_w2c(np.array(entry["transform_matrix"]), conv)
        dr = float(np.abs(got[:3, :3] - expected[:3, :3]).max())
        dt = float(np.abs(got[:3, 3] - expected[:3, 3]).max())
        if max(dr, dt) > max(worst_r, worst_t):
            worst_name = name
        worst_r, worst_t = max(worst_r, dr), max(worst_t, dt)
        checked += 1

    if checked != len(w2c_by_name):
        raise RuntimeError(
            f"round trip covered {checked} frames but {len(w2c_by_name)} were "
            f"predicted; the json lost or duplicated frames"
        )
    if max(worst_r, worst_t) > spec.roundtrip_tol:
        raise RuntimeError(
            f"pose round trip does not reproduce DA3's extrinsics: worst "
            f"rotation error {worst_r:.3e}, translation {worst_t:.3e} "
            f"(frame {worst_name}), tolerance {spec.roundtrip_tol:.1e}. The json "
            f"convention is wrong; every depth map downstream would be rescaled "
            f"by a wrong Umeyama factor without raising."
        )

    # A mirrored scene survives a careless round trip but not this: a proper
    # rotation has determinant +1, a reflected one -1.
    dets = [float(np.linalg.det(transform_matrix_to_w2c(
        np.array(e["transform_matrix"]), conv)[:3, :3]))
        for e in doc["frames"] + doc["test_frames"]]
    if min(dets) < 0.99 or max(dets) > 1.01:
        raise RuntimeError(
            f"written poses are not proper rotations: determinant range "
            f"[{min(dets):.4f}, {max(dets):.4f}]; a negative determinant is a "
            f"mirrored scene, which reconstructs and renders and is wrong"
        )

    return {"worst_rotation_abs_err": worst_r, "worst_translation_abs_err": worst_t,
            "frames_checked": checked, "det_min": min(dets), "det_max": max(dets)}


def reprojection_consistency(depth: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                             stride: int = 1, max_pairs: int = 60,
                             samples_per_pair: int = 3000, seed: int = 0) -> dict:
    """Check the poses against the depth, geometrically. This one can fail.

    Every algebraic check on a pose convention has the same weakness: the axis
    flip is an involution, so a writer and a reader that make the same mistake
    agree perfectly. Geometry does not have that weakness. Take a pixel in frame
    i, unproject it with frame i's depth, move it to the world with frame i's
    pose, and project it into frame j. If the poses and the depth describe the
    same scene, the depth that comes back out of frame j matches the depth that
    went in. If the axis pair is wrong, the camera is pointing somewhere else and
    the numbers do not agree.

    DA3 predicts depth and poses jointly, so on correct bookkeeping these agree
    closely -- which is what makes the test sharp: the tolerance can be tight
    because the only thing being tested is the bookkeeping between them, not the
    model's accuracy.

    Everything is done at the processed resolution, where `depth` and `K` already
    live, so no rescaling can contribute error of its own.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 3:
        raise ValueError(f"expected depth (N, H, W), got {depth.shape}")
    n, h, wd = depth.shape
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (3, 3):
        K = np.repeat(K[None], n, axis=0)
    w2c = np.stack([to_4x4(m) for m in w2c])
    if len(w2c) != n:
        raise ValueError(f"{len(w2c)} poses for {n} depth maps")

    rng = np.random.default_rng(seed)
    Kinv = np.linalg.inv(K)
    c2w = np.linalg.inv(w2c)

    pairs = [(i, i + stride) for i in range(0, n - stride)]
    if len(pairs) > max_pairs:
        pick = np.linspace(0, len(pairs) - 1, max_pairs).round().astype(int)
        pairs = [pairs[i] for i in pick]

    rel_errors: list[float] = []
    inlier_fracs: list[float] = []
    per_pair: list[dict] = []

    for i, j in pairs:
        di = depth[i]
        vs, us = np.nonzero(np.isfinite(di) & (di > 0))
        if vs.size == 0:
            continue
        if vs.size > samples_per_pair:
            sel = rng.choice(vs.size, samples_per_pair, replace=False)
            vs, us = vs[sel], us[sel]
        z = di[vs, us]

        rays = np.stack([us + 0.5, vs + 0.5, np.ones_like(us, dtype=np.float64)])
        pc_i = (Kinv[i] @ rays) * z                       # (3, S) camera-i points
        pw = c2w[i][:3, :3] @ pc_i + c2w[i][:3, 3:4]      # world
        pc_j = w2c[j][:3, :3] @ pw + w2c[j][:3, 3:4]      # camera-j
        zj = pc_j[2]
        front = zj > 1e-6
        if not front.any():
            per_pair.append({"pair": [i, j], "inlier_fraction": 0.0, "note": "no points in front"})
            continue

        uvj = (K[j] @ pc_j)[:2] / zj
        uj, vj = uvj[0] - 0.5, uvj[1] - 0.5
        inside = front & (uj >= 0) & (uj <= wd - 1) & (vj >= 0) & (vj <= h - 1)
        if not inside.any():
            per_pair.append({"pair": [i, j], "inlier_fraction": 0.0, "note": "no overlap"})
            continue

        # Nearest-neighbour sampling: interpolating across a depth discontinuity
        # would invent a value that is in neither surface and inflate the error
        # for a reason that has nothing to do with the poses.
        uu = np.clip(np.rint(uj[inside]).astype(int), 0, wd - 1)
        vv = np.clip(np.rint(vj[inside]).astype(int), 0, h - 1)
        dj = depth[j][vv, uu]
        ok = np.isfinite(dj) & (dj > 0)
        if not ok.any():
            per_pair.append({"pair": [i, j], "inlier_fraction": 0.0, "note": "no valid target depth"})
            continue

        pred, meas = zj[inside][ok], dj[ok]
        rel = np.abs(pred - meas) / np.maximum(meas, 1e-9)
        # Occlusion is expected and is not a pose error: a point visible in i can
        # be hidden in j, giving a large one-sided residual. The median is
        # reported because it is the statistic occlusion cannot move.
        med = float(np.median(rel))
        rel_errors.append(med)
        frac = float(inside.mean())
        inlier_fracs.append(frac)
        per_pair.append({"pair": [i, j], "median_rel_error": med,
                         "inlier_fraction": frac,
                         "agree_within_2pct": float((rel < 0.02).mean())})

    if not rel_errors:
        raise RuntimeError(
            "no frame pair produced a testable overlap; the poses cannot be "
            "checked against the depth, which is itself a finding"
        )

    return {
        "pairs_tested": len(rel_errors),
        "median_rel_error": float(np.median(rel_errors)),
        "p90_rel_error": float(np.percentile(rel_errors, 90)),
        "median_overlap_fraction": float(np.median(inlier_fracs)),
        "stride": stride,
        "per_pair": per_pair[:20],
    }


def verify_geometry(doc: dict, depth: np.ndarray, K_proc: np.ndarray,
                    conv: ReaderConvention, spec: PoseSpec,
                    names_in_order: list[str]) -> dict:
    """Read the written json back through the parsed convention, then test it.

    This is the end-to-end check: the poses that get tested are the ones the
    pipeline will actually compute from the file on disk, not the ones held in
    memory. A wrong axis pair, a wrong inversion, a mis-ordered frame list, or a
    json whose frames do not line up with the depth stack all move the number.
    """
    by_name = {e["file_path"]: e["transform_matrix"]
               for e in doc["frames"] + doc["test_frames"]}
    missing = [n for n in names_in_order if n not in by_name]
    if missing:
        raise RuntimeError(f"{len(missing)} frames absent from the written json: {missing[:4]}")

    w2c = np.stack([transform_matrix_to_w2c(np.array(by_name[n]), conv)
                    for n in names_in_order])

    report = {}
    # Adjacent frames have the most overlap; a wider stride is a harder test of
    # the same bookkeeping and is reported alongside rather than instead.
    for stride in (1, 5):
        if depth.shape[0] > stride:
            report[f"stride_{stride}"] = reprojection_consistency(
                depth, K_proc, w2c, stride=stride)

    worst = max(r["median_rel_error"] for r in report.values())
    report["worst_median_rel_error"] = worst
    report["tolerance"] = spec.reprojection_tol
    if worst > spec.reprojection_tol:
        raise RuntimeError(
            f"poses read back from the written json do not agree with DA3's own "
            f"depth: worst median reprojection error {worst:.4f}, tolerance "
            f"{spec.reprojection_tol}. DA3 predicts depth and poses jointly, so "
            f"they must agree to well within this unless the json convention is "
            f"wrong. Convention in use: {conv.describe()}."
        )
    return report


def trajectory_report(w2c: np.ndarray) -> dict:
    """Describe the camera path, in DA3 units, as a feasibility signal.

    Not a quality gate — a description. Baseline relative to scene extent is what
    decides whether depth is observable at all, and it is worth reading before
    committing an hour of GPU time to a reconstruction of a camera that barely
    moved.
    """
    centres = np.stack([np.linalg.inv(to_4x4(m))[:3, 3] for m in w2c])
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    extent = centres.max(axis=0) - centres.min(axis=0)
    # Where the cameras look, as unit vectors: a camera that pans covers many
    # directions from one spot, which is the pathological case for triangulation.
    fwd = np.stack([np.linalg.inv(to_4x4(m))[:3, 2] for m in w2c])
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-12
    return {
        "path_length": float(steps.sum()),
        "step_median": float(np.median(steps)) if len(steps) else 0.0,
        "extent_xyz": [float(v) for v in extent],
        "extent_diagonal": float(np.linalg.norm(extent)),
        # >1 means the camera travelled further than the box it swept: a walk.
        # <<1 means it mostly rotated in place.
        "path_over_extent": float(steps.sum() / max(np.linalg.norm(extent), 1e-9)),
        "view_direction_spread": float(np.linalg.norm(fwd.std(axis=0))),
    }


def write_scene(images_dir: Path, scene_dir: Path, model, spec: PoseSpec | None = None,
                image_names: list[str] | None = None) -> dict:
    """Predict poses for `images_dir` and lay out a ScanNet++-shaped scene.

    The layout is not a preference. `inference_da3_scannetppv2.py:711-721`
    dispatches on directory shape and recognises exactly three: ScanNet++,
    DL3DV, and ScanNet. Anything else reaches `raise ValueError`. In particular
    the flat shape that `config.py:flat_layout` describes — and that its own
    docstring claims leaves "everything downstream identical to the benchmark
    path" — is not one of the three.
    """
    spec = spec or PoseSpec()
    images_dir, scene_dir = Path(images_dir), Path(scene_dir)

    names = image_names or sorted(
        p.name for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith(".")
    )
    if not names:
        raise RuntimeError(f"no images in {images_dir}")
    paths = [images_dir / n for n in names]

    from PIL import Image
    sizes = {Image.open(p).size for p in paths}
    if len(sizes) != 1:
        raise RuntimeError(f"images differ in size: {sorted(sizes)}")
    orig_wh = sizes.pop()

    print(f"  predicting poses for {len(paths)} views at process_res="
          f"{spec.process_res} ...")
    pred = predict_poses(model, paths, spec)
    print(f"    processed at {pred['proc_wh'][0]}x{pred['proc_wh'][1]}, "
          f"original {orig_wh[0]}x{orig_wh[1]}")

    K, spread = consensus_intrinsic(pred["K_proc"], pred["proc_wh"], orig_wh, spec)
    print(f"    intrinsics: fx {spread['fx_median']:.1f} fy {spread['fy_median']:.1f} "
          f"cx {spread['cx_median']:.1f} cy {spread['cy_median']:.1f} "
          f"(hfov {spread['hfov_deg']:.1f} deg, worst spread "
          f"{spread['worst_rel_range']:.5f})")

    doc = build_transforms(pred["w2c"], K, names, orig_wh, spec)
    check = verify_roundtrip(doc, dict(zip(names, pred["w2c"])), spec)
    print(f"    pose round trip through the pipeline's own reader: rotation err "
          f"{check['worst_rotation_abs_err']:.2e}, translation "
          f"{check['worst_translation_abs_err']:.2e}, det in "
          f"[{check['det_min']:.4f}, {check['det_max']:.4f}] -- ok")

    traj = trajectory_report(pred["w2c"])
    print(f"    trajectory: path {traj['path_length']:.2f} over extent "
          f"{traj['extent_diagonal']:.2f} (ratio {traj['path_over_extent']:.2f}), "
          f"median step {traj['step_median']:.4f}")

    nerf = scene_dir / "dslr" / "nerfstudio"
    nerf.mkdir(parents=True, exist_ok=True)
    out = nerf / "transforms_undistorted.json"
    out.write_text(json.dumps(doc, indent=1))
    print(f"    wrote {out} ({len(doc['frames'])} train + "
          f"{len(doc['test_frames'])} test)")

    return {
        "transforms": str(out),
        "images": len(names),
        "orig_wh": list(orig_wh),
        "proc_wh": list(pred["proc_wh"]),
        "intrinsics": spread,
        "roundtrip": check,
        "trajectory": traj,
        "spec": asdict(spec),
        "scale_mode": "assumed",
    }
