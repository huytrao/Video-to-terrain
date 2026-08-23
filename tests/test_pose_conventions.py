"""Prove the pose conventions, including that the checker catches the mistakes.

A test that only exercises the correct path is the pattern this project has been
bitten by: a check that cannot fail carries no information. So every positive
case here is paired with a negative control — the specific wrong thing someone
would plausibly write instead — and the test asserts the verifier rejects it.

Runs on CPU in under a second. No GPU, no model, no kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v3d.ingest.poses import (  # noqa: E402
    PoseSpec,
    build_transforms,
    consensus_intrinsic,
    rescale_intrinsic,
    transform_matrix_to_w2c,
    verify_roundtrip,
    w2c_to_transform_matrix,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'pass' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def expect_raise(name: str, fn) -> None:
    """The negative controls. A verifier that accepts bad input is decoration."""
    try:
        fn()
    except Exception as e:
        check(name, True, f"rejected: {type(e).__name__}")
        return
    check(name, False, "accepted input that should have been refused")


def random_w2c(n: int, seed: int = 0) -> np.ndarray:
    """n plausible world-to-camera matrices with proper rotations."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        # A random proper rotation via QR, sign-fixed so det = +1.
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        m = np.eye(4)
        m[:3, :3] = q
        m[:3, 3] = rng.normal(scale=2.0, size=3)
        out.append(m)
    return np.stack(out)


def main() -> int:
    spec = PoseSpec()
    w2c = random_w2c(12, seed=7)
    names = [f"frame_{i:05d}.JPG" for i in range(len(w2c))]
    orig_wh = (1920, 1080)
    K = np.array([[1400.0, 0, 960.0], [0, 1400.0, 540.0], [0, 0, 1]])

    print("=== the axis flip is its own inverse ===")
    for i, m in enumerate(w2c):
        back = transform_matrix_to_w2c(w2c_to_transform_matrix(m))
        if not np.allclose(back, m, atol=1e-9):
            check(f"round trip frame {i}", False, f"max err {np.abs(back-m).max():.2e}")
            break
    else:
        check("w2c -> transform_matrix -> w2c is exact for 12 random poses", True)

    print()
    print("=== the written json survives the pipeline's own reader ===")
    doc = build_transforms(w2c, K, names, orig_wh, spec)
    by_name = dict(zip(names, w2c))
    report = verify_roundtrip(doc, by_name, spec)
    check("verify_roundtrip accepts a correctly written scene", True,
          f"rot {report['worst_rotation_abs_err']:.1e} "
          f"trans {report['worst_translation_abs_err']:.1e}")
    check("both frames and test_frames are populated",
          len(doc["frames"]) > 0 and len(doc["test_frames"]) > 0,
          f"{len(doc['frames'])} train + {len(doc['test_frames'])} test")
    check("every frame appears exactly once across the two keys",
          sorted(e["file_path"] for e in doc["frames"] + doc["test_frames"]) == sorted(names))
    check("scene-level intrinsic fields are present and match K",
          (doc["fl_x"], doc["fl_y"], doc["cx"], doc["cy"]) == (1400.0, 1400.0, 960.0, 540.0))
    check("w/h are the original image size, not the processed one",
          (doc["w"], doc["h"]) == orig_wh)

    print()
    print("=== negative controls: each mistake someone would actually make ===")

    def omit_flip():
        bad = build_transforms(w2c, K, names, orig_wh, spec)
        for e in bad["frames"] + bad["test_frames"]:
            m = np.array(e["transform_matrix"])
            m[:3, 1:3] *= -1              # undo the flip -> plain OpenCV c2w
            e["transform_matrix"] = m.tolist()
        verify_roundtrip(bad, by_name, spec)
    expect_raise("writing c2w without the OpenGL flip is caught", omit_flip)

    def wrong_columns():
        bad = build_transforms(w2c, K, names, orig_wh, spec)
        for e in bad["frames"] + bad["test_frames"]:
            m = np.array(e["transform_matrix"])
            m[:3, 1:3] *= -1
            m[:3, 0:2] *= -1              # flipped columns 0:2 instead of 1:3
            e["transform_matrix"] = m.tolist()
        verify_roundtrip(bad, by_name, spec)
    expect_raise("flipping the wrong two columns is caught", wrong_columns)

    def transpose_not_inverse():
        bad = build_transforms(w2c, K, names, orig_wh, spec)
        for e, m in zip(bad["frames"] + bad["test_frames"], w2c):
            c2w = m.T.copy()              # transpose of the full 4x4, not its inverse
            c2w[:3, 1:3] *= -1
            e["transform_matrix"] = c2w.tolist()
        verify_roundtrip(bad, by_name, spec)
    expect_raise("transposing instead of inverting is caught", transpose_not_inverse)

    def wrote_w2c():
        bad = build_transforms(w2c, K, names, orig_wh, spec)
        for e, m in zip(bad["frames"] + bad["test_frames"], w2c):
            c2w = m.copy()                # forgot to invert at all
            c2w[:3, 1:3] *= -1
            e["transform_matrix"] = c2w.tolist()
        verify_roundtrip(bad, by_name, spec)
    expect_raise("writing w2c where c2w was meant is caught", wrote_w2c)

    def mirrored():
        bad = build_transforms(w2c, K, names, orig_wh, spec)
        for e in bad["frames"] + bad["test_frames"]:
            m = np.array(e["transform_matrix"])
            m[:3, 0] *= -1                # a reflection: det becomes -1
            e["transform_matrix"] = m.tolist()
        verify_roundtrip(bad, by_name, spec)
    expect_raise("a mirrored scene is caught by the determinant check", mirrored)

    def dropped_frame():
        bad = build_transforms(w2c, K, names, orig_wh, spec)
        bad["frames"].pop()
        verify_roundtrip(bad, by_name, spec)
    expect_raise("a lost frame is caught by the coverage count", dropped_frame)

    print()
    print("=== intrinsic rescaling is the exact inverse of DA3's _resize_ixt ===")
    proc_wh = (504, 284)
    down = rescale_intrinsic(K, orig_wh, proc_wh)     # what _resize_ixt does
    up = rescale_intrinsic(down, proc_wh, orig_wh)    # what we do to undo it
    check("rescale_intrinsic(orig->proc->orig) is identity", np.allclose(up, K, atol=1e-9),
          f"max err {np.abs(up - K).max():.2e}")
    check("fx scales with width, fy with height",
          np.isclose(down[0, 0], K[0, 0] * 504 / 1920) and
          np.isclose(down[1, 1], K[1, 1] * 284 / 1080))

    print()
    print("=== scatter is pooled; a real zoom is refused ===")
    rng = np.random.default_rng(11)
    n = 60
    # A fixed camera whose focal length is estimated independently per frame:
    # a few percent of scatter, no trend. This is what DA3 actually produced on
    # the factory clip (fx range 2.0%, fy 4.0%, principal point pinned exactly).
    noisy = np.stack([down.copy() for _ in range(n)])
    noisy[:, 0, 0] *= 1 + rng.normal(0, 0.012, n)
    noisy[:, 1, 1] *= 1 + rng.normal(0, 0.020, n)
    Kc, spread = consensus_intrinsic(noisy, proc_wh, orig_wh, spec)
    check("a noisy but fixed camera is pooled, not refused",
          spread["pooled"] and spread["worst_rel_range"] > spec.max_focal_drift,
          f"range {spread['worst_rel_range']:.4f} pooled; "
          f"fx drift {spread['fx_structure']['rel_drift']:+.4f} = "
          f"{spread['fx_structure']['drift_over_noise']:.1f}x noise")
    check("the pooled focal length is the median, rescaled to the original size",
          np.isclose(Kc[0, 0], np.median(noisy[:, 0, 0]) * orig_wh[0] / proc_wh[0]),
          f"{Kc[0,0]:.1f} vs {np.median(noisy[:,0,0]) * orig_wh[0] / proc_wh[0]:.1f}")
    check("pooling records why it was allowed",
          "drift" in spread["pooling_justification"], spread["pooling_justification"])

    def ramp_zoom():
        bad = np.stack([down.copy() for _ in range(n)])
        # A smooth 25% zoom across the clip, plus the same scatter as above, so
        # the only thing distinguishing it is the trend.
        ramp = 1 + np.linspace(0, 0.25, n)
        bad[:, 0, 0] *= ramp * (1 + rng.normal(0, 0.012, n))
        bad[:, 1, 1] *= ramp * (1 + rng.normal(0, 0.012, n))
        consensus_intrinsic(bad, proc_wh, orig_wh, spec)
    expect_raise("a gradual zoom is caught by its trend, not its spread", ramp_zoom)

    def step_zoom():
        bad = np.stack([down.copy() for _ in range(n)])
        bad[n // 2:, 0, 0] *= 1.30
        bad[n // 2:, 1, 1] *= 1.30
        consensus_intrinsic(bad, proc_wh, orig_wh, spec)
    expect_raise("a focal length that steps mid-scene is refused", step_zoom)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed, including every negative control")
    return 0


def test_pose_conventions():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
