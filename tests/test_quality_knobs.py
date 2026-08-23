"""Check the derived knobs against the constants they replace.

The strongest available test of a derivation is that it lands in a defensible
place *in the setting the constant was tuned for*, and moves for the right reason
elsewhere. Both directions are checked here, plus the clamps, plus the control
number that says how much geometry the inherited constant would have erased.

Resolution matters to these tests and cannot be faked small: the voxel derivation
is a statement about the world-space size of one pixel, so a synthetic 80-px-wide
camera would correctly produce a coarse mesh and tell us nothing about a real one.
The scenes below therefore run at ScanNet++ DSLR resolution with a matching focal
length.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v3d.geometry.quality import KnobPolicy, depth_statistics, derive  # noqa: E402

# ScanNet++ DSLR, undistorted and resized: the camera the paper's constants were
# tuned against.
DSLR_WH = (1752, 1168)
DSLR_F = 1250.0

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'pass' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def expect_raise(name: str, fn) -> None:
    try:
        fn()
    except Exception as e:
        check(name, True, f"rejected: {type(e).__name__}")
        return
    check(name, False, "accepted input that should have been refused")


def intrinsic(wh=DSLR_WH, f=DSLR_F) -> np.ndarray:
    return np.array([[f, 0, wh[0] / 2], [0, f, wh[1] / 2], [0, 0, 1]], dtype=np.float64)


def synthetic_scene(n_views: int, near: float, far: float, cam_extent: float,
                    seed: int = 0, wh=DSLR_WH):
    """Depth maps and a camera walk with a known depth range and known extent."""
    rng = np.random.default_rng(seed)
    w, h = wh
    # A beta-shaped spread over [near, far] so the p98 sits just inside `far`,
    # the way a real room's depth histogram does: most surfaces mid-range, a thin
    # tail at the far wall.
    depth = (near + (far - near) * rng.beta(2.2, 1.6, size=(n_views, h, w))).astype(np.float32)
    # A sprinkling of the far-tail garbage every monocular predictor emits.
    tail = rng.random(depth.shape) < 0.004
    depth[tail] = far * rng.uniform(1.5, 4.0, size=int(tail.sum()))
    # And genuinely invalid samples, which must not count toward any percentile.
    depth[rng.random(depth.shape) < 0.01] = 0.0

    t = np.linspace(0, 1, n_views)
    centres = np.stack([cam_extent * t, 0.3 * cam_extent * np.sin(3 * t),
                        np.zeros_like(t)], axis=1)
    w2c = []
    for c in centres:
        m = np.eye(4)
        m[:3, 3] = -c            # identity rotation, so inv() puts the centre back
        w2c.append(m)
    return depth, np.stack(w2c)


def main() -> int:
    policy = KnobPolicy()
    K = intrinsic()

    print("=== invalid depth is excluded from the statistics ===")
    d = np.array([[0.0, -1.0, np.nan, np.inf, 2.0, 4.0]], dtype=np.float32)
    st = depth_statistics(d)
    check("only the two positive finite samples survive", st["samples"] == 2,
          f"samples={st['samples']}, min={st['min']}, max={st['max']}")
    expect_raise("all-invalid depth is refused rather than defaulted",
                 lambda: depth_statistics(np.zeros((4, 4), np.float32)))

    print()
    print("=== a ScanNet++-sized bedroom: fusion distance reproduces the paper's 4.0 ===")
    depth, w2c = synthetic_scene(6, near=0.4, far=4.2, cam_extent=2.0, seed=1)
    k = derive(depth, w2c, K, policy=policy)
    ctl = k.evidence["against_the_inherited_constants"]
    vr = k.evidence["voxel_reasoning"]
    check("derived fusion distance lands near the tuned constant",
          3.0 <= k.render_max_depth <= 5.0,
          f"derived {k.render_max_depth:.2f} vs paper 4.0")
    check("the paper's constant erases almost nothing on a bedroom",
          ctl["fraction_of_valid_depth_erased_at_4.0"] < 0.10,
          f"{ctl['fraction_of_valid_depth_erased_at_4.0']*100:.2f}% erased at 4.0")
    check("mask distance keeps the paper's 1.5x ratio to fusion distance",
          np.isclose(k.mask_max_depth / k.render_max_depth, 1.5),
          f"{k.mask_max_depth:.2f} / {k.render_max_depth:.2f}")

    print()
    print("=== voxel: finer than the paper, because the input supports it ===")
    check("the paper's 0.006 sits at a few pixels on this camera",
          1.5 <= vr["paper_0.006_in_pixels"] <= 6.0,
          f"0.006 = {vr['paper_0.006_in_pixels']:.2f} px at typical depth "
          f"{vr['typical_depth']:.2f}")
    check("derived voxel is finer than the paper's, and not absurdly so",
          k.voxel_size < 0.006 and k.voxel_size > 0.006 / 6,
          f"voxel {k.voxel_size:.5f} = {ctl['derived_over_paper_voxel']:.2f}x paper")
    check("the input's resolution is what bound the voxel, not a backstop",
          vr["bound_by"] == "input resolution", vr["bound_by"])
    check("voxel is the requested multiple of the pixel footprint",
          np.isclose(k.voxel_size,
                     policy.pixels_per_voxel * vr["pixel_footprint_at_typical_depth"]),
          f"{policy.pixels_per_voxel} x {vr['pixel_footprint_at_typical_depth']:.6f}")

    print()
    print("=== a factory bay must not come back as 4.0 ===")
    depthf, w2cf = synthetic_scene(6, near=0.8, far=14.0, cam_extent=9.0, seed=2)
    kf = derive(depthf, w2cf, K, policy=policy)
    ctlf = kf.evidence["against_the_inherited_constants"]
    check("derived fusion distance follows the scene, not the constant",
          kf.render_max_depth > 8.0,
          f"derived {kf.render_max_depth:.2f} vs paper 4.0")
    check("the control reports the geometry the constant would have deleted",
          ctlf["fraction_of_valid_depth_erased_at_4.0"] > 0.5,
          f"{ctlf['fraction_of_valid_depth_erased_at_4.0']*100:.1f}% of believable "
          f"depth erased at 4.0")
    check("the derived value erases only the far tail",
          ctlf["fraction_of_valid_depth_erased_at_derived"] < 0.05,
          f"{ctlf['fraction_of_valid_depth_erased_at_derived']*100:.2f}% erased")
    check("a deeper scene coarsens the voxel, because its pixels cover more world",
          kf.voxel_size > k.voxel_size,
          f"factory {kf.voxel_size:.5f} > bedroom {k.voxel_size:.5f}")
    check("the depth filter is on and the cluster cull is no longer 1",
          kf.use_depth_filter and kf.num_cluster > 1,
          f"use_depth_filter={kf.use_depth_filter}, num_cluster={kf.num_cluster}")

    print()
    print("=== a sharper camera earns a finer mesh, with nothing else changed ===")
    k4 = derive(depth, w2c, intrinsic(f=DSLR_F * 4), policy=policy)
    vr4 = k4.evidence["voxel_reasoning"]
    # The proportionality is a property of the derivation, so it is checked on the
    # derived value before the rails act. Checking it on the final voxel would
    # fail whenever a rail legitimately binds -- which is what happens here, and
    # is the code being right rather than wrong.
    check("quadrupling the focal length quarters the voxel the input supports",
          np.isclose(vr4["voxel_at_input_resolution"],
                     vr["voxel_at_input_resolution"] / 4, rtol=1e-9),
          f"{vr4['voxel_at_input_resolution']:.6f} vs "
          f"{vr['voxel_at_input_resolution']:.6f}")
    check("and the memory guard is what stops it going further, and says so",
          k4.voxel_size >= vr4["voxel_at_input_resolution"]
          and "memory guard" in vr4["bound_by"],
          f"final {k4.voxel_size:.6f}, bound_by={vr4['bound_by']}")
    check("fusion distance is unchanged by the camera",
          np.isclose(k4.render_max_depth, k.render_max_depth))

    print()
    print("=== the rails are counts, not lengths, so they cannot contradict ===")
    huge, w_h = synthetic_scene(4, near=50.0, far=900.0, cam_extent=600.0, seed=4)
    kh = derive(huge, w_h, K, policy=policy)
    vrh = kh.evidence["voxel_reasoning"]
    check("a scene 200x larger still respects the memory guard",
          vrh["voxels_across_diagonal"] <= policy.max_voxels_across_diagonal * 1.001,
          f"{vrh['voxels_across_diagonal']:.0f} across diagonal, "
          f"bound_by={vrh['bound_by']}")
    tiny, w_t = synthetic_scene(4, near=0.01, far=0.05, cam_extent=0.02, seed=3)
    kt = derive(tiny, w_t, intrinsic(f=DSLR_F * 20), policy=policy)
    vrt = kt.evidence["voxel_reasoning"]
    check("a scene 100x smaller still respects the memory guard",
          vrt["voxels_across_diagonal"] <= policy.max_voxels_across_diagonal * 1.001,
          f"{vrt['voxels_across_diagonal']:.0f} across diagonal, "
          f"bound_by={vrt['bound_by']}")
    check("every scene stays above the usefulness cap too",
          all(kx.evidence["voxel_reasoning"]["voxels_across_diagonal"]
              >= policy.min_voxels_across_diagonal * 0.999 for kx in (k, kf, kh, kt)),
          str([round(kx.evidence["voxel_reasoning"]["voxels_across_diagonal"])
               for kx in (k, kf, kh, kt)]))
    check("no absolute length survives in the policy",
          not any(n.startswith("voxel_m") for n in vars(policy)),
          f"policy fields: {sorted(vars(policy))}")
    expect_raise("crossed rails are refused rather than silently reordered",
                 lambda: derive(depth, w2c, K, policy=KnobPolicy(
                     max_voxels_across_diagonal=100.0, min_voxels_across_diagonal=200.0)))
    expect_raise("a nonsense focal length is refused, not defaulted",
                 lambda: derive(depth, w2c, intrinsic(f=0.0), policy=policy))

    print()
    print("=== the units are labelled, because they are not metres ===")
    check("assumed scale carries a warning into the evidence",
          k.evidence["scale_warning"] is not None, k.evidence["scale_warning"])
    check("a measured scale carries none",
          derive(depth, w2c, K, policy=policy,
                 scale_mode="measured").evidence["scale_warning"] is None)
    cli = k.as_cli()
    check("stage arguments come out as strings for the runner",
          all(isinstance(v, str) for v in cli.values()), str(cli))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


def test_quality_knobs():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
