"""PGSR / 3DGS training — the stage that actually decides surface quality.

Everything before this produced inputs; this is where the surface is fitted, and
it is the longest stage by an order of magnitude. Two things make this run
different from the paper's, and both are derived from the scene rather than
inherited:

**The schedule.** PGSR samples one image per iteration, so 30,000 iterations is
115 views of each image at the paper's 261 frames and only 50 at 600. Holding it
fixed while the frame count grows quietly reduces per-image supervision, which is
the opposite of what more input should buy. `v3d.geometry.quality.pgsr_schedule`
scales the iteration count and every milestone with the frame count, and ties
`position_lr_max_steps` to it — without which the extra iterations run at a
position learning rate that has already decayed to its floor, so colour and
opacity keep improving while the geometry cannot move.

**The view-pairing distances.** `multi_view_min_dis = 0.01` and
`multi_view_max_dis = 1.5` (`arguments/__init__.py:61-62`) are *lengths in scene
units*, and this scene's units are DA3's, not the paper's. They decide which
neighbouring views a frame is asked to be geometrically consistent with, and
multi-view consistency is the mechanism the paper credits for a clean surface. Too
small a maximum and no pair qualifies, so the regulariser silently does nothing;
too large and pairs share no overlap, so it enforces agreement between views that
see different things. Neither failure raises, and neither is visible in the loss.

Both are printed against the paper's values, because a derived number with nothing
to compare it against is just a different guess.
"""

from __future__ import annotations

import json
import os
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


def main() -> None:
    import torch

    from v3d.config import PipelineConfig, detect_layout
    from v3d import paper_runner as pr

    cfg = PipelineConfig()
    layout = detect_layout(SCENE, cfg.data_root)
    if not layout.point_cloud.exists():
        raise SystemExit(f"run the DA3 stage first: {layout.point_cloud} is missing")

    print("=" * 78)
    print("PGSR / 3DGS training")
    print("=" * 78)
    print(f"  scene        {layout.scene_id}")
    print(f"  frames       {len(list(layout.images_dir.glob('*.JPG')))}")
    print(f"  init cloud   {layout.point_cloud} "
          f"({layout.point_cloud.stat().st_size / 1e6:.1f} MB)")

    print("\n--- patching the paper's code ---")
    for p in pr.apply_offline_patches(cfg):
        print(f"  {p.describe()}")

    # PGSR/scene/gaussian_model.py:23 does `from pytorch3d.transforms import
    # quaternion_to_matrix`, and that is the *only* pytorch3d symbol the whole
    # paper repo uses. Building pytorch3d against torch 2.10 / sm_120 for one
    # function of pure tensor arithmetic is a large risk for no benefit.
    #
    # The convention is the thing worth checking rather than assuming: pytorch3d
    # takes quaternions real-part-FIRST (w, x, y, z) while scipy and most of the
    # graphics world put it last. Getting that backwards yields rotations that look
    # plausible and are wrong — every gaussian's orientation, silently. So verify()
    # is run first, and the swapped ordering is asserted to be *large*, because a
    # check that cannot fail proves nothing.
    print("\n--- pytorch3d shim ---")
    from v3d.geometry import pytorch3d_shim

    v = pytorch3d_shim.verify()
    print(f"  vs scipy: correct order {v['max_abs_diff_correct_order']:.2e}, "
          f"swapped order {v['max_abs_diff_swapped_order']:.4f} (control), "
          f"orthonormality {v['orthonormal_error']:.2e}")
    if v["max_abs_diff_correct_order"] > 1e-6:
        raise SystemExit(
            f"the shim disagrees with scipy by {v['max_abs_diff_correct_order']:.2e}; "
            f"every gaussian orientation would be wrong"
        )
    if v["max_abs_diff_swapped_order"] < 0.1:
        raise SystemExit(
            f"the swapped-ordering control only differs by "
            f"{v['max_abs_diff_swapped_order']:.4f}, so this test cannot tell the two "
            f"conventions apart and its pass means nothing"
        )
    pkg = pytorch3d_shim.install("/kaggle/working/src")
    print(f"  installed {pkg} (re-exports the verified function)")

    # Depth supervision is a term PGSR uses (train.py:226-236) and loses *silently*
    # when it cannot find the files: cameras.py prints one warning per camera and
    # leaves depth_map None, and training then runs to completion with a falling
    # loss and one fewer supervision term. That happened on the first attempt here —
    # 600 warnings scrolled past during camera loading and nothing else reported it.
    #
    # So resolve the path with the paper's own (now patched) function and require
    # the file to exist, before an hour of GPU time rather than after.
    print("\n--- depth supervision reachability ---")
    sys.path.insert(0, str(cfg.paper_root / "PGSR"))
    from scene.cameras import infer_depth_path  # noqa: E402

    sample = sorted(layout.images_dir.glob("*.JPG"))[:3]
    resolved = [(p.name, infer_depth_path(str(p))) for p in sample]
    missing = [(n, d) for n, d in resolved if not d or not Path(d).exists()]
    for n, d in resolved:
        print(f"  {n} -> {d}  {'ok' if d and Path(d).exists() else 'MISSING'}")
    if missing:
        raise SystemExit(
            f"PGSR cannot reach the DA3 depth maps for {len(missing)}/{len(resolved)} "
            f"sampled frames. Training would run to completion with the depth L1 term "
            f"disabled and report nothing but a falling photometric loss — which is "
            f"the whole resolution budget spent for no effect on the surface. "
            f"Expected them beside the scene at {layout.depth_dir}."
        )
    n_depth = len(list(layout.depth_dir.glob("*.npy")))
    n_img = len(list(layout.images_dir.glob("*.JPG")))
    print(f"  {n_depth} depth maps for {n_img} frames")
    if n_depth < n_img:
        raise SystemExit(f"only {n_depth} depth maps for {n_img} frames")

    print("\n--- deriving the schedule and the mesh knobs from this scene ---")
    knobs, schedule = pr.derive_knobs(layout, cfg)

    model_dir = cfg.gs_root / layout.scene_id
    done = model_dir / "point_cloud" / f"iteration_{schedule['iterations']}"
    if done.is_dir():
        print(f"\n--- already trained to {schedule['iterations']:,}: {done} ---")
        print("    delete the model directory to retrain")
    else:
        print(f"\n--- training to {schedule['iterations']:,} iterations ---")
        print(f"    every point in the {layout.point_cloud.stat().st_size / 1e6:.0f} MB "
              f"cloud becomes an initial gaussian: gaussian_model.py:170-189 does no "
              f"downsampling, so this run starts from 8 M rather than the paper's 4 M")
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        rc = pr.stage_pgsr(layout, cfg, schedule=schedule)
        mins = (time.time() - t) / 60
        log = cfg.log_root / f"{layout.scene_id}_pgsr.log"
        print(f"    exit {rc} after {mins:.1f} min; log {log}")
        if rc != 0:
            tail = log.read_text().splitlines()[-50:] if log.exists() else ["<no log>"]
            raise SystemExit("stage_pgsr failed:\n" + "\n".join(tail))

    # --- postconditions -------------------------------------------------------
    print("\n--- checking what it produced ---")
    iters = sorted(
        (int(p.name.split("_")[1]), p) for p in (model_dir / "point_cloud").glob("iteration_*")
    ) if (model_dir / "point_cloud").is_dir() else []
    if not iters:
        raise SystemExit(f"no gaussians under {model_dir / 'point_cloud'}")
    last_it, last_dir = iters[-1]
    ply = last_dir / "point_cloud.ply"
    print(f"  checkpoints      {[i for i, _ in iters]}")
    print(f"  final            iteration {last_it:,}, {ply.stat().st_size / 1e6:.1f} MB")
    if last_it != schedule["iterations"]:
        raise SystemExit(
            f"the newest checkpoint is iteration {last_it:,} but the schedule asked "
            f"for {schedule['iterations']:,}. render.py loads the newest with "
            f"--iteration -1, so the mesh would be built from a partial run and "
            f"nothing downstream would say so."
        )

    # Gaussian count from the PLY header — what a consumer actually sees.
    with ply.open("rb") as fh:
        header = b""
        while b"end_header" not in header and len(header) < 8192:
            block = fh.read(256)
            if not block:
                break
            header += block
    n_gauss = next((int(line.split()[-1])
                    for line in header.decode("ascii", "replace").splitlines()
                    if line.startswith("element vertex")), None)
    print(f"  gaussians        {n_gauss:,}" if n_gauss else "  gaussians        unknown")

    # PSNR from the training log, which is the only place it exists. Reported, not
    # gated: PSNR rewards texture fidelity and is nearly blind to a wall bowed by
    # 10 cm, so it cannot stand in for surface accuracy.
    log = cfg.log_root / f"{layout.scene_id}_pgsr.log"
    psnr_lines = []
    if log.exists():
        for line in log.read_text().splitlines():
            if "PSNR" in line or "L1" in line:
                psnr_lines.append(line.strip())
    for line in psnr_lines[-6:]:
        print(f"  eval             {line[:110]}")

    summary = {
        "scene": SCENE,
        "iterations": schedule["iterations"],
        "checkpoints": [i for i, _ in iters],
        "final_iteration": last_it,
        "gaussians": n_gauss,
        "gaussian_ply_mb": round(ply.stat().st_size / 1e6, 1),
        "schedule": schedule,
        "knobs": {k: v for k, v in vars(knobs).items() if k != "evidence"},
        "knob_evidence": knobs.evidence,
        "eval_lines": psnr_lines[-12:],
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    out = WORK / f"pgsr_summary_{SCENE}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  -> {out}")
    print("\nPGSR_OK")


main()
