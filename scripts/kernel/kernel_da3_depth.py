"""DA3 depth and the point cloud, at the highest resolution this GPU measured.

Runs the paper's `inference_da3_scannetppv2.py` unmodified except for the
substitutions `v3d.paper_runner.apply_offline_patches` makes, all of which turn a
hardcoded literal into an environment read whose default is that literal. So an
unset environment reproduces the paper exactly, and what this run actually used is
printed and recorded rather than remembered.

The configuration is not chosen here. It is read from the two reports the
measurement scripts wrote:

  da3_resolution_<scene>.json   the highest process_res that fit, and at what
                                chunk size, both measured rather than predicted
  chunk_scale_<scene>.json      whether independently-inferred chunks agree on
                                depth scale, which is what makes chunking safe

If the second says they disagree, this refuses the chunked configuration and
falls back, because a point cloud whose regions carry different scales is worse
than a coarser one that is consistent — and PGSR reads these same depth maps as
training supervision (`PGSR/scene/cameras.py:89`), so the inconsistency would be
baked into the surface where nothing later looks for it.

Then it checks its own output: depth finite and in range, the point cloud's
count against what was asked for, and a depth video so the result can be seen
rather than inferred from statistics.
"""

from __future__ import annotations

import json
import os
import subprocess
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
RES_REPORT = Path(os.environ.get("V3D_RES_REPORT", WORK / f"da3_resolution_{SCENE}.json"))
CHUNK_REPORT = Path(os.environ.get("V3D_CHUNK_REPORT", WORK / f"chunk_scale_{SCENE}.json"))
# How many points the delivered cloud should carry. The user's ask is double the
# paper's 4 M; see the note in paper_runner on why those are measurements rather
# than interpolation.
TARGET_POINTS = int(os.environ.get("V3D_TARGET_POINTS", "8000000"))
SCALE_TOL_PCT = float(os.environ.get("V3D_CHUNK_SCALE_TOL_PCT", "1.0"))

import numpy as np  # noqa: E402


def scale_error_pct(row: dict) -> float:
    """How far a chunk's *scale* is from its neighbour's, in percent.

    The right statistic is the deviation of the **median** ratio from 1.0, not the
    worst per-frame ratio. `prediction.depth /= scale` (`api.py:362`) applies one
    scalar to a whole inference call, so a chunk can only be wrong by one number;
    the frame-to-frame spread around that number is DA3's own prediction noise,
    which is present within a single chunk too and is not something chunking
    caused. Gating on the worst frame would therefore reject a configuration for
    noise it did not introduce — and on the measured data it does exactly that:
    200v@1008 shows a worst frame of 1.001% around a median of 0.11%.
    """
    return abs(row["median_ratio"] - 1.0) * 100.0


def pick_configuration() -> dict:
    """Read the measured reports and decide process_res and chunk size."""
    if not RES_REPORT.exists():
        raise SystemExit(f"no resolution report at {RES_REPORT}; "
                         f"run scripts/kernel/kernel_da3_maxres.py first")
    res = json.loads(RES_REPORT.read_text())
    chosen = res["chosen"]
    process_res = chosen["process_res"]
    method = chosen["process_res_method"]
    chunk = chosen["views_per_chunk"]
    frames = res["frames_available"]
    reason = "the highest resolution that fit, measured"

    if CHUNK_REPORT.exists():
        rep = json.loads(CHUNK_REPORT.read_text())
        rows = {r["label"]: r for r in rep.get("results", [])}
        for lbl, r in sorted(rows.items()):
            print(f"  chunk-scale {lbl:12} scale error {scale_error_pct(r):.3f}%  "
                  f"(per-frame spread {r['spread_pct']:.2f}%, "
                  f"{r['frames_compared']} shared frames)")
        native = rows.get(f"{chunk}v@{process_res}")
        if native is None:
            print(f"  note: no chunk-scale row for {chunk}v@{process_res}; "
                  f"cross-chunk scale is UNVERIFIED for this exact configuration")
        elif scale_error_pct(native) > SCALE_TOL_PCT:
            # Candidates that pass, best resolution first — a fallback should give
            # up as little detail as it has to.
            passing = sorted(
                (r for lbl, r in rows.items()
                 if lbl != f"{chunk}v@{process_res}" and scale_error_pct(r) <= SCALE_TOL_PCT),
                key=lambda r: -int(r["label"].split("@")[1]),
            )
            if not passing:
                raise SystemExit(
                    f"chunks at {chunk}v@{process_res} disagree on scale by "
                    f"{scale_error_pct(native):.2f}% (tolerance {SCALE_TOL_PCT}%), and "
                    f"no tested configuration agrees. Running anyway would produce a "
                    f"cloud whose regions carry different scales, and PGSR would "
                    f"train on it. Re-test with a larger chunk at a lower process_res."
                )
            best = passing[0]
            lbl = best["label"]
            chunk, process_res = (int(x) for x in lbl.replace("v@", " ").split())
            reason = (f"native chunking disagreed on scale by "
                      f"{scale_error_pct(native):.2f}%; this agrees to "
                      f"{scale_error_pct(best):.2f}%")
        else:
            reason = (f"the highest resolution that fit; chunks agree on scale to "
                      f"{scale_error_pct(native):.2f}%")
    else:
        print(f"  note: no chunk-scale report at {CHUNK_REPORT}; cross-chunk depth "
              f"scale is UNVERIFIED")

    return {"process_res": process_res, "method": method, "chunk": chunk,
            "frames": frames, "reason": reason}


def depth_video(depth_dir: Path, images_dir: Path, out: Path, max_frames: int = 240) -> Path | None:
    """RGB beside its depth map, on one shared colour scale across the whole clip.

    A per-frame scale would make every frame look equally well-resolved and hide
    exactly what this video is for: whether depth is consistent as the camera
    moves. The scale is therefore fixed from a global percentile range, and stated
    in the overlay so the picture carries its own units.
    """
    import cv2

    npys = sorted(depth_dir.glob("*.npy"))
    if not npys:
        return None
    if len(npys) > max_frames:
        idx = np.linspace(0, len(npys) - 1, max_frames).round().astype(int)
        npys = [npys[i] for i in sorted(set(idx.tolist()))]

    # One pass to fix the range, over a subsample: reading 600 full-resolution
    # maps twice to set a colour scale is not worth the minutes.
    probe = npys[:: max(1, len(npys) // 24)]
    vals = np.concatenate([np.load(p)[::8, ::8].reshape(-1) for p in probe])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    lo, hi = np.percentile(vals, [2, 98])
    print(f"    depth video colour scale {lo:.3f}-{hi:.3f} (DA3 units, 2nd-98th pct)")

    frames_dir = out.parent / f"_{out.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()

    written = 0
    for i, p in enumerate(npys):
        img_path = next((images_dir / f"{p.stem}{ext}"
                         for ext in (".JPG", ".jpg", ".png")
                         if (images_dir / f"{p.stem}{ext}").exists()), None)
        if img_path is None:
            continue
        rgb = cv2.imread(str(img_path))
        d = np.load(p)
        norm = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
        # Invalid depth is painted black rather than mapped to the near end of the
        # colour map, so a hole reads as a hole instead of as a close surface.
        colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colour[~(np.isfinite(d) & (d > 0))] = 0
        if colour.shape[:2] != rgb.shape[:2]:
            colour = cv2.resize(colour, (rgb.shape[1], rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        pair = np.hstack([rgb, colour])
        pair = cv2.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2))
        cv2.putText(pair, f"{p.stem}   depth {lo:.2f}-{hi:.2f} DA3 units (scale assumed, not metres)",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(frames_dir / f"{written:05d}.png"), pair)
        written += 1

    if not written:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "15",
           "-i", str(frames_dir / "%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    ffmpeg failed: {r.stderr[-400:]}")
        return None
    return out


def main() -> None:
    from v3d.config import PipelineConfig, detect_layout
    from v3d import paper_runner as pr

    cfg = PipelineConfig()
    layout = detect_layout(SCENE, cfg.data_root)
    problems = layout.missing()
    if problems:
        raise SystemExit(f"layout incomplete: {problems}")

    print("=" * 78)
    print("DA3 depth and point cloud")
    print("=" * 78)
    conf = pick_configuration()
    os.environ["V3D_PROCESS_RES"] = str(conf["process_res"])
    os.environ["V3D_PROCESS_RES_METHOD"] = conf["method"]
    os.environ["V3D_MAX_CHUNK_SIZE"] = str(conf["chunk"])
    os.environ["V3D_TARGET_POINTS"] = str(TARGET_POINTS)

    n_chunks = -(-conf["frames"] // conf["chunk"])
    print(f"  process_res       {conf['process_res']} ({conf['method']})")
    print(f"  views per chunk   {conf['chunk']}  -> {n_chunks} chunk(s) for "
          f"{conf['frames']} frames")
    print(f"  target points     {TARGET_POINTS:,}  (the paper hardcodes 4,000,000)")
    print(f"  why              {conf['reason']}")

    print("\n--- patching the paper's code ---")
    for p in pr.apply_offline_patches(cfg):
        print(f"  {p.describe()}")

    cloud = layout.point_cloud
    if cloud.exists():
        print(f"\n--- depth + cloud already present: {cloud} "
              f"({cloud.stat().st_size / 1e6:.1f} MB) ---")
        print(f"    delete it to re-run (the paper's script skips a scene whose "
              f"pointcloud_da3.ply exists)")
    else:
        print(f"\n--- running stage_da3 ({n_chunks} chunks) ---")
        t = time.time()
        rc = pr.stage_da3(layout, cfg)
        log = cfg.log_root / f"{layout.scene_id}_da3.log"
        print(f"  exit {rc} in {time.time() - t:.0f}s; log {log}")
        if rc != 0:
            tail = log.read_text().splitlines()[-40:] if log.exists() else ["<no log>"]
            raise SystemExit("stage_da3 failed:\n" + "\n".join(tail))

    # --- postconditions, computed rather than assumed -------------------------
    print("\n--- checking what it produced ---")
    npys = sorted(layout.depth_dir.glob("*.npy"))
    n_frames = len(list(layout.images_dir.glob("*.JPG")))
    print(f"  depth maps        {len(npys)} for {n_frames} frames")
    if len(npys) != n_frames:
        raise SystemExit(
            f"{len(npys)} depth maps for {n_frames} frames. Every later stage "
            f"indexes depth by frame, so a gap is silent corruption, not a "
            f"partial result."
        )

    probe = [npys[i] for i in
             np.linspace(0, len(npys) - 1, min(40, len(npys))).round().astype(int)]
    finite_frac, lo, hi = [], np.inf, -np.inf
    shapes = set()
    for p in probe:
        d = np.load(p)
        shapes.add(d.shape)
        good = np.isfinite(d) & (d > 0)
        finite_frac.append(float(good.mean()))
        if good.any():
            lo, hi = min(lo, float(d[good].min())), max(hi, float(d[good].max()))
    print(f"  depth shape       {sorted(shapes)}")
    print(f"  finite & positive {np.mean(finite_frac) * 100:.2f}% "
          f"(worst frame {min(finite_frac) * 100:.2f}%)")
    print(f"  range             {lo:.4f} - {hi:.4f} DA3 units (scale is 'assumed')")
    if len(shapes) != 1:
        raise SystemExit(f"depth maps differ in shape: {sorted(shapes)}")
    if np.mean(finite_frac) < 0.5:
        raise SystemExit(
            f"only {np.mean(finite_frac) * 100:.1f}% of depth is finite and "
            f"positive; the reconstruction would be built on holes"
        )

    # Point count read from the PLY header, which is what a consumer sees.
    with cloud.open("rb") as fh:
        header = b""
        while b"end_header" not in header:
            chunk = fh.read(256)
            if not chunk:
                break
            header += chunk
    n_pts = next((int(line.split()[-1])
                  for line in header.decode("ascii", "replace").splitlines()
                  if line.startswith("element vertex")), None)
    print(f"  point cloud       {n_pts:,} points, {cloud.stat().st_size / 1e6:.1f} MB")
    print(f"                    -> {cloud}")
    if n_pts is None or n_pts < 0.5 * TARGET_POINTS:
        raise SystemExit(
            f"asked for {TARGET_POINTS:,} points and got {n_pts}. The patch that "
            f"makes target_num_points configurable may not have applied, in which "
            f"case this is the paper's 4,000,000 being reported as yours."
        )

    # --- the video ------------------------------------------------------------
    print("\n--- visualisation ---")
    vid = depth_video(layout.depth_dir, layout.images_dir,
                      WORK / "videos" / f"01_depth_{SCENE}.mp4")
    if vid:
        print(f"  {vid}  ({vid.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  depth video not produced")

    summary = {
        "scene": SCENE,
        "configuration": conf,
        "target_points": TARGET_POINTS,
        "depth_maps": len(npys),
        "depth_shape": [list(s) for s in shapes],
        "depth_finite_fraction_mean": float(np.mean(finite_frac)),
        "depth_finite_fraction_min": float(min(finite_frac)),
        "depth_range": [lo, hi],
        "point_cloud": str(cloud),
        "points": n_pts,
        "point_cloud_mb": round(cloud.stat().st_size / 1e6, 1),
        "scale_mode": "assumed",
        "video": str(vid) if vid else None,
    }
    out = WORK / f"da3_summary_{SCENE}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  -> {out}")
    print("\nDA3_OK")


main()
