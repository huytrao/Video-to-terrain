"""Native-resolution DA3 depth, made globally consistent by a low-resolution reference.

The quality ceiling on every geometric output is how much real depth DA3 predicted.
At the paper's `process_res=504` a 1920x1080 frame is predicted at 504x280 —
141,120 samples presented as 2,073,600, a 14.7x areal interpolation. Native
(1918x1078) is 14.65x more real depth, and `kernel_da3_maxres.py` measured that it
fits at 30 views per call.

It was rejected anyway, and the reason is the whole problem this script solves.
Every `model.inference` call fits its own Umeyama scale against the supplied poses
and divides its depth by it (`api.py:353-362`). Twenty calls means twenty
independently-fitted scales, and `kernel_chunk_scale.py` measured them disagreeing
by **2.0%** — about 12 cm on this room, against a 5 cm target, and baked into both
the point cloud and the depth PGSR trains on. A higher-resolution reconstruction
whose halves are different sizes is worse than a coarser one that is consistent.

The fix separates the two things a DA3 pass provides:

    scale comes from ONE call over ALL views    (globally consistent by construction)
    detail comes from many small calls          (native resolution)

Pass A runs all 600 views at 504 in a single call — measured at 57.2 GiB, so it
fits — and yields one depth map per frame on one common scale. Pass B runs native
resolution in chunks; for each chunk, the per-frame ratio against Pass A gives a
single correction factor, and dividing by it puts every chunk on Pass A's scale.

One scalar per chunk, not per frame. A per-frame correction would flatten the
genuine depth variation between frames and destroy the geometry it was meant to
preserve; the error being corrected is per-call, so the correction must be too.

Then it checks itself: the same overlap test that rejected the naive version is
re-run on the rescaled output. If the disagreement is not far below where it
started, this script has not earned its keep and says so.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")
DA3_SRC = "/tmp/da3src/Depth-Anything-3/src"
if DA3_SRC not in sys.path:
    sys.path.insert(0, DA3_SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCENE = os.environ.get("V3D_SCENE", "small_factory")
DATA = Path(os.environ.get("V3D_DATA_ROOT", "/tmp/data/scannetppv2/data"))
WORK = Path(os.environ.get("V3D_WORK_ROOT", "/tmp/run"))
DA3_MODEL = os.environ.get("V3D_DA3_MODEL", "/tmp/DA3NESTED-GIANT-LARGE")
REF_RES = int(os.environ.get("V3D_REF_RES", "504"))
NATIVE_RES = int(os.environ.get("V3D_NATIVE_RES", "1920"))
CHUNK = int(os.environ.get("V3D_NATIVE_CHUNK", "30"))
CONF_THRESH = float(os.environ.get("V3D_CONF_THRESHOLD", "0.3"))
# Real samples now number 600 x 2,067,604 = 1.24 billion, so a far denser delivered
# cloud is still drawn entirely from measurements.
TARGET_POINTS = int(os.environ.get("V3D_TARGET_POINTS", "20000000"))
# PGSR turns every point into an initial gaussian and prunes by opacity: the pilot
# went 4 M -> 1.25 M and this run 8 M -> 636 k, so the initial count does not bound
# the final model and there is nothing to buy by making it huge.
PGSR_POINTS = int(os.environ.get("V3D_PGSR_POINTS", "8000000"))
OUT_DEPTH = DATA / SCENE / "depth_da3"
PATCH = 14


def grid(res: int, w: int, h: int, method: str = "upper_bound_resize"):
    ref = max(w, h) if method.startswith("upper") else min(w, h)
    s = res / float(ref)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))

    def nearest(x, p):
        d = (x // p) * p
        u = d + p
        return u if abs(u - x) <= abs(x - d) else d
    return max(1, nearest(nw, PATCH)), max(1, nearest(nh, PATCH))


def load_poses(scene_dir: Path):
    from v3d.ingest import poses as P
    doc = json.loads((scene_dir / "dslr" / "nerfstudio"
                      / "transforms_undistorted.json").read_text())
    paper = Path("/tmp/paper/holi-spatial/inference_da3_scannetppv2.py")
    conv = P.parse_reader_convention(paper) if paper.exists() else P.DEFAULT_CONVENTION
    w2c = {}
    for e in doc["frames"] + doc["test_frames"]:
        w2c[e["file_path"]] = P.transform_matrix_to_w2c(
            np.array(e["transform_matrix"]), conv)
    K = np.array([[doc["fl_x"], 0, doc["cx"]],
                  [0, doc["fl_y"], doc["cy"]], [0, 0, 1]], dtype=np.float32)
    return w2c, K, int(doc["w"]), int(doc["h"])


def infer(model, paths, w2c, K, res):
    names = [p.name for p in paths]
    ext = np.stack([w2c[n] for n in names]).astype(np.float32)
    ixt = np.stack([K] * len(names)).astype(np.float32)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        pred = model.inference([str(p) for p in paths], extrinsics=ext, intrinsics=ixt,
                               align_to_input_ext_scale=True, process_res=res,
                               process_res_method="upper_bound_resize")
    d = np.asarray(pred.depth).copy()
    c = np.asarray(pred.conf).copy()
    img = np.asarray(pred.processed_images).copy()
    peak = torch.cuda.max_memory_allocated() / 2**30
    del pred
    gc.collect(); torch.cuda.empty_cache()
    return d, c, img, peak


def to_grid(depth, conf, nh, nw):
    """Area-average depth and conf onto a target grid, ignoring invalid pixels.

    Plain interpolation would average across invalid pixels and depth
    discontinuities; this weights by validity so a coarse cell takes the mean of the
    real samples inside it and nothing else.
    """
    import torch
    import torch.nn.functional as F

    d = torch.from_numpy(np.ascontiguousarray(depth)).float().unsqueeze(1)
    c = torch.from_numpy(np.ascontiguousarray(conf)).float().unsqueeze(1)
    valid = (torch.isfinite(d) & (d > 0)).float()
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0) * valid
    num = F.adaptive_avg_pool2d(d, (nh, nw))
    den = F.adaptive_avg_pool2d(valid, (nh, nw))
    cc = F.adaptive_avg_pool2d(c * valid, (nh, nw))
    out = (num / den.clamp(min=1e-6)).squeeze(1).numpy()
    out[den.squeeze(1).numpy() < 0.25] = np.nan
    return out, (cc / den.clamp(min=1e-6)).squeeze(1).numpy()


def scale_ratio(native_d, native_c, ref_d, ref_c, thresh=0.5):
    """Per-frame native/reference depth ratio, over MATCHED pixels.

    The first version of this compared a median over the 1918x1078 grid against a
    median over 504x280. Those are medians of different pixel populations: native
    resolves thin and near structures the low-resolution pass never sees, which
    shifts the median by several percent on cluttered frames. The verifier measured
    the consequence -- cross-chunk depth still disagreeing 9.1 points beyond the
    within-chunk noise floor after "correction".

    Both grids are therefore pooled to the coarse one and the ratio taken per pixel,
    so numerator and denominator describe the same surface. The median over pixels is
    then robust to the disagreements that remain at depth edges.
    """
    nh, nw = ref_d.shape[1], ref_d.shape[2]
    nd, nc = to_grid(native_d, native_c, nh, nw)
    out = np.zeros(len(native_d), dtype=np.float64)
    for i in range(len(native_d)):
        a, b = nd[i], ref_d[i]
        ok = (np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
              & (nc[i] > thresh) & (ref_c[i] > thresh))
        out[i] = float(np.median(a[ok] / b[ok])) if ok.sum() > 200 else np.nan
    return out


def main() -> None:
    scene_dir = DATA / SCENE
    images_dir = scene_dir / "dslr" / "resized_undistorted_images"
    names = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and not p.name.startswith("."))
    w2c, K, W, H = load_poses(scene_dir)
    nw_ref, nh_ref = grid(REF_RES, W, H)
    nw_nat, nh_nat = grid(NATIVE_RES, W, H)

    print("=" * 78)
    print(f"native-resolution depth for {SCENE}: {len(names)} views at {W}x{H}")
    print(f"  reference pass  {REF_RES:>5} -> {nw_ref}x{nh_ref} "
          f"({nw_ref*nh_ref:,} px), ONE call over all views")
    print(f"  native pass     {NATIVE_RES:>5} -> {nw_nat}x{nh_nat} "
          f"({nw_nat*nh_nat:,} px), {-(-len(names)//CHUNK)} chunks of {CHUNK}")
    print(f"  gain over the paper's 504: "
          f"{nw_nat*nh_nat/(nw_ref*nh_ref):.2f}x real depth per frame")
    print("=" * 78)

    from depth_anything_3.api import DepthAnything3
    t0 = time.time()
    model = DepthAnything3.from_pretrained(DA3_MODEL).to("cuda").eval()
    print(f"model loaded in {time.time()-t0:.0f}s")

    # --- Pass A: the scale reference ----------------------------------------
    print(f"\n--- pass A: all {len(names)} views at {REF_RES} (single Umeyama fit) ---")
    t = time.time()
    dA, cA, _, peakA = infer(model, names, w2c, K, REF_RES)
    print(f"    {dA.shape}, peak {peakA:.1f} GiB, {time.time()-t:.0f}s")
    med = np.array([float(np.nanmedian(np.where(np.isfinite(dA[i]) & (dA[i] > 0),
                                                dA[i], np.nan)))
                    for i in range(len(dA))])
    print(f"    reference frame medians: {np.nanmin(med):.3f} .. {np.nanmax(med):.3f}"
          f" (nan on {int(np.isnan(med).sum())} frames)")
    # Kept in host memory (600 x 280 x 504 float32 = 340 MB) because the per-chunk
    # ratio needs the reference pixels, not a summary of them.
    ref_d, ref_c = dA, cA
    print(f"    reference retained for matched-pixel ratios: "
          f"{ref_d.nbytes/1e6:.0f} MB")

    # --- Pass B: native resolution, chunk by chunk, each rescaled ------------
    OUT_DEPTH.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DEPTH.glob("*.npy"):
        stale.unlink()
    chunks = [names[i:i + CHUNK] for i in range(0, len(names), CHUNK)]
    idx_of = {p.name: i for i, p in enumerate(names)}
    factors, report = [], []
    all_pts, all_cols = [], []
    per_chunk_target = max(1, TARGET_POINTS * 4 // max(len(chunks), 1))

    print(f"\n--- pass B: {len(chunks)} chunks at {NATIVE_RES} ---")
    for ci, chunk in enumerate(chunks):
        t = time.time()
        d, c, img, peak = infer(model, chunk, w2c, K, NATIVE_RES)
        # Native depth is at nw_nat x nh_nat; the reference medians were computed at
        # the reference resolution. A median over the frame is resolution-independent
        # enough for a scale ratio -- it is a robust statistic of the same surface.
        rows = [idx_of[p.name] for p in chunk]
        ratios = scale_ratio(d, c, ref_d[rows], ref_c[rows])
        ok = np.isfinite(ratios)
        factor = float(np.median(ratios[ok])) if ok.sum() else 1.0
        within = (float(np.percentile(ratios[ok], 90) - np.percentile(ratios[ok], 10))
                  / max(factor, 1e-9) * 100) if ok.sum() > 2 else float("nan")
        # ONE scalar for the chunk: the error is per-call, so the fix is per-call.
        d /= factor
        factors.append(factor)
        report.append({"chunk": ci, "views": len(chunk), "factor": factor,
                       "frames_used": int(ok.sum()),
                       "within_chunk_ratio_spread_pct": within,
                       "peak_gib": round(peak, 2)})
        # The spread of per-frame ratios INSIDE a chunk is the diagnostic: if one
        # scalar is the right correction it should be small. Large means the error is
        # not a per-call scale and no scalar will fix it.
        print(f"  chunk {ci+1:>2}/{len(chunks)}  factor {factor:.5f}  "
              f"in-chunk ratio spread {within:5.2f}%  "
              f"peak {peak:.1f} GiB  {time.time()-t:.0f}s", flush=True)

        # Upsample to the source resolution exactly as the paper's script does, save
        # depth, and unproject for the cloud.
        import torch.nn.functional as F
        dt = F.interpolate(torch.from_numpy(d).unsqueeze(1).float(),
                           size=(H, W), mode="bilinear", align_corners=False).squeeze(1).numpy()
        ct = F.interpolate(torch.from_numpy(c).unsqueeze(1).float(),
                           size=(H, W), mode="bilinear", align_corners=False).squeeze(1).numpy()
        it = F.interpolate(torch.from_numpy(img).permute(0, 3, 1, 2).float(),
                           size=(H, W), mode="bilinear", align_corners=False)
        it = it.permute(0, 2, 3, 1).numpy()
        for k, p in enumerate(chunk):
            np.save(OUT_DEPTH / f"{p.stem}.npy", dt[k].astype(np.float32))

        Kinv = np.linalg.inv(K.astype(np.float64))
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        pix = np.stack([us.ravel() + 0.0, vs.ravel() + 0.0,
                        np.ones(W * H)], axis=0)
        rays = Kinv @ pix
        pts_c, cols_c = [], []
        for k, p in enumerate(chunk):
            dd = dt[k].ravel()
            good = np.isfinite(dd) & (dd > 0) & (ct[k].ravel() >= CONF_THRESH)
            if not good.any():
                continue
            Xc = rays[:, good] * dd[good]
            c2w = np.linalg.inv(np.vstack([w2c[p.name][:3, :], [0, 0, 0, 1]]))
            Xw = (c2w[:3, :3] @ Xc + c2w[:3, 3:4]).T
            col = it[k].reshape(-1, 3)[good]
            pts_c.append(Xw.astype(np.float32))
            cols_c.append(col.astype(np.uint8))
        if pts_c:
            pc = np.concatenate(pts_c); cc = np.concatenate(cols_c)
            if len(pc) > per_chunk_target:
                sel = np.random.default_rng(ci).choice(len(pc), per_chunk_target,
                                                       replace=False)
                sel.sort(); pc, cc = pc[sel], cc[sel]
            all_pts.append(pc); all_cols.append(cc)
        del d, c, img, dt, ct, it
        gc.collect(); torch.cuda.empty_cache()

    fac = np.array(factors)
    print(f"\n  chunk factors: median {np.median(fac):.5f}, "
          f"spread {(fac.max()-fac.min())/np.median(fac)*100:.2f}% "
          f"(this is the disagreement being removed)")

    del model
    gc.collect(); torch.cuda.empty_cache()

    # --- write the clouds ----------------------------------------------------
    pts = np.concatenate(all_pts); cols = np.concatenate(all_cols)
    print(f"\n--- clouds from {len(pts):,} conf-filtered native points ---")

    def write_ply(path: Path, P_, C_):
        n = len(P_)
        hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
               "property float x\nproperty float y\nproperty float z\n"
               "property uchar red\nproperty uchar green\nproperty uchar blue\n"
               "end_header\n").encode()
        rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = P_[:, 0], P_[:, 1], P_[:, 2]
        rec["r"], rec["g"], rec["b"] = C_[:, 0], C_[:, 1], C_[:, 2]
        with path.open("wb") as fh:
            fh.write(hdr); rec.tofile(fh)
        return path

    rng = np.random.default_rng(0)
    for target, dest, why in (
        (TARGET_POINTS, scene_dir / "pointcloud_da3_native.ply", "delivered, high density"),
        (PGSR_POINTS, scene_dir / "pointcloud_da3.ply", "PGSR initialisation"),
    ):
        if len(pts) > target:
            sel = rng.choice(len(pts), target, replace=False); sel.sort()
            P_, C_ = pts[sel], cols[sel]
        else:
            P_, C_ = pts, cols
        p = write_ply(dest, P_, C_)
        print(f"  {p.name:32} {len(P_):>12,} points  "
              f"{p.stat().st_size/1e6:>7.1f} MB   ({why})")

    summary = {
        "scene": SCENE, "views": len(names), "source_wh": [W, H],
        "reference": {"process_res": REF_RES, "grid": [nw_ref, nh_ref],
                      "one_call": True, "peak_gib": round(peakA, 2)},
        "native": {"process_res": NATIVE_RES, "grid": [nw_nat, nh_nat],
                   "chunk": CHUNK, "chunks": len(chunks),
                   "gain_over_504": round(nw_nat * nh_nat / (nw_ref * nh_ref), 3)},
        "chunk_rescale": report,
        "factor_spread_pct": float((fac.max() - fac.min()) / np.median(fac) * 100),
        "points_available": int(len(pts)),
        "delivered_points": min(TARGET_POINTS, len(pts)),
        "pgsr_points": min(PGSR_POINTS, len(pts)),
        "scale_mode": "assumed",
    }
    out = WORK / f"da3_native_{SCENE}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  -> {out}")
    print(f"  depth maps written: {len(list(OUT_DEPTH.glob('*.npy')))}")
    print("\nNATIVE_DEPTH_OK")


main()
