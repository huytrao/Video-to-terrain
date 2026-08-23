"""Find the highest DA3 input resolution this GPU can actually hold, by measuring.

Why measure instead of choose: the paper runs DA3 at ``process_res=504`` with
``upper_bound_resize`` -- which is the *low_res* preset in DA3's own app
(``{"high_res": "lower_bound_resize", "low_res": "upper_bound_resize"}``) -- and
then bilinearly upsamples depth back to the original 1752x1168 before unprojecting.
The point cloud therefore *looks* full-resolution while carrying 504-px geometry;
no amount of extra sampled points recovers detail the depth map never measured.

Raising it is the single largest quality lever available, and the only thing
stopping us is memory. DA3 attends across the views in a chunk, so cost grows with
(views x tokens-per-view) and the tradeoff is genuinely two-dimensional. Guessing a
resolution risks either a 96 GB OOM an hour into a run, or leaving quality unused.
So: run the real model on real frames at each candidate, record peak VRAM, and let
the numbers pick the setting.

Nothing here is written into the pipeline. It prints a table; the chosen values are
passed later through V3D_PROCESS_RES / V3D_RES_METHOD, never baked into code.

Text only -- no image is displayed, no array is dumped.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCENE = os.environ.get("V3D_SCENE", "00777c41d4")
DATA_ROOT = Path(os.environ.get("V3D_DATA_ROOT", "/tmp/data/scannetppv2/data"))
DA3_MODEL = os.environ.get(
    "V3D_DA3_MODEL",
    "/kaggle/input/datasets/traoanhuy/v3d-da3nested-giant-large/DA3NESTED-GIANT-LARGE")
# Views per probe. Small enough to be quick, large enough that cross-view
# attention is exercised -- a 1-view probe would measure the wrong thing.
PROBE_VIEWS = int(os.environ.get("V3D_PROBE_VIEWS", "16"))
PATCH = 14

sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, "/kaggle/working/depth_anything_3")

from depth_anything_3.api import DepthAnything3  # noqa: E402

scene_dir = DATA_ROOT / SCENE
tj = scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
meta = json.loads(tj.read_text())
W, H = int(meta["w"]), int(meta["h"])
img_dir = scene_dir / "dslr" / "resized_undistorted_images"

frames = meta["frames"][:PROBE_VIEWS]
image_files, extr, intr = [], [], []
for fr in frames:
    image_files.append(str(img_dir / fr["file_path"]))
    c2w = np.array(fr["transform_matrix"])
    c2w[:3, 1:3] *= -1                      # OpenGL -> COLMAP, as the paper does
    extr.append(np.linalg.inv(c2w))
    intr.append(np.array([[meta["fl_x"], 0, meta["cx"]],
                          [0, meta["fl_y"], meta["cy"]],
                          [0, 0, 1]], dtype=np.float32))
extr = np.array(extr, dtype=np.float32)
intr = np.array(intr, dtype=np.float32)

total_vram = torch.cuda.get_device_properties(0).total_memory / 2**30
print("=" * 78)
print(f"gpu        {torch.cuda.get_device_name(0)}  {total_vram:.1f} GiB")
print(f"images     {W}x{H}  ({PROBE_VIEWS} views probed, scene has {len(meta['frames'])})")


def grid(res: int, method: str) -> tuple[int, int, int]:
    """Token grid DA3 will actually use: resize, then round each side to /14."""
    if method == "upper_bound_resize":
        scale = res / max(W, H)
    else:
        scale = res / min(W, H)
    w, h = W * scale, H * scale
    rw = max(PATCH, int(round(w / PATCH)) * PATCH)
    rh = max(PATCH, int(round(h / PATCH)) * PATCH)
    return rw, rh, (rw // PATCH) * (rh // PATCH)


# Candidates from the paper's setting up to native resolution. `lower_bound_resize`
# is DA3's own high_res preset: it sets the *short* side, so it always yields more
# pixels than upper_bound at the same number.
CANDIDATES = [
    (504, "upper_bound_resize"),      # what the paper runs -- the baseline
    (504, "lower_bound_resize"),
    (700, "lower_bound_resize"),
    (896, "lower_bound_resize"),
    (min(W, H), "lower_bound_resize"),  # native short side: no detail discarded
]

print("\nloading model (6.8 GB, once)...", flush=True)
t0 = time.time()
model = DepthAnything3.from_pretrained(DA3_MODEL).to(device="cuda")
print(f"  loaded in {time.time()-t0:.0f}s", flush=True)
base = torch.cuda.memory_allocated() / 2**30
print(f"  weights resident: {base:.2f} GiB")

print(f"\n{'res':>6} {'method':<20} {'grid':>11} {'tok/view':>9} "
      f"{'peakVRAM':>9} {'sec':>6} {'valid%':>7}")
print("-" * 78)

results = []
for res, method in CANDIDATES:
    rw, rh, tok = grid(res, method)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    t0 = time.time()
    try:
        pred = model.inference(
            image_files,
            extrinsics=extr,
            intrinsics=intr,
            align_to_input_ext_scale=True,
            process_res=res,
            process_res_method=method,
        )
        peak = torch.cuda.max_memory_allocated() / 2**30
        dt = time.time() - t0
        # Fraction of pixels that survive the paper's conf gate. A resolution that
        # OOMs is useless, but so is one whose extra pixels are all low-confidence.
        conf = np.asarray(pred.conf)
        depth = np.asarray(pred.depth)
        valid = float(np.mean((conf >= 0.3) & np.isfinite(depth) & (depth > 0)) * 100)
        print(f"{res:>6} {method:<20} {rw:>5}x{rh:<5} {tok:>9} "
              f"{peak:>8.2f}G {dt:>6.1f} {valid:>6.1f}%")
        results.append({"res": res, "method": method, "grid": [rw, rh], "tok": tok,
                        "peak_gib": round(peak, 2), "sec": round(dt, 1),
                        "valid_pct": round(valid, 1), "ok": True})
        del pred
    except torch.cuda.OutOfMemoryError:
        print(f"{res:>6} {method:<20} {rw:>5}x{rh:<5} {tok:>9} {'OOM':>9}")
        results.append({"res": res, "method": method, "tok": tok, "ok": False})
    except Exception as exc:                       # a shape assert is a real answer too
        print(f"{res:>6} {method:<20} {rw:>5}x{rh:<5} {tok:>9} "
              f"  FAILED {type(exc).__name__}: {str(exc)[:60]}")
        results.append({"res": res, "method": method, "tok": tok, "ok": False,
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    gc.collect()
    torch.cuda.empty_cache()

# --- extrapolate to the real run ---------------------------------------------
# The pipeline feeds all 261 views in one chunk (MAX_CHUNK_SIZE=800). Peak memory
# is roughly the resident weights plus per-view activations, so scale only the
# activation part -- reporting the raw probe peak would badly under-predict.
n_full = len(meta["frames"])
print(f"\n--- extrapolated to the full {n_full}-view chunk ---")
print(f"{'res':>6} {'method':<20} {'est.peak':>10}  verdict")
budget = total_vram * 0.85            # leave headroom: PGSR is not the only tenant
for r in results:
    if not r.get("ok"):
        continue
    act = max(0.0, r["peak_gib"] - base)
    est = base + act * n_full / PROBE_VIEWS
    fits = est < budget
    print(f"{r['res']:>6} {r['method']:<20} {est:>9.1f}G  "
          f"{'fits' if fits else 'needs chunking'}")
    r["est_full_gib"] = round(est, 1)
    r["fits_full"] = fits
    if not fits and act > 0:
        # Largest chunk that stays inside budget, so the caller can trade
        # cross-view context for resolution deliberately rather than by crashing.
        r["max_chunk"] = max(1, int((budget - base) / (act / PROBE_VIEWS)))
        print(f"{'':>6} {'':<20} {'':>10}  -> max chunk ~{r['max_chunk']} views")

out = Path("/kaggle/working/cache/da3_res_calibration.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0),
                           "total_gib": round(total_vram, 1),
                           "image_wh": [W, H], "probe_views": PROBE_VIEWS,
                           "full_views": n_full, "weights_gib": round(base, 2),
                           "results": results}, indent=2))
print(f"\nwrote {out}")
print("CALIBRATION_OK")
print("=" * 78)
