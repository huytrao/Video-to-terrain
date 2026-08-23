"""Find the highest-quality DA3 configuration this GPU can actually run.

The quality knob is `process_res`, and the paper leaves it at DA3's default 504
with the note "this can lead to OOM issue". At 504 with `upper_bound_resize` a
1920x1080 frame is predicted at **504x280** and then bilinearly upsampled back to
1920x1080 before unprojection, so 141,120 real depth samples are presented as
2,073,600 — a 14.7x areal interpolation. That interpolation is the ceiling on the
point cloud, and on the depth supervision PGSR reads from `depth_da3/` via
`PGSR/scene/cameras.py:89`. Raising it is the single largest quality lever
available.

The constraint is memory, and the question this script answers is *what memory
actually scales with*. One measurement already exists: 600 views at 504x280 —
720 tokens per view at PATCH_SIZE 14, so 432,000 tokens — peaked at 57.2 GiB.
From one point you cannot tell a token-linear cost from a view-quadratic one, and
the two give wildly different answers for the final run: if views attend to each
other pairwise, trading views for resolution is nearly free, and if the cost is
in the token count, it is a straight exchange.

So three probes, chosen to separate exactly that:

  A  600 views @ 504   (432k tokens)  the measurement already taken
  B  300 views @ 504   (216k tokens)  halves views at fixed resolution
  C  150 views @ 1008  (443k tokens)  same token count as A, 4.1x the resolution

If C ≈ A, cost tracks tokens and the exchange rate is one-for-one. If C >> A,
resolution is dearer than views and the ladder has to stop lower. B gives the
fixed cost (weights plus workspace) so the variable part can be separated from
it.

Then it escalates: starting from the highest resolution the fitted model says
should fit, it actually runs it, and on OOM steps *down the chunk size* rather
than the resolution — because with poses supplied and `align_to_input_ext_scale`
on, every chunk is registered to the same global pose set, so chunking costs
cross-view context while resolution loss is unrecoverable. That ordering is the
"different calculation" an OOM demands: back off on the axis whose loss the rest
of the pipeline can absorb.

Nothing here writes a point cloud. It reports a configuration.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, "/kaggle/working/src")
DA3_SRC = "/tmp/da3src/Depth-Anything-3/src"
if DA3_SRC not in sys.path:
    sys.path.insert(0, DA3_SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCENE = os.environ.get("V3D_SCENE", "small_factory")
DATA_ROOT = os.environ.get("V3D_DATA_ROOT", "/tmp/data/scannetppv2/data")
DA3_MODEL = os.environ.get("V3D_DA3_MODEL", "/tmp/DA3NESTED-GIANT-LARGE")
OUT = os.environ.get("V3D_RES_REPORT", f"/tmp/run/da3_resolution_{SCENE}.json")

PATCH = 14
# Leave this much of the card unused. The peak reported by
# `max_memory_allocated` counts tensors PyTorch allocated and not the blocks its
# caching allocator holds but has not handed out, nor anything allocated outside
# PyTorch (cuDNN workspaces, the CUDA context). Fragmentation also means a peak
# comfortably under the card can still fail to find a contiguous block. So the
# headroom is deliberately generous rather than shaved.
HEADROOM_GIB = 12.0

# The frame size is READ FROM THE SCENE, never assumed. It was a hardcoded
# 1920x1080 -- the pilot clip's size -- and every derived figure inherited it:
# "% of native", the areal-interpolation factor, and the choice of ladder rung.
# Run against a 920x690 capture, that constant reported 6.8% of native where the
# truth was 30%, and would have argued for a resolution the scene cannot use,
# because upper_bound_resize never upsamples past the source. A wrong number that
# looks like a measurement is worse than no number.
SRC_W, SRC_H = 0, 0          # set by read_source_size() before anything uses them


def read_source_size(images_dir) -> tuple[int, int]:
    """Actual pixel size of the frames this scene will be run on."""
    import cv2

    names = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not names:
        raise SystemExit(f"no frames in {images_dir}")
    img = cv2.imread(str(names[0]))
    if img is None:
        raise SystemExit(f"unreadable frame {names[0]}")
    h, w = img.shape[:2]
    return w, h


def grid(process_res: int, method: str = "upper_bound_resize",
         w: int | None = None, h: int | None = None) -> tuple[int, int, int]:
    """Reproduce InputProcessor's sizing exactly, and return (W, H, tokens).

    Mirrors `_resize_longest_side` / `_resize_shortest_side` then
    `_make_divisible_by_resize`, from
    `depth_anything_3/utils/io/input_processor.py:316-381`. Written out rather
    than imported because the point is to compare a prediction against what the
    processor really does — the model is asked for its own shape afterwards, and
    a disagreement is reported.
    """
    w = SRC_W if w is None else w
    h = SRC_H if h is None else h
    if not (w and h):
        raise RuntimeError("frame size unknown; read_source_size() must run first")
    ref = max(w, h) if method.startswith("upper") else min(w, h)
    scale = process_res / float(ref)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))

    def nearest_multiple(x: int, p: int) -> int:
        down = (x // p) * p
        up = down + p
        return up if abs(up - x) <= abs(x - down) else down

    nw, nh = max(1, nearest_multiple(nw, PATCH)), max(1, nearest_multiple(nh, PATCH))
    return nw, nh, (nw // PATCH) * (nh // PATCH)


def gib(x: int) -> float:
    return x / 2**30


def free_vram() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure(model, paths, process_res, method, label):
    """Run one inference and report peak VRAM, or report the OOM.

    Returns a dict; `ok` False means it did not fit. An OOM is caught rather than
    fatal because the whole point is to find the boundary, and a crashed process
    would lose every measurement taken before it.
    """
    nw, nh, tok = grid(process_res, method)
    total = tok * len(paths)
    free_vram()
    before = torch.cuda.memory_allocated()
    t = time.time()
    try:
        with torch.no_grad():
            pred = model.inference(
                [str(p) for p in paths],
                extrinsics=None, intrinsics=None,
                process_res=process_res, process_res_method=method,
            )
        peak = gib(torch.cuda.max_memory_allocated())
        got_h, got_w = int(pred.depth.shape[1]), int(pred.depth.shape[2])
        agree = (got_w, got_h) == (nw, nh)
        del pred
        free_vram()
        row = {"label": label, "views": len(paths), "process_res": process_res,
               "method": method, "predicted_wh": [nw, nh], "actual_wh": [got_w, got_h],
               "grid_agrees": agree, "tokens_per_view": tok, "total_tokens": total,
               "peak_gib": round(peak, 2), "resident_before_gib": round(gib(before), 2),
               "seconds": round(time.time() - t, 1), "ok": True}
        flag = "" if agree else "  !! grid prediction disagrees with the model"
        print(f"  [ok]   {label:22} {len(paths):4d}v @ {process_res:4d} -> "
              f"{got_w}x{got_h}  {total:>9,} tok  peak {peak:5.1f} GiB  "
              f"{row['seconds']:5.1f}s{flag}", flush=True)
        return row
    except torch.cuda.OutOfMemoryError as exc:
        free_vram()
        print(f"  [OOM]  {label:22} {len(paths):4d}v @ {process_res:4d} -> "
              f"{nw}x{nh}  {total:>9,} tok  ({str(exc).splitlines()[0][:70]})", flush=True)
        return {"label": label, "views": len(paths), "process_res": process_res,
                "method": method, "predicted_wh": [nw, nh], "tokens_per_view": tok,
                "total_tokens": total, "ok": False, "error": "OOM"}


def main() -> None:
    from pathlib import Path

    images_dir = Path(DATA_ROOT) / SCENE / "dslr" / "resized_undistorted_images"
    names = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and not p.name.startswith("."))
    if not names:
        raise SystemExit(f"no frames in {images_dir}")
    n_all = len(names)

    global SRC_W, SRC_H
    SRC_W, SRC_H = read_source_size(images_dir)
    native = max(SRC_W, SRC_H)

    def evenly(k: int):
        """k views spread across the whole clip, so a probe sees the real scene."""
        if k >= n_all:
            return names
        idx = np.linspace(0, n_all - 1, k).round().astype(int)
        return [names[i] for i in sorted(set(idx.tolist()))]

    total_gib = gib(torch.cuda.get_device_properties(0).total_memory)
    budget = total_gib - HEADROOM_GIB
    print("=" * 78)
    print(f"{torch.cuda.get_device_name(0)}  {total_gib:.1f} GiB, "
          f"budget {budget:.1f} GiB after {HEADROOM_GIB:.0f} GiB headroom")
    print(f"{n_all} frames at {SRC_W}x{SRC_H}")
    print("=" * 78)

    # `upper_bound_resize` scales by process_res/longest_side, so a rung above the
    # source UPSAMPLES the input: more tokens, more memory, and not one extra bit
    # of scene information. Native is the ceiling, and it is a real rung.
    rungs = [r for r in (504, 700, 896, 1008, 1120, 1456, 1920) if r < native]
    rungs.append(native)

    print("\nwhat each resolution would mean per frame:")
    for res in rungs:
        nw, nh, tok = grid(res)
        print(f"  process_res {res:4d} -> {nw:5d}x{nh:<5d} "
              f"{nw*nh:>9,} px  {tok:>6,} tok/view  "
              f"{nw*nh/(SRC_W*SRC_H)*100:5.1f}% of native")
    nw504, nh504, _ = grid(504)
    print(f"\n  the paper's 504 predicts {nw504}x{nh504} and upsamples to "
          f"{SRC_W}x{SRC_H}: {SRC_W*SRC_H/(nw504*nh504):.1f}x areal interpolation")

    from depth_anything_3.api import DepthAnything3
    t = time.time()
    model = DepthAnything3.from_pretrained(DA3_MODEL).to("cuda").eval()
    print(f"\nmodel loaded in {time.time() - t:.0f}s, "
          f"resident {gib(torch.cuda.memory_allocated()):.1f} GiB")

    rows = []

    # --- separate token cost from view cost ----------------------------------
    print("\n--- probes: is the cost in the tokens or in the views? ---")
    rows.append(measure(model, evenly(600), 504, "upper_bound_resize", "A 600v@504"))
    rows.append(measure(model, evenly(300), 504, "upper_bound_resize", "B 300v@504"))
    rows.append(measure(model, evenly(150), 1008, "upper_bound_resize", "C 150v@1008"))

    good = [r for r in rows if r["ok"]]
    if len(good) < 2:
        raise SystemExit("not enough successful probes to fit a cost model")

    # Least squares on peak = fixed + slope * total_tokens. Two unknowns, three
    # points, so the residual is itself informative: a large one means the cost is
    # not token-linear and the ladder below should not be trusted far from the
    # measured range.
    X = np.array([[1.0, r["total_tokens"]] for r in good])
    y = np.array([r["peak_gib"] for r in good])
    (fixed, slope), *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ np.array([fixed, slope])
    resid = float(np.abs(pred - y).max())
    print(f"\n  fitted  peak_gib = {fixed:.2f} + {slope * 1e6:.4f} per Mtoken   "
          f"worst residual {resid:.2f} GiB")
    a = next((r for r in good if r["label"].startswith("A")), None)
    c = next((r for r in good if r["label"].startswith("C")), None)
    if a and c:
        print(f"  A vs C at equal token count: {a['peak_gib']:.1f} vs {c['peak_gib']:.1f} GiB "
              f"-> {'token-driven, views and resolution trade one-for-one' if abs(a['peak_gib']-c['peak_gib']) < 0.15*a['peak_gib'] else 'NOT token-driven; resolution costs more than views'}")

    def fits(views: int, res: int) -> float:
        _, _, tok = grid(res)
        return fixed + slope * tok * views

    # --- escalate to the highest resolution that fits ------------------------
    # Resolution descends; for each, the largest chunk the model says fits. The
    # chunk is what gives way, per the module docstring.
    print("\n--- escalating: highest resolution first, chunk size gives way ---")
    LADDER = sorted(set(rungs), reverse=True)   # capped at this scene's native size
    CHUNKS = [600, 400, 300, 200, 150, 100, 75, 50, 40, 30, 24, 16, 12, 8]
    chosen = None
    for res in LADDER:
        candidate = next((c for c in CHUNKS if fits(c, res) <= budget), None)
        if candidate is None:
            _, _, tok = grid(res)
            print(f"  process_res {res:4d}: predicted {fits(8, res):.0f} GiB even at "
                  f"8 views — skipped")
            continue
        print(f"  process_res {res:4d}: trying {candidate} views/chunk "
              f"(predicted {fits(candidate, res):.1f} GiB)")
        row = measure(model, evenly(candidate), res, "upper_bound_resize",
                      f"try {candidate}v@{res}")
        rows.append(row)
        if row["ok"]:
            chosen = row
            break
        # The prediction was optimistic. Step down the chunk, which is the axis
        # whose loss the rest of the pipeline can absorb.
        for smaller in [c for c in CHUNKS if c < candidate]:
            print(f"    backing off to {smaller} views/chunk")
            row = measure(model, evenly(smaller), res, "upper_bound_resize",
                          f"try {smaller}v@{res}")
            rows.append(row)
            if row["ok"]:
                chosen = row
                break
        if chosen:
            break

    if chosen is None:
        raise SystemExit("no configuration fit, not even the paper's — something is wrong")

    _, _, tok = grid(chosen["process_res"])
    real_per_view = chosen["actual_wh"][0] * chosen["actual_wh"][1]
    baseline = nw504 * nh504
    report = {
        "device": torch.cuda.get_device_name(0),
        "total_gib": round(total_gib, 1),
        "budget_gib": round(budget, 1),
        "frames_available": n_all,
        "source_wh": [SRC_W, SRC_H],
        "cost_model": {"fixed_gib": round(float(fixed), 3),
                       "gib_per_megatoken": round(float(slope) * 1e6, 4),
                       "worst_residual_gib": round(resid, 3)},
        "measurements": rows,
        "chosen": {
            "process_res": chosen["process_res"],
            "process_res_method": chosen["method"],
            "views_per_chunk": chosen["views"],
            "depth_wh": chosen["actual_wh"],
            "peak_gib": chosen["peak_gib"],
            "chunks_for_all_frames": -(-n_all // chosen["views"]),
            "real_samples_per_frame": real_per_view,
            "real_samples_all_frames": real_per_view * n_all,
            "gain_over_paper_504": round(real_per_view / baseline, 2),
            "areal_interpolation_remaining": round(SRC_W * SRC_H / real_per_view, 2),
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)

    ch = report["chosen"]
    print("\n" + "=" * 78)
    print("CHOSEN CONFIGURATION")
    print("=" * 78)
    print(f"  process_res            {ch['process_res']}  ({ch['process_res_method']})")
    print(f"  depth predicted at     {ch['depth_wh'][0]}x{ch['depth_wh'][1]} "
          f"= {ch['real_samples_per_frame']:,} real samples/frame")
    print(f"  vs the paper's 504     {nw504}x{nh504} = {baseline:,} "
          f"-> {ch['gain_over_paper_504']}x more real depth")
    print(f"  remaining upsample     {ch['areal_interpolation_remaining']}x areal "
          f"(was {SRC_W*SRC_H/baseline:.1f}x)")
    print(f"  views per chunk        {ch['views_per_chunk']}  "
          f"({ch['chunks_for_all_frames']} chunk(s) for {n_all} frames)")
    print(f"  measured peak          {ch['peak_gib']} GiB of {total_gib:.1f}")
    print(f"  real samples, total    {ch['real_samples_all_frames']:,}")
    print(f"\n  -> {OUT}")
    print("\nRESOLUTION_OK")


main()
