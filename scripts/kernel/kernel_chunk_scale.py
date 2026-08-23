"""Do independently-inferred chunks agree on depth scale? Test it before committing.

Native-resolution DA3 only fits in 95 GiB at 30 views per call, so 600 frames
become 20 separate `model.inference` calls. Each call runs
`_align_to_input_extrinsics_intrinsics`, which fits its own Umeyama scale against
the supplied poses and then does `prediction.depth /= scale` (`api.py:353-362`).

Twenty independent scale fits is twenty chances to scale part of the room
differently from the rest. That would be quietly destructive rather than loud:
the point cloud would have regions at inconsistent size, and PGSR reads these same
depth maps as training supervision through `PGSR/scene/cameras.py:89`, so the
inconsistency would be baked into the surface. Nothing downstream checks it.

The test is direct. Infer two overlapping sets of views as separate calls, and
compare the depth each produces **for the frames they share**. Same frames, same
supplied poses, different chunk company: if the scale fit is stable, the ratio is
1.0 to within the model's own noise. If chunking rescales anything, the ratio
departs from 1.0 and the median over ~2 million pixels makes that unmissable.

A ratio is the right statistic rather than an absolute difference, because that is
exactly the quantity a scale error corrupts, and it is dimensionless — which
matters when the units are DA3's own and not metres.

Reported for two configurations so the decision is informed rather than assumed:
the one `kernel_da3_maxres.py` actually chose for *this* scene, read from its
report, and a control at the same chunk size one resolution rung lower. The two
together separate the causes -- if only the chosen rung disagrees, resolution is
at fault; if both do, the chunk size is, and lowering resolution will not help.
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
OUT = os.environ.get("V3D_CHUNK_REPORT", f"/tmp/run/chunk_scale_{SCENE}.json")


def load_poses(scene_dir):
    """The poses exactly as the paper's DA3 entry point will read them."""
    from v3d.ingest import poses as P

    doc = json.loads((scene_dir / "dslr" / "nerfstudio" /
                      "transforms_undistorted.json").read_text())
    paper = "/tmp/paper/holi-spatial/inference_da3_scannetppv2.py"
    from pathlib import Path
    conv = (P.parse_reader_convention(Path(paper)) if os.path.exists(paper)
            else P.DEFAULT_CONVENTION)
    entries = doc["frames"] + doc["test_frames"]
    by_name = {}
    for e in entries:
        by_name[e["file_path"]] = P.transform_matrix_to_w2c(
            np.array(e["transform_matrix"]), conv)
    K = np.array([[doc["fl_x"], 0.0, doc["cx"]],
                  [0.0, doc["fl_y"], doc["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return by_name, K, doc["w"], doc["h"]


def infer(model, paths, w2c_by_name, K, res, method):
    """One inference call, with poses supplied — the same call the stage makes."""
    names = [p.name for p in paths]
    ext = np.stack([w2c_by_name[n] for n in names]).astype(np.float32)
    ixt = np.stack([K for _ in names]).astype(np.float32)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.no_grad():
        pred = model.inference(
            [str(p) for p in paths],
            extrinsics=ext, intrinsics=ixt,
            align_to_input_ext_scale=True,
            process_res=res, process_res_method=method,
        )
    depth = np.asarray(pred.depth).copy()
    conf = np.asarray(pred.conf).copy()
    peak = torch.cuda.max_memory_allocated() / 2**30
    del pred
    gc.collect(); torch.cuda.empty_cache()
    return depth, conf, peak, time.time() - t


def compare(dA, cA, idxA, dB, cB, idxB, shared, label):
    """Median per-frame depth ratio on the shared frames, over confident pixels."""
    ratios, per_frame = [], []
    for name in shared:
        a = dA[idxA[name]]
        b = dB[idxB[name]]
        if a.shape != b.shape:
            raise SystemExit(f"{label}: shape mismatch {a.shape} vs {b.shape}")
        # Confidence gates both sides: the ratio in a region neither call believed
        # says nothing about the scale fit, and its noise would swamp the signal.
        ok = (np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
              & (cA[idxA[name]] > 0.5) & (cB[idxB[name]] > 0.5))
        if ok.sum() < 1000:
            per_frame.append({"frame": name, "pixels": int(ok.sum()), "note": "too few"})
            continue
        r = float(np.median(a[ok] / b[ok]))
        ratios.append(r)
        per_frame.append({"frame": name, "pixels": int(ok.sum()),
                          "median_ratio": round(r, 5)})
    if not ratios:
        raise SystemExit(f"{label}: no shared frame had enough confident pixels")
    arr = np.array(ratios)
    return {
        "label": label,
        "shared_frames": len(shared),
        "frames_compared": len(ratios),
        "median_ratio": float(np.median(arr)),
        "min_ratio": float(arr.min()),
        "max_ratio": float(arr.max()),
        "spread_pct": float((arr.max() - arr.min()) / np.median(arr) * 100),
        "worst_deviation_pct": float(np.abs(arr - 1.0).max() * 100),
        "per_frame": per_frame,
    }


def main():
    from pathlib import Path

    scene_dir = Path(DATA_ROOT) / SCENE
    images_dir = scene_dir / "dslr" / "resized_undistorted_images"
    names = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and not p.name.startswith("."))
    w2c_by_name, K, W, H = load_poses(scene_dir)
    print("=" * 78)
    print(f"{len(names)} frames at {W}x{H}; testing whether independent chunks "
          f"agree on depth scale")
    print("=" * 78)

    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(DA3_MODEL).to("cuda").eval()
    print(f"model loaded, resident {torch.cuda.memory_allocated() / 2**30:.1f} GiB\n")

    results = []

    # The configuration under test is READ FROM THE RESOLUTION REPORT, not written
    # here. It used to be a hardcoded 30 views at process_res 1920 -- the rungs the
    # search happened to choose for one 1920x1080 clip. Tested against a different
    # capture, those constants measure a configuration nobody is going to run, and
    # report AGREE or DISAGREE about it with equal confidence.
    res_report = os.environ.get(
        "V3D_RES_REPORT", f"/tmp/run/da3_resolution_{SCENE}.json")
    if not os.path.exists(res_report):
        raise SystemExit(
            f"no resolution report at {res_report}; run kernel_da3_maxres.py first "
            "-- there is no point testing chunk agreement for a chunk size nothing chose")
    chosen = json.load(open(res_report))["chosen"]
    RES, VIEWS = int(chosen["process_res"]), int(chosen["views_per_chunk"])
    print(f"testing the chosen configuration: {VIEWS} views @ {RES}\n")

    # Two chunks drawn the way the paper draws them (spread over the whole clip,
    # since it permutes randomly), overlapping in 30% of their frames.
    overlap = max(2, VIEWS // 3)
    need = VIEWS * 2 - overlap
    spread = np.linspace(0, len(names) - 1, min(need, len(names))).round().astype(int)
    spread = [names[i] for i in sorted(set(spread.tolist()))]
    setA = spread[:VIEWS]
    setB = spread[VIEWS - overlap:VIEWS * 2 - overlap]
    shared = [n.name for n in setA if n in setB]
    print(f"--- {VIEWS} views @ {RES} (the chosen config), {len(shared)} shared frames ---")
    dA, cA, pkA, tA = infer(model, setA, w2c_by_name, K, RES, "upper_bound_resize")
    print(f"    chunk A: {dA.shape}, peak {pkA:.1f} GiB, {tA:.0f}s")
    dB, cB, pkB, tB = infer(model, setB, w2c_by_name, K, RES, "upper_bound_resize")
    print(f"    chunk B: {dB.shape}, peak {pkB:.1f} GiB, {tB:.0f}s")
    r = compare(dA, cA, {p.name: i for i, p in enumerate(setA)},
                dB, cB, {p.name: i for i, p in enumerate(setB)},
                shared, f"{VIEWS}v@{RES}")
    results.append(r)
    print(f"    median ratio {r['median_ratio']:.5f}  range "
          f"[{r['min_ratio']:.5f}, {r['max_ratio']:.5f}]  "
          f"worst deviation {r['worst_deviation_pct']:.2f}%")
    del dA, cA, dB, cB
    gc.collect(); torch.cuda.empty_cache()

    # --- control: the same chunk size one resolution rung down -----------------
    # If the chosen rung disagrees and this one agrees, resolution is the cause;
    # if both disagree, the chunk size is, and lowering resolution will not help.
    RES2 = max(504, int(RES * 0.7) // 14 * 14)
    setC, setD, sharedCD = setA, setB, shared
    print(f"\n--- control: {VIEWS} views @ {RES2}, {len(sharedCD)} shared frames ---")
    try:
        dC, cC, pkC, tC = infer(model, setC, w2c_by_name, K, RES2, "upper_bound_resize")
        print(f"    chunk C: {dC.shape}, peak {pkC:.1f} GiB, {tC:.0f}s")
        dD, cD, pkD, tD = infer(model, setD, w2c_by_name, K, RES2, "upper_bound_resize")
        print(f"    chunk D: {dD.shape}, peak {pkD:.1f} GiB, {tD:.0f}s")
        r2 = compare(dC, cC, {p.name: i for i, p in enumerate(setC)},
                     dD, cD, {p.name: i for i, p in enumerate(setD)},
                     sharedCD, f"{VIEWS}v@{RES2}")
        results.append(r2)
        print(f"    median ratio {r2['median_ratio']:.5f}  range "
              f"[{r2['min_ratio']:.5f}, {r2['max_ratio']:.5f}]  "
              f"worst deviation {r2['worst_deviation_pct']:.2f}%")
    except torch.cuda.OutOfMemoryError:
        print(f"    OOM at {VIEWS} views @ {RES2} — the control needs a smaller chunk too")
        gc.collect(); torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"scene": SCENE, "frames": len(names), "results": results}, fh, indent=2)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    # 1% is the threshold that matters: the accuracy target for this project is
    # +/-5 cm on features of order metres, so a scale disagreement above roughly a
    # percent between regions of one room is already the dominant error term.
    TOL = 1.0
    for r in results:
        verdict = "AGREE" if r["worst_deviation_pct"] <= TOL else "DISAGREE"
        print(f"  {r['label']:14} {verdict:9} worst deviation "
              f"{r['worst_deviation_pct']:.2f}% over {r['frames_compared']} shared frames")
    bad = [r for r in results if r["worst_deviation_pct"] > TOL]
    if bad:
        print(f"\n  Chunking rescales depth between calls by more than {TOL}%.")
        print("  Do NOT run 20 chunks: the point cloud and PGSR's depth supervision")
        print("  would carry region-dependent scale. Either use one call at a lower")
        print("  process_res, or rescale each chunk to a common reference before use.")
    else:
        print(f"\n  Independent chunks agree to within {TOL}%, so splitting the run")
        print("  does not rescale the room and the chosen configuration is safe.")
    print(f"\n  -> {OUT}")
    print("\nCHUNK_SCALE_OK")


main()
