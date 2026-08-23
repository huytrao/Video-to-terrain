"""Prove the rescaled native depth is globally consistent. Independent of how it was made.

`kernel_da3_native.py` corrects each chunk by one scalar fitted against a
low-resolution reference pass. That is an argument, not evidence. This measures the
result the same way the original problem was found, and it reads only the depth maps
on disk -- so it cannot inherit an assumption from the code that produced them.

The test: for pairs of frames that see the same surface, unproject frame i's depth,
move it into frame j with the recorded poses, and compare the depth that comes back
against frame j's own. If two frames came out of chunks scaled differently, their
depths disagree by that ratio and the median residual shows it. If the rescaling
worked, cross-chunk pairs look like within-chunk pairs.

Reporting both populations is the point. A single overall number cannot distinguish
"consistent" from "uniformly bad", whereas within-chunk error is the noise floor the
model itself sets, and cross-chunk error above that floor is the scaling defect.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")
import numpy as np  # noqa: E402

SCENE = os.environ.get("V3D_SCENE", "small_factory")
DATA = Path(os.environ.get("V3D_DATA_ROOT", "/tmp/data/scannetppv2/data"))
WORK = Path(os.environ.get("V3D_WORK_ROOT", "/tmp/run"))
CHUNK = int(os.environ.get("V3D_NATIVE_CHUNK", "30"))
SAMPLES = 4000
TOL_PCT = float(os.environ.get("V3D_CONSISTENCY_TOL_PCT", "1.0"))


def main() -> None:
    from v3d.ingest import poses as P

    scene_dir = DATA / SCENE
    doc = json.loads((scene_dir / "dslr" / "nerfstudio"
                      / "transforms_undistorted.json").read_text())
    paper = Path("/tmp/paper/holi-spatial/inference_da3_scannetppv2.py")
    conv = P.parse_reader_convention(paper) if paper.exists() else P.DEFAULT_CONVENTION
    w2c = {e["file_path"]: P.transform_matrix_to_w2c(np.array(e["transform_matrix"]), conv)
           for e in doc["frames"] + doc["test_frames"]}
    K = np.array([[doc["fl_x"], 0, doc["cx"]], [0, doc["fl_y"], doc["cy"]], [0, 0, 1]],
                 dtype=np.float64)

    depth_dir = scene_dir / "depth_da3"
    names = sorted(p.stem for p in depth_dir.glob("*.npy"))
    print(f"{len(names)} depth maps, chunk size {CHUNK}")
    if len(names) < 4:
        raise SystemExit("too few depth maps to test")

    # Which chunk each frame came from, reproducing the writer's contiguous split.
    order = sorted(w2c)                      # the order the writer iterated in
    chunk_of = {n: i // CHUNK for i, n in enumerate(order)}

    Kinv = np.linalg.inv(K)
    rng = np.random.default_rng(0)
    cache: dict[str, np.ndarray] = {}

    def depth(stem: str) -> np.ndarray:
        if stem not in cache:
            if len(cache) > 24:
                cache.pop(next(iter(cache)))
            cache[stem] = np.load(depth_dir / f"{stem}.npy")
        return cache[stem]

    def residual(a: str, b: str) -> float | None:
        """Median relative depth disagreement between two frames, or None."""
        fa, fb = f"{a}.JPG", f"{b}.JPG"
        if fa not in w2c or fb not in w2c:
            return None
        da, db = depth(a), depth(b)
        H, W = da.shape
        vs, us = np.nonzero(np.isfinite(da) & (da > 0))
        if vs.size < 500:
            return None
        pick = rng.choice(vs.size, min(SAMPLES, vs.size), replace=False)
        vs, us = vs[pick], us[pick]
        z = da[vs, us]
        rays = np.stack([us + 0.5, vs + 0.5, np.ones_like(us, dtype=float)])
        pc = (Kinv @ rays) * z
        Wa = np.vstack([w2c[fa][:3, :], [0, 0, 0, 1]])
        Wb = np.vstack([w2c[fb][:3, :], [0, 0, 0, 1]])
        c2wa = np.linalg.inv(Wa)
        pw = c2wa[:3, :3] @ pc + c2wa[:3, 3:4]
        pcb = Wb[:3, :3] @ pw + Wb[:3, 3:4]
        zb = pcb[2]
        front = zb > 1e-6
        if not front.any():
            return None
        uv = (K @ pcb)[:2] / zb
        u, v = uv[0] - 0.5, uv[1] - 0.5
        inside = front & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        if inside.sum() < 200:
            return None
        uu = np.clip(np.rint(u[inside]).astype(int), 0, W - 1)
        vv = np.clip(np.rint(v[inside]).astype(int), 0, H - 1)
        meas = db[vv, uu]
        pred = zb[inside]
        ok = np.isfinite(meas) & (meas > 0)
        if ok.sum() < 200:
            return None
        # Median: occlusion gives large one-sided residuals that a mean would absorb.
        return float(np.median(np.abs(pred[ok] - meas[ok]) / meas[ok]))

    within, cross = [], []
    # Pair each frame with a near neighbour (same chunk) and with a frame a chunk
    # away, so both populations come from comparable viewpoint separations.
    step = max(1, CHUNK // 3)
    for i in range(0, len(names) - CHUNK - step, step):
        a = names[i]
        b_in = names[i + 2]                       # same chunk, adjacent
        b_out = names[i + CHUNK]                  # one chunk along
        if chunk_of.get(f"{a}.JPG") == chunk_of.get(f"{b_in}.JPG"):
            r = residual(a, b_in)
            if r is not None:
                within.append(r)
        if chunk_of.get(f"{a}.JPG") != chunk_of.get(f"{b_out}.JPG"):
            r = residual(a, b_out)
            if r is not None:
                cross.append(r)

    if not within or not cross:
        raise SystemExit(f"not enough testable pairs (within {len(within)}, "
                         f"cross {len(cross)})")

    w = np.array(within) * 100
    c = np.array(cross) * 100
    print(f"\nwithin-chunk pairs  n={len(w):>4}  median {np.median(w):6.3f}%  "
          f"p90 {np.percentile(w,90):6.3f}%   <- the model's own noise floor")
    print(f"cross-chunk  pairs  n={len(c):>4}  median {np.median(c):6.3f}%  "
          f"p90 {np.percentile(c,90):6.3f}%   <- scaling defect shows up here")
    excess = float(np.median(c) - np.median(w))
    print(f"\nexcess attributable to chunk scaling: {excess:+.3f} percentage points")

    report = {"scene": SCENE, "chunk": CHUNK,
              "within_chunk": {"n": len(w), "median_pct": float(np.median(w)),
                               "p90_pct": float(np.percentile(w, 90))},
              "cross_chunk": {"n": len(c), "median_pct": float(np.median(c)),
                              "p90_pct": float(np.percentile(c, 90))},
              "excess_pct_points": excess, "tolerance_pct": TOL_PCT}
    out = WORK / f"native_consistency_{SCENE}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  -> {out}")

    if excess > TOL_PCT:
        print(f"\nFAIL: cross-chunk depth still disagrees {excess:.2f} points beyond the "
              f"within-chunk floor, above the {TOL_PCT}% tolerance. The per-chunk "
              f"rescaling has not made the reconstruction globally consistent; do not "
              f"build a mesh on this.")
        raise SystemExit(1)
    print(f"\nCONSISTENCY_OK — cross-chunk agrees with within-chunk to "
          f"{excess:+.3f} points, inside {TOL_PCT}%")


main()
