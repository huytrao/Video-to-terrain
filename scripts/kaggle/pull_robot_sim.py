#!/usr/bin/env python3
"""Pull the robot-simulation subset from the published Kaggle dataset, then run it.

The GPU kernel is not needed and neither is its proxy: this goes through Kaggle's own
download path, which has none of the 30-second read timeout that made the kernel's
Contents API unusable for anything over a few megabytes.

It fetches only what a simulator consumes — the scene descriptions, the terrain, and a
convex collider per object — not the 39.8 M-triangle visual mesh, which would make
contact resolution useless and is included only as a decimated reference.

Then it loads the MJCF and steps it, because a scene file that has not been compiled
and stepped is not known to work.

    pip install kaggle
    python3 scripts/kaggle/pull_robot_sim.py --out outputs/robot_sim
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = "traoanhuy/v3d-small-factory-scene"

# Exact files, plus one prefix for the per-object colliders.
WANTED = [
    "mujoco/scene.xml",
    "mujoco/heightfield.png",
    "exchange/scene.urdf",
    "terrain/terrain.json",
    "terrain/height_map.npy",
    "objects.json",
    "README.md",
    "manifest.json",
]
COLLIDER_RE = re.compile(r"^objects/object_\d+_.*_collision\.ply$")
VISUAL_REFERENCE = "geometry/mesh_decimated_1.5M.ply"


def credentials() -> tuple[str, str]:
    """From the environment, then .env, then the example file. Never printed."""
    u, k = os.environ.get("KAGGLE_USERNAME", ""), os.environ.get("KAGGLE_KEY", "")
    if u and k:
        return u, k
    for path, pat_u, pat_k in (
        (ROOT / ".env", r"^KAGGLE_USERNAME=(.+)$", r"^KAGGLE_KEY=(.+)$"),
        (ROOT / "scripts/kaggle/kaggle_upload_example.py",
         r'KAGGLE_USERNAME\s*=\s*["\']([^"\']+)', r'KAGGLE_KEY\s*=\s*["\']([^"\']+)'),
    ):
        if not path.exists():
            continue
        t = path.read_text()
        mu = re.search(pat_u, t, re.M)
        mk = re.search(pat_k, t, re.M)
        if mu and mk:
            if path.name != ".env":
                print(f"note: credentials read from {path.name}, which holds a live "
                      f"key in plain text — move it to .env and rotate it")
            return mu.group(1).strip(), mk.group(1).strip()
    sys.exit("set KAGGLE_USERNAME and KAGGLE_KEY, or put them in .env")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "robot_sim")
    ap.add_argument("--with-visual", action="store_true",
                    help="also fetch the 41 MB decimated visual mesh")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    if not subprocess.run(["bash", "-lc", "command -v kaggle"],
                          capture_output=True).returncode == 0:
        sys.exit("kaggle CLI not found. Run: pip install kaggle")

    user, key = credentials()
    env = dict(os.environ, KAGGLE_USERNAME=user, KAGGLE_KEY=key)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"listing {DATASET} ...")
    r = subprocess.run(["kaggle", "datasets", "files", DATASET, "--page-size", "500"],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"kaggle listing failed:\n{r.stdout}\n{r.stderr}")
    names = [l.split()[0] for l in r.stdout.splitlines()
             if l and not l.startswith(("name", "---"))]
    colliders = sorted(n for n in names if COLLIDER_RE.match(n))
    todo = [n for n in WANTED if n in names] + colliders
    if args.with_visual and VISUAL_REFERENCE in names:
        todo.append(VISUAL_REFERENCE)
    missing = [n for n in WANTED if n not in names]
    print(f"  {len(names)} files in the dataset; fetching {len(todo)} "
          f"({len(colliders)} colliders)")
    if missing:
        print(f"  NOT PRESENT: {missing}")

    for n in todo:
        dest = out / n
        if dest.exists() and dest.stat().st_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        rr = subprocess.run(["kaggle", "datasets", "download", DATASET,
                             "-f", n, "-p", str(dest.parent), "--unzip", "--force"],
                            capture_output=True, text=True, env=env)
        if rr.returncode != 0:
            print(f"  [FAIL] {n}: {(rr.stdout + rr.stderr).strip()[-160:]}")
    got = sorted(p for p in out.rglob("*") if p.is_file())
    print(f"  {len(got)} files, {sum(p.stat().st_size for p in got)/1e6:.1f} MB in {out}")

    # --- prove it runs ------------------------------------------------------
    mjcf = out / "mujoco" / "scene.xml"
    hf = out / "mujoco" / "heightfield.png"
    if not mjcf.exists():
        sys.exit("scene.xml did not download; cannot verify")
    # scene.xml references the heightfield by bare filename, so MuJoCo resolves it
    # relative to the XML. Both must sit in the same directory.
    if hf.exists() and not (mjcf.parent / hf.name).exists():
        import shutil
        shutil.copy2(hf, mjcf.parent / hf.name)

    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("mujoco/numpy not installed locally; skipping the run check")
        return 0

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    peak, series = 0.0, []
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
        if data.qvel.size:
            v = float(np.abs(data.qvel).max())
            peak = max(peak, v)
            series.append(v)
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    print(f"\nMJCF loads: {model.nbody} bodies, {model.ngeom} geoms, "
          f"{model.nq} dof")
    print(f"{args.steps} steps, sim time {data.time:.2f}s, finite {finite}, "
          f"peak |qvel| {peak:.2f}")
    if series:
        k = max(1, len(series) // 5)
        early, late = float(np.mean(series[:k])), float(np.mean(series[-k:]))
        print(f"settling: {early:.2f} -> {late:.2f} (ratio {late/max(early,1e-9):.3f})")
        if late > 0.5 * early:
            print("  the scene is NOT settling — objects spawn interpenetrating. "
                  "Expected; see README 'Two known defects'.")
    doc = json.loads((out / "objects.json").read_text()) if (out / "objects.json").exists() else {}
    if doc:
        print(f"objects.json: {len(doc.get('objects', []))} objects, "
              f"scale_mode={doc.get('scale_mode')}")
        print("  LENGTHS ARE IN DA3 UNITS, NOT METRES — every mass and force in "
              "scene.xml is therefore unanchored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
