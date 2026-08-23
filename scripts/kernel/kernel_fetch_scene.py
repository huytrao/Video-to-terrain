"""Pull one ScanNet++ scene out of the Holi-Spatial release, on a kernel with internet.

Runs on the internet-enabled notebook. The image shards are 14.4 GB each and must
never be downloaded whole -- `fetch_scene` finds the right shard by binary search
over first-member names and then extracts only this scene's members by byte range,
about ten range requests in total.

The scene id is taken from V3D_SCENE (default: the pilot), not written into the
code: this same path has to serve an arbitrary video's scene later, and a literal
here would be the first thing to break.

Output is a single tar, because the relay moves one file. Text only -- nothing here
opens a JPG.
"""
import os
import subprocess
import sys
import tarfile
from pathlib import Path

SCENE = os.environ.get("V3D_SCENE", "00777c41d4")
DATA = Path("/tmp/data")
SCENES = DATA / "scannetppv2" / "data"
GT = DATA / "gt"
TAR = Path("/tmp/scene.tar")
SRC = Path("/kaggle/working/src")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from v3d.data.scannetpp import fetch_gt_boxes, fetch_scene  # noqa: E402

print(f"=== scene {SCENE} ===", flush=True)

target = SCENES / SCENE
already = sorted(target.rglob("*.JPG")) if target.exists() else []
if already:
    print(f"  already fetched: {len(already)} images -- skipping the shard walk", flush=True)
    assets = None
else:
    assets = fetch_scene(SCENE, SCENES)
    print(f"  images     {assets.image_count}", flush=True)
    print(f"  colmap     {len(assets.colmap_files)} files", flush=True)
    for rel in assets.colmap_files[:12]:
        print(f"      {rel}")
    if len(assets.colmap_files) > 12:
        print(f"      ... {len(assets.colmap_files) - 12} more")

gt_path = GT / f"{SCENE}.json"
if gt_path.is_file():
    print(f"  gt already present: {gt_path.stat().st_size/1e6:.1f} MB", flush=True)
else:
    fetch_gt_boxes(SCENE, gt_path)
    print(f"  gt         {gt_path.stat().st_size/1e6:.1f} MB", flush=True)

# The two paths the runner's layout detection actually looks for. If either is
# absent the fetch "succeeded" into a shape nothing downstream can read.
print("\n=== what the runner will look for ===", flush=True)
checks = {
    "images dir": target / "dslr" / "resized_undistorted_images",
    "transforms": target / "dslr" / "nerfstudio" / "transforms_undistorted.json",
    "gt boxes": gt_path,
}
bad = []
for name, p in checks.items():
    if p.is_dir():
        n = len(list(p.iterdir()))
        print(f"  [ok]      {name:12} {p}  ({n} entries)")
    elif p.is_file():
        print(f"  [ok]      {name:12} {p}  ({p.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  [MISSING] {name:12} {p}")
        bad.append(name)
if bad:
    sys.exit(f"ERROR: fetch produced an unusable layout, missing: {bad}")

files = [p for p in DATA.rglob("*") if p.is_file()]
total = sum(p.stat().st_size for p in files)
print(f"\n=== tar {len(files)} files, {total/1e6:.1f} MB ===", flush=True)

# Uncompressed: the payload is 261 JPEGs, and gzip on already-compressed image
# data costs minutes of CPU to save nothing. The relay verifies byte counts.
if TAR.exists():
    TAR.unlink()
with tarfile.open(TAR, "w") as tf:
    tf.add(DATA, arcname="data")
size = TAR.stat().st_size
print(f"  {TAR}  {size:,} bytes", flush=True)

# The relay reads this path on this kernel; sha lets the far side prove the bytes
# arrived intact rather than trusting a length that matched by luck.
sha = subprocess.run(["sha256sum", str(TAR)], capture_output=True, text=True).stdout.split()[0]
print(f"  sha256 {sha}", flush=True)
print(f"RELAY_SRC {TAR} {size} {sha}")
