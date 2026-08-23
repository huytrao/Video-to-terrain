"""Make the RUN kernel ready for the paper stages, and prove it — don't assume it.

The relay reported RELAY_PUSH_OK, which proves the *bytes* arrived. It says nothing
about whether they landed in the shape `detect_layout` recognises: the tar carries
an `arcname="data"` prefix, so extracting one directory too deep produces
`/tmp/data/data/...` and every stage then fails with "no frames" much later, after
the model weights have already loaded. Check the layout here, cheaply, first.

Also places the cached VLM output. Stage `vlm` needs a network-served model and
this kernel has no internet, so the labels for the pilot scene are carried in as
data. Copying it to work_root is what lets `sam3` start.

Scene id and roots come from the environment, never a literal -- the same script
has to serve an arbitrary video's scene later.

Text only: prints paths, counts and sizes. Opens no JPG.
"""
import json
import os
import shutil
import sys
from pathlib import Path

SCENE = os.environ.get("V3D_SCENE", "00777c41d4")
DATA_ROOT = Path(os.environ.get("V3D_DATA_ROOT", "/tmp/data/scannetppv2/data"))
WORK_ROOT = Path(os.environ.get("V3D_WORK_ROOT", "/tmp/run"))
GT_DIR = Path(os.environ.get("V3D_GT_DIR", "/tmp/data/gt"))
# Durable copy: /kaggle/working survives a kernel restart, /tmp does not. After a
# restart this file is still here, so the VLM stage stays skippable without a refetch.
VLM_CACHE = Path(os.environ.get(
    "V3D_VLM_CACHE", f"/kaggle/working/cache/vlm_classes_{SCENE}.json"))

SRC = Path("/kaggle/working/src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

print("=" * 66)
print(f"scene      {SCENE}")
print(f"data_root  {DATA_ROOT}")
print(f"work_root  {WORK_ROOT}")

# --- 1. did the tar land at the right depth? ---------------------------------
# A wrong-depth extract is the failure this whole check exists to catch, so name
# it explicitly rather than only reporting "images dir missing".
stray = DATA_ROOT.parent.parent / "data" / "scannetppv2"
if stray.exists() and not (DATA_ROOT / SCENE).exists():
    sys.exit(f"ERROR: tar extracted one level too deep -- found {stray}.\n"
             f"       Re-run relay_push with --extract-to /tmp (not /tmp/data).")

print("\n--- what detect_layout needs ---")
scene_dir = DATA_ROOT / SCENE
checks = {
    "images dir": scene_dir / "dslr" / "resized_undistorted_images",
    "transforms": scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json",
    "colmap": scene_dir / "dslr" / "colmap" / "cameras.txt",
    "gt boxes": GT_DIR / f"{SCENE}.json",
}
missing = []
for name, p in checks.items():
    if p.is_dir():
        n = len(list(p.iterdir()))
        print(f"  [ok]      {name:12} {p}  ({n} entries)")
    elif p.is_file():
        print(f"  [ok]      {name:12} {p}  ({p.stat().st_size/1e6:.2f} MB)")
    else:
        print(f"  [MISSING] {name:12} {p}")
        missing.append(name)
if missing:
    sys.exit(f"ERROR: relayed data is unusable, missing: {missing}")

# --- 2. ask the runner itself, not a path guess ------------------------------
# The checks above encode my reading of the layout rules; this makes the runner's
# own detector agree. If the two ever diverge, the detector is right.
os.environ.setdefault("V3D_PAPER_ROOT", "/tmp/paper/holi-spatial")
from v3d.config import PipelineConfig, detect_layout  # noqa: E402

cfg = PipelineConfig()
layout = detect_layout(SCENE, cfg.data_root)
imgs = sorted(p for p in layout.images_dir.iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
print(f"\ndetect_layout -> {type(layout).__name__}"
      f"  scene_id={layout.scene_id}  frames={len(imgs)}")
if not imgs:
    sys.exit("ERROR: detect_layout found the scene but no readable frames in it")

# --- 3. place the VLM cache so stage sam3 can start --------------------------
dest = cfg.work_root / "scene_objects" / f"{SCENE}.json"
print("\n--- vlm cache ---")
if dest.is_file():
    print(f"  already in place: {dest}")
else:
    if not VLM_CACHE.is_file():
        sys.exit(f"ERROR: no VLM cache at {VLM_CACHE}. This kernel has no internet, "
                 f"so stage vlm cannot run here -- upload the cache first.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VLM_CACHE, dest)
    print(f"  {VLM_CACHE} -> {dest}")

# Shape, not just presence: sam3 reads `vocabulary` and `per_image`, and a cache
# whose frames are from a different scene would sail through an exists() check.
doc = json.loads(dest.read_text())
for key in ("scene", "per_image", "vocabulary"):
    if key not in doc:
        sys.exit(f"ERROR: vlm cache is missing key {key!r}")
if doc["scene"] != SCENE:
    sys.exit(f"ERROR: vlm cache is for scene {doc['scene']!r}, not {SCENE!r}")
labelled = set(doc["per_image"])
have = {p.name for p in imgs}
overlap = labelled & have
print(f"  scene={doc['scene']}  vocabulary={len(doc['vocabulary'])} classes")
print(f"  labelled frames {len(labelled)}, present frames {len(have)}, "
      f"overlap {len(overlap)}")
if not overlap:
    sys.exit("ERROR: vlm cache names no frame that exists -- wrong scene or wrong "
             "image set; sam3 would produce zero masks.")
if len(overlap) < len(have):
    print(f"  note: {len(have) - len(overlap)} present frames are unlabelled; "
          f"sam3 will skip them")

print("\nPREPARE_OK")
print("=" * 66)
