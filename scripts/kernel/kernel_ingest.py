"""Turn the factory video into the posed frame set every later stage consumes.

Two halves, and the split matters. Frame selection is CPU work with no model in
it, so it is cheap to inspect and cheap to redo; pose prediction is a DA3
inference whose memory cost is the thing that decides how much resolution the
*depth* stage can afford later. Running them in one script means the numbers from
the first inform the second in the same log.

What it does not do is choose a scale. Nothing in a video fixes metres — a room
and a photograph of a room twice its size produce identical images — so the
transforms file is written with `v3d_scale_mode: assumed` and every length below
is in DA3's own units. `v3d.ingest.poses.build_transforms` records that, and
Stage J refuses to report an absolute error against it. This project has already
shipped a room measured 8.6 m tall with a validator passing it 16/16; the
guard against repeating that is refusing to call these numbers metres.

Environment:
    V3D_SCENE          scene id (default small_factory)
    V3D_VIDEO          path to the video
    V3D_DATA_ROOT      where the scene layout is written
    V3D_MAX_FRAMES     ceiling on kept frames
    V3D_POSE_RES       process_res for pose prediction
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/kaggle/working/src")
# `kernel_stage.py` puts DA3 on a .pth in site-packages, which every *new* process
# picks up — but this script is exec'd inside a long-lived IPython kernel whose
# sys.path was fixed before that file existed. Adding it here rather than relying
# on the .pth is what makes the two cases behave the same.
DA3_SRC = "/tmp/da3src/Depth-Anything-3/src"
if DA3_SRC not in sys.path:
    sys.path.insert(0, DA3_SRC)
# And hand the same two paths to the stage subprocesses, which inherit no sys.path
# edit made in a notebook cell.
os.environ.setdefault("V3D_IMPORT_PATHS", f"/kaggle/working/src{os.pathsep}{DA3_SRC}")

SCENE = os.environ.get("V3D_SCENE", "small_factory")
VIDEO = os.environ.get("V3D_VIDEO", "/tmp/input/small_factory_final.mp4")
DATA_ROOT = os.environ.get("V3D_DATA_ROOT", "/tmp/data/scannetppv2/data")
MAX_FRAMES = int(os.environ.get("V3D_MAX_FRAMES", "600"))
POSE_RES = int(os.environ.get("V3D_POSE_RES", "504"))
DA3_MODEL = os.environ.get("V3D_DA3_MODEL", "/tmp/DA3NESTED-GIANT-LARGE")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from v3d.ingest import frames as F  # noqa: E402
from v3d.ingest import poses as P  # noqa: E402

# The ScanNet++ shape, because that is the only layout the paper's DA3 entry point
# recognises: `inference_da3_scannetppv2.py:711-721` dispatches on directory shape
# over exactly three known layouts and raises ValueError on anything else. The
# flat shape `config.flat_layout` describes is not one of them, despite its
# docstring's claim — so the video's frames are presented in the benchmark's own
# layout and every downstream stage runs unchanged.
scene_dir = Path(DATA_ROOT) / SCENE
images_dir = scene_dir / "dslr" / "resized_undistorted_images"
manifest_path = scene_dir / "ingest_manifest.json"

print("=" * 70)
print(f"scene       {SCENE}")
print(f"video       {VIDEO}")
print(f"layout      {scene_dir}")
print(f"max_frames  {MAX_FRAMES}   pose process_res {POSE_RES}")
print("=" * 70)

if not os.path.exists(VIDEO):
    raise SystemExit(f"no video at {VIDEO}")

# --- 1. frames ---------------------------------------------------------------

if manifest_path.exists() and images_dir.is_dir() and any(images_dir.iterdir()):
    manifest = json.loads(manifest_path.read_text())
    n = len(list(images_dir.glob("*.JPG")))
    print(f"\n--- 1. frames: reusing {n} already extracted ---")
    print(f"    (delete {manifest_path} to force a re-extract)")
else:
    print("\n--- 1. frames ---")
    t = time.time()
    spec = F.FrameSpec(max_frames=MAX_FRAMES)
    manifest = F.ingest(Path(VIDEO), images_dir, spec)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"    {time.time() - t:.0f}s -> {manifest_path}")

names = sorted(p.name for p in images_dir.iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
               and not p.name.startswith("."))
print(f"    {len(names)} frames on disk, first {names[0]}, last {names[-1]}")

# --- 2. poses ----------------------------------------------------------------

print("\n--- 2. poses (DA3, no pose input) ---")
transforms = scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
if transforms.exists():
    doc = json.loads(transforms.read_text())
    print(f"    reusing {transforms}: {len(doc['frames'])} train + "
          f"{len(doc['test_frames'])} test")
    print(f"    (delete it to re-predict)")
else:
    from depth_anything_3.api import DepthAnything3

    t = time.time()
    print(f"    loading {DA3_MODEL} ...")
    model = DepthAnything3.from_pretrained(DA3_MODEL).to("cuda").eval()
    print(f"    loaded in {time.time() - t:.0f}s")

    # The paper's own DA3 entry point is patched to read the convention from
    # `load_scannetppv2_poses`; do the same here so the writer cannot disagree
    # with the reader about which columns to negate. Parsed, not remembered.
    paper_file = Path("/tmp/paper/holi-spatial/inference_da3_scannetppv2.py")
    conv = (P.parse_reader_convention(paper_file) if paper_file.exists()
            else P.DEFAULT_CONVENTION)
    print(f"    pose convention: {conv.describe()}")

    spec = P.PoseSpec(process_res=POSE_RES, paper_file=str(paper_file))

    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.no_grad():
        report = P.write_scene(images_dir, scene_dir, model, spec, image_names=names)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"    pose prediction: {time.time() - t:.0f}s, peak VRAM {peak:.1f} GiB "
          f"for {len(names)} views at process_res {POSE_RES}")
    report["peak_vram_gib"] = peak
    report["views"] = len(names)
    (scene_dir / "pose_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"    -> {scene_dir / 'pose_report.json'}")

    del model
    torch.cuda.empty_cache()

# --- 3. does detect_layout agree? --------------------------------------------
# The checks above encode this script's reading of the layout rules. Make the
# runner's own detector agree, because if the two diverge the detector is what
# every stage actually uses.
print("\n--- 3. what the pipeline sees ---")
from v3d.config import PipelineConfig, detect_layout  # noqa: E402

cfg = PipelineConfig()
layout = detect_layout(SCENE, cfg.data_root)
problems = layout.missing()
print(f"    scene_id     {layout.scene_id}")
print(f"    source_dir   {layout.source_dir}")
print(f"    images_dir   {layout.images_dir}")
print(f"    transforms   {layout.transforms}")
if problems:
    raise SystemExit(f"detect_layout is not satisfied: {problems}")
print(f"    missing()    [] — layout is complete")

print("\nINGEST_OK")
