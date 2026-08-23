"""Turn a ScanNet++ iPhone RGB-D capture into the posed frame set the pipeline reads.

This is the second input route into Pipeline 1. The first route starts from a
video and has to *estimate* poses; this one starts from a capture that already
carries them, so the work is not estimation but translation — from the layout the
capture ships in, to `flat_layout`, which every later stage already understands.

Three things here are not obvious and cost real time when got wrong.

**The pose convention.** `scene_iphone_metadata.npz` stores `trajectories` as
camera-to-world in the OpenCV convention. That was not assumed: frame 100's
LiDAR depth was reprojected into frame 103 under all three candidate readings,
and the photometric error separates them cleanly — 21.5 as-is, 50.3 with a
GL->CV flip, 68.0 read as world-to-camera. `build_transforms` wants w2c, so the
matrices are inverted on the way in, and it applies the flip the *reader's own
source* asks for.

**Which frames to drop.** The capture is already subsampled to every tenth frame
of the iPhone stream, so consecutive frames are ~5 cm apart and there is no
redundancy left to throw away. The only frames worth dropping are the ones that
would poison depth and masks: genuinely blurred, or blown out. The sharpness
threshold is a fraction of *this clip's own* median, never an absolute number,
because a threshold measured in one room is a threshold for one room.

**The masks are a different size.** RGB is 920x690 and `refined_ins_ids` is
1440x1920 — the capture keeps the instance labels at full sensor resolution.
They are downsampled with nearest-neighbour, never interpolation: an interpolated
instance id is an id that names no instance.

Environment:
    V3D_SCENE         scene id to write as (default scannetpp_00777c41d4)
    V3D_IPHONE_DIR    the extracted capture (default /tmp/scene/_bad_00777c41d4)
    V3D_DATA_ROOT     where the scene layout is written
    V3D_SHARP_FRAC    sharpness floor as a fraction of the clip median
    V3D_MAX_BLOWN     reject a frame if this fraction of pixels is clipped white
    V3D_LAYOUT        "scannetpp" (default) or "flat"

**Which layout to write.** `flat_layout` is v3d's own convention and every v3d
stage reads it, but the DA3 stage does not run v3d code — it runs the paper's
`inference_da3_scannetppv2.py`, which sniffs the directory itself and accepts
exactly three shapes: `dslr/nerfstudio/transforms_undistorted.json`,
`dense/cam`, or `color/` + `pose/`. A flat directory is none of them and the
script raises before it loads a single frame. So the default here is the
ScanNet++ shape, which is the one that route has actually been run on.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")

import cv2
import numpy as np

from v3d.config import LAYOUTS
from v3d.ingest.poses import (
    DEFAULT_CONVENTION,
    PoseSpec,
    build_transforms,
    verify_roundtrip,
)

try:  # the convention the reader will actually use, parsed from its source
    from v3d.ingest.poses import parse_reader_convention
except ImportError:  # pragma: no cover - older checkout
    parse_reader_convention = None

SCENE = os.environ.get("V3D_SCENE", "scannetpp_00777c41d4")
SRC = Path(os.environ.get("V3D_IPHONE_DIR", "/tmp/scene/_bad_00777c41d4"))
DATA_ROOT = Path(os.environ.get("V3D_DATA_ROOT", "/tmp/data/scannetppv2/data"))
SHARP_FRAC = float(os.environ.get("V3D_SHARP_FRAC", "0.25"))
MAX_BLOWN = float(os.environ.get("V3D_MAX_BLOWN", "0.12"))
PAPER = Path(os.environ.get("V3D_PAPER_DIR", "/tmp/paper/holi-spatial"))
LAYOUT = os.environ.get("V3D_LAYOUT", "scannetpp")


def measure(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Sharpness and blown-highlight fraction for every frame, in order."""
    sharp, blown = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"unreadable frame {p}")
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
        blown.append(float((g >= 250).mean()))
    return np.asarray(sharp), np.asarray(blown)


def main() -> int:
    meta = np.load(SRC / "scene_iphone_metadata.npz", allow_pickle=True)
    c2w = meta["trajectories"]          # camera-to-world, OpenCV — see module docstring
    K_all = meta["intrinsics"]
    names = [str(n) for n in meta["images"]]

    if not (len(c2w) == len(K_all) == len(names)):
        raise RuntimeError(f"npz disagrees with itself: {len(c2w)}/{len(K_all)}/{len(names)}")

    det = np.linalg.det(c2w[:, :3, :3])
    if not np.allclose(det, 1.0, atol=1e-5):
        raise RuntimeError(f"pose rotations are not proper: det in [{det.min()}, {det.max()}]")
    if not np.isfinite(c2w).all():
        raise RuntimeError("non-finite pose in the capture")

    # A single scene intrinsic is all the json can carry. These captures hold one
    # fixed lens, so pooling is exact rather than a compromise — but check it.
    if not np.allclose(K_all, K_all[0], rtol=1e-6):
        raise RuntimeError("per-frame intrinsics vary; the json format cannot carry that")
    K = K_all[0]

    paths = [SRC / "images" / n for n in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frames named in the npz are absent, e.g. {missing[0]}")

    sharp, blown = measure(paths)
    med = float(np.median(sharp))
    keep = (sharp >= med * SHARP_FRAC) & (blown <= MAX_BLOWN)
    print(f"frames        {len(names)}")
    print(f"sharpness     median {med:.1f}, floor {med * SHARP_FRAC:.1f} ({SHARP_FRAC:g}x)")
    print(f"dropped       {int((sharp < med * SHARP_FRAC).sum())} blurred, "
          f"{int((blown > MAX_BLOWN).sum())} blown")
    print(f"KEPT          {int(keep.sum())} / {len(names)}")
    if keep.sum() < 30:
        raise RuntimeError(f"only {int(keep.sum())} frames survived the gate; refusing to continue")

    if LAYOUT not in LAYOUTS:
        raise SystemExit(f"V3D_LAYOUT must be one of {sorted(LAYOUTS)}, got {LAYOUT!r}")
    layout = LAYOUTS[LAYOUT](SCENE, DATA_ROOT)
    layout.images_dir.mkdir(parents=True, exist_ok=True)
    layout.transforms.parent.mkdir(parents=True, exist_ok=True)
    print(f"layout        {LAYOUT}: images -> {layout.images_dir}")

    kept_names, w2c = [], []
    for i, (name, ok) in enumerate(zip(names, keep)):
        if not ok:
            continue
        shutil.copy2(paths[i], layout.images_dir / name)
        kept_names.append(name)
        w2c.append(np.linalg.inv(c2w[i]))
    w2c = np.stack(w2c)

    h, w = cv2.imread(str(paths[0])).shape[:2]
    conv = DEFAULT_CONVENTION
    if parse_reader_convention is not None:
        reader = PAPER / "inference_da3_scannetppv2.py"
        if reader.exists():
            conv = parse_reader_convention(reader)
    print(f"convention    {conv.describe()}")

    spec = PoseSpec()
    doc = build_transforms(w2c, K, kept_names, (w, h), spec, conv)
    report = verify_roundtrip(doc, dict(zip(kept_names, w2c)), spec, conv)
    print(f"roundtrip     {report}")

    layout.transforms.write_text(json.dumps(doc))
    print(f"wrote         {layout.transforms} "
          f"({len(doc['frames'])} train + {len(doc['test_frames'])} held out)")

    # Depth and instance ids are not read by DA3, but they are the only metric
    # reference and the only per-frame ground truth this scene has. Carry them
    # along now; re-fetching them later costs a download and a relay.
    for sub, interp in (("depth", cv2.INTER_NEAREST), ("refined_ins_ids", cv2.INTER_NEAREST)):
        src_dir = SRC / sub
        if not src_dir.is_dir():
            continue
        dst = layout.scene_dir / sub
        dst.mkdir(parents=True, exist_ok=True)
        for name in kept_names:
            stem = Path(name).stem
            f = src_dir / f"{stem}.png"
            if not f.exists():
                raise FileNotFoundError(f"{sub} missing {f.name}, which the frame list needs")
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=interp)
            cv2.imwrite(str(dst / f"{stem}.png"), img)
        print(f"carried       {sub} -> {dst} ({len(kept_names)} at {w}x{h})")

    print("INGEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
