"""Unpack the two staged tars into the layout the runner expects, and prove it.

The paper checkout goes to /tmp/paper/holi-spatial (V3D_PAPER_ROOT's default) and
src/v3d to /kaggle/working/src/v3d, which persists. /tmp does not survive a session
restart, so this script is expected to be re-run; it is idempotent.

Verification is by import and by the runner's own patcher, not by file count: a
tree that unpacked cleanly can still be the wrong tree, and apply_offline_patches
reports per-patch whether the anchor text it needs was actually found. Run in
process rather than through a subprocess -c string -- nesting quotes inside an
inline snippet is how the last attempt turned into a SyntaxError in code that was
correct on disk.
"""
import os
import shutil
import sys
import tarfile
from pathlib import Path

STAGE = Path("/kaggle/working/_stage")
PAPER = Path("/tmp/paper/holi-spatial")
SRC = Path("/kaggle/working/src")


def unpack(tar: Path, dest: Path, wipe: Path | None = None) -> None:
    if not tar.is_file():
        sys.exit(f"ERROR: {tar} missing -- the upload step did not land")
    if wipe and wipe.exists():
        shutil.rmtree(wipe)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf:
        tf.extractall(dest, filter="data")
    files = [p for p in dest.rglob("*") if p.is_file()]
    print(f"  {tar.name} -> {dest}  {len(files)} files  "
          f"{sum(p.stat().st_size for p in files)/1e6:.1f} MB")


print("=== unpack ===")
unpack(STAGE / "paper.tgz", PAPER, wipe=PAPER)
unpack(STAGE / "v3d.tgz", SRC, wipe=SRC / "v3d")

print("\n=== the paper scripts the runner invokes ===")
missing = []
for rel in ("inference_da3_scannetppv2.py", "PGSR/train.py", "PGSR/render.py",
            "sam3.py", "classic_vllm.py", "3d_bounding_instance_gs_rerun_da3.py",
            "postprocess_3d_bbox_aabb.py", "PGSR/scene/dataset_readers.py"):
    ok = (PAPER / rel).is_file()
    print(f"  {'[ok]      ' if ok else '[MISSING] '}{rel}")
    if not ok:
        missing.append(rel)

print("\n=== v3d imports, and the patcher agrees with the checkout ===")
os.environ["V3D_PAPER_ROOT"] = str(PAPER)
for stale in [m for m in sys.modules if m == "v3d" or m.startswith("v3d.")]:
    del sys.modules[stale]           # a re-run must not see the previous tree
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from v3d.paper_runner import apply_offline_patches

    for p in apply_offline_patches():
        state = "applied" if p.applied else "already-patched-or-anchor-absent"
        print(f"  {state:32} {p.path.name}")
except Exception as exc:
    import traceback
    traceback.print_exc()
    sys.exit(f"ERROR: v3d does not import from {SRC}: {type(exc).__name__}: {exc}")

if missing:
    sys.exit(f"ERROR: paper checkout incomplete: {missing}")
print("\nOK: code is in place")
