"""What the three attached datasets actually contain, and can Open3D be installed.

The pipeline expects model directories at specific paths; the datasets were built
in an earlier session and the layout was recorded only as a total size. Names and
sizes only -- nothing here opens a weight file.
"""
from pathlib import Path

DS = Path("/kaggle/input/datasets/traoanhuy")

for name in ("v3d-da3nested-giant-large", "v3d-sam31-multiplex", "v3d-build-cache-sm120"):
    root = DS / name
    print(f"=== {name} ===")
    if not root.exists():
        print("  ABSENT")
        continue
    # top two levels only; these hold thousands of files in the sam3 case
    for p in sorted(root.iterdir()):
        if p.is_file():
            print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")
        else:
            kids = sorted(p.iterdir())
            files = [q for q in p.rglob("*") if q.is_file()]
            print(f"  {p.name}/  {len(files)} files  {sum(q.stat().st_size for q in files)/1e6:.1f} MB")
            for q in kids[:6]:
                mark = "/" if q.is_dir() else ""
                size = "" if q.is_dir() else f"  {q.stat().st_size/1e6:.1f} MB"
                print(f"      {q.name}{mark}{size}")
            if len(kids) > 6:
                print(f"      ... {len(kids) - 6} more")
    print()

print("=== wheels available offline (for open3d and friends) ===")
wanted = ("open3d", "trimesh", "pymeshlab", "plyfile", "opencv", "imageio", "pillow")
found = {}
for p in DS.rglob("*.whl"):
    low = p.name.lower()
    for w in wanted:
        if low.startswith(w):
            found.setdefault(w, []).append(p)
for w in wanted:
    hits = found.get(w, [])
    print(f"  {w:10} {len(hits)} wheel(s)")
    for h in hits[:3]:
        print(f"      {h.relative_to(DS)}  {h.stat().st_size/1e6:.1f} MB")

print("\n=== what imports right now ===")
for mod in ("open3d", "trimesh", "cv2", "imageio", "PIL", "plyfile", "sklearn", "scipy"):
    try:
        m = __import__(mod)
        print(f"  {mod:10} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        print(f"  {mod:10} MISSING ({type(exc).__name__})")
