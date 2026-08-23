"""Full text inventory of a Kaggle kernel's persistent working directory.

Written as a file and uploaded rather than passed as an inline snippet: quoting a
multi-line script through a shell into `exec` corrupts it, and the failure looks
like a syntax error in code that is fine locally.

Text only. Never opens a file's contents — sizes and names are all that is needed
to decide what still has to be transferred, and the standing instruction is not to
read binaries or print image bytes.
"""
from pathlib import Path

WORK = Path("/kaggle/working")


def mb(p: Path) -> float:
    return p.stat().st_size / 1e6


print("--- /kaggle/working, per top-level entry ---")
for d in sorted(WORK.iterdir()):
    if d.is_dir():
        files = [p for p in d.rglob("*") if p.is_file()]
        print(f"  {d.name + '/':30} {len(files):6} files  {sum(mb(p) for p in files):9.1f} MB")
    else:
        print(f"  {d.name:30} {'':13} {mb(d):9.1f} MB")

print("\n--- deliver_out (the shipped artefacts) ---")
out = WORK / "deliver_out"
if out.exists():
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  {mb(p):.1f} MB")
else:
    print("  absent")

print("\n--- videos anywhere under /kaggle/working ---")
for p in sorted(WORK.rglob("*.mp4")):
    print(f"  {p}  {mb(p):.1f} MB")

print("\n--- inputs the benchmark still needs, searched for ---")
for pat in ("scannetppv2", "holi-spatial", "instances.json", "scene_objects",
            "*vlm_classes*", "transforms_undistorted.json", "*.ply"):
    hits = sorted(WORK.rglob(pat))[:6]
    print(f"  {pat:28} {len(hits)} hit(s)")
    for h in hits:
        size = f"{mb(h):.1f} MB" if h.is_file() else "dir"
        print(f"      {h}  {size}")

print("\n--- /tmp (does NOT persist across restarts) ---")
tmp = Path("/tmp")
for name in ("paper", "data", "work", "gs"):
    p = tmp / name
    if p.exists():
        files = [q for q in p.rglob("*") if q.is_file()]
        print(f"  {p}/  {len(files)} files  {sum(mb(q) for q in files):.1f} MB")
    else:
        print(f"  {p}/  absent")

print("\n--- attached datasets ---")
inp = Path("/kaggle/input/datasets")
if inp.exists():
    for owner in sorted(inp.iterdir()):
        for ds in sorted(owner.iterdir()):
            files = [p for p in ds.rglob("*") if p.is_file()]
            print(f"  {ds}  {len(files)} files  {sum(mb(p) for p in files):.1f} MB")
else:
    print("  /kaggle/input/datasets absent")
