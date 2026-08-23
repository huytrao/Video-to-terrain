"""What survived on the Kaggle kernel — run on the kernel, not locally.

Kaggle keeps /kaggle/working across session restarts but rebuilds the Python
environment, so the artefacts persist and the compiled extensions do not. After
every proxy change the first question is which half is missing, and answering it
by hand through shell-quoted one-liners has already produced two false negatives.

    python kaggle_executor.py run scripts/kernel/kernel_inventory.py
"""

from __future__ import annotations

import glob
import importlib.util
import os
import subprocess

W = "/kaggle/working"
INTEREST = (
    ".ply", ".glb", ".blend", ".json", ".mp4", ".png", ".npy", ".whl", ".tar",
)


def tree(rel: str, max_depth: int = 3) -> None:
    root0 = os.path.join(W, rel)
    if not os.path.exists(root0):
        print(f"  MISSING  {rel}")
        return
    for root, dirs, files in os.walk(root0):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".ipynb_checkpoints"}]
        if root[len(root0):].count(os.sep) >= max_depth:
            dirs[:] = []
        sized = sorted(
            ((f, os.path.getsize(os.path.join(root, f))) for f in files),
            key=lambda x: -x[1],
        )
        shown = [f"{f} {s/1e6:.1f}MB" for f, s in sized[:4]]
        print(f"  {root[len(W):] or '/'}  ({len(files)} files) {shown}")


def main() -> None:
    print("== GPU")
    print(" ", subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip())

    print("== key artefacts")
    for rel in ("outputs", "deliver_out"):
        print(f"-- {rel}")
        tree(rel)

    print("== compiled extensions (these do NOT survive a restart)")
    for mod in ("torch", "open3d", "diff_plane_rasterization", "simple_knn",
                "nvdiffrast", "pytorch3d", "sam3", "transformers"):
        spec = None
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, ValueError):
            pass
        print(f"  {mod:28s} {'present' if spec else 'ABSENT'}")

    print("== wheels cached under /kaggle/working (reinstallable without internet)")
    wheels = glob.glob(W + "/**/*.whl", recursive=True)
    for w in sorted(wheels):
        print(f"  {w[len(W):]}  {os.path.getsize(w)/1e6:.1f}MB")
    if not wheels:
        print("  none — a rebuild needs the BUILD kernel")

    print("== biggest files overall")
    found = []
    for root, dirs, files in os.walk(W):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for f in files:
            if f.endswith(INTEREST):
                p = os.path.join(root, f)
                try:
                    found.append((os.path.getsize(p), p[len(W):]))
                except OSError:
                    pass
    for s, p in sorted(found, reverse=True)[:30]:
        print(f"  {s/1e6:9.1f}MB  {p}")

    print("== disk")
    print(subprocess.run(["df", "-h", W, "/tmp"], capture_output=True, text=True).stdout)


main()
