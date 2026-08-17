"""Probe a Kaggle kernel: what hardware, what disk, what data is already there.

Text only — no binary reads, no image dumps. Prints what determines whether the
benchmark can run here, and nothing else.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or out.stderr.strip()[:120] or "no output"
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def internet():
    """Reachability, not authorisation.

    huggingface.co/api/whoami-v2 answers 401 without a token, and urllib raises
    HTTPError for that -- so a bare `except` reported "NO internet" on a kernel
    whose network was fine. An HTTP status of any kind proves the request reached
    the host; only URLError/socket errors mean it did not.
    """
    import urllib.error
    import urllib.request
    url = "https://huggingface.co/api/whoami-v2"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            return f"reachable (HTTP {r.status})"
    except urllib.error.HTTPError as exc:
        return f"reachable (HTTP {exc.code} -- a status means the host answered)"
    except Exception as exc:
        return f"NO internet ({type(exc).__name__}: {exc})"


def tree(root, depth=2, limit=24):
    root = Path(root)
    if not root.exists():
        return [f"{root}: absent"]
    rows = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if len(rel.parts) > depth:
            continue
        if p.is_dir():
            rows.append(f"  {rel}/")
        else:
            rows.append(f"  {rel}  {p.stat().st_size / 1e6:.1f} MB")
        if len(rows) >= limit:
            rows.append(f"  ... (truncated at {limit})")
            break
    return rows or [f"{root}: empty"]


print("=" * 62)
print("gpu      :", gpu())
print("internet :", internet())
print("python   :", sys.version.split()[0])
for mod in ("torch", "open3d", "numpy"):
    try:
        m = __import__(mod)
        extra = ""
        if mod == "torch":
            extra = f"  cuda={m.cuda.is_available()}  arch={m.cuda.get_arch_list()[-3:] if m.cuda.is_available() else '-'}"
        print(f"{mod:9}: {m.__version__}{extra}")
    except Exception as exc:
        print(f"{mod:9}: MISSING ({type(exc).__name__})")

for mount in ("/kaggle/working", "/tmp", "/kaggle/input"):
    try:
        t, u, f = shutil.disk_usage(mount)
        print(f"disk {mount:16} {f/1e9:7.1f} GB free of {t/1e9:.0f} GB")
    except Exception as exc:
        print(f"disk {mount:16} unavailable ({exc})")

print("\n--- /kaggle/input (attached datasets) ---")
for line in tree("/kaggle/input", depth=2, limit=30):
    print(line)

print("\n--- /kaggle/working ---")
for line in tree("/kaggle/working", depth=2, limit=30):
    print(line)

print("\n--- things the benchmark needs ---")
# Attached datasets live under /kaggle/input/datasets/<owner>/<name>, not
# /kaggle/input/<name>. Probing the shorter path reported all three model
# datasets ABSENT when every one of them was attached and readable, and that
# false negative cost a session's worth of re-uploading.
DS = "/kaggle/input/datasets/traoanhuy"
NEEDS = {
    "paper checkout": "/tmp/paper/holi-spatial",
    "v3d source": "/kaggle/working/src/v3d",
    "DA3 weights": f"{DS}/v3d-da3nested-giant-large",
    "SAM3 weights": f"{DS}/v3d-sam31-multiplex",
    "build cache": f"{DS}/v3d-build-cache-sm120",
    "scannet++ scene": "/tmp/data/scannetppv2",
    "ground truth": "/tmp/data/gt",
    "vlm classes": "/kaggle/working/work/scene_objects",
}
for name, path in NEEDS.items():
    p = Path(path)
    if p.exists():
        n = sum(1 for _ in p.rglob("*")) if p.is_dir() else 1
        print(f"  [have]    {name:16} {path}  ({n} entries)")
    else:
        print(f"  [MISSING] {name:16} {path}")
print("=" * 62)
