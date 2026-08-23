#!/usr/bin/env python3
"""
examples/hunyuan3d/run_hunyuan3d.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example: run Tencent Hunyuan3D-2 image → 3D mesh on a Kaggle GPU,
using kaggle-runtime-agent-skills as the transport layer.

Usage:
    export KAGGLE_PROXY_URL='https://kkb-production.../proxy'

    python examples/hunyuan3d/run_hunyuan3d.py \
        --front  photo_front.png \
        --side   photo_side.png  \
        --out    ./output_3d     \
        --steps  30

    # With texture painting (needs ~14 GB VRAM)
    python examples/hunyuan3d/run_hunyuan3d.py \
        --front  photo_front.png \
        --out    ./output_3d     \
        --textured
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Use the top-level toolkit modules
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from kaggle_executor import KaggleExecutor  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Kaggle-side script (injected as a string, runs on the GPU node)
# ─────────────────────────────────────────────────────────────────────────────

_KAGGLE_SCRIPT = '''\
import subprocess, sys, os, time, shutil
from pathlib import Path

print("=" * 70)
print("Hunyuan3D-2  |  kaggle-runtime-agent-skills example")
print("=" * 70)
os.chdir("/kaggle/working")

# ── GPU info ─────────────────────────────────────────────────────────────────
r = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
     "--format=csv,noheader"],
    capture_output=True, text=True,
)
print(f"\\n[GPU] {r.stdout.strip()}")

# ── Clone Hunyuan3D-2 ────────────────────────────────────────────────────────
REPO = Path("/kaggle/working/Hunyuan3D-2")
if not REPO.exists():
    print("\\n[Clone] Hunyuan3D-2 ...")
    subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/Tencent-Hunyuan/Hunyuan3D-2", str(REPO)],
        check=True, capture_output=True,
    )
    print("  done.")
else:
    print("\\n[Clone] already present — skip")

# ── Install ──────────────────────────────────────────────────────────────────
print("\\n[Install] torch + hy3dgen ...")
for cmd in [
    [sys.executable, "-m", "pip", "install", "-q",
     "torch==2.3.0", "torchvision==0.18.0",
     "--index-url", "https://download.pytorch.org/whl/cu121"],
    [sys.executable, "-m", "pip", "install", "-q",
     "-r", str(REPO / "requirements.txt")],
    [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)],
]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("  WARN:", r.stderr[-300:])
print("  done.")

# ── Image paths ──────────────────────────────────────────────────────────────
IN  = Path("/kaggle/working/_input_imgs")
OUT = Path("/kaggle/working/_output_3d")
IN.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

front = IN / "front.png"
side  = IN / "side.png"
print(f"\\n[Images]  front={front.exists()}  side={side.exists()}")

# ── Inference ────────────────────────────────────────────────────────────────
print("\\n[Inference] loading pipeline ...")
t0 = time.time()

try:
    sys.path.insert(0, str(REPO))
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    print(f"  torch={torch.__version__}  CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        d = torch.cuda.get_device_properties(0)
        print(f"  {d.name}  {d.total_memory/1e9:.1f} GB VRAM")

    img = Image.open(str(front)).convert("RGBA")
    print(f"  input: {img.size}")

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        torch_dtype=torch.float16,
        device="cuda:0",
    )
    print("  pipeline loaded!")

    TEXTURED = {TEXTURED}
    STEPS    = {STEPS}
    SEED     = {SEED}

    outputs = pipe(
        image=img,
        num_inference_steps=STEPS,
        seed=SEED,
        octree_resolution=512,
        guidance_scale=3.0,
    )
    mesh = outputs[0]
    mesh.export(str(OUT / "result.glb"))
    mesh.export(str(OUT / "result.obj"))

    if TEXTURED:
        print("\\n[Texture] painting ...")
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        paint = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            torch_dtype=torch.float16,
            device="cuda:0",
        )
        tex = paint(mesh_path=str(OUT / "result.glb"),
                    image_path=str(front))[0]
        tex.export(str(OUT / "result_textured.glb"))

    elapsed = time.time() - t0
    print(f"\\n✅  Done in {elapsed:.1f}s")

except Exception as e:
    import traceback
    print(f"\\n❌  {e}")
    traceback.print_exc()

print("\\n[Output files]")
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
'''


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    front_image: str,
    side_image: str | None,
    out_dir: str,
    textured: bool = False,
    steps: int = 30,
    seed: int = 42,
) -> None:
    proxy_url = os.environ.get("KAGGLE_PROXY_URL", "")
    if not proxy_url:
        sys.exit("ERROR: set KAGGLE_PROXY_URL first.")

    ex = KaggleExecutor(proxy_url)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Uploading images...")
    ex.upload_file(front_image, "_input_imgs/front.png")
    ex.upload_file(side_image or front_image, "_input_imgs/side.png")

    script = (
        _KAGGLE_SCRIPT
        .replace("{TEXTURED}", str(textured))
        .replace("{STEPS}",    str(steps))
        .replace("{SEED}",     str(seed))
    )

    print("\n[2/4] Running on Kaggle GPU (10–20 min on first run)...")
    kid = ex.get_or_create_kernel()
    ex.execute(kid, script, timeout=2400)

    print("\n[3/4] Downloading results...")
    n = ex.download_dir("_output_3d", str(out), extensions=(".glb", ".obj"))

    print(f"\n[4/4] ✅  {n} file(s) saved to {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Hunyuan3D-2 image→3D on Kaggle GPU (example)"
    )
    p.add_argument("--front",    required=True, help="Front-view image (PNG/JPG)")
    p.add_argument("--side",     default=None,  help="Side-view image (improves quality)")
    p.add_argument("--out",      default="./output_3d")
    p.add_argument("--textured", action="store_true",
                   help="Run texture-paint stage (~14 GB VRAM)")
    p.add_argument("--steps",    type=int, default=30)
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()
    run(args.front, args.side, args.out, args.textured, args.steps, args.seed)


if __name__ == "__main__":
    main()
