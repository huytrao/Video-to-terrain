#!/usr/bin/env python3
"""Run Depth Anything 3 monocular depth inference on a Kaggle video.

Kaggle usage (GPU T4 x2 and Internet must be enabled):
    !python /kaggle/working/run_kaggle.py

The default input is the video supplied for this task. Outputs are written to
/kaggle/working/depth_anything_3_output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/bytedance-seed/depth-anything-3.git"
DEFAULT_VIDEO = (
    "/kaggle/input/datasets/traoanhuy/test-depthanythingv3/"
    "20260810_150244_color.mp4"
)
DEFAULT_REPO = "/kaggle/working/depth-anything-3"
DEFAULT_OUTPUT = "/kaggle/working/depth_anything_3_output"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command)


def setup_repository(repo_dir: Path) -> None:
    """Clone DA3 and install only runtime dependencies needed by this pipeline."""
    if not (repo_dir / ".git").is_dir():
        if repo_dir.exists() and any(repo_dir.iterdir()):
            raise RuntimeError(
                f"{repo_dir} exists but is not a git clone. Remove it or pass --repo-dir."
            )
        run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])

    # Use this interpreter so installation is visible to this script/kernel.
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo_dir)])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "einops",
            "huggingface_hub",
            "omegaconf",
            "addict",
            "e3nn",
            "evo",
            "natsort",
            "colorama",
            "pyyaml",
            "safetensors",
            "tqdm",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input MP4 path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--repo-dir", default=DEFAULT_REPO, help="DA3 clone directory")
    parser.add_argument("--model", default="depth-anything/DA3MONO-LARGE")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2, help="Use 1 if T4 runs out of memory")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--max-output-height", type=int, default=720)
    parser.add_argument("--skip-install", action="store_true", help="Do not clone/install dependencies")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0 or args.batch_size <= 0:
        raise ValueError("--sample-fps and --batch-size must be positive")

    repo_dir = Path(args.repo_dir)
    if not args.skip_install:
        setup_repository(repo_dir)

    repo_src = repo_dir / "src"
    if not repo_src.is_dir():
        raise FileNotFoundError(f"DA3 source directory not found: {repo_src}")
    sys.path.insert(0, str(repo_src))

    # This pipeline only needs prediction.depth. Avoid importing optional
    # binary exporters (GLB/COLMAP/3DGS) that may not match Kaggle's NumPy.
    export_init = repo_src / "depth_anything_3" / "utils" / "export" / "__init__.py"
    export_init.write_text(
        '"""Minimal exporter for the Kaggle monocular-video pipeline."""\n\n'
        'def export(*args, **kwargs):\n'
        '    raise RuntimeError("Export is disabled in this pipeline; export_dir must be None.")\n'
    )

    # Imports must happen after the project has been installed / added to sys.path.
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm.auto import tqdm

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.visualize import visualize_depth

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is unavailable. In Kaggle, set Accelerator to T4 x2 GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name} (compute capability {major}.{minor})")
    if (major, minor) < (7, 0):
        raise RuntimeError(
            "This PyTorch build does not support the selected legacy GPU reliably. "
            "Select T4 x2 GPU in Kaggle Settings, restart the session, and rerun."
        )

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_dir = Path(args.output_dir)
    depth_dir = output_dir / "depth_arrays"
    frame_dir = output_dir / "depth_frames"
    depth_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if source_fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot read video metadata: {video_path}")

    sample_every = max(1, round(source_fps / args.sample_fps))
    actual_fps = source_fps / sample_every
    output_height = min(height, args.max_output_height)
    output_width = int(width * output_height / height)
    output_width -= output_width % 2  # MP4 encoders require an even width.

    print(f"Video: {width}x{height}, {source_fps:.2f} FPS, {frame_count} frames")
    print(f"Sampling: every {sample_every} frame ({actual_fps:.2f} FPS)")
    print(f"Loading model: {args.model}")
    model = DepthAnything3.from_pretrained(args.model).to("cuda").eval()

    video_output = output_dir / "depth_video.mp4"
    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {video_output}")

    def process_batch(images_rgb: list[np.ndarray], output_ids: list[int]) -> None:
        prediction = model.inference(
            image=images_rgb,
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        for depth, output_id in zip(prediction.depth, output_ids):
            np.save(depth_dir / f"{output_id:06d}.npy", depth.astype(np.float32))
            depth_rgb = visualize_depth(depth).astype(np.uint8)
            depth_rgb = cv2.resize(
                depth_rgb, (output_width, output_height), interpolation=cv2.INTER_NEAREST
            )
            Image.fromarray(depth_rgb).save(frame_dir / f"{output_id:06d}.jpg", quality=95)
            writer.write(cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))

    cap = cv2.VideoCapture(str(video_path))
    images_rgb: list[np.ndarray] = []
    output_ids: list[int] = []
    saved = 0
    try:
        for source_index in tqdm(range(frame_count), desc="Depth inference"):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if source_index % sample_every:
                continue

            images_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            output_ids.append(saved)
            saved += 1
            if len(images_rgb) >= args.batch_size:
                process_batch(images_rgb, output_ids)
                images_rgb, output_ids = [], []
                torch.cuda.empty_cache()

        if images_rgb:
            process_batch(images_rgb, output_ids)
    finally:
        cap.release()
        writer.release()

    print(f"Completed {saved} sampled frames.")
    print(f"Depth video: {video_output}")
    print(f"Depth arrays: {depth_dir}")


if __name__ == "__main__":
    main()
