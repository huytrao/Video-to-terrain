import sys
import subprocess
from pathlib import Path

# ==================== CẤU HÌNH ====================
REPO_URL = "https://github.com/bytedance-seed/depth-anything-3.git"
REPO_DIR = Path("/kaggle/working/depth-anything-3")

VIDEO_PATH = Path(
    "/kaggle/input/datasets/traoanhuy/test-depthanythingv3/"
    "20260810_150244_color.mp4"
)

OUTPUT_DIR = Path("/kaggle/working/depth_anything_3_output")
MODEL_ID = "depth-anything/DA3MONO-LARGE"

SAMPLE_FPS = 2.0
BATCH_SIZE = 2
PROCESS_RES = 504
MAX_OUTPUT_HEIGHT = 720
# ====================================================


def run_command(command):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.check_call(command)


# Clone Depth Anything 3
if not (REPO_DIR / ".git").is_dir():
    run_command([
        "git", "clone", "--depth", "1",
        REPO_URL,
        str(REPO_DIR),
    ])

# Cài dependencies vào Python kernel Kaggle hiện tại
run_command([
    sys.executable, "-m", "pip", "install",
    "-q", "--no-deps", "-e", str(REPO_DIR),
])

run_command([
    # Do not let pip resolve dependencies: Kaggle's bundled torch/torchvision
    # CUDA pair must remain untouched.
    sys.executable, "-m", "pip", "install", "-q", "--no-deps",
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
])

# Import sau khi clone/cài đặt
REPO_SRC = REPO_DIR / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

# Pipeline này chỉ cần prediction.depth, không cần xuất GLB/COLMAP/3DGS.
# Thay exporter của upstream bằng stub để các optional packages binary không
# được import (trimesh, pycolmap, gsplat) trên Kaggle Python 3.12.
export_init = REPO_SRC / "depth_anything_3" / "utils" / "export" / "__init__.py"
export_init.write_text(
    '"""Minimal exporter for the Kaggle monocular-video pipeline."""\n\n'
    'def export(*args, **kwargs):\n'
    '    raise RuntimeError("Export is disabled in this pipeline; export_dir must be None.")\n'
)

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

# Kiểm tra GPU T4
if not torch.cuda.is_available():
    raise RuntimeError("Hãy bật T4 x2 GPU trong Kaggle Settings → Accelerator.")

gpu_name = torch.cuda.get_device_name(0)
gpu_capability = torch.cuda.get_device_capability(0)

print(f"GPU: {gpu_name}")
print(f"Compute capability: {gpu_capability}")

if gpu_capability[0] < 7:
    raise RuntimeError("Không dùng P100. Hãy chuyển Kaggle Accelerator sang T4 x2 GPU.")

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy video: {VIDEO_PATH}")

# Tạo thư mục output
DEPTH_DIR = OUTPUT_DIR / "depth_arrays"
FRAME_DIR = OUTPUT_DIR / "depth_frames"

DEPTH_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

# Đọc metadata video
cap = cv2.VideoCapture(str(VIDEO_PATH))

source_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.release()

if source_fps <= 0 or frame_count <= 0:
    raise RuntimeError("Không thể đọc metadata video.")

sample_every = max(1, round(source_fps / SAMPLE_FPS))
actual_fps = source_fps / sample_every

output_height = min(height, MAX_OUTPUT_HEIGHT)
output_width = int(width * output_height / height)
output_width -= output_width % 2

print(f"Video: {width}x{height}")
print(f"FPS gốc: {source_fps:.2f}")
print(f"FPS xử lý: {actual_fps:.2f}")
print(f"Model: {MODEL_ID}")

# Tải model
model = DepthAnything3.from_pretrained(MODEL_ID).to("cuda").eval()

# Tạo video depth output
depth_video_path = OUTPUT_DIR / "depth_video.mp4"

writer = cv2.VideoWriter(
    str(depth_video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    actual_fps,
    (output_width, output_height),
)

if not writer.isOpened():
    raise RuntimeError("Không thể tạo file depth_video.mp4")


def process_batch(images_rgb, output_ids):
    prediction = model.inference(
        image=images_rgb,
        process_res=PROCESS_RES,
        process_res_method="upper_bound_resize",
        export_dir=None,
    )

    for depth, output_id in zip(prediction.depth, output_ids):
        np.save(
            DEPTH_DIR / f"{output_id:06d}.npy",
            depth.astype(np.float32),
        )

        depth_rgb = visualize_depth(depth).astype(np.uint8)

        depth_rgb = cv2.resize(
            depth_rgb,
            (output_width, output_height),
            interpolation=cv2.INTER_NEAREST,
        )

        Image.fromarray(depth_rgb).save(
            FRAME_DIR / f"{output_id:06d}.jpg",
            quality=95,
        )

        writer.write(cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))


# Inference video
cap = cv2.VideoCapture(str(VIDEO_PATH))

batch_images = []
batch_ids = []
saved_frames = 0

try:
    for source_index in tqdm(range(frame_count), desc="Depth inference"):
        ok, frame_bgr = cap.read()

        if not ok:
            break

        if source_index % sample_every != 0:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        batch_images.append(frame_rgb)
        batch_ids.append(saved_frames)
        saved_frames += 1

        if len(batch_images) >= BATCH_SIZE:
            process_batch(batch_images, batch_ids)

            batch_images = []
            batch_ids = []

            torch.cuda.empty_cache()

    if batch_images:
        process_batch(batch_images, batch_ids)

finally:
    cap.release()
    writer.release()

print(f"\nHoàn tất {saved_frames} frames.")
print(f"Video depth: {depth_video_path}")
print(f"Depth maps: {DEPTH_DIR}")
print(f"Depth images: {FRAME_DIR}")
