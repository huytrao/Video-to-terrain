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
OUTPUT_DIR = Path("/kaggle/working/da3_3d_glb_output")

# DA3-BASE là any-view model phù hợp VRAM 16 GB của T4.
MODEL_ID = "depth-anything/DA3-BASE"
# Keep this deliberately conservative for a 16-GB T4. The all-view attention
# cost scales quadratically in both number of views and image tokens.
PROCESS_RES = 378
MAX_VIEWS = 8
MAX_POINTS = 500_000
# ====================================================


def run_command(command):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.check_call(command)


# Clone và cài các Python dependency tối thiểu. --no-deps giữ nguyên cặp
# torch/torchvision CUDA do Kaggle cài sẵn.
if not (REPO_DIR / ".git").is_dir():
    run_command(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

run_command([
    sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR),
])
run_command([
    sys.executable, "-m", "pip", "install", "-q", "--no-deps",
    "addict", "e3nn", "einops", "evo", "huggingface_hub",
    "natsort", "colorama", "omegaconf", "pyyaml", "safetensors",
    "tqdm", "trimesh",
])

REPO_SRC = REPO_DIR / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

# Upstream exporter imports COLMAP and 3D Gaussian dependencies eagerly.
# Chỉ giữ GLB exporter để tránh pycolmap/gsplat không cần thiết.
export_init = REPO_SRC / "depth_anything_3" / "utils" / "export" / "__init__.py"
export_init.write_text(
    '"""GLB-only exporter for Kaggle video reconstruction."""\n\n'
    'from .glb import export_to_glb\n\n'
    'def export(prediction, export_format, export_dir, **kwargs):\n'
    '    if export_format != "glb":\n'
    '        raise ValueError("This Kaggle pipeline supports GLB export only.")\n'
    '    return export_to_glb(prediction, export_dir, **kwargs.get("glb", {}))\n'
)

# A previous depth-video cell may have imported DA3 with a minimal exporter.
# Purge its modules so this cell reloads the GLB-only exporter written above.
for module_name in list(sys.modules):
    if module_name == "depth_anything_3" or module_name.startswith("depth_anything_3."):
        del sys.modules[module_name]

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from depth_anything_3.api import DepthAnything3

if not torch.cuda.is_available():
    raise RuntimeError("Hãy bật T4 x2 GPU: Kaggle Settings → Accelerator.")

gpu_name = torch.cuda.get_device_name(0)
gpu_capability = torch.cuda.get_device_capability(0)
print(f"GPU: {gpu_name}; compute capability: {gpu_capability}")
if gpu_capability[0] < 7:
    raise RuntimeError("Không dùng P100. Chọn T4 x2 GPU và Restart Session.")

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy video: {VIDEO_PATH}")

FRAMES_DIR = OUTPUT_DIR / "input_frames"
GLB_DIR = OUTPUT_DIR / "glb"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
GLB_DIR.mkdir(parents=True, exist_ok=True)

# Lấy các frame phân bố đều theo toàn bộ video. Multi-view DA3 cần một tập
# frame vừa phải; đưa quá nhiều frame cùng lúc sẽ làm đầy VRAM.
cap = cv2.VideoCapture(str(VIDEO_PATH))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
source_fps = cap.get(cv2.CAP_PROP_FPS)
if frame_count <= 0 or source_fps <= 0:
    cap.release()
    raise RuntimeError("Không thể đọc metadata video.")

num_views = min(MAX_VIEWS, frame_count)
target_indices = np.linspace(0, frame_count - 1, num_views, dtype=np.int64)
target_set = set(int(index) for index in target_indices)

frame_paths = []
saved = 0
for source_index in tqdm(range(frame_count), desc="Trích frame cho 3D"):
    ok, frame_bgr = cap.read()
    if not ok:
        break
    if source_index not in target_set:
        continue

    frame_path = FRAMES_DIR / f"{saved:04d}.jpg"
    if not cv2.imwrite(str(frame_path), frame_bgr):
        raise RuntimeError(f"Không thể ghi frame: {frame_path}")
    frame_paths.append(str(frame_path))
    saved += 1
cap.release()

if len(frame_paths) < 2:
    raise RuntimeError("Cần ít nhất 2 frame để dựng scene 3D.")

print(f"Đã chọn {len(frame_paths)} frame từ {frame_count} frame ({source_fps:.2f} FPS).")
print(f"Tải model: {MODEL_ID}")

model = DepthAnything3.from_pretrained(MODEL_ID).to("cuda").eval()

print("Đang dựng geometry và xuất GLB...")
prediction = model.inference(
    image=frame_paths,
    process_res=PROCESS_RES,
    process_res_method="upper_bound_resize",
    export_dir=str(GLB_DIR),
    export_format="glb",
    use_ray_pose=False,
    conf_thresh_percentile=40.0,
    num_max_points=MAX_POINTS,
    show_cameras=True,
)

glb_path = GLB_DIR / "scene.glb"
if not glb_path.is_file():
    raise RuntimeError(f"DA3 không tạo được GLB tại: {glb_path}")

print("\nHoàn tất reconstruction 3D.")
print(f"GLB: {glb_path}")
print(f"Kích thước: {glb_path.stat().st_size / 1024 / 1024:.1f} MB")
print("Mở file scene.glb bằng Blender, MeshLab hoặc Babylon.js Sandbox.")
