"""Stage B: Depth Anything 3 - depth maps, intrinsics, poses, point cloud."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from src.v2t.config import CFG

# Import error handler
from src.v2t.geometry.da3_error_handler import (
    DA3ErrorHandler, DA3Config, clamp_depth_maps, validate_camera_params,
    process_point_cloud_chunked, ErrorType
)

# Import local DA3 (vendored under <project_root>/depth-anything-3, portable
# across machines/Kaggle instead of a hardcoded absolute dev path)
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DA3_REPO = _PROJECT_ROOT / "depth-anything-3" / "src"
if str(DA3_REPO) not in sys.path:
    sys.path.insert(0, str(DA3_REPO))

from depth_anything_3.api import DepthAnything3


def run_da3(frames_dir: Path, depth_dir: Path, geometry_dir: Path):
    """Run DA3 on sampled frames with comprehensive error handling."""
    depth_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)

    # Load frames
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        raise RuntimeError(f"No frames found in {frames_dir}")

    images = []
    for f in frame_files:
        img = Image.open(f).convert("RGB")
        images.append(np.array(img))

    print(f"Loading {len(images)} frames for DA3...")

    # Device
    device = CFG.da3_device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    print(f"Using device: {device}")

    # Configure error handler
    error_config = DA3Config(
        model=CFG.da3_model,
        process_res=CFG.da3_process_res,
        batch_size=2,  # DA3 processes all frames at once, but we can chunk if needed
        device=device,
        max_retries=3,
    )
    handler = DA3ErrorHandler(error_config, verbose=True)

    def _run_inference(config_dict):
        """Internal inference function wrapped by error handler."""
        # Load model with error handling
        print(f"Loading DA3 model: {config_dict['model']}...")
        try:
            model = DepthAnything3.from_pretrained(f"depth-anything/{config_dict['model']}")
            model = model.to(device=config_dict['device'])
            model.eval()
        except Exception as e:
            # Let error handler diagnose
            raise

        # Run inference with error handling
        print("Running DA3 inference...")
        try:
            with torch.inference_mode():
                prediction = model.inference(
                    images,
                    extrinsics=None,
                    intrinsics=None,
                    align_to_input_ext_scale=False,
                    infer_gs=False,
                    use_ray_pose=True,
                    process_res=config_dict['process_res'],
                    export_dir=str(depth_dir),
                    export_format="mini_npz"
                )
        except Exception as e:
            # Clear GPU cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            raise

        return prediction

    # Run with error handling and fallback
    prediction = handler.run_with_fallback(_run_inference)

    # Extract outputs
    depth_maps = prediction.depth  # (N, H, W)
    conf_maps = prediction.conf    # (N, H, W) or None for metric models
    extrinsics = prediction.extrinsics  # (N, 3, 4) or (N, 4, 4) or None
    intrinsics = prediction.intrinsics  # (N, 3, 3) or None
    processed_images = prediction.processed_images  # (N, H, W, 3)

    # Handle metric models that don't return confidence (e.g., da3metric-large)
    if conf_maps is None:
        print("DA3 metric model: no confidence map returned, creating dummy (all ones)")
        conf_maps = np.ones_like(depth_maps, dtype=np.float32)

    # Handle models that don't return extrinsics/intrinsics
    if extrinsics is None:
        print("DA3: no extrinsics returned, creating dummy identity poses")
        N = depth_maps.shape[0]
        extrinsics = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
    if intrinsics is None:
        print("DA3: no intrinsics returned, creating dummy intrinsics")
        N = depth_maps.shape[0]
        H, W = depth_maps.shape[1], depth_maps.shape[2]
        # Approximate focal length from process_res
        f = max(H, W) * 1.2
        cx, cy = W / 2, H / 2
        intrinsics = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)[None].repeat(N, axis=0)

    print(f"Depth: {depth_maps.shape}, Conf: {conf_maps.shape}")
    print(f"Extrinsics: {extrinsics.shape}, Intrinsics: {intrinsics.shape}")

    # ROBUST FIX 1: Clamp depth maps to valid range (handle NaN/Inf)
    depth_maps = clamp_depth_maps(depth_maps, min_val=0.01, max_val=100.0)

    # Ensure extrinsics is (N, 4, 4)
    if extrinsics.shape[-2:] == (3, 4):
        N = extrinsics.shape[0]
        extrinsics_4x4 = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
        extrinsics_4x4[:, :3, :] = extrinsics
        extrinsics = extrinsics_4x4

    # ROBUST FIX 2: Validate and fix camera parameters
    intrinsics, extrinsics = validate_camera_params(intrinsics, extrinsics)

    # Save depth as .npy (clamped)
    for i, (d, c) in enumerate(zip(depth_maps, conf_maps)):
        np.save(depth_dir / f"depth_{i:06d}.npy", d.astype(np.float32))
        np.save(depth_dir / f"conf_{i:06d}.npy", c.astype(np.float32))

    # Save camera parameters
    camera_intrinsics = {
        "intrinsics": intrinsics.tolist(),
        "width": processed_images.shape[2],
        "height": processed_images.shape[1],
        "note": "Estimated by DA3 ray pose (no ground truth calibration)"
    }
    with open(geometry_dir / "camera_intrinsics.json", 'w') as f:
        json.dump(camera_intrinsics, f, indent=2)

    camera_poses = {
        "extrinsics": extrinsics.tolist(),  # world-to-camera
        "note": "Estimated by DA3 ray pose (metric scale relative)"
    }
    with open(geometry_dir / "camera_poses.json", 'w') as f:
        json.dump(camera_poses, f, indent=2)

    # ROBUST FIX 3: Generate point cloud with chunked processing (memory safe)
    print("Generating point cloud (chunked)...")
    points, colors = process_point_cloud_chunked(
        depth_maps=depth_maps,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=processed_images,
        max_points_per_chunk=500_000,
        max_total_points=2_000_000,
    )

    if len(points) > 0:
        # Save PLY
        ply_path = geometry_dir / "point_cloud.ply"
        save_ply(ply_path, points, colors)
        print(f"Point cloud saved: {ply_path} ({len(points)} points)")
    else:
        print("WARNING: No valid points generated")

    return {
        "depth_dir": str(depth_dir),
        "camera_intrinsics": str(geometry_dir / "camera_intrinsics.json"),
        "camera_poses": str(geometry_dir / "camera_poses.json"),
        "point_cloud": str(geometry_dir / "point_cloud.ply")
    }


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    """Save point cloud as binary PLY."""
    import struct
    n = len(points)
    with open(path, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {n}\n'.encode())
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property uchar red\n')
        f.write(b'property uchar green\n')
        f.write(b'property uchar blue\n')
        f.write(b'end_header\n')
        for i in range(n):
            f.write(struct.pack('<fffBBB',
                float(points[i, 0]), float(points[i, 1]), float(points[i, 2]),
                int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])))


def run_stage_b():
    paths = CFG.paths
    return run_da3(paths.frames, paths.depth, paths.geometry)


if __name__ == "__main__":
    run_stage_b()
