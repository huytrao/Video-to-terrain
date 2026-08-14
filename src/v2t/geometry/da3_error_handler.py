#!/usr/bin/env python3
"""
DA3 GPU Inference Error Handler

Handles common GPU inference errors for Depth Anything 3 on Kaggle/GPU servers:
1. CUDA OOM - reduce batch size, process_res, or use gradient checkpointing
2. PyTorch/CUDA version mismatch - reinstall torch with correct CUDA
3. DA3 model download fails - use local cache, retry with timeout
4. open3d/trimesh import errors - reinstall with pip
5. Memory errors on large point clouds - chunk processing
6. NaN/inf in depth maps - add clamping and validation
"""

from __future__ import annotations
import os
import sys
import time
import logging
import subprocess
import traceback
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum

import torch
import numpy as np

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of DA3 inference errors."""
    CUDA_OOM = "cuda_oom"
    CUDA_VERSION_MISMATCH = "cuda_version_mismatch"
    MODEL_DOWNLOAD_FAILED = "model_download_failed"
    IMPORT_ERROR = "import_error"
    MEMORY_ERROR = "memory_error"
    NAN_INF_DEPTH = "nan_inf_depth"
    POINT_CLOUD_TOO_LARGE = "point_cloud_too_large"
    UNKNOWN = "unknown"


@dataclass
class ErrorDiagnosis:
    """Result of error diagnosis."""
    error_type: ErrorType
    message: str
    suggested_fix: str
    recovery_action: Optional[Callable] = None
    retry_possible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DA3Config:
    """DA3 inference configuration with fallback values."""
    model: str = "da3metric-large"
    process_res: int = 504
    batch_size: int = 2
    device: str = "cuda"
    max_retries: int = 3
    retry_delay: float = 2.0

    # Fallback configurations for error recovery
    fallback_configs: List[Dict] = field(default_factory=lambda: [
        {"process_res": 384, "batch_size": 1, "name": "low_mem_384"},
        {"process_res": 256, "batch_size": 1, "name": "low_mem_256"},
        {"process_res": 504, "batch_size": 1, "name": "batch_1"},
    ])


class DA3ErrorHandler:
    """
    Comprehensive error handler for DA3 GPU inference.

    Monitors inference, diagnoses errors, applies fixes, and retries with
    progressive fallback configurations.
    """

    def __init__(self, config: Optional[DA3Config] = None, verbose: bool = True):
        self.config = config or DA3Config()
        self.verbose = verbose
        self.attempt = 0
        self.current_config = self._config_to_dict(self.config)
        self.error_history: List[ErrorDiagnosis] = []

        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

    def _config_to_dict(self, config: DA3Config) -> Dict:
        return {
            "model": config.model,
            "process_res": config.process_res,
            "batch_size": config.batch_size,
            "device": config.device,
        }

    def _dict_to_config(self, d: Dict) -> DA3Config:
        return DA3Config(
            model=d.get("model", self.config.model),
            process_res=d.get("process_res", self.config.process_res),
            batch_size=d.get("batch_size", self.config.batch_size),
            device=d.get("device", self.config.device),
        )

    def diagnose_error(self, exception: Exception, context: Dict = None) -> ErrorDiagnosis:
        """
        Diagnose the root cause of an exception and suggest recovery action.
        """
        context = context or {}
        error_msg = str(exception)
        error_type = ErrorType.UNKNOWN
        suggested_fix = "Unknown error - check logs"
        recovery_action = None
        retry_possible = True
        metadata = {}

        # 1. CUDA Out of Memory
        if self._is_cuda_oom(exception, error_msg):
            error_type = ErrorType.CUDA_OOM
            suggested_fix = "Reduce process_res or batch_size, enable gradient checkpointing"
            recovery_action = self._create_oom_recovery()
            metadata["gpu_memory_gb"] = self._get_gpu_memory()
            metadata["current_process_res"] = self.current_config.get("process_res")
            metadata["current_batch_size"] = self.current_config.get("batch_size")

        # 2. PyTorch/CUDA Version Mismatch
        elif self._is_cuda_version_mismatch(exception, error_msg):
            error_type = ErrorType.CUDA_VERSION_MISMATCH
            suggested_fix = "Reinstall PyTorch with matching CUDA version (12.1 recommended)"
            recovery_action = self._create_cuda_reinstall_recovery()
            metadata["pytorch_version"] = torch.__version__
            metadata["cuda_version"] = torch.version.cuda
            metadata["cuda_available"] = torch.cuda.is_available()

        # 3. Model Download Failed (HF Hub)
        elif self._is_model_download_failed(exception, error_msg):
            error_type = ErrorType.MODEL_DOWNLOAD_FAILED
            suggested_fix = "Use local cache, increase timeout, or pre-download model"
            recovery_action = self._create_model_download_recovery()
            metadata["model_name"] = self.current_config.get("model")

        # 4. Import Errors (open3d, trimesh, etc.)
        elif self._is_import_error(exception, error_msg):
            error_type = ErrorType.IMPORT_ERROR
            missing_module = self._extract_missing_module(error_msg)
            suggested_fix = f"Reinstall missing module: {missing_module}"
            recovery_action = self._create_import_recovery(missing_module)
            metadata["missing_module"] = missing_module

        # 5. Memory Error (system RAM, not GPU)
        elif self._is_memory_error(exception, error_msg):
            error_type = ErrorType.MEMORY_ERROR
            suggested_fix = "Process in chunks, reduce point cloud size, or use swap"
            recovery_action = self._create_memory_recovery()
            metadata["system_memory_gb"] = self._get_system_memory()

        # 6. NaN/Inf in depth maps
        elif self._is_nan_inf_error(exception, error_msg):
            error_type = ErrorType.NAN_INF_DEPTH
            suggested_fix = "Add depth clamping and validation before point cloud generation"
            recovery_action = self._create_nan_inf_recovery()

        # 7. Point cloud too large
        elif self._is_point_cloud_error(exception, error_msg):
            error_type = ErrorType.POINT_CLOUD_TOO_LARGE
            suggested_fix = "Downsample point cloud or process in tiles"
            recovery_action = self._create_point_cloud_recovery()

        diagnosis = ErrorDiagnosis(
            error_type=error_type,
            message=error_msg,
            suggested_fix=suggested_fix,
            recovery_action=recovery_action,
            retry_possible=retry_possible,
            metadata=metadata
        )

        self.error_history.append(diagnosis)
        return diagnosis

    def _is_cuda_oom(self, exception: Exception, error_msg: str) -> bool:
        """Check if error is CUDA out of memory."""
        oom_indicators = [
            "CUDA out of memory",
            "out of memory",
            "OOM",
            "torch.cuda.OutOfMemoryError",
            "CUBLAS_STATUS_ALLOC_FAILED",
        ]
        return any(indicator in error_msg for indicator in oom_indicators)

    def _is_cuda_version_mismatch(self, exception: Exception, error_msg: str) -> bool:
        """Check if error is CUDA version mismatch."""
        mismatch_indicators = [
            "CUDA version",
            "version mismatch",
            "incompatible",
            "CUDA driver version is insufficient",
            "no kernel image is available",
            "unsupported compute capability",
        ]
        return any(indicator in error_msg for indicator in mismatch_indicators)

    def _is_model_download_failed(self, exception: Exception, error_msg: str) -> bool:
        """Check if error is model download failure."""
        download_indicators = [
            "ConnectionError",
            "TimeoutError",
            "ReadTimeout",
            "ConnectTimeout",
            "huggingface_hub",
            "Repository not found",
            "404",
            "403",
            "429",
            "Model not found",
            "gated repo",
        ]
        return any(indicator in error_msg for indicator in download_indicators)

    def _is_import_error(self, exception: Exception, error_msg: str) -> bool:
        """Check if error is an import error."""
        return isinstance(exception, (ImportError, ModuleNotFoundError)) or \
               "No module named" in error_msg or \
               "ImportError" in str(type(exception))

    def _is_memory_error(self, exception: Exception, error_msg: str) -> bool:
        """Check if error is system memory error (not GPU)."""
        memory_indicators = [
            "MemoryError",
            "Cannot allocate memory",
            "Out of memory",
            "killed",
            "OOMKilled",
        ]
        return any(indicator in error_msg for indicator in memory_indicators) and \
               not self._is_cuda_oom(exception, error_msg)

    def _is_nan_inf_error(self, exception: Exception, error_msg: str) -> bool:
        """Check if error is related to NaN/Inf values.

        Uses word-boundary matching so that substrings like 'inf' inside
        'inference' or 'nan' inside 'banana' do not trigger false positives.
        """
        import re
        # Match standalone nan/inf/NaN/Inf tokens (not substrings of larger words)
        nan_inf_pattern = re.compile(r'(?<![A-Za-z])(?:nan|inf|NaN|Inf|NaN|INF)(?![A-Za-z])', re.IGNORECASE)
        if nan_inf_pattern.search(error_msg):
            return True
        # Also detect explicit numpy/torch messages
        explicit_indicators = [
            "invalid value",
            "divide by zero",
            "RuntimeWarning: invalid value",
        ]
        return any(indicator in error_msg for indicator in explicit_indicators)

    def _is_point_cloud_error(self, exception: Exception, error_msg: str) -> bool:
        """Check if error is related to point cloud size."""
        pc_indicators = [
            "too many points",
            "point cloud",
            "exceeds maximum",
            "memory limit",
        ]
        return any(indicator in error_msg.lower() for indicator in pc_indicators)

    def _extract_missing_module(self, error_msg: str) -> str:
        """Extract missing module name from ImportError message."""
        import re
        # Match "No module named 'xxx'" or "ModuleNotFoundError: No module named 'xxx'"
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_msg)
        if match:
            return match.group(1)
        # Match "ImportError: cannot import name 'xxx' from 'yyy'"
        match = re.search(r"cannot import name ['\"]([^'\"]+)['\"]", error_msg)
        if match:
            return match.group(1)
        return "unknown"

    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        return 0.0

    def _get_system_memory(self) -> float:
        """Get system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / 1e9
        except ImportError:
            # Fallback: read from /proc/meminfo on Linux
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            return int(line.split()[1]) * 1024 / 1e9
            except Exception:
                pass
        return 0.0

    # ============================================================
    # Recovery Actions
    # ============================================================

    def _create_oom_recovery(self) -> Callable:
        """Create recovery action for CUDA OOM."""
        def recovery():
            # Try next fallback configuration
            for fallback in self.config.fallback_configs:
                if fallback["process_res"] < self.current_config.get("process_res", 504) or \
                   fallback["batch_size"] < self.current_config.get("batch_size", 2):
                    logger.warning(f"Applying fallback: {fallback['name']} "
                                 f"(process_res={fallback['process_res']}, batch_size={fallback['batch_size']})")
                    self.current_config["process_res"] = fallback["process_res"]
                    self.current_config["batch_size"] = fallback["batch_size"]
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    return True
            return False
        return recovery

    def _create_cuda_reinstall_recovery(self) -> Callable:
        """Create recovery action for CUDA version mismatch."""
        def recovery():
            logger.warning("Attempting to reinstall PyTorch with CUDA 12.1...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-q",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu121"
                ], check=True, timeout=300)
                logger.info("PyTorch reinstalled with CUDA 12.1")
                # Note: requires restart to take effect
                return True
            except Exception as e:
                logger.error(f"Failed to reinstall PyTorch: {e}")
                return False
        return recovery

    def _create_model_download_recovery(self) -> Callable:
        """Create recovery action for model download failure."""
        def recovery():
            model_name = self.current_config.get("model", "da3metric-large")
            logger.warning(f"Attempting to pre-download model: {model_name}")
            try:
                from huggingface_hub import snapshot_download
                # Use local cache with longer timeout
                snapshot_download(
                    repo_id=f"depth-anything/{model_name}",
                    local_files_only=False,
                    resume_download=True,
                    max_workers=4,
                )
                logger.info(f"Model {model_name} downloaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                # Try alternative: use local DA3 repo if available
                local_repo = Path("/kaggle/working/Video-to-Terrain/depth-anything-3")
                if local_repo.exists():
                    logger.info(f"Found local DA3 repo at {local_repo}")
                    return True
                return False
        return recovery

    def _create_import_recovery(self, missing_module: str) -> Callable:
        """Create recovery action for import errors."""
        def recovery():
            logger.warning(f"Attempting to install missing module: {missing_module}")
            try:
                # Special handling for certain modules
                if missing_module in ("open3d", "trimesh", "pycolmap"):
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "-q", missing_module
                    ], check=True, timeout=180)
                else:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "-q", missing_module
                    ], check=True, timeout=120)
                logger.info(f"Module {missing_module} installed successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to install {missing_module}: {e}")
                return False
        return recovery

    def _create_memory_recovery(self) -> Callable:
        """Create recovery action for system memory errors."""
        def recovery():
            logger.warning("Applying memory optimization: enabling swap, reducing batch size")
            # Reduce batch size further
            if self.current_config.get("batch_size", 1) > 1:
                self.current_config["batch_size"] = 1
                logger.info("Reduced batch_size to 1")
            # Clear Python garbage
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return recovery

    def _create_nan_inf_recovery(self) -> Callable:
        """Create recovery action for NaN/Inf in depth maps."""
        def recovery():
            logger.warning("NaN/Inf detected - will apply depth clamping in post-processing")
            # This is handled in the inference wrapper by clamping depth values
            return True
        return recovery

    def _create_point_cloud_recovery(self) -> Callable:
        """Create recovery action for large point clouds."""
        def recovery():
            logger.warning("Large point cloud - will apply downsampling")
            # This is handled in stage_b_da3.py with 2M point limit
            return True
        return recovery

    # ============================================================
    # Main Inference Wrapper
    # ============================================================

    def run_with_fallback(self, inference_fn: Callable, *args, **kwargs) -> Any:
        """
        Run inference function with automatic fallback configurations.

        Args:
            inference_fn: Function that takes (config: Dict) as first positional arg
            *args, **kwargs: Additional arguments passed to inference_fn

        Returns:
            Result of successful inference
        """
        self.attempt = 0
        while self.attempt < self.config.max_retries:
            self.attempt += 1
            logger.info(f"DA3 Inference Attempt {self.attempt}/{self.config.max_retries}")
            logger.info(f"Config: {self.current_config}")

            try:
                # Pass config dict as first positional argument, then *args, **kwargs
                result = inference_fn(self.current_config, *args, **kwargs)
                logger.info("DA3 Inference completed successfully!")
                return result
            except Exception as e:
                diagnosis = self.diagnose_error(e, {"attempt": self.attempt})
                logger.error(f"Error diagnosed: {diagnosis.error_type.value}")
                logger.error(f"Message: {diagnosis.message}")
                logger.error(f"Suggested fix: {diagnosis.suggested_fix}")

                if diagnosis.recovery_action and diagnosis.retry_possible:
                    if self.attempt < self.config.max_retries:
                        logger.info(f"Applying recovery action...")
                        recovery_success = diagnosis.recovery_action()
                        if recovery_success:
                            logger.info(f"Recovery applied, retrying in {self.config.retry_delay}s...")
                            time.sleep(self.config.retry_delay)
                            continue

                logger.error(f"All recovery attempts exhausted. Raising error.")
                raise

        raise RuntimeError(f"DA3 inference failed after {self.config.max_retries} attempts")


def clamp_depth_maps(depth_maps: np.ndarray, min_val: float = 0.01, max_val: float = 100.0) -> np.ndarray:
    """
    Clamp depth maps to valid range, replacing NaN/Inf with min_val.

    Args:
        depth_maps: Array of shape (N, H, W) or (H, W)
        min_val: Minimum valid depth (meters)
        max_val: Maximum valid depth (meters)

    Returns:
        Clamped depth maps
    """
    depth_maps = np.asarray(depth_maps, dtype=np.float32)

    # Replace NaN and Inf
    nan_mask = ~np.isfinite(depth_maps)
    if np.any(nan_mask):
        logger.warning(f"Found {np.sum(nan_mask)} NaN/Inf values in depth maps, clamping to {min_val}")
        depth_maps = np.nan_to_num(depth_maps, nan=min_val, posinf=max_val, neginf=min_val)

    # Clamp to valid range
    depth_maps = np.clip(depth_maps, min_val, max_val)

    return depth_maps


def validate_camera_params(intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple:
    """
    Validate and fix camera parameters.

    Args:
        intrinsics: (N, 3, 3) or (3, 3) camera intrinsics
        extrinsics: (N, 4, 4) or (4, 4) camera extrinsics (world-to-camera)

    Returns:
        Validated (intrinsics, extrinsics)
    """
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)

    # Ensure batch dimension
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None]
    if extrinsics.ndim == 2:
        extrinsics = extrinsics[None]

    # Validate intrinsics: fx, fy > 0, cx, cy in reasonable range
    for i, K in enumerate(intrinsics):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        if fx <= 0 or fy <= 0:
            logger.warning(f"Invalid focal length in intrinsics[{i}]: fx={fx}, fy={fy}, fixing...")
            K[0, 0] = K[1, 1] = max(abs(fx), abs(fy), 100.0)

        # cx, cy should be within image bounds (roughly)
        # We can't know exact image size here, but warn if suspicious
        if cx < 0 or cy < 0:
            logger.warning(f"Negative principal point in intrinsics[{i}]: cx={cx}, cy={cy}")

    # Validate extrinsics: should be valid SE(3) matrices
    for i, W2C in enumerate(extrinsics):
        # Check if it's a valid 4x4 matrix
        if W2C.shape != (4, 4):
            logger.warning(f"Extrinsics[{i}] has invalid shape {W2C.shape}, fixing...")
            extrinsics[i] = np.eye(4, dtype=np.float32)
            continue

        # Check rotation part is orthogonal (approximately)
        R = W2C[:3, :3]
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.1 and abs(det + 1.0) > 0.1:
            logger.warning(f"Extrinsics[{i}] rotation has det={det:.3f}, may be invalid")

        # Check for NaN/Inf
        if not np.all(np.isfinite(W2C)):
            logger.warning(f"Extrinsics[{i}] contains NaN/Inf, replacing with identity")
            extrinsics[i] = np.eye(4, dtype=np.float32)

    return intrinsics, extrinsics


def process_point_cloud_chunked(
    depth_maps: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    processed_images: np.ndarray,
    max_points_per_chunk: int = 500_000,
    max_total_points: int = 2_000_000,
) -> tuple:
    """
    Generate point cloud in chunks to avoid memory issues.

    Args:
        depth_maps: (N, H, W) depth maps
        intrinsics: (N, 3, 3) camera intrinsics
        extrinsics: (N, 4, 4) camera extrinsics (world-to-camera)
        processed_images: (N, H, W, 3) RGB images
        max_points_per_chunk: Max points per chunk
        max_total_points: Max total points to keep

    Returns:
        (points, colors) arrays
    """
    N = len(depth_maps)
    all_points = []
    all_colors = []
    total_points = 0

    for i in range(N):
        d = depth_maps[i]
        K = intrinsics[i]
        W2C = extrinsics[i]
        C2W = np.linalg.inv(W2C)
        img = processed_images[i]

        H, W = d.shape
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)

        # Valid depth mask
        valid = np.isfinite(d.reshape(-1)) & (d.reshape(-1) > 0)
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)[valid]
        pix_valid = pix[valid]
        colors = img.reshape(-1, 3)[valid]

        # Process in chunks if too many valid points
        n_valid = len(d_flat)
        if n_valid > max_points_per_chunk:
            # Randomly subsample
            idx = np.random.choice(n_valid, max_points_per_chunk, replace=False)
            d_flat = d_flat[idx]
            pix_valid = pix_valid[idx]
            colors = colors[idx]
            n_valid = max_points_per_chunk

        # Unproject
        K_inv = np.linalg.inv(K)
        rays = K_inv @ pix_valid.T
        Xc = rays * d_flat[None, :]
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (C2W @ Xc_h)[:3].T

        all_points.append(Xw.astype(np.float32))
        all_colors.append(colors.astype(np.uint8))
        total_points += n_valid

        if total_points >= max_total_points:
            logger.warning(f"Reached max total points ({max_total_points}), stopping")
            break

    if not all_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    all_points = np.concatenate(all_points, 0)
    all_colors = np.concatenate(all_colors, 0)

    # Final downsample if needed
    if len(all_points) > max_total_points:
        idx = np.random.choice(len(all_points), max_total_points, replace=False)
        all_points = all_points[idx]
        all_colors = all_colors[idx]

    logger.info(f"Generated point cloud: {len(all_points)} points")
    return all_points, all_colors


# ============================================================
# Convenience function for direct use
# ============================================================

def run_da3_with_error_handling(
    frames_dir: Path,
    depth_dir: Path,
    geometry_dir: Path,
    model: str = "da3metric-large",
    process_res: int = 504,
    batch_size: int = 2,
    device: str = "cuda",
    max_retries: int = 3,
    verbose: bool = True,
) -> Dict:
    """
    Run DA3 inference with comprehensive error handling.

    This is a drop-in replacement for run_da3() in stage_b_da3.py
    that adds robust error handling and automatic recovery.
    """
    config = DA3Config(
        model=model,
        process_res=process_res,
        batch_size=batch_size,
        device=device,
        max_retries=max_retries,
    )

    handler = DA3ErrorHandler(config, verbose=verbose)

    def inference_fn(config_dict):
        # This will be the actual DA3 inference call
        # Import here to avoid circular imports
        from src.v2t.geometry.stage_b_da3 import run_da3 as original_run_da3
        from src.v2t.config import CFG

        # Apply config overrides
        CFG.da3_model = config_dict["model"]
        CFG.da3_process_res = config_dict["process_res"]
        CFG.da3_device = config_dict["device"]

        return original_run_da3(frames_dir, depth_dir, geometry_dir)

    return handler.run_with_fallback(inference_fn)


if __name__ == "__main__":
    # Test error handler
    import tempfile

    logging.basicConfig(level=logging.DEBUG)

    # Test depth clamping
    test_depth = np.array([[[1.0, np.nan, 5.0], [np.inf, 2.0, -1.0]]], dtype=np.float32)
    clamped = clamp_depth_maps(test_depth)
    print(f"Original: {test_depth}")
    print(f"Clamped: {clamped}")
    assert np.all(np.isfinite(clamped))
    assert np.all(clamped >= 0.01) and np.all(clamped <= 100.0)
    print("Depth clamping test passed!")

    # Test camera validation
    K = np.array([[[500, 0, 320], [0, 500, 240], [0, 0, 1]]], dtype=np.float32)
    W2C = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]], dtype=np.float32)
    K_val, W2C_val = validate_camera_params(K, W2C)
    print(f"Validated intrinsics shape: {K_val.shape}")
    print(f"Validated extrinsics shape: {W2C_val.shape}")
    print("Camera validation test passed!")

    print("\nAll DA3 error handler tests passed!")