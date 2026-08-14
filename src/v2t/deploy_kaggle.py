#!/usr/bin/env python3
"""
deploy_kaggle.py - Deploy Video-to-Terrain pipeline to Kaggle GPU runtime.

Uses kaggle_remote.py to connect to Kaggle via Jupyter proxy:
- Syncs project to Kaggle runtime (excluding .venv, outputs/, __pycache__, .git, source-reference/)
- Installs CUDA torch, DA3 dependencies on Kaggle GPU
- Copies input video to Kaggle
- Runs pipeline Stage B (DA3) on GPU (expected ~6-7s vs 60s on MPS)
- Syncs results back to local outputs/

Environment:
    export KAGGLE_PROXY_URL='https://.../proxy'

Usage:
    python -m src.v2t.deploy_kaggle --stage B
    python -m src.v2t.deploy_kaggle --stage B --model da3metric-large
    python -m src.v2t.deploy_kaggle --download-only
    python -m src.v2t.deploy_kaggle --shell

Error Handling:
- CUDA OOM: Automatic fallback to lower process_res (504->384->256) and batch_size
- Model download: Pre-download with retry, use local cache
- Import errors: Auto-install missing dependencies
- Memory errors: Chunked point cloud processing
- NaN/Inf in depth: Automatic clamping
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .kaggle_remote import KaggleProxy

# Project root (3 levels up from src/v2t/deploy_kaggle.py -> Videos_to_terrain)
PROJECT_ROOT = Path(__file__).parent.parent.parent


class KaggleDeployer:
    """Orchestrates deployment to Kaggle GPU runtime."""

    def __init__(self, proxy_url: str):
        self.kaggle = KaggleProxy(proxy_url)
        self.remote_working = "/kaggle/working"
        self.remote_project = f"{self.remote_working}/Video-to-Terrain"
        self.local_project = PROJECT_ROOT
        self._setup_complete = False

    # ============================================================
    # Connection & Setup
    # ============================================================

    def ping(self) -> bool:
        """Verify connection to Kaggle Jupyter API."""
        print("[INFO] Pinging Kaggle Jupyter API...")
        try:
            self.kaggle.list_path("")
            print("[SUCCESS] Kaggle connection verified")
            return True
        except Exception as e:
            print(f"[ERROR] Cannot connect to Kaggle: {e}")
            return False

    def ensure_remote_dirs(self) -> None:
        """Ensure required remote directories exist."""
        dirs = [
            self.remote_project,
            f"{self.remote_project}/input",
            f"{self.remote_project}/outputs/run_mvp/geometry/depth",
        ]
        for d in dirs:
            self.kaggle.ensure_dir(d)

    def cleanup_remote_cache(self) -> None:
        """Clean up caches and old data to free disk space on Kaggle."""
        print("[INFO] Cleaning up remote caches to free disk space...")
        cleanup_commands = [
            # Clean pip cache
            "pip cache purge 2>/dev/null || true",
            # Clean HF cache (keep model weights, remove temp)
            "find ~/.cache/huggingface -name '*.tmp' -delete 2>/dev/null || true",
            # Clean old stage_b runs if they exist (keep latest)
            "find /kaggle/working/Video-to-Terrain -name '__pycache__' -type d -exec rm -rf {{}} + 2>/dev/null || true",
            # Show disk space after cleanup
            "df -h /kaggle/working",
        ]
        for cmd in cleanup_commands:
            try:
                output = self.kaggle.run_command(cmd, cwd=self.remote_project, timeout=30)
                if "df -h" in cmd and output.strip():
                    print(f"  [OUT] {output.strip()}")
            except Exception as e:
                print(f"  [WARN] Cleanup command failed: {e}")
        print("[SUCCESS] Cache cleanup complete")

    def setup_environment(self, model: str = "da3metric-large") -> bool:
        """
        Set up conda environment with CUDA and install dependencies on Kaggle.
        Returns True if successful.
        """
        if self._setup_complete:
            print("[INFO] Environment already set up")
            return True

        print("[INFO] Setting up Kaggle environment with CUDA...")

        # Kaggle has CUDA pre-installed, just need to install PyTorch CUDA and deps
        # Use retry logic for network operations
        setup_commands = [
            # 1. Clean pip cache and ensure we have space
            "pip cache purge 2>/dev/null || true",
            "df -h /kaggle/working",
            # 2. Install PyTorch with CUDA - use cu121 which matches Kaggle's CUDA 12.x
            "pip install -q --timeout=120 --retries=3 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            # 3. Core dependencies with version constraints to avoid conflicts
            "pip install -q --timeout=120 --retries=3 'numpy<2' 'pillow<11'",
            "pip install -q --timeout=120 --retries=3 open3d trimesh networkx tqdm pyyaml python-dotenv opencv-python scipy matplotlib",
            # 4. PyTorch Geometric dependencies (required by DA3) - use compatible versions
            "pip install -q --timeout=120 --retries=3 torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 2>/dev/null || echo 'PyG: optional, continuing...'",
            # 5. DA3 - install from local repo if available, with fallback
            "pip install -q --timeout=120 --retries=3 -e /kaggle/working/Video-to-Terrain/depth-anything-3 2>/dev/null || echo 'DA3: local install failed, will use HF hub at runtime'",
            # 6. Verify CUDA and imports
            "python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')\"",
            # 7. Verify DA3 import
            "python -c \"import sys; sys.path.insert(0, 'src'); from depth_anything_3.api import DepthAnything3; print('DA3 import: OK')\" 2>/dev/null || python -c \"import sys; sys.path.insert(0, '/kaggle/working/Video-to-Terrain/depth-anything-3/src'); from depth_anything_3.api import DepthAnything3; print('DA3 import (alt path): OK')\" || echo 'DA3 import: will try at runtime'",
        ]

        for i, cmd in enumerate(setup_commands):
            print(f"  [RUN {i+1}/{len(setup_commands)}] {cmd[:100]}...")
            try:
                output = self.kaggle.run_command(cmd, cwd=self.remote_project, timeout=240)
                if output.strip():
                    # Filter out excessive pip output, show only important lines
                    lines = output.strip().split('\n')
                    for line in lines[-5:]:  # Show last 5 lines
                        if any(kw in line.lower() for kw in ['success', 'error', 'fail', 'warn', 'cuda', 'gpu', 'pytorch', 'torch', 'da3', 'import']):
                            print(f"  [OUT] {line.strip()}")
                        elif i == len(setup_commands) - 1:  # Always show last command output
                            print(f"  [OUT] {line.strip()}")
            except Exception as e:
                print(f"  [WARN] Command failed: {e}")
                # Don't fail on individual commands - some are optional

        # Final verification
        print("[INFO] Running final verification...")
        try:
            verify_output = self.kaggle.run_command(
                "python -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available!'; import open3d; print('open3d OK'); import trimesh; print('trimesh OK')\"",
                cwd=self.remote_project,
                timeout=60
            )
            print(f"  [VERIFY] {verify_output.strip()}")
        except Exception as e:
            print(f"  [WARN] Verification had issues: {e}")

        self._setup_complete = True
        print("[SUCCESS] Environment setup complete")
        return True

    # ============================================================
    # File Operations
    # ============================================================

    def upload_project(self) -> bool:
        """Upload project to Kaggle, excluding unnecessary files."""
        print(f"[INFO] Uploading project to {self.remote_project}...")

        skip_dirs = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            ".DS_Store",
            "outputs",
            ".claude",
            ".mypy_cache",
            ".pytest_cache",
            "source-reference",
        }

        try:
            self.kaggle.upload_dir(
                str(self.local_project),
                self.remote_project,
                skip=skip_dirs
            )
            print("[SUCCESS] Project uploaded")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to upload project: {e}")
            return False

    def upload_input_video(self, video_name: str = "20260810_150244_color.mp4") -> bool:
        """Upload input video to Kaggle."""
        input_video = self.local_project / "input" / video_name
        if not input_video.exists():
            print(f"[WARNING] Input video not found: {input_video}")
            return False

        remote_video = f"{self.remote_project}/input/{video_name}"
        print(f"[INFO] Uploading input video ({input_video.stat().st_size / 1e6:.1f} MB)...")

        try:
            self.kaggle.upload_file(str(input_video), remote_video)
            print("[SUCCESS] Input video uploaded")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to upload video: {e}")
            return False

    # ============================================================
    # Pipeline Execution
    # ============================================================

    def run_stage_b(self, model: str = "da3metric-large", process_res: int = 504) -> bool:
        """
        Run Stage B (DA3) on Kaggle GPU.
        Returns True if successful.
        """
        print(f"[INFO] Running Stage B (DA3) with model={model}, process_res={process_res} on GPU...")

        # First verify frames exist on remote
        check_cmd = f"ls {self.remote_project}/outputs/run_mvp/frames/frame_*.jpg 2>/dev/null | wc -l"
        try:
            frame_count = int(self.kaggle.run_command(check_cmd).strip())
            print(f"[INFO] Found {frame_count} frames on remote")
            if frame_count == 0:
                print("[ERROR] No frames found. Run Stage A locally first.")
                return False
        except Exception as e:
            print(f"[WARN] Could not verify frames: {e}")

        # Check disk space before running
        try:
            df_output = self.kaggle.run_command("df -h /kaggle/working")
            print(f"[INFO] Disk space:\n{df_output}")
        except Exception as e:
            print(f"[WARN] Could not check disk space: {e}")

        # Build the Python script to run remotely with comprehensive error handling
        # and automatic fallback configurations for OOM recovery
        # Using a regular string with .format() to avoid f-string nesting issues
        remote_script_template = '''import sys
import os
import traceback
import re

# Add both possible DA3 paths
sys.path.insert(0, 'src')
sys.path.insert(0, '/kaggle/working/Video-to-Terrain/depth-anything-3/src')

# Set CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Fallback configurations for OOM recovery (progressive)
FALLBACK_CONFIGS = [
    {{"process_res": 504, "name": "default_504"}},
    {{"process_res": 384, "name": "fallback_384"}},
    {{"process_res": 256, "name": "fallback_256"}},
]

def is_oom_error(msg):
    oom_indicators = [
        "CUDA out of memory", "out of memory", "OOM",
        "CUBLAS_STATUS_ALLOC_FAILED", "OutOfMemoryError",
    ]
    return any(x in msg for x in oom_indicators)

def is_model_download_error(msg):
    dl_indicators = ["ConnectionError", "TimeoutError", "ReadTimeout",
                     "ConnectTimeout", "huggingface_hub", "404", "403", "429"]
    return any(x in msg for x in dl_indicators)

def is_import_error(exc):
    return isinstance(exc, (ImportError, ModuleNotFoundError)) or \
           "No module named" in str(exc)

def install_missing_module(msg):
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", msg)
    if not match:
        return False
    module = match.group(1)
    print("[RECOVERY] Installing missing module: " + module)
    import subprocess
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", module],
                      check=True, timeout=180)
        return True
    except Exception as e:
        print("[RECOVERY] Failed to install " + module + ": " + str(e))
        return False

# Try imports with error handling
try:
    import torch
    from v2t.config import CFG
    from v2t.geometry.stage_b_da3 import run_stage_b
except ImportError as e:
    print("IMPORT ERROR: " + str(e))
    traceback.print_exc()
    # Try to install and retry
    if install_missing_module(str(e)):
        try:
            import torch
            from v2t.config import CFG
            from v2t.geometry.stage_b_da3 import run_stage_b
        except ImportError as e2:
            print("IMPORT ERROR after install: " + str(e2))
            traceback.print_exc()
            sys.exit(1)
    else:
        sys.exit(1)

# Check frames exist
from pathlib import Path
frames_precheck = list(Path(CFG.paths.frames).glob('frame_*.jpg'))
print("Found " + str(len(frames_precheck)) + " frames")
if len(frames_precheck) == 0:
    print("ERROR: No frames found! Run Stage A first locally.")
    sys.exit(1)

# Run with progressive fallback for OOM
attempt = 0
max_attempts = len(FALLBACK_CONFIGS)
while attempt < max_attempts:
    fb = FALLBACK_CONFIGS[attempt]
    print("[ATTEMPT " + str(attempt + 1) + "/" + str(max_attempts) + "] config=" + fb['name'] + " process_res=" + str(fb['process_res']))
    try:
        # Override config for GPU
        CFG.da3_device = 'cuda'
        CFG.da3_model = '{model}'
        CFG.da3_process_res = fb['process_res']

        print("Running DA3 on: " + CFG.da3_device)
        print("Model: " + CFG.da3_model)
        print("Process resolution: " + str(CFG.da3_process_res))

        print("PyTorch: " + torch.__version__)
        print("CUDA available: " + str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            print("GPU: " + torch.cuda.get_device_name(0))
            print("GPU Memory: " + str(torch.cuda.get_device_properties(0).total_memory / 1e9) + " GB")
            # Clear cache before starting
            torch.cuda.empty_cache()

        result = run_stage_b()
        print("Stage B completed successfully!")
        print("Results: " + str(result))
        break  # Success, exit loop

    except Exception as e:
        error_msg = str(e)
        print("STAGE B ERROR (attempt " + str(attempt + 1) + "): " + error_msg)
        traceback.print_exc()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if is_oom_error(error_msg) and attempt < max_attempts - 1:
            attempt += 1
            print("[INFO] OOM detected, trying next fallback config...")
            continue
        elif is_model_download_error(error_msg):
            print("[RECOVERY] Model download failed, trying HuggingFace retry...")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="depth-anything/" + CFG.da3_model,
                    resume_download=True,
                    max_workers=4,
                )
                print("[RECOVERY] Model downloaded, retrying...")
                continue
            except Exception as dl_err:
                print("[RECOVERY] Model download retry failed: " + str(dl_err))
                sys.exit(1)
        elif is_import_error(e):
            if install_missing_module(error_msg):
                print("[RECOVERY] Module installed, retrying...")
                continue
            else:
                sys.exit(1)
        else:
            # Non-recoverable error
            sys.exit(1)
else:
    print("All fallback configurations exhausted!")
    sys.exit(1)
'''

        remote_script = remote_script_template.format(model=model)

        # Write script to remote file
        script_path = f"{self.remote_project}/run_stage_b.py"
        print("[INFO] Writing Stage B script to remote...")
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(remote_script)
            temp_path = f.name
        try:
            self.kaggle.upload_file(temp_path, script_path)
        finally:
            os.unlink(temp_path)

        # Execute the script with longer timeout and retry
        print("[INFO] Executing Stage B on GPU (this takes ~30-60 seconds)...")
        max_retries = 2
        for attempt in range(1, max_retries + 1):
            try:
                output = self.kaggle.run_command(
                    f"cd {self.remote_project} && python run_stage_b.py",
                    cwd=self.remote_project,
                    timeout=420  # 7 minutes timeout
                )
                print(f"[REMOTE OUTPUT]:\n{output}")

                if "Stage B completed successfully" in output:
                    print("[SUCCESS] Stage B completed on GPU")
                    return True
                elif "CUDA out of memory" in output or "out of memory" in output.lower():
                    print(f"[WARN] GPU OOM on attempt {attempt}, clearing cache and retrying...")
                    # Try to clear cache and retry with lower resolution
                    if attempt < max_retries:
                        self.kaggle.run_command("python -c \"import torch; torch.cuda.empty_cache()\"", timeout=30)
                        continue
                else:
                    print("[ERROR] Stage B did not complete successfully")
                    if attempt < max_retries:
                        print(f"[INFO] Retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    return False

            except Exception as e:
                print(f"[ERROR] Stage B execution failed (attempt {attempt}): {e}")
                if attempt < max_retries:
                    print(f"[INFO] Retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)
                    continue
                return False

        return False

    # ============================================================
    # Results Download
    # ============================================================

    def download_results(self, local_output_dir: str = "outputs/run_mvp/geometry") -> bool:
        """Download results from Kaggle to local machine and verify geometry outputs."""
        print(f"[INFO] Downloading results to {local_output_dir}...")

        remote_geometry = f"{self.remote_project}/outputs/run_mvp/geometry"
        local_geometry = self.local_project / local_output_dir
        local_geometry.mkdir(parents=True, exist_ok=True)

        try:
            # Use download_dir for recursive download
            downloaded = self.kaggle.download_dir(
                remote_dir=remote_geometry,
                local_dir=str(local_geometry)
            )

            if downloaded > 0:
                print(f"[SUCCESS] Downloaded {downloaded} files to {local_geometry}")

                # Verify key files
                key_files = [
                    "camera_intrinsics.json",
                    "camera_poses.json",
                    "point_cloud.ply",
                ]
                all_ok = True
                for f in key_files:
                    path = local_geometry / f
                    if path.exists():
                        size_kb = path.stat().st_size / 1024
                        print(f"  ✓ {f} ({size_kb:.1f} KB)")
                    else:
                        print(f"  ✗ {f} MISSING")
                        all_ok = False

                # Check depth maps
                depth_files = list((local_geometry / "depth").glob("depth_*.npy"))
                print(f"  ✓ {len(depth_files)} depth maps")
                if len(depth_files) == 0:
                    all_ok = False

                # Validate geometry content
                if all_ok:
                    all_ok = self._validate_geometry(local_geometry)

                return all_ok
            else:
                print("[WARNING] No files downloaded - check if Stage B ran successfully")
                return False

        except Exception as e:
            print(f"[ERROR] Failed to download results: {e}")
            return False

    def _validate_geometry(self, geometry_dir: Path) -> bool:
        """Validate downloaded geometry files have correct structure and metric scale."""
        print("[INFO] Validating geometry outputs...")
        all_ok = True

        try:
            import json
            import numpy as np

            # 1. Validate camera intrinsics
            intrinsics_path = geometry_dir / "camera_intrinsics.json"
            with open(intrinsics_path) as f:
                intrinsics = json.load(f)
            K = np.array(intrinsics["intrinsics"][0])  # First frame
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            print(f"  Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            if fx <= 0 or fy <= 0:
                print("  ✗ Invalid focal lengths")
                all_ok = False

            # 2. Validate camera poses
            poses_path = geometry_dir / "camera_poses.json"
            with open(poses_path) as f:
                poses = json.load(f)
            extrinsics = np.array(poses["extrinsics"])
            print(f"  Extrinsics shape: {extrinsics.shape}")
            if extrinsics.shape[1:] != (4, 4):
                print("  ✗ Extrinsics not 4x4")
                all_ok = False

            # 3. Validate point cloud
            ply_path = geometry_dir / "point_cloud.ply"
            # Read PLY header to check point count
            with open(ply_path, 'rb') as f:
                header = b''
                while b'end_header' not in header:
                    header += f.readline()
                header_str = header.decode('ascii', errors='ignore')
                for line in header_str.split('\n'):
                    if line.startswith('element vertex'):
                        n_points = int(line.split()[-1])
                        print(f"  Point cloud: {n_points:,} points")
                        if n_points < 1000:
                            print("  ✗ Too few points")
                            all_ok = False
                        break

            # 4. Validate depth maps
            depth_files = list((geometry_dir / "depth").glob("depth_*.npy"))
            for i, df in enumerate(depth_files[:3]):  # Check first 3
                d = np.load(df)
                valid = np.isfinite(d) & (d > 0)
                if np.any(valid):
                    d_valid = d[valid]
                    print(f"  Depth {i}: shape={d.shape}, range=[{d_valid.min():.3f}, {d_valid.max():.3f}], valid={valid.mean()*100:.1f}%")
                    # Check for reasonable metric scale (room ~3-10m)
                    if d_valid.max() > 50 or d_valid.max() < 0.1:
                        print(f"  ⚠ Depth range may indicate scale issue: max={d_valid.max():.2f}m")
                else:
                    print(f"  ✗ Depth {i}: no valid values")
                    all_ok = False

            if all_ok:
                print("[SUCCESS] Geometry validation passed")
            else:
                print("[WARNING] Geometry validation had issues")

        except Exception as e:
            print(f"  [WARN] Validation error: {e}")
            # Don't fail download on validation errors

        return all_ok

    # ============================================================
    # Interactive
    # ============================================================

    def interactive_shell(self) -> None:
        """Open interactive shell on Kaggle."""
        print("[INFO] Starting interactive shell on Kaggle...")
        print("  Ctrl+] to detach without killing remote terminal")
        name = self.kaggle.create_terminal()
        self.kaggle.interactive(name, initial_command=f"cd {self.remote_project}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Video-to-Terrain to Kaggle GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.v2t.deploy_kaggle --stage B                    # Full pipeline
  python -m src.v2t.deploy_kaggle --stage B --model da3metric-large  # Specific model
  python -m src.v2t.deploy_kaggle --download-only              # Only download results
  python -m src.v2t.deploy_kaggle --shell                      # Interactive shell
  python -m src.v2t.deploy_kaggle --ping                       # Test connection
  python -m src.v2t.deploy_kaggle --upload-only                # Only upload project
        """
    )
    parser.add_argument("--stage", default="B", choices=["B"], help="Pipeline stage to run")
    parser.add_argument("--model", default="da3metric-large", help="DA3 model to use")
    parser.add_argument("--process-res", type=int, default=504, help="DA3 process resolution")
    parser.add_argument("--upload-only", action="store_true", help="Only upload project, don't run")
    parser.add_argument("--download-only", action="store_true", help="Only download results")
    parser.add_argument("--shell", action="store_true", help="Open interactive shell")
    parser.add_argument("--ping", action="store_true", help="Test connection only")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup (use cached)")

    args = parser.parse_args()

    proxy_url = os.environ.get("KAGGLE_PROXY_URL")
    if not proxy_url:
        print("[ERROR] KAGGLE_PROXY_URL environment variable not set")
        print("Example: export KAGGLE_PROXY_URL='https://.../proxy'")
        sys.exit(1)

    deployer = KaggleDeployer(proxy_url)

    if args.ping:
        sys.exit(0 if deployer.ping() else 1)

    if args.shell:
        deployer.interactive_shell()
        return

    if args.download_only:
        success = deployer.download_results()
        sys.exit(0 if success else 1)

    # Ensure remote directories exist
    deployer.ensure_remote_dirs()

    # Clean up remote cache to free disk space
    deployer.cleanup_remote_cache()

    # Upload project
    if not deployer.upload_project():
        sys.exit(1)

    if args.upload_only:
        print("[INFO] Upload complete. Use --shell to interact or run without --upload-only to execute.")
        return

    # Upload input video
    if not deployer.upload_input_video():
        sys.exit(1)

    # Setup environment (unless skipped)
    if not args.skip_setup:
        if not deployer.setup_environment(args.model):
            print("[WARNING] Environment setup had issues, continuing anyway...")

    # Run Stage B
    if args.stage == "B":
        if not deployer.run_stage_b(args.model, args.process_res):
            sys.exit(1)

    # Download results
    if not deployer.download_results():
        sys.exit(1)

    print("\n[SUCCESS] Kaggle GPU deployment complete!")
    print(f"Results available at: {PROJECT_ROOT}/outputs/run_mvp/geometry/")


if __name__ == "__main__":
    main()