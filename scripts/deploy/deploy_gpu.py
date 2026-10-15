#!/usr/bin/env python3
"""
GPU Server Deployment Script for Video-to-Terrain Pipeline
Uses paramiko/fabric for programmatic deployment with:
- SSH connection with password
- File sync via SFTP/rsync
- Remote command execution
- Progress reporting
- Error handling and retry
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import getpass
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

try:
    import paramiko
    from paramiko import SSHClient, AutoAddPolicy, SFTPClient
    from paramiko.ssh_exception import SSHException, AuthenticationException
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None
    SSHClient = None
    AutoAddPolicy = None
    SFTPClient = None
    SSHException = Exception
    AuthenticationException = Exception

# Optional: fabric for higher-level operations
try:
    from fabric import Connection, Config
    from fabric.transfer import Transfer
    FABRIC_AVAILABLE = True
except ImportError:
    FABRIC_AVAILABLE = False
    Connection = None
    Config = None
    Transfer = None

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ServerConfig:
    """Server connection configuration."""
    host: str = "10.188.189.205"
    user: str = "phnh"
    password: Optional[str] = field(default_factory=lambda: os.environ.get("SERVER_PASS") or os.environ.get("SERVER_PASSWORD"))
    remote_dir: str = "/home/phnh/Video-to-Terrain"
    port: int = 22
    connect_timeout: int = 30


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    # This file lives at scripts/deploy/; the project root is two levels up.
    local_project_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    conda_env_name: str = "v2t-gpu"
    da3_model: str = "da3metric-large"
    python_version: str = "3.11"
    cuda_version: str = "12.1"

    # Rsync excludes
    rsync_excludes: List[str] = field(default_factory=lambda: [
        ".venv", "outputs/", "__pycache__/", ".git/", "source-reference/",
        ".claude/", "*.pyc", ".pytest_cache/", ".mypy_cache/", "*.log",
        "scripts/deploy/deploy_gpu.sh", "scripts/deploy/deploy_gpu.py"
    ])

    # Input video
    input_video: Path = field(default_factory=lambda:
        Path(__file__).resolve().parents[2] / "input" / "Small-factory" / "small_factory_final.mp4"
    )


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up colored logging."""
    logger = logging.getLogger("deploy_gpu")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # Colored formatter
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[0;36m',    # Cyan
                'INFO': '\033[0;32m',     # Green
                'WARNING': '\033[1;33m',  # Yellow
                'ERROR': '\033[0;31m',    # Red
                'CRITICAL': '\033[1;31m', # Bold Red
            }
            RESET = '\033[0m'

            def format(self, record):
                color = self.COLORS.get(record.levelname, '')
                record.levelname = f"{color}{record.levelname}{self.RESET}"
                return super().format(record)

        formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# =============================================================================
# PROGRESS REPORTING
# =============================================================================

class ProgressReporter:
    """Progress reporting with callbacks."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.current_step = 0
        self.total_steps = 0
        self.step_name = ""

    def start_step(self, step_name: str, step_num: int, total: int):
        self.current_step = step_num
        self.total_steps = total
        self.step_name = step_name
        self.logger.info(f"[{step_num}/{total}] {step_name}")

    def progress(self, message: str, level: int = logging.INFO):
        self.logger.log(level, f"  {message}")

    def success(self, message: str):
        self.logger.info(f"  ✓ {message}")

    def warning(self, message: str):
        self.logger.warning(f"  ⚠ {message}")

    def error(self, message: str):
        self.logger.error(f"  ✗ {message}")

    def complete(self):
        self.logger.info(f"[{self.current_step}/{self.total_steps}] {self.step_name} - COMPLETE")


# =============================================================================
# SSH CONNECTION MANAGER
# =============================================================================

class SSHManager:
    """Manages SSH connections with retry logic."""

    def __init__(self, config: ServerConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client: Optional[SSHClient] = None
        self.sftp: Optional[SFTPClient] = None
        self._fabric_conn = None

        if not PARAMIKO_AVAILABLE:
            self.logger.error("paramiko is not installed. Install with: pip install paramiko")
            return False

        if not self.config.password:
            self.logger.error("Authentication requires a password. Set SERVER_PASS environment variable or pass --password.")
            return False

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.debug(f"Connecting to {self.config.user}@{self.config.host}:{self.config.port} (attempt {attempt}/{max_retries})")

                self.client = SSHClient()
                self.client.set_missing_host_key_policy(AutoAddPolicy())
                self.client.connect(
                    hostname=self.config.host,
                    port=self.config.port,
                    username=self.config.user,
                    password=self.config.password,
                    timeout=self.config.connect_timeout,
                    allow_agent=False,
                    look_for_keys=False
                )

                self.sftp = self.client.open_sftp()

                # Also create fabric connection if available
                if FABRIC_AVAILABLE:
                    fabric_config = Config(overrides={
                        'run': {'echo': True, 'warn': True},
                        'sudo': {'password': self.config.password}
                    })
                    self._fabric_conn = Connection(
                        host=self.config.host,
                        user=self.config.user,
                        port=self.config.port,
                        connect_kwargs={"password": self.config.password},
                        config=fabric_config
                    )

                self.logger.debug("SSH connection established")
                return True

            except AuthenticationException:
                self.logger.error("Authentication failed. Check username/password.")
                return False
            except SSHException as e:
                self.logger.warning(f"SSH connection failed (attempt {attempt}): {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to connect after {max_retries} attempts")
                    return False
            except Exception as e:
                self.logger.error(f"Unexpected connection error: {e}")
                return False

        return False

    def disconnect(self):
        """Close SSH connections."""
        if self.sftp:
            try:
                self.sftp.close()
            except Exception:
                pass
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        if self._fabric_conn:
            try:
                self._fabric_conn.close()
            except Exception:
                pass
        self.logger.debug("SSH connections closed")

    def execute(self, command: str, timeout: int = 300, sudo: bool = False) -> tuple[int, str, str]:
        """Execute command on remote server."""
        if not self.client:
            raise RuntimeError("Not connected")

        full_cmd = command
        if sudo:
            full_cmd = f"echo {self.config.password} | sudo -S {command}"

        self.logger.debug(f"Executing: {full_cmd[:100]}...")

        stdin, stdout, stderr = self.client.exec_command(full_cmd, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode('utf-8', errors='replace')
        stderr_text = stderr.read().decode('utf-8', errors='replace')

        return exit_code, stdout_text, stderr_text

    def execute_streaming(self, command: str, timeout: int = 300,
                          callback: Optional[Callable[[str], None]] = None) -> int:
        """Execute command with streaming output."""
        if not self.client:
            raise RuntimeError("Not connected")

        transport = self.client.get_transport()
        channel = transport.open_session()
        channel.settimeout(timeout)
        channel.exec_command(command)

        exit_code = -1
        while True:
            if channel.exit_status_ready():
                exit_code = channel.recv_exit_status()
                break

            # Read stdout
            if channel.recv_ready():
                data = channel.recv(4096).decode('utf-8', errors='replace')
                if callback:
                    callback(data)
                else:
                    print(data, end='', flush=True)

            # Read stderr
            if channel.recv_stderr_ready():
                data = channel.recv_stderr(4096).decode('utf-8', errors='replace')
                if callback:
                    callback(f"STDERR: {data}")
                else:
                    print(data, end='', flush=True, file=sys.stderr)

            time.sleep(0.1)

        # Drain remaining output
        while channel.recv_ready():
            data = channel.recv(4096).decode('utf-8', errors='replace')
            if callback:
                callback(data)
        while channel.recv_stderr_ready():
            data = channel.recv_stderr(4096).decode('utf-8', errors='replace')
            if callback:
                callback(f"STDERR: {data}")

        channel.close()
        return exit_code

    @contextmanager
    def sftp_session(self):
        """Context manager for SFTP operations."""
        if not self.sftp:
            self.sftp = self.client.open_sftp()
        try:
            yield self.sftp
        finally:
            pass  # Keep SFTP open for reuse


# =============================================================================
# FILE SYNC OPERATIONS
# =============================================================================

class FileSyncer:
    """Handles file synchronization via rsync and SFTP."""

    def __init__(self, ssh_manager: SSHManager, config: DeploymentConfig,
                 server_config: ServerConfig, logger: logging.Logger):
        self.ssh = ssh_manager
        self.config = config
        self.server_config = server_config
        self.logger = logger

    def rsync_to_server(self, local_path: Path, remote_path: str,
                        excludes: Optional[List[str]] = None) -> bool:
        """Sync local directory to server using rsync over SSH."""
        if not self.server_config.password:
            self.logger.error("Password is required for rsync authentication. Set SERVER_PASS or pass --password.")
            return False
        excludes = excludes or self.config.rsync_excludes

        # Build rsync command
        exclude_args = []
        for pattern in excludes:
            exclude_args.extend(["--exclude", pattern])

        # Ensure trailing slash for directory sync
        local_str = str(local_path)
        if not local_str.endswith('/'):
            local_str += '/'

        remote_str = f"{self.server_config.user}@{self.server_config.host}:{remote_path}"
        if not remote_str.endswith('/'):
            remote_str += '/'

        cmd = [
            "rsync", "-avz", "--progress",
            "-e", f"sshpass -p '{self.server_config.password}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
            *exclude_args,
            local_str,
            remote_str
        ]

        self.logger.info(f"Syncing {local_path} -> {remote_path}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                self.logger.debug("Rsync completed successfully")
                return True
            else:
                self.logger.error(f"Rsync failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("Rsync timed out")
            return False
        except FileNotFoundError:
            self.logger.error("rsync or sshpass not found. Install with: brew install rsync sshpass")
            return False

    def rsync_from_server(self, remote_path: str, local_path: Path) -> bool:
        """Sync from server to local using rsync."""
        if not self.server_config.password:
            self.logger.error("Password is required for rsync authentication. Set SERVER_PASS or pass --password.")
            return False
        remote_str = f"{self.server_config.user}@{self.server_config.host}:{remote_path}"
        if not remote_str.endswith('/'):
            remote_str += '/'

        local_str = str(local_path)
        if not local_str.endswith('/'):
            local_str += '/'

        cmd = [
            "rsync", "-avz", "--progress",
            "-e", f"sshpass -p '{self.server_config.password}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
            remote_str,
            local_str
        ]

        self.logger.info(f"Syncing {remote_path} -> {local_path}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Rsync from server failed: {e}")
            return False

    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload single file via SFTP."""
        try:
            with self.ssh.sftp_session() as sftp:
                # Ensure remote directory exists
                remote_dir = str(Path(remote_path).parent)
                self._mkdir_p(sftp, remote_dir)

                self.logger.debug(f"Uploading {local_path} -> {remote_path}")
                sftp.put(str(local_path), remote_path)
                return True
        except Exception as e:
            self.logger.error(f"SFTP upload failed: {e}")
            return False

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Download single file via SFTP."""
        try:
            with self.ssh.sftp_session() as sftp:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Downloading {remote_path} -> {local_path}")
                sftp.get(remote_path, str(local_path))
                return True
        except Exception as e:
            self.logger.error(f"SFTP download failed: {e}")
            return False

    def _mkdir_p(self, sftp: SFTPClient, path: str):
        """Create remote directory recursively."""
        dirs = []
        current = path
        while current != '/' and current != '.':
            try:
                sftp.stat(current)
                break
            except FileNotFoundError:
                dirs.append(current)
                current = str(Path(current).parent)

        for d in reversed(dirs):
            try:
                sftp.mkdir(d)
            except Exception:
                pass


# =============================================================================
# DEPLOYMENT ORCHESTRATOR
# =============================================================================

class GPUDeployer:
    """Main deployment orchestrator."""

    def __init__(self, server_config: ServerConfig, deploy_config: DeploymentConfig,
                 logger: Optional[logging.Logger] = None):
        self.server_config = server_config
        self.deploy_config = deploy_config
        self.logger = logger or setup_logging()
        self.progress = ProgressReporter(self.logger)

        self.ssh = SSHManager(server_config, self.logger)
        self.syncer = FileSyncer(self.ssh, deploy_config, server_config, self.logger)

    def deploy(self) -> bool:
        """Run full deployment pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("Starting GPU Deployment for Video-to-Terrain")
        self.logger.info(f"Server: {self.server_config.user}@{self.server_config.host}")
        self.logger.info(f"Remote dir: {self.server_config.remote_dir}")
        self.logger.info(f"DA3 Model: {self.deploy_config.da3_model} (CUDA)")
        self.logger.info("=" * 60)

        # Define deployment steps
        steps = [
            ("Verify local input", self._step_verify_input),
            ("Connect to server", self._step_connect),
            ("Sync project to server", self._step_sync_project),
            ("Setup conda environment", self._step_setup_conda),
            ("Install DA3 dependencies", self._step_install_da3),
            ("Copy input video", self._step_copy_video),
            ("Run Stage B (DA3) on GPU", self._step_run_da3),
            ("Sync results back", self._step_sync_results),
            ("Verify results", self._step_verify_results),
        ]

        self.progress.total_steps = len(steps)

        for i, (name, step_fn) in enumerate(steps, 1):
            self.progress.start_step(name, i, len(steps))
            try:
                if not step_fn():
                    self.progress.error(f"Step failed: {name}")
                    return False
                self.progress.complete()
            except Exception as e:
                self.progress.error(f"Step exception: {name} - {e}")
                self.logger.exception("Step failed with exception")
                return False

        self.logger.info("=" * 60)
        self.logger.info("GPU Deployment Completed Successfully!")
        self.logger.info("=" * 60)
        return True

    def _step_verify_input(self) -> bool:
        """Verify local input video exists."""
        video_path = self.deploy_config.input_video
        if not video_path.exists():
            self.logger.error(f"Input video not found: {video_path}")
            return False

        size_mb = video_path.stat().st_size / (1024 * 1024)
        self.progress.success(f"Input video found: {video_path.name} ({size_mb:.1f} MB)")
        return True

    def _step_connect(self) -> bool:
        """Establish SSH connection."""
        if not self.ssh.connect():
            return False
        self.progress.success("SSH connection established")
        return True

    def _step_sync_project(self) -> bool:
        """Sync project files to server."""
        # Ensure remote directory exists
        self.ssh.execute(f"mkdir -p {self.server_config.remote_dir}")

        success = self.syncer.rsync_to_server(
            self.deploy_config.local_project_dir,
            self.server_config.remote_dir
        )

        if success:
            self.progress.success("Project synced to server")
        return success

    def _step_setup_conda(self) -> bool:
        """Set up conda environment with CUDA on server."""
        conda_setup_script = f'''
set -euo pipefail

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || {{
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p ~/miniconda3
    source ~/miniconda3/etc/profile.d/conda.sh
}}

# Create/update environment
ENV_NAME="{self.deploy_config.conda_env_name}"
PYTHON_VER="{self.deploy_config.python_version}"
CUDA_VER="{self.deploy_config.cuda_version}"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Updating existing environment..."
else
    echo "Creating new environment..."
fi

conda env update -n $ENV_NAME -f - << 'ENVCONFIG'
name: $ENV_NAME
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=$PYTHON_VER
  - pytorch::pytorch
  - pytorch::torchvision
  - pytorch::torchaudio
  - pytorch-cuda=$CUDA_VER
  - numpy
  - pillow
  - opencv
  - scipy
  - matplotlib
  - tqdm
  - pyyaml
  - python-dotenv
  - pip
  - pip:
    - torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    - torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    - torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    - torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    - torch-geometric
    - open3d
    - trimesh
    - networkx
    - wandb
ENVCONFIG

# Verify CUDA
conda activate $ENV_NAME
python -c "
import torch
print(f'PyTorch: {{torch.__version__}}')
print(f'CUDA available: {{torch.cuda.is_available()}}')
print(f'CUDA version: {{torch.version.cuda}}')
if torch.cuda.is_available():
    print(f'GPU: {{torch.cuda.get_device_name(0)}}')
    print(f'GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB')
"
'''

        self.logger.info("Setting up conda environment (this may take 2-5 minutes)...")
        exit_code, stdout, stderr = self.ssh.execute(conda_setup_script, timeout=600)

        if exit_code != 0:
            self.logger.error(f"Conda setup failed: {stderr}")
            return False

        self.logger.debug(stdout)
        self.progress.success("Conda environment ready with CUDA")
        return True

    def _step_install_da3(self) -> bool:
        """Install DA3 and additional dependencies."""
        install_script = f'''
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate {self.deploy_config.conda_env_name}

cd {self.server_config.remote_dir}

# Install DA3 from local repo if available
if [[ -d "/home/phnh/depth-anything-3" ]]; then
    echo "Installing DA3 from local repo..."
    pip install -e /home/phnh/depth-anything-3
elif [[ -d "depth-anything-3" ]]; then
    echo "Installing DA3 from project subdirectory..."
    pip install -e depth-anything-3
else
    echo "DA3 repo not found locally, will use HuggingFace hub download at runtime"
fi

# Ensure numpy compatibility
pip install "numpy<2"

# Verify DA3 import
python -c "
try:
    import sys
    sys.path.insert(0, 'src')
    from depth_anything_3.api import DepthAnything3
    print('DA3 import successful')
except ImportError as e:
    print(f'DA3 import failed: {{e}}')
    exit(1)
"
'''

        exit_code, stdout, stderr = self.ssh.execute(install_script, timeout=300)

        if exit_code != 0:
            self.logger.warning(f"DA3 install had issues: {stderr}")
            # Don't fail - DA3 might be downloaded at runtime

        self.logger.debug(stdout)
        self.progress.success("DA3 dependencies installed")
        return True

    def _step_copy_video(self) -> bool:
        """Copy input video to server."""
        remote_video_dir = f"{self.server_config.remote_dir}/input"
        self.ssh.execute(f"mkdir -p {remote_video_dir}")

        video_name = self.deploy_config.input_video.name
        success = self.syncer.upload_file(
            self.deploy_config.input_video,
            f"{remote_video_dir}/{video_name}"
        )

        if success:
            self.progress.success(f"Input video copied: {video_name}")
        return success

    def _step_run_da3(self) -> bool:
        """Run Stage B (DA3) on GPU with OOM fallback."""
        # Fallback configurations for OOM recovery
        fallback_configs = [
            {"process_res": 504, "name": "default_504"},
            {"process_res": 384, "name": "fallback_384"},
            {"process_res": 256, "name": "fallback_256"},
        ]

        for attempt, fb in enumerate(fallback_configs):
            self.progress.start_step(f"Run DA3 (attempt {attempt + 1}/{len(fallback_configs)}: {fb['name']})",
                                   attempt + 1, len(fallback_configs))

            run_script = f'''
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate {self.deploy_config.conda_env_name}

cd {self.server_config.remote_dir}

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run Stage B with CUDA
python -c "
import sys
import torch
sys.path.insert(0, 'src')
from v2t.config import CFG
from v2t.geometry.stage_b_da3 import run_stage_b

# Override device to CUDA
CFG.da3_device = 'cuda'
CFG.da3_model = '{self.deploy_config.da3_model}'
CFG.da3_process_res = {fb['process_res']}

print(f'Running DA3 on: {{CFG.da3_device}}')
print(f'Model: {{CFG.da3_model}}')
print(f'Process resolution: {{CFG.da3_process_res}}')
print(f'Input video: {{CFG.input_video}}')
print(f'Frames dir: {{CFG.paths.frames}}')
print(f'Depth dir: {{CFG.paths.depth}}')
print(f'Geometry dir: {{CFG.paths.geometry}}')

print(f'PyTorch: {{torch.__version__}}')
print(f'CUDA available: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    print(f'GPU: {{torch.cuda.get_device_name(0)}}')
    print(f'GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB')
    torch.cuda.empty_cache()

# Check frames exist
from pathlib import Path
frames = list(Path(CFG.paths.frames).glob('frame_*.jpg'))
print(f'Found {{len(frames)}} frames')

if len(frames) == 0:
    print('ERROR: No frames found! Run Stage A first.')
    sys.exit(1)

result = run_stage_b()
print('Stage B completed successfully!')
print(f'Results: {{result}}')
"
'''

            self.logger.info(f"Running DA3 inference on GPU (config: {fb['name']}, process_res={fb['process_res']})...")

            def output_callback(line: str):
                if line.strip():
                    self.logger.info(f"  [remote] {line.strip()}")

            exit_code = self.ssh.execute_streaming(run_script, timeout=600, callback=output_callback)

            if exit_code == 0:
                self.progress.success(f"DA3 succeeded with {fb['name']}")
                return True
            else:
                self.progress.warning(f"DA3 failed with {fb['name']} (exit code: {exit_code})")
                if attempt < len(fallback_configs) - 1:
                    self.logger.info("Clearing GPU cache and trying next fallback...")
                    # Try to clear GPU cache
                    clear_script = '''
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print('GPU cache cleared')
"
'''
                    self.ssh.execute(clear_script, timeout=30)
                else:
                    self.logger.error("All fallback configurations exhausted!")
                    return False

        return False

        self.logger.info("Running DA3 inference on GPU (expected 30-60 seconds for 100 frames)...")

        # Use streaming execution for progress visibility
        def output_callback(line: str):
            if line.strip():
                self.logger.info(f"  [remote] {line.strip()}")

        exit_code = self.ssh.execute_streaming(run_script, timeout=600, callback=output_callback)

        if exit_code != 0:
            self.logger.error("DA3 inference failed")
            return False

        self.progress.success("Stage B (DA3) completed on GPU")
        return True

    def _step_sync_results(self) -> bool:
        """Sync results back from server."""
        local_outputs = self.deploy_config.local_project_dir / "outputs"
        remote_outputs = f"{self.server_config.remote_dir}/outputs"

        success = self.syncer.rsync_from_server(remote_outputs, local_outputs)

        if success:
            self.progress.success("Results synced to local outputs/")
        return success

    def _step_verify_results(self) -> bool:
        """Verify results locally."""
        run_dir = self.deploy_config.local_project_dir / "outputs" / "run_mvp"
        geometry_dir = run_dir / "geometry"

        checks = [
            (geometry_dir / "depth", "depth_*.npy", "Depth maps"),
            (geometry_dir / "point_cloud.ply", None, "Point cloud"),
            (geometry_dir / "camera_intrinsics.json", None, "Camera intrinsics"),
            (geometry_dir / "camera_poses.json", None, "Camera poses"),
        ]

        all_ok = True
        for path, pattern, name in checks:
            if pattern:
                count = len(list(path.glob(pattern))) if path.exists() else 0
                if count > 0:
                    self.progress.success(f"{name}: {count} files")
                else:
                    self.progress.warning(f"{name}: Not found")
                    all_ok = False
            else:
                if path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    self.progress.success(f"{name}: {size_mb:.1f} MB")
                else:
                    self.progress.warning(f"{name}: Not found")
                    all_ok = False

        return all_ok


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU Deployment for Video-to-Terrain Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/deploy/deploy_gpu.py                    # Full deployment
  python scripts/deploy/deploy_gpu.py --verify-only      # Only verify existing results
  python scripts/deploy/deploy_gpu.py --sync-only        # Only sync project (no DA3 run)
  python scripts/deploy/deploy_gpu.py --run-da3-only     # Only run DA3 on already-synced project
  python scripts/deploy/deploy_gpu.py -v                 # Verbose output
        """
    )

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--host', default='10.188.189.205', help='Server hostname/IP')
    parser.add_argument('--user', default='phnh', help='SSH username')
    parser.add_argument('--password', default=os.environ.get("SERVER_PASS") or os.environ.get("SERVER_PASSWORD"),
                        help='SSH/sudo password (defaults to SERVER_PASS or SERVER_PASSWORD env var)')
    parser.add_argument('--remote-dir', default='/home/phnh/Video-to-Terrain',
                        help='Remote project directory')
    parser.add_argument('--da3-model', default='da3metric-large', help='DA3 model name')
    parser.add_argument('--conda-env', default='v2t-gpu', help='Conda environment name')

    # Partial deployment options
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify local results')
    parser.add_argument('--sync-only', action='store_true',
                        help='Only sync project to server')
    parser.add_argument('--run-da3-only', action='store_true',
                        help='Only run DA3 on server (assumes project synced)')
    parser.add_argument('--sync-results-only', action='store_true',
                        help='Only sync results back from server')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)

    # Prompt or error if password authentication is required without a password
    if not args.verify_only and not args.password:
        if sys.stdin.isatty():
            try:
                args.password = getpass.getpass("Enter SSH/sudo password: ")
            except Exception:
                pass
        if not args.password:
            logger.error("Password is required for server deployment. Please set SERVER_PASS environment variable or specify --password.")
            sys.exit(1)

    # Configuration
    server_config = ServerConfig(
        host=args.host,
        user=args.user,
        password=args.password,
        remote_dir=args.remote_dir
    )

    deploy_config = DeploymentConfig(
        conda_env_name=args.conda_env,
        da3_model=args.da3_model
    )

    deployer = GPUDeployer(server_config, deploy_config, logger)

    try:
        # Handle partial deployment modes
        if args.verify_only:
            logger.info("Running verification only...")
            success = deployer._step_verify_results()
            sys.exit(0 if success else 1)

        if args.sync_only:
            logger.info("Running project sync only...")
            if not deployer.ssh.connect():
                sys.exit(1)
            success = deployer._step_sync_project()
            deployer.ssh.disconnect()
            sys.exit(0 if success else 1)

        if args.run_da3_only:
            logger.info("Running DA3 only...")
            if not deployer.ssh.connect():
                sys.exit(1)
            success = deployer._step_run_da3()
            deployer.ssh.disconnect()
            sys.exit(0 if success else 1)

        if args.sync_results_only:
            logger.info("Syncing results only...")
            if not deployer.ssh.connect():
                sys.exit(1)
            success = deployer._step_sync_results()
            deployer.ssh.disconnect()
            sys.exit(0 if success else 1)

        # Full deployment
        success = deployer.deploy()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.warning("Deployment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Deployment failed with exception: {e}")
        sys.exit(1)
    finally:
        deployer.ssh.disconnect()


if __name__ == "__main__":
    main()