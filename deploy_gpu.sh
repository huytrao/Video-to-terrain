#!/usr/bin/env bash
# =============================================================================
# GPU Server Deployment Script for Video-to-Terrain Pipeline
# Deploys project to GPU server (10.188.189.205), runs DA3 inference on CUDA,
# and syncs results back locally.
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
SERVER_USER="phnh"
SERVER_HOST="10.188.189.205"
SERVER_PASS="1"
SERVER_DIR="/home/phnh/Video-to-Terrain"
LOCAL_PROJECT_DIR="/Users/traoanhuy/source_code/Full-pipeline-scan-terrain/Videos_to_terrain"
CONDA_ENV_NAME="v2t-gpu"
DA3_MODEL="da3metric-large"
PYTHON_VERSION="3.11"

# SSH/rsync options
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30"
RSYNC_EXCLUDES=(
    "--exclude=.venv"
    "--exclude=outputs/"
    "--exclude=__pycache__/"
    "--exclude=.git/"
    "--exclude=source-reference/"
    "--exclude=.claude/"
    "--exclude=*.pyc"
    "--exclude=.pytest_cache/"
    "--exclude=.mypy_cache/"
    "--exclude=*.log"
    "--exclude=deploy_gpu.sh"
    "--exclude=deploy_gpu.py"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

run_ssh() {
    sshpass -p "${SERVER_PASS}" ssh ${SSH_OPTS} "${SERVER_USER}@${SERVER_HOST}" "$@"
}

run_rsync() {
    sshpass -p "${SERVER_PASS}" rsync -avz --progress -e "ssh ${SSH_OPTS}" "${RSYNC_EXCLUDES[@]}" "$@"
}

check_sshpass() {
    if ! command -v sshpass &> /dev/null; then
        log_error "sshpass not found. Install with: brew install sshpass (macOS) or apt-get install sshpass (Ubuntu)"
        exit 1
    fi
}

check_rsync() {
    if ! command -v rsync &> /dev/null; then
        log_error "rsync not found. Please install rsync."
        exit 1
    fi
}

# =============================================================================
# MAIN DEPLOYMENT STEPS
# =============================================================================
main() {
    log_info "Starting GPU deployment for Video-to-Terrain pipeline"
    log_info "Server: ${SERVER_USER}@${SERVER_HOST}"
    log_info "Remote dir: ${SERVER_DIR}"
    log_info "DA3 Model: ${DA3_MODEL} (CUDA)"

    check_sshpass
    check_rsync

    # Step 1: Verify local input video exists
    log_info "Step 1: Verifying local input video..."
    INPUT_VIDEO="${LOCAL_PROJECT_DIR}/input/20260810_150244_color.mp4"
    if [[ ! -f "${INPUT_VIDEO}" ]]; then
        log_error "Input video not found: ${INPUT_VIDEO}"
        exit 1
    fi
    log_success "Input video found: $(basename ${INPUT_VIDEO}) ($(du -h ${INPUT_VIDEO} | cut -f1))"

    # Step 2: Sync project to server
    log_info "Step 2: Syncing project to GPU server..."
    run_rsync "${LOCAL_PROJECT_DIR}/" "${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/"
    log_success "Project synced to server"

    # Step 3: Set up conda environment with CUDA on server
    log_info "Step 3: Setting up conda environment with CUDA on server..."
    run_ssh << 'ENDSSH'
set -euo pipefail

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || {
    echo "Conda not found, installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p ~/miniconda3
    source ~/miniconda3/etc/profile.d/conda.sh
}

# Create or update conda environment
if conda env list | grep -q "^v2t-gpu "; then
    echo "Environment v2t-gpu exists, updating..."
    conda env update -n v2t-gpu -f - << 'ENVCONFIG'
name: v2t-gpu
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch::pytorch
  - pytorch::torchvision
  - pytorch::torchaudio
  - pytorch-cuda=12.1
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
else
    echo "Creating new environment v2t-gpu..."
    conda env create -n v2t-gpu -f - << 'ENVCONFIG'
name: v2t-gpu
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch::pytorch
  - pytorch::torchvision
  - pytorch::torchaudio
  - pytorch-cuda=12.1
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
fi

# Activate environment and verify CUDA
conda activate v2t-gpu
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
ENDSSH
    log_success "Conda environment ready with CUDA support"

    # Step 4: Install DA3 and additional dependencies on server
    log_info "Step 4: Installing DA3 and additional dependencies..."
    run_ssh << 'ENDSSH'
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate v2t-gpu

cd /home/phnh/Video-to-Terrain

# Install DA3 from local depth-anything-3 repo if available, else from pip
if [[ -d "/home/phnh/depth-anything-3" ]]; then
    echo "Installing DA3 from local repo..."
    pip install -e /home/phnh/depth-anything-3
elif [[ -d "depth-anything-3" ]]; then
    echo "Installing DA3 from project subdirectory..."
    pip install -e depth-anything-3
else
    echo "DA3 repo not found locally, will use HuggingFace hub download at runtime"
fi

# Install any missing dependencies
pip install --upgrade pip
pip install "numpy<2"  # Ensure numpy compatibility
ENDSSH
    log_success "DA3 dependencies installed"

    # Step 5: Copy input video to server
    log_info "Step 5: Copying input video to server..."
    run_rsync "${INPUT_VIDEO}" "${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/input/"
    log_success "Input video copied to server"

    # Step 6: Run Stage B (DA3) on GPU with OOM fallback
    log_info "Step 6: Running Stage B (DA3) on GPU..."
    log_info "This may take 30-60 seconds for 100 frames..."

    run_ssh << 'ENDSSH'
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate v2t-gpu

cd /home/phnh/Video-to-Terrain

# Set CUDA device and memory management
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Fallback configurations for OOM recovery (progressive)
declare -A FALLBACK_RES=(
    [0]=504
    [1]=384
    [2]=256
)
FALLBACK_NAMES=("default_504" "fallback_384" "fallback_256")
MAX_FALLBACK=2

attempt=0
while [ $attempt -le $MAX_FALLBACK ]; do
    PROCESS_RES=${FALLBACK_RES[$attempt]}
    FALLBACK_NAME=${FALLBACK_NAMES[$attempt]}

    echo "[ATTEMPT $((attempt + 1))/$((MAX_FALLBACK + 1))] config=$FALLBACK_NAME process_res=$PROCESS_RES"

    python -c "
import sys
import torch
sys.path.insert(0, 'src')
from v2t.config import CFG
from v2t.geometry.stage_b_da3 import run_stage_b

# Override device to CUDA
CFG.da3_device = 'cuda'
CFG.da3_model = 'da3metric-large'
CFG.da3_process_res = $PROCESS_RES

print(f'Running DA3 on: {CFG.da3_device}')
print(f'Model: {CFG.da3_model}')
print(f'Process resolution: {CFG.da3_process_res}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    torch.cuda.empty_cache()

result = run_stage_b()
print('Stage B completed successfully!')
print(f'Results: {result}')
" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Stage B completed on attempt $((attempt + 1))"
        break
    else
        echo "FAILED (exit code: $EXIT_CODE)"
        if [ $attempt -lt $MAX_FALLBACK ]; then
            echo "Clearing GPU cache and trying next fallback..."
            python -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()" 2>/dev/null || true
            attempt=$((attempt + 1))
        else
            echo "All fallback configurations exhausted!"
            exit 1
        fi
    fi
done
ENDSSH
    log_success "Stage B (DA3) completed on GPU"

    # Step 7: Sync results back to local outputs/
    log_info "Step 7: Syncing results back to local machine..."
    LOCAL_OUTPUT_DIR="${LOCAL_PROJECT_DIR}/outputs"
    run_rsync "${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/outputs/" "${LOCAL_OUTPUT_DIR}/"
    log_success "Results synced to local outputs/"

    # Step 8: Verify results
    log_info "Step 8: Verifying results..."
    RUN_DIR="${LOCAL_OUTPUT_DIR}/run_mvp"
    if [[ -d "${RUN_DIR}/geometry/depth" ]]; then
        DEPTH_COUNT=$(ls -1 "${RUN_DIR}/geometry/depth"/depth_*.npy 2>/dev/null | wc -l)
        log_success "Depth maps generated: ${DEPTH_COUNT} files"
    fi
    if [[ -f "${RUN_DIR}/geometry/point_cloud.ply" ]]; then
        log_success "Point cloud generated: $(du -h ${RUN_DIR}/geometry/point_cloud.ply | cut -f1)"
    fi
    if [[ -f "${RUN_DIR}/geometry/camera_intrinsics.json" ]]; then
        log_success "Camera intrinsics saved"
    fi
    if [[ -f "${RUN_DIR}/geometry/camera_poses.json" ]]; then
        log_success "Camera poses saved"
    fi

    log_success "=========================================="
    log_success "GPU Deployment Complete!"
    log_success "Results available at: ${RUN_DIR}/geometry/"
    log_success "=========================================="
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi