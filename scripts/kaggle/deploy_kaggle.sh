#!/usr/bin/env bash
# =============================================================================
# scripts/kaggle/deploy_kaggle.sh - Robust Kaggle GPU Deployment Wrapper
#
# Manages Kaggle proxy connection and session lifecycle:
#   1. Detects proxy URL expiration (JWT timeout ~1-2 hours)
#   2. Retries with exponential backoff on connection drops
#   3. Handles kernel restarts (re-runs setup + inference)
#   4. Provides resume capability for interrupted uploads/downloads
# =============================================================================

set -uo pipefail  # No 'e' - we handle errors manually for retry logic

# =============================================================================
# CONFIGURATION
# =============================================================================
# This script lives at scripts/kaggle/; the project root is two levels up.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_PROXY_URL=""  # Set your default KAGGLE_PROXY_URL here or pass via env

# Proxy URL persistence (avoid re-prompting on every invocation)
PROXY_URL_FILE="$PROJECT_DIR/.kaggle_proxy_url_cache"

# Retry configuration
MAX_PIPELINE_RETRIES=3
BACKOFF_BASE=5      # seconds
BACKOFF_MAX=120     # seconds (2 min max)
BACKOFF_FACTOR=2    # exponential

# Session monitoring
SESSION_START_TIME=$(date +%s)
MAX_SESSION_SECONDS=7200  # 2 hours (JWT typical lifetime)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $(date +%H:%M:%S) $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $(date +%H:%M:%S) $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $(date +%H:%M:%S) $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $(date +%H:%M:%S) $*"; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $(date +%H:%M:%S) $*"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Video-to-Terrain pipeline to Kaggle GPU runtime.

Options:
    -u, --url URL              Kaggle Jupyter proxy URL (or set KAGGLE_PROXY_URL env)
    -s, --stage STAGE          Pipeline stage to run (default: B)
    -m, --model MODEL          DA3 model to use (default: da3metric-large)
    -r, --process-res RES      DA3 process resolution (default: 504)
    --upload-only              Only upload project, don't run pipeline
    --download-only            Only download results from previous run
    --shell                    Open interactive shell on Kaggle
    --ping                     Test connection only
    --skip-setup               Skip environment setup (use cached deps)
    --force-setup              Force re-setup environment (clear cached state)
    --use-robust               Use robust deployer with full retry/resume (default)
    --use-legacy               Use legacy deployer (no retry/logic)
    --max-retries N            Max pipeline retries on transient errors (default: 3)
    --clear-proxy-cache        Clear cached proxy URL (force re-prompt)
    -v, --verbose              Enable verbose/debug logging
    -h, --help                 Show this help

Session Management:
    --session-status           Check current session health (no operations)
    --keep-alive               Run in keep-alive mode (pings every 60s)
    --proxy-health             Show detailed proxy health check

Examples:
    # Full deploy with auto-resume
    $0 --url "https://.../proxy"

    # Deploy with specific model
    $0 --url "https://.../proxy" --stage B --model da3metric-large

    # Just test connection
    $0 --ping

    # Download results only (resume from previous run)
    $0 --download-only

    # Interactive shell
    $0 --shell

    # Force re-setup after kernel restart
    $0 --force-setup

    # Run with 5 max retries
    $0 --max-retries 5

Environment:
    KAGGLE_PROXY_URL    Kaggle Jupyter proxy URL (required unless --url provided)

Long-running session handling:
    - Proxy URL has JWT expiry ~1-2 hours
    - If proxy expires mid-deploy, script will detect and prompt for fresh URL
    - Kernel restarts are handled by re-running setup + inference
    - Interrupted uploads resume by checking file sizes

Expected workflow:
    1. Local:  $0 --url "https://.../proxy"
    2. Remote: runs DA3 on GPU (~6-7s vs 60s on MPS)
    3. Local:  results synced back to outputs/run_mvp/geometry/
EOF
}

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
STAGE="B"
MODEL="da3metric-large"
PROCESS_RES=504
UPLOAD_ONLY=false
DOWNLOAD_ONLY=false
SHELL_MODE=false
PING_ONLY=false
SKIP_SETUP=false
FORCE_SETUP=false
USE_ROBUST=true
USE_LEGACY=false
MAX_RETRIES="$MAX_PIPELINE_RETRIES"
CLEAR_PROXY_CACHE=false
VERBOSE=false
SESSION_STATUS=false
KEEP_ALIVE=false
PROXY_HEALTH=false
PROXY_URL="${KAGGLE_PROXY_URL:-$DEFAULT_PROXY_URL}"

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            PROXY_URL="$2"
            shift 2
            ;;
        -s|--stage)
            STAGE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -r|--process-res)
            PROCESS_RES="$2"
            shift 2
            ;;
        --upload-only)
            UPLOAD_ONLY=true
            shift
            ;;
        --download-only)
            DOWNLOAD_ONLY=true
            shift
            ;;
        --shell)
            SHELL_MODE=true
            shift
            ;;
        --ping)
            PING_ONLY=true
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --force-setup)
            FORCE_SETUP=true
            shift
            ;;
        --use-robust)
            USE_ROBUST=true
            USE_LEGACY=false
            shift
            ;;
        --use-legacy)
            USE_LEGACY=true
            USE_ROBUST=false
            shift
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --clear-proxy-cache)
            CLEAR_PROXY_CACHE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --session-status)
            SESSION_STATUS=true
            shift
            ;;
        --keep-alive)
            KEEP_ALIVE=true
            shift
            ;;
        --proxy-health)
            PROXY_HEALTH=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Check session age and warn if proxy might expire
check_session_age() {
    local now=$(date +%s)
    local elapsed=$((now - SESSION_START_TIME))
    local remaining=$((MAX_SESSION_SECONDS - elapsed))

    if [[ "$remaining" -lt 300 ]]; then
        log_warning "Session has been running ${elapsed}s (JWT likely expires ~${MAX_SESSION_SECONDS}s)"
        log_warning "Proxy URL may be expired. If operations fail, get fresh URL from Kaggle UI."
        return 1
    elif [[ "$remaining" -lt 600 ]]; then
        log_warning "Session running ${elapsed}s, ${remaining}s until typical JWT expiry"
    fi
    return 0
}

# Load cached proxy URL if not provided
load_cached_proxy() {
    if [[ -z "$PROXY_URL" && -f "$PROXY_URL_FILE" ]]; then
        PROXY_URL=$(cat "$PROXY_URL_FILE")
        log_info "Using cached proxy URL from $PROXY_URL_FILE"
    fi
}

# Save proxy URL to cache
save_proxy_cache() {
    if [[ -n "$PROXY_URL" ]]; then
        echo "$PROXY_URL" > "$PROXY_URL_FILE"
        chmod 600 "$PROXY_URL_FILE"  # restrict perms
    fi
}

# Clear proxy cache
clear_proxy_cache() {
    if [[ -f "$PROXY_URL_FILE" ]]; then
        rm -f "$PROXY_URL_FILE"
        log_info "Cleared proxy URL cache"
    fi
}

# Exponential backoff sleep
backoff_sleep() {
    local attempt=$1
    local delay=$((BACKOFF_BASE * (BACKOFF_FACTOR ** (attempt - 1))))
    if [[ "$delay" -gt "$BACKOFF_MAX" ]]; then
        delay=$BACKOFF_MAX
    fi
    log_info "Backing off for ${delay}s before retry..."
    sleep "$delay"
}

# Check if Python module exists
check_module() {
    local module=$1
    if python3 -c "import $module" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# VALIDATION
# =============================================================================

# Clear proxy cache if requested
if [[ "$CLEAR_PROXY_CACHE" == true ]]; then
    clear_proxy_cache
fi

# Load cached proxy URL
load_cached_proxy

if [[ -z "$PROXY_URL" ]]; then
    log_error "KAGGLE_PROXY_URL not set."
    log_info "Options:"
    log_info "  1. Pass via --url flag"
    log_info "  2. export KAGGLE_PROXY_URL='https://.../proxy' and re-run"
    log_info "  3. The script will cache the URL at $PROXY_URL_FILE for future runs"
    usage
    exit 1
fi

# Validate proxy URL format
if [[ ! "$PROXY_URL" =~ ^https?:// ]]; then
    log_error "Invalid proxy URL format. Must start with http:// or https://"
    exit 1
fi

# Debug: log proxy URL (truncated)
log_info "Proxy URL: ${PROXY_URL:0:60}..."

# Verify project directory exists
if [[ ! -d "$PROJECT_DIR" ]]; then
    log_error "Project directory not found: $PROJECT_DIR"
    exit 1
fi

# Verify input video exists (unless only doing non-pipeline operations)
INPUT_VIDEO="$PROJECT_DIR/../input/20260810_150244_color.mp4"
if [[ ! -f "$INPUT_VIDEO" && "$DOWNLOAD_ONLY" == false && "$PING_ONLY" == false && "$SHELL_MODE" == false ]]; then
    log_error "Input video not found: $INPUT_VIDEO"
    exit 1
fi

# Verify required Python packages
log_info "Checking required Python packages..."
MISSING_PACKAGES=()

for pkg in requests:requests websocket-client:websocket; do
    pip_name="${pkg%%:*}"
    import_name="${pkg##*:}"
    if ! check_module "$import_name"; then
        MISSING_PACKAGES+=("$pip_name")
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    log_error "Missing Python packages: ${MISSING_PACKAGES[*]}"
    log_info "Install with: pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi

# Verify robust deployer script exists
ROBUST_MODULE="src.v2t.kaggle_robust"
LEGACY_MODULE="src.v2t.deploy_kaggle"

PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

if [[ "$USE_LEGACY" == true ]]; then
    PYTHON_CMD=("$PYTHON_BIN" -m "$LEGACY_MODULE")
else
    PYTHON_CMD=("$PYTHON_BIN" -m "$ROBUST_MODULE")
fi

# =============================================================================
# SESSION STATUS & HEALTH CHECK MODE
# =============================================================================

if [[ "$SESSION_STATUS" == true || "$PROXY_HEALTH" == true ]]; then
    log_info "Checking Kaggle session status..."
    export KAGGLE_PROXY_URL="$PROXY_URL"
    check_session_age
    "${PYTHON_CMD[@]}" --ping
    exit $?
fi

# =============================================================================
# KEEP-ALIVE MODE
# =============================================================================

if [[ "$KEEP_ALIVE" == true ]]; then
    log_info "Starting keep-alive mode (ping every 60s, Ctrl+C to stop)..."
    export KAGGLE_PROXY_URL="$PROXY_URL"
    while true; do
        timestamp=$(date +"%H:%M:%S")
        if "${PYTHON_CMD[@]}" --ping >/dev/null 2>&1; then
            echo -e "${GREEN}[$timestamp]${NC} Kaggle proxy healthy"
        else
            echo -e "${RED}[$timestamp]${NC} Proxy unhealthy - may need fresh URL"
        fi
        sleep 60
    done
    exit 0
fi

# =============================================================================
# MAIN DEPLOYMENT WITH RETRY LOGIC
# =============================================================================

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [[ -d "$PROJECT_DIR/.venv" ]]; then
    log_info "Activating virtual environment..."
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Build Python command arguments
build_python_args() {
    local args=()

    if [[ "$VERBOSE" == true ]]; then
        args+=(--verbose)
    fi

    if [[ "$PING_ONLY" == true ]]; then
        args+=(--ping)
    elif [[ "$SHELL_MODE" == true ]]; then
        args+=(--shell)
    elif [[ "$DOWNLOAD_ONLY" == true ]]; then
        args+=(--download-only)
    elif [[ "$UPLOAD_ONLY" == true ]]; then
        args+=(--upload-only)
    else
        args+=(--stage "$STAGE" --model "$MODEL" --process-res "$PROCESS_RES")
        if [[ "$SKIP_SETUP" == true ]]; then
            args+=(--skip-setup)
        fi
        if [[ "$FORCE_SETUP" == true ]]; then
            args+=(--force-setup)
        fi
    fi

    echo "${args[@]}"
}

# Export proxy URL for the Python script
export KAGGLE_PROXY_URL="$PROXY_URL"

# Cache the proxy URL for future runs
save_proxy_cache

# =============================================================================
# RUN WITH RETRY LOOP
# =============================================================================
RETRY_COUNT=0
FINAL_EXIT_CODE=1

while [[ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))

    if [[ "$RETRY_COUNT" -gt 1 ]]; then
        log_warning "Retry attempt $RETRY_COUNT/$MAX_RETRIES"
        backoff_sleep "$RETRY_COUNT"

        # Check session age before retry
        check_session_age

        # For robust deployer, it handles its own proxy refresh
        # But if proxy changed, update env
        if [[ -f "$PROXY_URL_FILE" ]]; then
            CACHED_URL=$(cat "$PROXY_URL_FILE")
            if [[ "$CACHED_URL" != "$PROXY_URL" ]]; then
                log_info "Proxy URL updated since last attempt, using new URL"
                PROXY_URL="$CACHED_URL"
                export KAGGLE_PROXY_URL="$PROXY_URL"
            fi
        fi
    fi

    log_info "========================================"
    log_info "Attempt $RETRY_COUNT/$MAX_RETRIES"
    log_info "  Mode: $([[ "$USE_LEGACY" == true ]] && echo 'legacy' || echo 'robust')"
    log_info "  Stage: $STAGE"
    log_info "  Model: $MODEL"
    log_info "  Resolution: $PROCESS_RES"
    log_info "========================================"

    ARGS=$(build_python_args)
    FULL_CMD=("${PYTHON_CMD[@]}" $ARGS)

    log_info "Running: ${FULL_CMD[*]}"

    # Run command, capture exit code
    "${FULL_CMD[@]}"
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        log_success "Kaggle deployment completed successfully!"
        if [[ "$DOWNLOAD_ONLY" == false && "$PING_ONLY" == false && "$SHELL_MODE" == false && "$UPLOAD_ONLY" == false ]]; then
            log_success "Results available at: $PROJECT_DIR/outputs/run_mvp/geometry/"

            # Verify key output files
            GEOMETRY_DIR="$PROJECT_DIR/outputs/run_mvp/geometry"
            if [[ -d "$GEOMETRY_DIR" ]]; then
                log_info "Output verification:"
                for f in camera_intrinsics.json camera_poses.json point_cloud.ply; do
                    if [[ -f "$GEOMETRY_DIR/$f" ]]; then
                        size=$(du -h "$GEOMETRY_DIR/$f" | cut -f1)
                        log_success "  ✓ $f ($size)"
                    else
                        log_warning "  ✗ $f missing"
                    fi
                done
                depth_count=$(find "$GEOMETRY_DIR/depth" -name "depth_*.npy" 2>/dev/null | wc -l | tr -d ' ')
                log_info "  Depth maps: $depth_count"
            fi
        fi
        FINAL_EXIT_CODE=0
        break
    else
        log_error "Attempt $RETRY_COUNT failed with exit code $EXIT_CODE"

        # Check if this is a transient error worth retrying
        if [[ $RETRY_COUNT -lt "$MAX_RETRIES" ]]; then
            log_info "Will retry in a moment..."
        else
            log_error "All $MAX_RETRIES attempts exhausted."
            log_info ""
            log_info "Troubleshooting:"
            log_info "  1. If 'Proxy expired' or 401/403: Get fresh URL from Kaggle UI"
            log_info "  2. If 'Kernel restart': Use --force-setup to re-run setup"
            log_info "  3. If 'CUDA OOM': Use smaller --process-res (e.g., --process-res 252)"
            log_info "  4. For interactive debugging: $0 --shell"
        fi
        FINAL_EXIT_CODE=$EXIT_CODE
    fi
done

exit $FINAL_EXIT_CODE