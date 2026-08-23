# Robust Kaggle GPU Deployment

This document describes the robust deployment system for running Video-to-Terrain pipeline on Kaggle GPU runtime with full session lifecycle management.

## Overview

The robust deployment system (`kaggle_robust.py` + `scripts/kaggle/deploy_kaggle.sh`) handles:

1. **Proxy URL Expiration** - JWT tokens expire ~1-2 hours
2. **Connection Drops** - Automatic retry with exponential backoff
3. **Kernel Restarts** - Detection and recovery (re-run setup + inference)
4. **Resume Capability** - Skip already-uploaded/downloaded files
5. **Keep-Alive** - Session health monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    scripts/kaggle/deploy_kaggle.sh                         │
│  (Wrapper with retry logic, session monitoring, proxy cache)│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              src/v2t/kaggle_robust.py                       │
│  (RobustKaggleProxy + RobustKaggleDeployer)                 │
│  - Retry decorator with exponential backoff                 │
│  - Proxy health checks & auto-refresh prompting             │
│  - Kernel restart detection (PID monitoring)                │
│  - Resume upload/download via file size comparison          │
│  - Keep-alive during long commands                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
            ┌──────────┐  ┌──────────┐
            │  Legacy  │  │  Robust  │
            │ deployer │  │ deployer │
            └──────────┘  └──────────┘
```

## Key Components

### 1. RobustKaggleProxy

Enhanced version of `KaggleProxy` with:

- **Retry decorator** (`@with_retry`) on all network operations
- **Proxy health check** (`check_proxy_health()`) - returns session elapsed/remaining time
- **Proxy refresh** (`refresh_proxy_url()`) - swap URL without restarting script
- **Kernel restart detection** (`detect_kernel_restart()`) - monitors PID
- **Resume upload** - compares file sizes before uploading
- **Resume download** - skips files already present with matching size
- **Keep-alive during long commands** - sends harmless newlines to maintain session

### 2. RobustKaggleDeployer

High-level orchestrator with:

- **Automatic proxy validation** before each major operation
- **Interactive proxy refresh prompt** when JWT expires
- **Setup recovery** - re-runs environment setup after kernel restart
- **Progress callbacks** for upload/download tracking
- **Structured error handling** with specific exception types

### 3. scripts/kaggle/deploy_kaggle.sh Wrapper

Provides:

- **Pipeline-level retry** (default 3 attempts with exponential backoff)
- **Session age monitoring** - warns when JWT nearing expiry
- **Proxy URL caching** - avoids re-prompting on subsequent runs
- **Session status/keep-alive modes** for long-running sessions

## Exception Types

| Exception | When Raised | Recovery Action |
|-----------|-------------|-----------------|
| `ProxyExpiredError` | 401/403 with JWT/expired in response | Prompt for fresh URL, call `refresh_proxy_url()` |
| `ConnectionLostError` | 503 / network timeout | Retry with backoff |
| `KernelRestartedError` | PID changed during command | Re-run setup + inference (auto) |
| `UploadFailedError` | >10% upload failures | Abort, manual investigation |
| `DownloadFailedError` | >10% download failures | Retry with resume |

## Usage

### Basic Deploy (Recommended - Uses Robust Mode)

```bash
# First time - provide proxy URL
./scripts/kaggle/deploy_kaggle.sh --url "https://.../proxy"

# Subsequent runs - uses cached proxy URL
./scripts/kaggle/deploy_kaggle.sh
```

### Specific Operations

```bash
# Test connection only
./scripts/kaggle/deploy_kaggle.sh --ping

# Check session health without operations
./scripts/kaggle/deploy_kaggle.sh --session-status

# Check detailed proxy health
./scripts/kaggle/deploy_kaggle.sh --proxy-health

# Interactive shell for debugging
./scripts/kaggle/deploy_kaggle.sh --shell

# Force re-setup (after kernel restart)
./scripts/kaggle/deploy_kaggle.sh --force-setup

# Download results from previous run (resume supported)
./scripts/kaggle/deploy_kaggle.sh --download-only

# Upload only (for manual intervention)
./scripts/kaggle/deploy_kaggle.sh --upload-only

# Use legacy deployer (no retry/resume)
./scripts/kaggle/deploy_kaggle.sh --use-legacy

# Custom retry count
./scripts/kaggle/deploy_kaggle.sh --max-retries 5

# Clear cached proxy URL
./scripts/kaggle/deploy_kaggle.sh --clear-proxy-cache
```

### Long-Running Session Keep-Alive

```bash
# Run in background to keep session alive (pings every 60s)
./scripts/kaggle/deploy_kaggle.sh --keep-alive &
KEEP_ALIVE_PID=$!

# Do your work...

# Stop keep-alive
kill $KEEP_ALIVE_PID
```

### Environment Variable

```bash
export KAGGLE_PROXY_URL="https://.../proxy"
./scripts/kaggle/deploy_kaggle.sh
```

## Session Lifecycle

### Typical JWT Timeline

```
T=0        : Get fresh proxy URL from Kaggle UI
T=0-1h     : Normal operations (healthy)
T=1h-1h55m : Warning in logs (proxy nearing expiry)
T=2h       : JWT expires → 401/403 on API calls
T=2h+      : Script detects, prompts for fresh URL
```

### Proxy Refresh Flow

```
1. API call returns 401/403 with "jwt"/"expired" in response
2. RobustKaggleProxy raises ProxyExpiredError
3. RobustKaggleDeployer catches, calls prompt_for_fresh_url()
4. User pastes new URL from Kaggle UI
5. refresh_proxy_url() called - creates new session, re-bootsstraps
6. Operation retries automatically
```

### Kernel Restart Flow

```
1. run_command() detects PID change (via detect_kernel_restart())
2. KernelRestartedError raised
3. RobustKaggleDeployer catches, sets _setup_complete = False
4. Re-runs setup_environment() (installs deps again)
5. Re-runs Stage B inference
6. Max 2 kernel restarts before giving up
```

## Resume Capability

### Upload Resume

Before uploading each file, checks remote Jupyter Contents API:
- If file exists with same size → **SKIP**
- If file exists with different size → **OVERWRITE**
- If file doesn't exist → **UPLOAD**

This makes interrupted uploads restart quickly.

### Download Resume

Before downloading each file:
- If local file exists with same size as remote → **SKIP**
- Otherwise → **DOWNLOAD**

Use `--download-only` to resume after interruption.

## Configuration

### RetryConfig (in kaggle_robust.py)

```python
@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 2.0      # seconds
    max_delay: float = 60.0      # seconds
    exponential_base: float = 2.0
    jitter: float = 0.1          # 10% jitter

    http_timeout: float = 30.0
    upload_timeout: float = 120.0
    download_timeout: float = 180.0
    command_timeout: float = 300.0
```

### SessionConfig (in kaggle_robust.py)

```python
@dataclass
class SessionConfig:
    keepalive_interval: int = 60      # seconds between keepalive pings
    proxy_refresh_threshold: int = 300  # warn 5 min before JWT expiry
    max_session_duration: int = 7200    # 2 hours max session
```

### scripts/kaggle/deploy_kaggle.sh Settings

```bash
MAX_PIPELINE_RETRIES=3      # Total pipeline attempts
BACKOFF_BASE=5              # Initial backoff seconds
BACKOFF_MAX=120             # Max backoff seconds
MAX_SESSION_SECONDS=7200    # JWT lifetime estimate
```

## Troubleshooting

### "Proxy URL expired" / 401 Unauthorized

**Cause**: JWT token in proxy URL expired (typical 1-2 hour lifetime)

**Fix**: 
1. Go to Kaggle notebook in browser
2. Click "Copy Link" / proxy URL button
3. When prompted by script, paste new URL
4. Or: `./scripts/kaggle/deploy_kaggle.sh --clear-proxy-cache && ./scripts/kaggle/deploy_kaggle.sh --url "NEW_URL"`

### "Kernel restarted" / "KernelRestartedError"

**Cause**: Kaggle kernel restarted (OOM, timeout, maintenance)

**Fix**: 
- Script auto-recovers (re-runs setup + inference)
- Max 2 restarts before giving up
- Use `--force-setup` to manually trigger full re-setup

### "CUDA out of memory"

**Cause**: GPU memory insufficient for model/resolution

**Fix**: 
```bash
./scripts/kaggle/deploy_kaggle.sh --process-res 252  # Default is 504
# Or use smaller model
./scripts/kaggle/deploy_kaggle.sh --model da3metric-small
```

### Connection timeout / ReadTimeout

**Cause**: Network instability, Kaggle busy

**Fix**: 
- Script retries automatically (3x with backoff)
- Increase timeouts in RetryConfig if persistent
- Try `--use-legacy` to compare behavior

### Upload/Download stuck

**Cause**: Large files, slow connection

**Fix**:
- Resume capability auto-skips completed files
- Run `--download-only` to resume download
- Check `outputs/run_mvp/geometry/` for partial results

## Monitoring Commands

```bash
# Quick health check
./scripts/kaggle/deploy_kaggle.sh --ping

# Detailed health with session timing
./scripts/kaggle/deploy_kaggle.sh --proxy-health

# Continuous keep-alive (background)
./scripts/kaggle/deploy_kaggle.sh --keep-alive &
```

## Integration with CI/CD

For automated pipelines, set proxy URL via environment and use non-interactive mode:

```bash
# In CI, you need a way to refresh proxy URL programmatically
# Option 1: Use Kaggle API to get fresh URL
# Option 2: Store URL in secret, update via external process

export KAGGLE_PROXY_URL="$KAGGLE_PROXY_URL_SECRET"
./scripts/kaggle/deploy_kaggle.sh --max-retries 5 2>&1 | tee deploy.log

# Check exit code
if [ $? -eq 0 ]; then
    echo "Deploy successful"
else
    echo "Deploy failed - check logs"
fi
```

## Files

| File | Description |
|------|-------------|
| `src/v2t/kaggle_robust.py` | Core robust implementation |
| `src/v2t/deploy_kaggle.py` | Legacy deployer (for comparison) |
| `src/v2t/kaggle_remote.py` | Original proxy client |
| `scripts/kaggle/deploy_kaggle.sh` | Robust wrapper script |
| `.kaggle_proxy_url_cache` | Cached proxy URL (auto-created) |

## Testing

```bash
# Test proxy health
./scripts/kaggle/deploy_kaggle.sh --proxy-health

# Test with dry-run (ping only)
./scripts/kaggle/deploy_kaggle.sh --ping

# Test upload/download cycle
./scripts/kaggle/deploy_kaggle.sh --upload-only
./scripts/kaggle/deploy_kaggle.sh --download-only
```

## Migration from Legacy

The robust deployer is a drop-in replacement. Key differences:

| Feature | Legacy | Robust |
|---------|--------|--------|
| Retry logic | Manual | Automatic (exponential backoff) |
| Proxy expiry | Crashes | Prompts for fresh URL |
| Kernel restart | Fails silently | Auto-recovers (re-setup + re-run) |
| Upload resume | No | Yes (size comparison) |
| Download resume | No | Yes (size comparison) |
| Keep-alive | No | During long commands |
| Session monitoring | No | `--session-status`, `--proxy-health` |

To use legacy: `./scripts/kaggle/deploy_kaggle.sh --use-legacy`