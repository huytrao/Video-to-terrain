# GPU Deployment Guide: Kaggle

This guide covers deploying the Video-to-Terrain pipeline to Kaggle's free GPU runtime for accelerated Depth Anything 3 (DA3) inference.

## Overview

| Platform | Device | DA3 Inference (100 frames) |
|----------|--------|---------------------------|
| Local (Mac MPS) | Apple Silicon GPU | ~60 seconds |
| **Kaggle** | **NVIDIA GPU (Tesla P100/T4)** | **~6-7 seconds** |

Kaggle provides free GPU access (30 hours/week) with NVIDIA GPUs and CUDA pre-installed, making it ideal for DA3 metric depth estimation.

---

## Prerequisites

1. **Kaggle Account** - Create at [kaggle.com](https://www.kaggle.com)
2. **Kaggle Notebook** - Create a new notebook with GPU accelerator:
   - Settings → Accelerator → **GPU T4 x2** (or P100)
   - Enable **Internet** access
3. **Jupyter Proxy URL** - Get from running notebook:
   - Run notebook → Copy the URL from browser
   - Format: `https://kkb-production.jupyter-proxy.kaggle.net/k/{kernel_id}/.../proxy`

---

## Quick Start

### 1. Start Kaggle Notebook

```bash
# Create notebook at kaggle.com with:
# - GPU accelerator enabled
# - Internet access enabled
# - Run the notebook (keep it running)
```

### 2. Get Proxy URL

From the running notebook URL in browser:
```
https://www.kaggle.com/code/{username}/{notebook_name}/edit?kernelSessionId={session_id}
```

The proxy URL format:
```
https://kkb-production.jupyter-proxy.kaggle.net/k/{kernel_id}/{token}/proxy
```

### 3. Deploy from Local Machine

```bash
# Set proxy URL
export KAGGLE_PROXY_URL='https://kkb-production.jupyter-proxy.kaggle.net/k/.../proxy'

# Run full pipeline (upload → setup → DA3 → download)
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL"

# Or with specific model
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --model da3metric-large --process-res 504
```

---

## Command Reference

### scripts/kaggle/deploy_kaggle.sh Options

| Option | Description | Default |
|--------|-------------|---------|
| `-u, --url URL` | Kaggle Jupyter proxy URL | `$KAGGLE_PROXY_URL` |
| `-s, --stage STAGE` | Pipeline stage (B only) | `B` |
| `-m, --model MODEL` | DA3 model variant | `da3metric-large` |
| `-r, --process-res RES` | DA3 process resolution | `504` |
| `--upload-only` | Only upload project files | - |
| `--download-only` | Only download results | - |
| `--shell` | Open interactive shell | - |
| `--ping` | Test connection only | - |
| `--skip-setup` | Skip dependency installation | - |
| `-h, --help` | Show help | - |

### DA3 Model Options

| Model | Type | Speed | Quality |
|-------|------|-------|---------|
| `da3metric-large` | Metric (absolute scale) | ~6s | Best |
| `da3metric-base` | Metric | ~4s | Good |
| `da3metric-small` | Metric | ~2s | Fast |
| `depth-anything-v3-large` | Relative | ~5s | Good |

### Process Resolution

| Resolution | VRAM Usage | Speed | Quality |
|------------|-----------|-------|---------|
| `504` | ~6 GB | Baseline | Baseline |
| `768` | ~12 GB | Slower | Higher |
| `1024` | ~18 GB | Slowest | Highest |

---

## Workflow Examples

### Full Pipeline (Recommended)

```bash
# 1. Run Stage A locally (extract frames, run HOLI-Spatial)
python -m src.v2t.run_pipeline --stage A

# 2. Deploy Stage B to Kaggle GPU
export KAGGLE_PROXY_URL='your-proxy-url'
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL"

# 3. Results downloaded to outputs/run_mvp/geometry/
#    - camera_intrinsics.json
#    - camera_poses.json
#    - point_cloud.ply
#    - depth/depth_*.npy
```

### Partial Operations

```bash
# Test connection only
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --ping

# Upload project only (for manual interaction)
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --upload-only

# Open interactive shell on Kaggle
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --shell

# Download results from previous run
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --download-only

# Re-run DA3 with different model (skip setup for speed)
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --model da3metric-base --skip-setup
```

### Using Python Directly

```bash
export KAGGLE_PROXY_URL='your-proxy-url'

# Full pipeline
python -m src.v2t.deploy_kaggle --stage B --model da3metric-large

# Only download
python -m src.v2t.deploy_kaggle --download-only

# Interactive shell
python -m src.v2t.deploy_kaggle --shell
```

---

## Expected Outputs

After successful deployment, local `outputs/run_mvp/geometry/` contains:

```
outputs/run_mvp/geometry/
├── camera_intrinsics.json      # DA3-estimated intrinsics per frame
├── camera_poses.json           # DA3-estimated poses (world-to-camera)
├── point_cloud.ply             # Fused point cloud (colored)
└── depth/
    ├── depth_000000.npy        # Depth map (meters, metric scale)
    ├── depth_000001.npy
    ├── conf_000000.npy         # Confidence map
    └── ...
```

---

## Troubleshooting

### Connection Issues

```bash
# Test connection
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --ping

# Common errors:
# - "Cannot reach Jupyter API" → Notebook not running or wrong URL
# - "403 Forbidden" → Internet access not enabled in notebook settings
# - "Connection timeout" → Notebook idle, restart it
```

### DA3 Fails to Import

```bash
# Check DA3 installation on Kaggle
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --shell
# Then in shell:
pip install -e /kaggle/working/Video-to-Terrain/depth-anything-3
python -c "from depth_anything_3.api import DepthAnything3; print('OK')"
```

### No Frames Found

```
ERROR: No frames found! Run Stage A first locally.
```

**Fix:** Run Stage A locally first:
```bash
python -m src.v2t.run_pipeline --stage A
```

### CUDA Not Available

```bash
# Verify GPU on Kaggle
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --shell
# Then in shell:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Fix:** Ensure notebook has GPU accelerator enabled in settings.

### Out of Memory

```bash
# Reduce process resolution
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --process-res 504

# Or use smaller model
./scripts/kaggle/deploy_kaggle.sh --url "$KAGGLE_PROXY_URL" --model da3metric-base
```

### Slow Upload/Download

- Large video files: Upload input video separately if needed
- Many depth maps: First run downloads all; subsequent runs can use `--skip-setup`

---

## Architecture

```
Local Machine                          Kaggle GPU Runtime
┌─────────────────────┐                ┌─────────────────────┐
│ 1. Stage A:         │                │                     │
│    - Extract frames │                │                     │
│    - HOLI-Spatial   │                │                     │
│    - Save frames    │───────────────►│ 2. Upload project   │
│                     │   (HTTPS)      │    + input video    │
│                     │                │                     │
│                     │                │ 3. Install deps     │
│                     │                │    - torch CUDA     │
│                     │                │    - DA3 + deps     │
│                     │                │                     │
│                     │                │ 4. Run Stage B      │
│                     │                │    (DA3 on GPU)     │
│                     │                │    ~6-7 seconds     │
│                     │                │                     │
│ 5. Download         │◄───────────────│ 5. Save outputs     │
│    results          │   (HTTPS)      │    to /kaggle/      │
│                     │                │    working/...      │
└─────────────────────┘                └─────────────────────┘
```

---

## Security Notes

- **Proxy URL is a bearer token** - Treat like a password
- **Do not commit** proxy URLs to git
- **Rotate** by restarting Kaggle notebook (generates new URL)
- **Local .env** - Store in `.env` file (gitignored):
  ```bash
  echo "KAGGLE_PROXY_URL='https://.../proxy'" > .env
  source .env
  ./scripts/kaggle/deploy_kaggle.sh
  ```

---

## Cost & Limits

| Resource | Kaggle Free Tier |
|----------|-----------------|
| GPU Hours/Week | 30 hours |
| GPU Type | T4 (16 GB) or P100 (16 GB) |
| Disk | 20 GB |
| Internet | Yes (when enabled) |
| Max Session | 12 hours |

**Estimated usage per run:**
- Setup: ~2-3 minutes
- DA3 inference: ~30-60 seconds
- Total: ~5 minutes per deployment

---

## Advanced: Custom Kaggle Notebook

For production use, create a custom notebook with pre-installed dependencies:

```python
# In Kaggle notebook cell:
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q open3d trimesh networkx tqdm pyyaml python-dotenv numpy pillow opencv-python scipy matplotlib
!pip install -q -e /kaggle/working/Video-to-Terrain/depth-anything-3
```

Then use `--skip-setup` flag for faster subsequent runs.

---

## Related Files

| File | Purpose |
|------|---------|
| `src/v2t/deploy_kaggle.py` | Main Python deployment script |
| `scripts/kaggle/deploy_kaggle.sh` | Bash wrapper with colored output |
| `src/v2t/kaggle_remote.py` | Low-level Kaggle Jupyter API client |
| `src/v2t/geometry/stage_b_da3.py` | DA3 inference implementation |
| `src/v2t/config.py` | Pipeline configuration (CFG) |

---

## Support

- **Issues**: Check Kaggle notebook logs via `--shell`
- **DA3 Docs**: [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3)
- **Kaggle API**: [Jupyter REST API](https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html)