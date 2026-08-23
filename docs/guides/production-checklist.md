# Video-to-Terrain Production Checklist

This checklist ensures the pipeline is hardened for production use. Complete all items before deploying to production environments.

---

## 🔐 1. NVIDIA API Keys (CRITICAL)

### Required Actions
- [ ] **Get 3 real NVIDIA API keys** from [build.nvidia.com](https://build.nvidia.com)
  - Sign in with NVIDIA account
  - Navigate to NIM API keys
  - Create 3 separate keys for rotation pool
- [ ] **Update `.env` file** with real keys:
  ```bash
  cp .env.example .env
  # Edit .env and replace all nvapi-REPLACE_WITH_REAL_KEY_X
  ```
- [ ] **Verify key format**: Each key must start with `nvapi-` and be ~60+ characters
- [ ] **Test key rotation**: Run `python -m src.v2t.setup_production --stage keys`

### Validation
```bash
python -m src.v2t.setup_production --verify-only
# Should show: ✅ NVIDIA_API_KEY_1/2/3: Valid format
```

---

## 🤖 2. SAM3 Model (Stage D2)

### Required Actions
- [ ] **Install huggingface_hub**:
  ```bash
  pip install huggingface_hub
  ```
- [ ] **Download SAM3 model** (auto on first run, or pre-download):
  ```bash
  python -m src.v2t.setup_production --stage sam3
  ```
  - Downloads `facebook/sam3` from HuggingFace (~2-4 GB)
  - Caches to `~/.cache/huggingface/hub/models--facebook--sam3*`
- [ ] **Optional: Use local checkpoint** for air-gapped environments:
  ```bash
  export SAM3_CHECKPOINT=/path/to/sam3_checkpoint.pth
  ```

### Validation
```bash
python -m src.v2t.run_pipeline --only D2
# Should show: SAM3 model loaded successfully (not "STUB mask")
```

### Fallback Behavior
If SAM3 unavailable, Stage D2 falls back to depth-edge + connected components stub masks (`stub: true` in metadata).

---

## 🤖 3. Robot Policy Controllers (Stage H)

### Required Actions
- [ ] **Install MJLab and vinrobotics_mjlab** (for real policy configs):
  ```bash
  # From source (recommended for development)
  pip install -e /path/to/vinrobotics_mjlab
  
  # Or if published to PyPI
  pip install mjlab vinrobotics_mjlab
  ```
- [ ] **Configure policy type** in `.env`:
  ```bash
  ROBOT_POLICY_TYPE=tracking  # or "velocity" or "rough"
  ```
- [ ] **Optional: Add trained policy checkpoints** for inference:
  ```bash
  export POLICY_CHECKPOINT_DIR=/path/to/checkpoints
  # Should contain .pt or .pth files from RL training
  ```
- [ ] **Generate policy configs**:
  ```bash
  python -m src.v2t.setup_production --stage policy
  ```

### Policy Types
| Type | Use Case | Terrain |
|------|----------|---------|
| `tracking` | Motion tracking, flat terrain | Plane |
| `rough` | Motion tracking, uneven terrain | Rough/procedural |
| `velocity` | Locomotion commands | Plane |

### Output Files
- `outputs/run_mvp/simulation/policy/policy_config.json` — JSON config for programmatic use
- `outputs/run_mvp/simulation/policy/policy_cfg.py` — MJLab-compatible Python config
- `outputs/run_mvp/robot/policy/` — Duplicate for direct access

### Validation
```bash
python -m src.v2t.run_pipeline --only H
# Should generate both tracking and velocity configs
```

---

## 🖥️ 4. GPU / Compute (Stage B - DA3)

### Required Actions
- [ ] **Verify GPU availability**:
  ```bash
  python -m src.v2t.setup_production --stage gpu
  ```
- [ ] **CUDA (NVIDIA GPU)**: Preferred for production
  - DA3 runs ~10x faster on GPU (60s → ~6s for 100 frames)
  - Requires NVIDIA GPU with 8GB+ VRAM
- [ ] **MPS (Apple Silicon)**: Works for development
  - Slower than CUDA but functional
- [ ] **CPU fallback**: Works but very slow (not recommended for production)

### Production Deployment (GPU)
- [ ] **Kaggle GPU** (free tier): Use `scripts/kaggle/deploy_kaggle.sh` for remote GPU
  ```bash
  export KAGGLE_PROXY_URL='https://.../proxy'
  python src/v2t/kaggle_remote.py upload-dir . project
  python src/v2t/kaggle_remote.py shell --cwd /project
  ```
- [ ] **Cloud GPU** (AWS/GCP/Azure): Deploy with Docker
  ```dockerfile
  FROM nvidia/cuda:12.1-runtime-ubuntu22.04
  # ... install deps
  ```

---

## 📦 5. Dependencies

### Required Actions
- [ ] **Install all Python dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
  Key packages:
  - `torch`, `torchvision` (with CUDA/MPS support)
  - `opencv-python`, `pillow`
  - `trimesh`, `open3d`
  - `openai`, `python-dotenv`
  - `huggingface_hub`, `tqdm`, `pyyaml`

- [ ] **Verify imports**:
  ```bash
  python -m src.v2t.setup_production --stage deps
  ```

---

## 🎬 6. Input Data

### Required Actions
- [ ] **Place input video** at expected location:
  ```
  input/20260810_150244_color.mp4
  ```
  Or update `CFG.input_video` in config
- [ ] **Verify video properties**:
  - Format: MP4 (H.264)
  - Resolution: 1920x1080 or similar
  - Duration: 10-60 seconds recommended
  - Frame rate: 30 FPS

### Validation
```bash
python -m src.v2t.setup_production --stage video
```

---

## ⚙️ 7. Pipeline Configuration

### Required Actions
- [ ] **Review and adjust config** in `src/v2t/config.py`:
  - `MAX_FRAMES`: Increase from 10 to 100+ for production
  - `FRAME_SAMPLE_RATE`: Adjust based on video length
  - `SIM_STEPS`: Increase for longer validation (300 → 1000+)
  - `DA3_DEVICE`: Set to "cuda" for GPU

- [ ] **Validate full config**:
  ```bash
  python -m src.v2t.run_pipeline --validate-config
  ```

---

## 🧪 8. End-to-End Validation

### Required Actions
- [ ] **Run full pipeline** (dry run):
  ```bash
  python -m src.v2t.run_pipeline
  ```
  Expected: All 10 stages (A-J) complete with PASS

- [ ] **Check outputs**:
  ```
  outputs/run_mvp/
  ├── validation/report.json     # Should show 15+/16 checks PASS
  ├── mujoco/scene.xml           # Should compile in MuJoCo
  ├── mjlab/scene_cfg.py         # MJLab SceneConfig
  └── client_report.html         # Interactive HTML report
  ```

- [ ] **Verify no STUB stages** in production:
  - Stage D2: Should show "SAM3 model loaded" not "STUB mask"
  - Stage H: Should show MJLab configs generated

---

## 🚀 9. Deployment Checklist

### Pre-deployment
- [ ] All items above ✅
- [ ] `.env` file committed to secure secret manager (NOT git)
- [ ] CI/CD pipeline configured for automated testing
- [ ] Monitoring/alerting for pipeline failures
- [ ] Log aggregation configured (structured JSON logs)

### Production Environment
- [ ] GPU nodes available (CUDA 12.1+)
- [ ] Sufficient disk space (100GB+ for outputs)
- [ ] Network access to `integrate.api.nvidia.com` (NIM API)
- [ ] HuggingFace Hub access (for SAM3 download)
- [ ] MJLab/vinrobotics_mjlab installed

### Post-deployment
- [ ] First production run completes successfully
- [ ] Validation report shows PASS
- [ ] Client package generated (`delivery_*.zip`)
- [ ] Metrics logged (stage timings, GPU utilization, API costs)

---

## 🔍 10. Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| "No valid NVIDIA API keys" | Check `.env` has real `nvapi-` keys, not placeholders |
| "SAM3 unavailable" | Run setup_production --stage sam3; check HF Hub access |
| "mjlab not available" | `pip install -e /path/to/vinrobotics_mjlab` |
| "CUDA out of memory" | Reduce `MAX_FRAMES`, `DA3_PROCESS_RES`, or use CPU |
| "DA3 slow" | Ensure GPU detected; check `torch.cuda.is_available()` |
| "Policy config not generated" | Run setup_production --stage policy |
| "Validation FAIL" | Check `validation/report.json` for specific failed checks |

---

## 📋 Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| DevOps Engineer | | | |
| ML Engineer | | | |
| Robotics Engineer | | | |
| Tech Lead | | | |

---

**Version**: 1.0  
**Last Updated**: 2026-08-13  
**Pipeline Version**: Video-to-Terrain MVP