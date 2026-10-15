---
name: kaggle-runtime-toolkit
description: >-
  Orchestrate and execute remote GPU workloads on Kaggle Jupyter kernels
  (e.g., RTX PRO 6000 Blackwell / A100 / T4) via Jupyter WebSocket & Contents API.
  Use to run code remotely, upload datasets/scripts, stream logs live, and download output artifacts.
---

# Kaggle Runtime Toolkit (`kaggle-runtime-toolkit`)

The **Kaggle Runtime Toolkit** is a specialized skill for managing, streaming, and orchestrating remote high-performance GPU tasks (such as RL policy training, 3D shape generation, point cloud reconstruction, neural rendering, or deep learning inference) running on Kaggle Jupyter servers via WebSocket and Contents API bridges without requiring cloud API keys or Docker setup.

---

## 📁 Toolkit Directory Structure

```text
kaggle-runtime-toolkit/
├── SKILL.md                 # Antigravity skill specification & runbook
├── README.md                # Quickstart and overview documentation
├── LICENSE                  # MIT License
├── requirements.txt         # Dependencies (requests, websocket-client)
├── scripts/                 # Core executable CLI tools
│   ├── kaggle_executor.py   # WebSocket kernel code execution & live streaming
│   ├── kaggle_remote.py     # PTY interactive shell & remote file manager
│   ├── kaggle_relay.py      # Kernel-to-kernel zero-hop data transfer
│   └── kx.sh                # Resilient runner with auto-retry & backoff
├── references/              # Detailed guides (Progressive Disclosure)
│   ├── KERNEL_TO_KERNEL_TRANSFER.md  # High-speed data transfer between kernels
│   └── API_REFERENCE.md              # Jupyter WebSocket & Contents API specs
└── examples/                # Runnable workflow templates
    ├── gpu_benchmark.py     # CUDA & VRAM benchmark
    ├── file_sync.py         # Remote file execution & artifact download
    └── hunyuan3d/           # 3D shape generation workflow
        └── run_hunyuan3d.py
```

---

## ⚡ Quick Reference: Common Commands

All tools automatically resolve the active proxy URL from `.env` in the current or parent directory (checking `KAGGLE_PROXY_URL`, `KAGGLE_PROXY_RTX6000_NO_INTERNET`, `KAGGLE_WITHOUT_INTERNET`, `KAGGLE_T4X2_PROXY`, `KAGGLE_HAS_INTERNET`).

### 1. Execute Code on Remote GPU

```bash
# Execute Python snippet directly (auto-streams stdout/stderr live)
python scripts/kaggle_executor.py exec "
import torch
print('Device:', torch.cuda.get_device_name(0))
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

# Execute a standalone script on the remote kernel
python scripts/kaggle_executor.py run examples/gpu_benchmark.py

# Using the resilient auto-retry wrapper (scripts/kx.sh)
scripts/kx.sh exec "import torch; print(torch.cuda.is_available())"
```

### 2. File & Directory Transfers

```bash
# Upload a local file to remote /kaggle/working/
python scripts/kaggle_executor.py upload ./local_data.pt dataset/local_data.pt

# Download an artifact back to local machine
python scripts/kaggle_executor.py download output/mesh.glb ./mesh.glb

# Download an entire remote directory (filtered by file extensions)
python scripts/kaggle_executor.py download-dir output/ ./results/ --ext .glb .pt .json
```

### 3. Interactive Shell & Inspection

```bash
# Ping connection to Kaggle Jupyter proxy
python scripts/kaggle_remote.py ping

# Open an interactive PTY shell inside the Kaggle environment (detach with Ctrl+])
python scripts/kaggle_remote.py shell --cwd /kaggle/working

# List remote files in /kaggle/working
python scripts/kaggle_remote.py ls
```

### 4. High-Speed Kernel-to-Kernel Relay (Zero Local Bandwidth)

Transfer large datasets or model weights directly from an internet-enabled kernel to an offline GPU kernel:

```bash
# Relays data directly across Kaggle platform without passing through local machine
python scripts/kaggle_relay.py https://huggingface.co/path/to/weights.safetensors \
  --dest /kaggle/working/weights.safetensors \
  --workers 4
```

For advanced transfer techniques and measured throughput benchmarks, see [references/KERNEL_TO_KERNEL_TRANSFER.md](kaggle-runtime-toolkit-agent-skills/references/KERNEL_TO_KERNEL_TRANSFER.md).

---

## 🛠 Script Reference & Entrypoints

| Script | Purpose |
| :--- | :--- |
| [`scripts/kaggle_executor.py`](kaggle-runtime-toolkit-agent-skills/scripts/kaggle_executor.py) | Executes Python snippets and scripts via Jupyter WebSocket, streams live logs, and manages kernels. |
| [`scripts/kaggle_remote.py`](kaggle-runtime-toolkit-agent-skills/scripts/kaggle_remote.py) | Opens interactive PTY terminals (`shell`, `attach`, `terminals`) and handles file transfers. |
| [`scripts/kaggle_relay.py`](kaggle-runtime-toolkit-agent-skills/scripts/kaggle_relay.py) | Streams files between online and offline kernels via parallel Contents API PUTs. |
| [`scripts/kx.sh`](kaggle-runtime-toolkit-agent-skills/scripts/kx.sh) | Execution wrapper with exponential backoff and automatic retry on transient proxy disconnects. |

---

## 🔍 Troubleshooting & Best Practices

1. **Proxy URL Expiration (HTTP 404)**:
   - Kaggle Jupyter proxy URLs contain short-lived JWT tokens that expire when a session restarts.
   - If an HTTP 404 occurs on `/api`, reopen the Kaggle notebook, copy the new URL from **⋮ More settings** → **Kaggle Jupyter Server**, and update `KAGGLE_PROXY_URL` in `.env`.

2. **Large File Uploads via Contents API**:
   - Single-file Contents API PUTs over ~24 MB may truncate without raising HTTP errors.
   - For files >20 MB, use `kaggle_relay.py` or chunked multipart upload.

3. **Background Tasks**:
   - Long-running scripts can be launched in a detached terminal (`kaggle_remote.py shell`) or monitored asynchronously.
