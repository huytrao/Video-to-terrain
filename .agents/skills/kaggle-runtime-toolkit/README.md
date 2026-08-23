# kaggle-runtime-agent-skills

<div align="center">

<img src="https://img.shields.io/badge/Kaggle-GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
<img src="https://img.shields.io/github/stars/huytrao/kaggle-runtime-agent-skills?style=for-the-badge&color=f59e0b"/>

<br/><br/>

**Control any Kaggle GPU notebook from your local terminal.**
<br/>
Upload files. Execute code. Stream output live. Download results.
<br/>
No Kaggle API key. No Docker. No cloud bill.

<br/>

[**Quick Start**](#-quick-start) · [**Tools**](#-tools) · [**Examples**](#-examples) · [**How it works**](#-how-it-works)

</div>

---

<br/>

## ✨ What is this?

You have a MacBook. You need a GPU. Kaggle gives you a **free Tesla T4 (15 GB VRAM)**.

But Kaggle's UI is clunky for serious work — you can't easily run scripts, pipe output, or automate workflows from your local machine.

**kaggle-runtime-agent-skills** fixes that. It turns the Jupyter proxy URL of any running Kaggle notebook into a full remote control interface:

| What you can do | How |
|---|---|
| 📤 Upload any file or directory | Jupyter Contents API (base64, any size) |
| 📥 Download results to your machine | Same API — fast and reliable |
| ⚡ Execute Python code with **live output streaming** | Jupyter WebSocket kernel protocol |
| 🖥️ Open an interactive shell (PTY) | Terminado WebSocket |
| 🤖 Run full AI/GPU pipelines end-to-end | `kaggle_executor.py` + your script |

> **Zero Kaggle API key required.** Just the proxy URL from your browser.

<br/>

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/huytrao/kaggle-runtime-agent-skills.git
cd kaggle-runtime-agent-skills

# 2. Install (macOS / Linux)
pip install -r requirements.txt

# 3. Get your proxy URL
#    → Open a Kaggle notebook in your browser
#    → Copy the URL (it ends with /proxy)
export KAGGLE_PROXY_URL='https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/.../proxy'

# 4. Verify connection
python kaggle_remote.py ping
# ✓ Kaggle Jupyter API reachable

# 5. Run anything on the GPU
python kaggle_executor.py exec "
import torch
print(torch.cuda.get_device_name(0))
print(f'Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')
"
```

<br/>

---

## 🛠 Tools

### `kaggle_remote.py` — File manager + interactive shell

```
python kaggle_remote.py <command>

  ping                       Check connectivity to the Kaggle proxy
  ls [remote]                List files in a remote path
  upload <local> <remote>    Upload a single file
  upload-dir <local> <dir>   Upload a whole directory (skips .git, __pycache__)
  download <remote> <local>  Download a single file
  terminals                  List active terminal sessions
  shell [--cwd PATH]         Open a new interactive PTY shell  (Ctrl+] to detach)
  attach <name>              Reattach to an existing terminal
  kill <name>                Kill a terminal session
```

**Interactive shell example:**
```bash
python kaggle_remote.py shell --cwd /kaggle/working
# Kaggle terminal opens in your local iTerm/Terminal
# Ctrl+] detaches — terminal keeps running on Kaggle
python kaggle_remote.py attach 1   # resume anytime
```

---

### `kaggle_relay.py` — pull a file into a no-internet notebook

Two running notebooks, no local relay: a file streams **source → notebook
with internet → notebook without internet**, straight through Kaggle's own
Contents API. Your machine submits one small driver script and never sees
the file bytes.

```
python kaggle_relay.py <url> [--dest PATH] [--workers N] [--subchunk MB]
```

```bash
export KAGGLE_INTERNET='https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/.../proxy'          # has internet
export KAGGLE_WITHOUT_INTERNET='https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/.../proxy'  # internet OFF

python kaggle_relay.py https://huggingface.co/1038lab/sam3/resolve/main/sam3.safetensors
# 3.44 GB relayed in ~2 minutes, 4 parallel byte-range workers
```

See [How the no-internet relay works](#-how-the-no-internet-relay-works) below.

> **Moving a file the other way, or between two notebooks that both have internet?**
> See **[KERNEL_TO_KERNEL_TRANSFER.md](KERNEL_TO_KERNEL_TRANSFER.md)** — all four
> routes with measured throughput, and the limits that fail *silently* (a Contents
> API PUT over ~24 MB is truncated with no error, so pass `--subchunk 8`).

---

### `kaggle_executor.py` — GPU kernel executor

```
python kaggle_executor.py <command>

  kernels                         List active Jupyter kernels
  exec "<code>"                   Execute an inline Python snippet (streams live)
  run <script.py> [--timeout N]   Upload + execute a local script (streams live)
  upload <local> <remote>         Upload a file
  download <remote> <local>       Download a file
  download-dir <r> <l> [--ext …]  Download all files in a remote dir
  ls [remote]                     List remote path
```

**Live-stream a GPU training job:**
```bash
# Your training script stays on your machine — executor uploads + runs it
python kaggle_executor.py run train.py --timeout 3600
```

**Download only the files you need:**
```bash
python kaggle_executor.py download-dir output ./local_out --ext .pt .onnx .csv
```

<br/>

---

## 📚 Examples

### Check GPU specs
```bash
python kaggle_executor.py exec "
import torch, subprocess
subprocess.run(['nvidia-smi'])
print(torch.cuda.get_device_name(0))
"
```

### Run a custom training script
```bash
python kaggle_executor.py run my_model/train.py
```

### Hunyuan3D-2: Image → 3D mesh (on free Kaggle T4!)

```bash
python examples/hunyuan3d/run_hunyuan3d.py \
    --front  photo_front.png \
    --side   photo_side.png  \
    --out    ./output_3d     \
    --steps  30
```

This runs the full **Tencent Hunyuan3D-2** pipeline on Kaggle's free GPU:
1. Clones [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) on the GPU node
2. Installs PyTorch + hy3dgen
3. Downloads model weights (~4 GB) from HuggingFace
4. Runs diffusion shape generation (30 steps, ~3 min on T4)
5. Streams all output to your terminal in real-time
6. Downloads `.glb` + `.obj` mesh directly to your local machine

**Tested output: 30 MB `.glb` + 69 MB `.obj` in ~8 minutes total on Tesla T4.**

<br/>

---

## ⚙️ How it works

```
Your machine                          Kaggle GPU node
─────────────                         ──────────────────────────────────
kaggle_executor.py
  │
  ├── HTTP  /api/contents/*  ──────►  Upload files (Jupyter Contents API)
  │
  ├── WSS   /api/kernels/*/channels   Jupyter WebSocket protocol
  │          execute_request  ──────►  Runs your code on the GPU kernel
  │          ◄─── stream msgs          Streams stdout/stderr back live
  │
  └── HTTP  /api/contents/*  ◄──────  Download results
```

No tunnel, no VPN, no SSH — just HTTPS/WSS through the Kaggle proxy that's
already public while your notebook session is active.

<br/>

---

## 🔌 How the no-internet relay works

`kaggle_relay.py` moves a file **between two notebooks** — one with
internet, one without — using two *different* proxy URLs:

```
KAGGLE_INTERNET          → notebook that downloads the file (has internet)
KAGGLE_WITHOUT_INTERNET  → notebook that receives the file  (internet OFF)
```

```
Source URL                Notebook A (internet: ON)          Notebook B (internet: OFF)
───────────                ──────────────────────────         ──────────────────────────
HTTPS Range GET  ───────►  kaggle_relay driver script
 (parallel workers)         (submitted once by your machine)
                                    │
                                    │  HTTP PUT /api/contents/*   (Contents API)
                                    ▼
                            ─────────────────────────────►    part files land on B
                                                                        │
                                                            one kernel exec on B
                                                            concatenates the parts
                                                                        ▼
                                                                /kaggle/working/<dest>
```

**Why B's "Internet: off" doesn't block the incoming write:** that toggle
only gates network calls the kernel's *own* process initiates inside a cell
— e.g. `requests.get(...)`. It has no say over Kaggle's platform accepting
an authenticated file write pushed at it from outside. B's kernel never has
to reach out for this to work, so its network restriction never comes into
play — a plain DNS lookup from *inside* B's kernel still fails, verifiably.

**Why it's fast:** the parts are written via B's Jupyter *Contents API*
(`PUT /api/contents/<path>`, plain HTTP) rather than through B's kernel
`execute_request` channel. Writing through the kernel channel means
compiling a multi-MB code literal and waiting on kernel busy/idle status for
every chunk — in testing that capped out around **0.75 MB/s even with 6
parallel kernels**. Switching the same 4 workers to Contents-API PUTs
relayed a 3.44 GB file in **115 seconds** (~30 MB/s) — no kernel exec in the
hot path at all, just HTTP.

<br/>

---

## 📋 Requirements

- Python 3.10+
- `pip install requests websocket-client`
- A **running** Kaggle notebook (free tier — T4 GPU available)
- macOS or Linux *(the interactive shell uses `termios`/`tty`; file ops work on Windows)*

<br/>

---

## 📁 Project Structure

```
kaggle-runtime-agent-skills/
├── kaggle_remote.py        # File manager + interactive PTY shell
├── kaggle_executor.py      # Jupyter kernel executor (live streaming)
├── kaggle_relay.py         # Direct notebook-to-notebook relay (internet → no-internet)
├── requirements.txt
├── examples/
│   └── hunyuan3d/
│       └── run_hunyuan3d.py   # Hunyuan3D-2 image → 3D mesh pipeline
└── README.md
```

<br/>

---

## ❓ FAQ

**Q: Do I need a Kaggle API key?**
No. Only the proxy URL from your browser while the notebook is running.

**Q: What if my proxy URL expires?**
Kaggle proxy URLs are tied to the session. When the notebook resets, copy a new URL from the browser.

**Q: Does it work with any GPU model, not just Hunyuan3D?**
Yes — `kaggle_executor.py run my_script.py` runs any Python code. The Hunyuan3D pipeline is just one example.

**Q: Can I run multiple scripts in parallel?**
Each Jupyter kernel runs one cell at a time. You can create multiple kernels
(`get_or_create_kernel()` creates a new one if the existing one is busy).

**Q: How do I get a notebook's proxy URL?**
Open the notebook (session must be **Running**), then the ⋮ **More settings**
menu → **Kaggle Jupyter Server** panel has a ready-to-copy proxy URL
("VSCode Compatible URL" / "Colab Compatible URL"). No DevTools needed.

**Q: Can I relay a file into a notebook with internet turned off?**
Yes — that's what `kaggle_relay.py` is for. See
[How the no-internet relay works](#-how-the-no-internet-relay-works).

**Q: Windows support?**
File operations and `kaggle_executor.py` work on Windows. The interactive PTY shell
(`kaggle_remote.py shell`) requires POSIX (`termios`/`tty`) — macOS/Linux only.

<br/>

---

## 🤝 Contributing

Found a bug? Have a new example pipeline (Stable Diffusion, Whisper, LoRA fine-tuning...)?
PRs and issues are very welcome!

<br/>

---

## License

MIT — see [LICENSE](LICENSE).

<br/>

---

<div align="center">

### If this saved you a GPU rental bill or hours of frustration...

**Please give it a ⭐ star.** It takes 2 seconds, costs nothing,<br/>
and it's the only way other developers can find this tool.

## [⭐ Star this repo on GitHub](https://github.com/huytrao/kaggle-runtime-agent-skills/stargazers)

*Built for the community. Shared freely. Powered by your ⭐.*

</div>
