#!/usr/bin/env python3
"""
scripts/tools/setup_remote_hunyuan_gpu.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Automated 1-Click Fast Boot & Environment Provisioning for Hunyuan3D-2 / Hunyuan3D-2mv
on remote Kaggle RTX 6000 PRO Blackwell GPU (102 GB VRAM).

Run this script whenever a new session starts or when connecting to a fresh Kaggle kernel:
    python scripts/tools/setup_remote_hunyuan_gpu.py
"""
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kaggle-runtime-toolkit-agent-skills"))
from kaggle_executor import KaggleExecutor

def get_env_var(var_name: str, fallback: str = "") -> str:
    val = os.environ.get(var_name, "")
    if val:
        return val.strip()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{var_name}="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return fallback

def main():
    gpu_proxy = get_env_var("KAGGLE_PROXY_RTX6000_NO_INTERNET") or get_env_var("KAGGLE_PROXY_URL")
    online_proxy = get_env_var("KAGGLE_PROXY_HAS_INTERNET_NO_GPU")

    if not gpu_proxy:
        raise ValueError("Missing KAGGLE_PROXY_RTX6000_NO_INTERNET or KAGGLE_PROXY_URL in .env")

    print("=================================================================")
    print("🚀 PROVISIONING REMOTE GPU ENVIRONMENT FOR HUNYUAN3D-2 (1-CLICK)")
    print(f"  • GPU Kernel (Target): {gpu_proxy[:60]}...")
    print(f"  • Online Kernel (Bridge): {online_proxy[:60]}...")
    print("=================================================================\n")

    gpu_exec = KaggleExecutor(gpu_proxy)

    # 1. Check GPU kernel status
    print("1. Probing GPU kernel environment...")
    probe_code = """
import os, sys, torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)')

has_hunyuan = os.path.exists('/kaggle/working/Hunyuan3D-2')
has_dit = os.path.exists('/kaggle/working/models/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/model.fp16.safetensors')
has_vae = os.path.exists('/kaggle/working/models/Hunyuan3D-2/hunyuan3d-vae-v2-0/model.fp16.safetensors')

try:
    import pymeshlab, trimesh, xatlas, accelerate
    has_pkgs = True
except ImportError:
    has_pkgs = False

print(f'STATUS: Hunyuan3D-2={has_hunyuan}, DiT_weights={has_dit}, VAE_weights={has_vae}, Geometry_pkgs={has_pkgs}')
"""
    kid = gpu_exec.get_or_create_kernel()
    out = gpu_exec.execute(kid, probe_code)
    print(out)

    if "DiT_weights=True" in out and "VAE_weights=True" and "Geometry_pkgs=True" and "Hunyuan3D-2=True":
        print("\n✅ GPU Kernel is ALREADY FULLY PROVISIONED and ready for inference!")
        return

    # 2. If online proxy is available, run auto-relay
    if online_proxy:
        print("\n2. Provisioning missing dependencies via Online Kernel Bridge...")
        relay_cmd = [
            sys.executable,
            str(ROOT / "kaggle-runtime-toolkit-agent-skills" / "kaggle_relay.py"),
        ]
        # Relay VAE if needed
        if "VAE_weights=False" in out:
            print("  -> Relaying Hunyuan3D-2 VAE safetensors (428 MB)...")
            subprocess.run([
                *relay_cmd,
                "https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-vae-v2-0/model.fp16.safetensors",
                "--dest", "/kaggle/working/models/Hunyuan3D-2/hunyuan3d-vae-v2-0/model.fp16.safetensors",
                "--workers", "4"
            ], env={**os.environ, "KAGGLE_INTERNET": online_proxy, "KAGGLE_WITHOUT_INTERNET": gpu_proxy}, check=True)

        # Relay DiT if needed
        if "DiT_weights=False" in out:
            print("  -> Relaying Hunyuan3D-2mv DiT safetensors (4.9 GB)...")
            subprocess.run([
                *relay_cmd,
                "https://huggingface.co/tencent/Hunyuan3D-2mv/resolve/main/hunyuan3d-dit-v2-mv/model.fp16.safetensors",
                "--dest", "/kaggle/working/models/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/model.fp16.safetensors",
                "--workers", "8"
            ], env={**os.environ, "KAGGLE_INTERNET": online_proxy, "KAGGLE_WITHOUT_INTERNET": gpu_proxy}, check=True)

    print("\n✅ Setup complete! You can now run scripts/adapter/04_run_hunyuan_shape.py directly.")

if __name__ == "__main__":
    main()
