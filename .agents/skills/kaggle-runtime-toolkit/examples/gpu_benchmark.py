#!/usr/bin/env python3
"""
GPU & CUDA Benchmark Example for Kaggle Runtime Toolkit
Executes locally or remotely on Kaggle GPU kernel.
"""
import time
import torch

def benchmark():
    print("=" * 60)
    print("🚀 Kaggle GPU Kernel Benchmark")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this environment.")
        return

    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / (1024 ** 3)
    free_mem, _ = torch.cuda.mem_get_info()
    free_mem = free_mem / (1024 ** 3)

    print(f"Device: {device_name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total VRAM: {total_mem:.2f} GB")
    print(f"Free VRAM: {free_mem:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("-" * 60)

    # Matrix multiplication benchmark
    size = 8192
    print(f"Running FP16 MatMul Benchmark ({size}x{size})...")
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(5):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Timed run
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters

    tflops = (2 * (size ** 3)) / (elapsed * 1e12)
    print(f"Avg iteration time: {elapsed * 1000:.2f} ms")
    print(f"Compute Throughput: {tflops:.2f} TFLOPS (FP16 Tensor Core)")
    print("=" * 60)
    print("✓ Benchmark completed successfully.")

if __name__ == "__main__":
    benchmark()
