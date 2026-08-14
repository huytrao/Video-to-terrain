#!/usr/bin/env python3
"""
Local verification script for Video-to-Terrain geometry pipeline.
Run this to verify local environment before deploying to Kaggle.
"""
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        print("  ✓ Python version OK (3.10+)")
        return True
    else:
        print("  ✗ Python 3.10+ required")
        return False

def check_pytorch_cuda():
    """Check PyTorch and CUDA availability."""
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("  ⚠ CUDA not available (expected on Mac/CPU)")
            return True  # Not a failure for local dev
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False

def check_da3_import():
    """Check DA3 import works."""
    # Add local DA3 path
    da3_paths = [
        Path("/Users/traoanhuy/source_code/Full-pipeline-scan-terrain/depth-anything-3/src"),
        Path("depth-anything-3/src"),
        Path("../depth-anything-3/src"),
    ]

    for p in da3_paths:
        if p.exists():
            sys.path.insert(0, str(p))
            break

    try:
        from depth_anything_3.api import DepthAnything3
        print("  ✓ DA3 import OK")
        return True
    except ImportError as e:
        print(f"  ✗ DA3 import failed: {e}")
        return False

def check_geometry_deps():
    """Check geometry pipeline dependencies."""
    deps = [
        ("open3d", "open3d"),
        ("trimesh", "trimesh"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("PIL", "pillow"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("yaml", "pyyaml"),
        ("dotenv", "python-dotenv"),
        ("networkx", "networkx"),
    ]

    all_ok = True
    for import_name, pkg_name in deps:
        try:
            importlib.import_module(import_name)
            print(f"  ✓ {pkg_name}")
        except ImportError:
            print(f"  ✗ {pkg_name} not installed")
            all_ok = False
    return all_ok

def check_pyg_deps():
    """Check PyTorch Geometric dependencies (optional but needed for DA3)."""
    pyg_deps = [
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        "torch_geometric",
    ]

    for dep in pyg_deps:
        try:
            importlib.import_module(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ⚠ {dep} not installed (optional, DA3 may download at runtime)")

def check_project_structure():
    """Check project directory structure."""
    root = Path("/Users/traoanhuy/source_code/Full-pipeline-scan-terrain/Videos_to_terrain")

    checks = [
        ("Input video", root / "input" / "20260810_150244_color.mp4"),
        ("DA3 repo", root / "depth-anything-3"),
        ("Source code", root / "src" / "v2t"),
        ("Config", root / "src" / "v2t" / "config.py"),
        ("Stage B", root / "src" / "v2t" / "geometry" / "stage_b_da3.py"),
        ("Deploy script", root / "src" / "v2t" / "deploy_kaggle.py"),
        ("Kaggle remote", root / "src" / "v2t" / "kaggle_remote.py"),
    ]

    all_ok = True
    for name, path in checks:
        if path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} NOT FOUND")
            all_ok = False
    return all_ok

def check_kaggle_config():
    """Check Kaggle proxy URL is set."""
    import os
    proxy_url = os.environ.get("KAGGLE_PROXY_URL")
    if proxy_url:
        print(f"  ✓ KAGGLE_PROXY_URL set ({proxy_url[:60]}...)")
        return True
    else:
        print("  ⚠ KAGGLE_PROXY_URL not set (needed for remote deployment)")
        return True  # Warning only

def main():
    print("=" * 60)
    print("Video-to-Terrain Geometry Pipeline - Environment Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("PyTorch + CUDA", check_pytorch_cuda),
        ("DA3 Import", check_da3_import),
        ("Geometry Dependencies", check_geometry_deps),
        ("PyG Dependencies (optional)", check_pyg_deps),
        ("Project Structure", check_project_structure),
        ("Kaggle Config", check_kaggle_config),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ Check failed with exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ All critical checks passed! Ready for Kaggle deployment.")
        return 0
    else:
        print("✗ Some checks failed. Fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
