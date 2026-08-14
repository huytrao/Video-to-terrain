#!/usr/bin/env python3
"""Production Setup for Video-to-Terrain Pipeline.

This script prepares the pipeline for production use by:
1. Validating NVIDIA API keys are real (not placeholders)
2. Downloading and setting up real SAM3 model from HuggingFace
3. Configuring robot policy controllers (Stage H)
4. Verifying all dependencies and environment

Usage:
    python -m src.v2t.setup_production
    python -m src.v2t.setup_production --verify-only
    python -m src.v2t.setup_production --force-redownload
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.v2t.config import CFG, PROJECT_ROOT as CFG_PROJECT_ROOT


class ProductionSetup:
    """Handles production environment setup and validation."""

    def __init__(self, force_redownload: bool = False, verify_only: bool = False):
        self.force_redownload = force_redownload
        self.verify_only = verify_only
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.successes: list[str] = []

    def log_success(self, msg: str) -> None:
        self.successes.append(msg)
        print(f"✅ {msg}")

    def log_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"⚠️  {msg}")

    def log_error(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"❌ {msg}")

    def validate_nvidia_keys(self) -> bool:
        """Validate that real NVIDIA API keys are configured."""
        print("\n" + "=" * 60)
        print("Validating NVIDIA API Keys")
        print("=" * 60)

        keys = CFG.nvidia_api_keys
        placeholder_patterns = [
            "REPLACE_ME",
            "REPLACE_WITH_REAL_KEY",
            "nvapi-REPLACE",
        ]

        valid_keys = []
        for i, key in enumerate(keys, 1):
            if not key:
                self.log_error(f"NVIDIA_API_KEY_{i}: EMPTY - not set in .env")
                continue

            is_placeholder = any(p in key for p in placeholder_patterns)
            if is_placeholder:
                self.log_error(f"NVIDIA_API_KEY_{i}: PLACEHOLDER - {key[:20]}... (must be real nvapi- key)")
                continue

            if not key.startswith("nvapi-"):
                self.log_warning(f"NVIDIA_API_KEY_{i}: Unexpected format (expected nvapi- prefix): {key[:20]}...")

            # Basic validation - check it's a reasonable length
            if len(key) < 30:
                self.log_warning(f"NVIDIA_API_KEY_{i}: Suspiciously short ({len(key)} chars)")
            else:
                self.log_success(f"NVIDIA_API_KEY_{i}: Valid format ({key[:20]}...)")
                valid_keys.append(key)

        if not valid_keys:
            self.log_error("NO VALID NVIDIA API KEYS FOUND. Pipeline will fail at VLM stage.")
            self.log_error("Get keys from https://build.nvidia.com and update .env")
            return False

        self.log_success(f"Found {len(valid_keys)} valid NVIDIA API key(s)")
        return True

    def validate_env_file(self) -> bool:
        """Validate .env file exists and has required variables."""
        print("\n" + "=" * 60)
        print("Validating .env Configuration")
        print("=" * 60)

        env_path = CFG_PROJECT_ROOT / ".env"
        if not env_path.exists():
            self.log_error(f".env file not found at {env_path}")
            self.log_error("Run: cp .env.example .env and fill in values")
            return False

        self.log_success(f".env file found at {env_path}")

        # Check critical vars
        required_vars = [
            "NVIDIA_API_KEY_1",
            "NVIDIA_API_KEY_2",
            "NVIDIA_API_KEY_3",
            "NVIDIA_BASE_URL",
            "NVIDIA_VLM_MODEL",
        ]

        from dotenv import dotenv_values
        env_vars = dotenv_values(env_path)

        missing = [var for var in required_vars if not env_vars.get(var)]
        if missing:
            self.log_error(f"Missing required env vars: {missing}")
            return False

        self.log_success("All required env vars present")
        return True

    def setup_sam3_model(self) -> bool:
        """Download and setup SAM3 model from HuggingFace."""
        print("\n" + "=" * 60)
        print("Setting up SAM3 Model (HuggingFace)")
        print("=" * 60)

        # Check if already cached
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        sam3_cache = list(cache_dir.glob("models--facebook--sam3*"))

        if sam3_cache and not self.force_redownload:
            self.log_success(f"SAM3 model already cached at {sam3_cache[0]}")
            return True

        if self.verify_only:
            self.log_warning("SAM3 model not cached (verify-only mode)")
            return True

        try:
            # Try importing and downloading via HF
            print("Downloading SAM3 model from HuggingFace (facebook/sam3)...")
            print("This may take several minutes on first run...")

            from huggingface_hub import snapshot_download

            # Download SAM3 model
            model_path = snapshot_download(
                repo_id="facebook/sam3",
                local_dir=None,  # Use cache
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            self.log_success(f"SAM3 model downloaded to {model_path}")

            # Verify we can import sam3
            try:
                # The holi-spatial repo has sam3 package
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
                self.log_success("SAM3 Python package importable")
            except ImportError as e:
                self.log_warning(f"SAM3 Python package not directly importable: {e}")
                self.log_warning("Ensure holi-spatial repo is available at /tmp/holi-spatial-inspect")

            return True

        except ImportError:
            self.log_error("huggingface_hub not installed. Run: pip install huggingface_hub")
            return False
        except Exception as e:
            self.log_error(f"Failed to download SAM3 model: {e}")
            self.log_warning("Will fall back to stub masks in Stage D2")
            return False

    def setup_robot_policy(self) -> bool:
        """Configure robot policy controllers for Stage H."""
        print("\n" + "=" * 60)
        print("Configuring Robot Policy Controllers (Stage H)")
        print("=" * 60)

        # Check MJLab availability
        try:
            import mjlab
            self.log_success("mjlab package available")
            mjlab_available = True
        except ImportError:
            self.log_warning("mjlab not installed - policy config will be template only")
            self.log_warning("Install with: pip install -e /path/to/vinrobotics_mjlab")
            mjlab_available = False

        # Check vinrobotics_mjlab
        try:
            import vinrobotics_mjlab
            self.log_success("vinrobotics_mjlab available")
            vinrobotics_available = True
        except ImportError:
            self.log_warning("vinrobotics_mjlab not found - using config import paths only")
            vinrobotics_available = False

        # Create policy output directories
        policy_dir = CFG.paths.simulation / "policy"
        robot_policy_dir = CFG.paths.run_dir / "robot" / "policy"

        for d in [policy_dir, robot_policy_dir]:
            d.mkdir(parents=True, exist_ok=True)
            self.log_success(f"Created policy directory: {d}")

        # Generate default policy configs for both types
        if not self.verify_only:
            try:
                from src.v2t.robot.stage_h_policy import run_stage_h
                run_stage_h("tracking")
                run_stage_h("velocity")
                self.log_success("Generated tracking and velocity policy configs")
            except Exception as e:
                self.log_error(f"Failed to generate policy configs: {e}")
                return False

        return True

    def validate_dependencies(self) -> bool:
        """Validate all Python dependencies are installed."""
        print("\n" + "=" * 60)
        print("Validating Python Dependencies")
        print("=" * 60)

        # (import_name, display_name)
        required_packages = [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("numpy", "NumPy"),
            ("cv2", "OpenCV"),
            ("PIL", "Pillow"),
            ("trimesh", "Trimesh"),
            ("open3d", "Open3D"),
            ("openai", "OpenAI client"),
            ("dotenv", "python-dotenv"),
            ("huggingface_hub", "HuggingFace Hub"),
            ("tqdm", "tqdm"),
            ("yaml", "PyYAML"),
        ]

        missing = []
        for pkg, name in required_packages:
            try:
                __import__(pkg)
                self.log_success(f"{name}: installed")
            except ImportError:
                missing.append(name)
                self.log_error(f"{name}: MISSING")

        if missing:
            self.log_error(f"Missing packages: {', '.join(missing)}")
            self.log_error("Run: pip install -r requirements.txt")
            return False

        return True

    def validate_gpu(self) -> bool:
        """Validate GPU availability for DA3 inference."""
        print("\n" + "=" * 60)
        print("Validating GPU / Compute")
        print("=" * 60)

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.log_success(f"CUDA GPU: {gpu_name} ({gpu_mem:.1f} GB)")
                return True
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.log_success("MPS (Apple Silicon) available")
                return True
            else:
                self.log_warning("No GPU detected - DA3 will run on CPU (slow)")
                return True  # Not fatal, just slow
        except ImportError:
            self.log_error("PyTorch not installed")
            return False

    def validate_input_video(self) -> bool:
        """Validate input video exists."""
        print("\n" + "=" * 60)
        print("Validating Input Video")
        print("=" * 60)

        if CFG.input_video.exists():
            size_mb = CFG.input_video.stat().st_size / 1e6
            self.log_success(f"Input video: {CFG.input_video} ({size_mb:.1f} MB)")
            return True
        else:
            self.log_error(f"Input video not found: {CFG.input_video}")
            self.log_error("Place video at input/20260810_150244_color.mp4 or update CFG.input_video")
            return False

    def verify_pipeline_config(self) -> bool:
        """Run pipeline config validation."""
        print("\n" + "=" * 60)
        print("Verifying Pipeline Configuration")
        print("=" * 60)

        try:
            CFG.validate()
            self.log_success("Pipeline config validation passed")
            return True
        except Exception as e:
            self.log_error(f"Config validation failed: {e}")
            return False

    def run(self) -> int:
        """Run all production setup steps."""
        print("=" * 60)
        print("VIDEO-TO-TERRAIN PRODUCTION SETUP")
        print("=" * 60)

        if self.verify_only:
            print("Mode: VERIFY ONLY (no downloads or modifications)")

        steps = [
            ("Environment file", self.validate_env_file),
            ("NVIDIA API Keys", self.validate_nvidia_keys),
            ("Input video", self.validate_input_video),
            ("Python dependencies", self.validate_dependencies),
            ("GPU / Compute", self.validate_gpu),
            ("Pipeline config", self.verify_pipeline_config),
        ]

        if not self.verify_only:
            steps.extend([
                ("SAM3 model", self.setup_sam3_model),
                ("Robot policy", self.setup_robot_policy),
            ])

        all_passed = True
        for name, step_fn in steps:
            try:
                if not step_fn():
                    all_passed = False
            except Exception as e:
                self.log_error(f"{name} check failed with exception: {e}")
                all_passed = False

        # Summary
        print("\n" + "=" * 60)
        print("SETUP SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {len(self.successes)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")

        if self.errors:
            print("\nERRORS:")
            for e in self.errors:
                print(f"  - {e}")

        if self.warnings:
            print("\nWARNINGS:")
            for w in self.warnings:
                print(f"  - {w}")

        print("\n" + "=" * 60)
        if all_passed and not self.errors:
            print("🎉 PRODUCTION SETUP COMPLETE - Pipeline ready for production!")
            return 0
        else:
            print("❌ SETUP INCOMPLETE - Fix errors above before running pipeline")
            return 1


def main():
    parser = argparse.ArgumentParser(description="Video-to-Terrain Production Setup")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify configuration, don't download models")
    parser.add_argument("--force-redownload", action="store_true",
                        help="Force re-download of SAM3 model from HuggingFace")
    parser.add_argument("--stage", choices=["all", "keys", "sam3", "policy", "deps", "gpu"],
                        default="all", help="Run specific setup stage only")

    args = parser.parse_args()

    setup = ProductionSetup(
        force_redownload=args.force_redownload,
        verify_only=args.verify_only,
    )

    if args.stage != "all":
        # Run single stage
        stage_map = {
            "keys": setup.validate_nvidia_keys,
            "sam3": setup.setup_sam3_model,
            "policy": setup.setup_robot_policy,
            "deps": setup.validate_dependencies,
            "gpu": setup.validate_gpu,
        }
        if args.stage in stage_map:
            result = stage_map[args.stage]()
            return 0 if result else 1
        else:
            print(f"Unknown stage: {args.stage}")
            return 1

    return setup.run()


if __name__ == "__main__":
    sys.exit(main())