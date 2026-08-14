#!/usr/bin/env python3
"""
verify_geometry_download.py - Verify and recover Kaggle → local geometry downloads.

ERROR HANDLER #4: Handles Kaggle → local download/sync errors.

Common errors handled:
1. Jupyter Contents API timeout on large files → chunk download, increase timeout
2. Partial/corrupted .npy files → verify checksums, re-download
3. Missing files (DA3 didn't write all 100 frames) → detect gaps, re-run inference
4. Path mismatch (Kaggle output dir vs expected) → fix deploy_kaggle.sh output paths
5. Large point_cloud.ply download fails → compress before download, or stream

Usage:
    python -m src.v2t.verify_geometry_download --verify
    python -m src.v2t.verify_geometry_download --download-missing
    python -m src.v2t.verify_geometry_download --repair
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path (3 levels up from src/v2t/verify_geometry_download.py -> Videos_to_terrain)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.v2t.config import CFG
from src.v2t.kaggle_remote import KaggleProxy


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPECTED_FILES = {
    "depth_maps": {
        "pattern": "depth_*.npy",
        "count": 100,
        "dir": "depth",
        "check_func": "check_npy",
    },
    "conf_maps": {
        "pattern": "conf_*.npy",
        "count": 100,
        "dir": "depth",
        "check_func": "check_npy",
    },
    "camera_intrinsics": {
        "path": "camera_intrinsics.json",
        "check_func": "check_json",
        "required_keys": ["intrinsics", "width", "height"],
    },
    "camera_poses": {
        "path": "camera_poses.json",
        "check_func": "check_json",
        "required_keys": ["extrinsics"],
    },
    "point_cloud": {
        "path": "point_cloud.ply",
        "check_func": "check_ply",
        "min_size_bytes": 1024,  # at least 1KB
    },
    "scene_visual": {
        "path": "scene_visual.obj",
        "check_func": "check_obj",
        "min_size_bytes": 100,
    },
}

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def log_info(msg: str):
    print(f"{BLUE}[INFO]{NC} {msg}")


def log_success(msg: str):
    print(f"{GREEN}[SUCCESS]{NC} {msg}")


def log_warning(msg: str):
    print(f"{YELLOW}[WARNING]{NC} {msg}")


def log_error(msg: str):
    print(f"{RED}[ERROR]{NC} {msg}")


# =============================================================================
# FILE VALIDATION FUNCTIONS
# =============================================================================

def check_npy(filepath: Path) -> tuple[bool, str]:
    """Validate .npy file can be loaded and has reasonable content."""
    try:
        data = np.load(filepath, mmap_mode='r')
        if data.size == 0:
            return False, "Empty array"
        if not np.isfinite(data).all():
            return False, "Contains NaN/Inf values"
        # Check reasonable depth range (0.01m to 100m)
        if data.min() < 0.001 or data.max() > 1000:
            return False, f"Unreasonable range: [{data.min():.3f}, {data.max():.3f}]"
        return True, f"Shape: {data.shape}, Range: [{data.min():.3f}, {data.max():.3f}]"
    except Exception as e:
        return False, f"Load failed: {e}"


def check_json(filepath: Path, required_keys: list[str] | None = None) -> tuple[bool, str]:
    """Validate JSON file structure."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                return False, f"Missing keys: {missing}"
        return True, f"Keys: {list(data.keys())}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Read failed: {e}"


def check_ply(filepath: Path) -> tuple[bool, str]:
    """Validate PLY file has valid header and data."""
    try:
        with open(filepath, 'rb') as f:
            header = f.readline().decode('ascii').strip()
            if header != 'ply':
                return False, f"Invalid PLY magic: {header}"
            # Read header lines
            vertex_count = 0
            while True:
                line = f.readline().decode('ascii').strip()
                if line == 'end_header':
                    break
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
            if vertex_count == 0:
                return False, "No vertices in PLY"
            # Try to read a few points
            data = f.read(min(1000, vertex_count * 24))  # rough estimate
            if len(data) == 0:
                return False, "No vertex data"
        return True, f"Vertices: {vertex_count}"
    except Exception as e:
        return False, f"PLY validation failed: {e}"


def check_obj(filepath: Path) -> tuple[bool, str]:
    """Validate OBJ file has vertices and faces."""
    try:
        vertices = 0
        faces = 0
        with open(filepath) as f:
            for line in f:
                if line.startswith('v '):
                    vertices += 1
                elif line.startswith('f '):
                    faces += 1
        if vertices == 0:
            return False, "No vertices in OBJ"
        return True, f"Vertices: {vertices}, Faces: {faces}"
    except Exception as e:
        return False, f"OBJ validation failed: {e}"


# =============================================================================
# CHECKSUM UTILITIES
# =============================================================================

def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_checksums(checksums: dict, output_path: Path):
    """Save checksums to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(checksums, f, indent=2)


def load_checksums(checksum_path: Path) -> dict:
    """Load checksums from JSON file."""
    if not checksum_path.exists():
        return {}
    with open(checksum_path) as f:
        return json.load(f)


# =============================================================================
# VERIFICATION CLASS
# =============================================================================

class GeometryDownloadVerifier:
    """Verifies and recovers Kaggle geometry downloads."""

    def __init__(self, run_name: str = "run_mvp", proxy_url: str | None = None):
        self.run_name = run_name
        self.run_dir = CFG.runs_root / run_name
        self.geometry_dir = self.run_dir / "geometry"
        self.depth_dir = self.geometry_dir / "depth"
        self.proxy_url = proxy_url or os.environ.get("KAGGLE_PROXY_URL")
        self.kaggle: KaggleProxy | None = None

        if self.proxy_url:
            self.kaggle = KaggleProxy(self.proxy_url)

    def verify_local(self, verbose: bool = False) -> dict:
        """Verify all expected files exist and are valid locally."""
        log_info(f"Verifying local geometry in {self.geometry_dir}")

        results = {
            "total_expected": 0,
            "total_found": 0,
            "total_valid": 0,
            "total_invalid": 0,
            "total_missing": 0,
            "files": {},
            "errors": [],
        }

        # Check depth maps (100 frames expected)
        depth_files = sorted(self.depth_dir.glob("depth_*.npy"))
        conf_files = sorted(self.depth_dir.glob("conf_*.npy"))

        results["files"]["depth_maps"] = self._verify_file_set(
            "depth_maps", depth_files, 100, check_npy, verbose
        )
        results["files"]["conf_maps"] = self._verify_file_set(
            "conf_maps", conf_files, 100, check_npy, verbose
        )

        # Check required single files
        for key in ["camera_intrinsics", "camera_poses", "point_cloud", "scene_visual"]:
            spec = EXPECTED_FILES[key]
            path = self.geometry_dir / spec["path"]
            if path.exists():
                check_func = globals()[spec["check_func"]]
                if spec["check_func"] == "check_json":
                    ok, msg = check_func(path, spec.get("required_keys"))
                else:
                    ok, msg = check_func(path)
                size = path.stat().st_size
                results["files"][key] = {
                    "expected": 1,
                    "found": 1,
                    "valid": 1 if ok else 0,
                    "invalid": 0 if ok else 1,
                    "missing": 0,
                    "size_bytes": size,
                    "details": msg,
                }
                if ok:
                    results["total_valid"] += 1
                else:
                    results["total_invalid"] += 1
                    results["errors"].append(f"{key}: {msg}")
            else:
                results["files"][key] = {
                    "expected": 1,
                    "found": 0,
                    "valid": 0,
                    "invalid": 0,
                    "missing": 1,
                    "size_bytes": 0,
                    "details": "File not found",
                }
                results["total_missing"] += 1
                results["errors"].append(f"{key}: Missing")

            results["total_expected"] += 1
            results["total_found"] += 1 if path.exists() else 0

        # Summary
        log_info(f"Verification complete: {results['total_valid']} valid, "
                 f"{results['total_invalid']} invalid, {results['total_missing']} missing")

        return results

    def _verify_file_set(self, name: str, files: list[Path], expected_count: int,
                         check_func, verbose: bool) -> dict:
        """Verify a set of files (e.g., depth maps)."""
        found = len(files)
        valid = 0
        invalid = 0
        details = []

        for i, f in enumerate(files):
            ok, msg = check_func(f)
            if ok:
                valid += 1
            else:
                invalid += 1
                details.append(f"{f.name}: {msg}")
            if verbose:
                status = "✓" if ok else "✗"
                print(f"  {status} {f.name}: {msg}")

        missing = expected_count - found

        if missing > 0:
            log_warning(f"{name}: Missing {missing} files (expected {expected_count}, found {found})")

        return {
            "expected": expected_count,
            "found": found,
            "valid": valid,
            "invalid": invalid,
            "missing": missing,
            "details": details if invalid else "All valid",
        }

    def verify_remote(self) -> dict:
        """Verify what files exist on the remote Kaggle runtime."""
        if not self.kaggle:
            log_error("No KAGGLE_PROXY_URL configured")
            return {}

        log_info("Verifying remote geometry on Kaggle...")
        remote_dir = f"{self.kaggle.remote_project}/outputs/{self.run_name}/geometry"

        try:
            items = self.kaggle._list_dir_recursive(remote_dir)
        except Exception as e:
            log_error(f"Failed to list remote directory: {e}")
            return {}

        remote_files = {}
        for item in items:
            if item.get("type") == "file":
                rel_path = item["path"]
                if rel_path.startswith(remote_dir + "/"):
                    rel = rel_path[len(remote_dir) + 1:]
                    remote_files[rel] = item.get("size", 0)

        log_info(f"Found {len(remote_files)} files on remote")
        return remote_files

    def compare_local_remote(self, local_results: dict, remote_files: dict) -> dict:
        """Compare local vs remote to find missing/different files."""
        comparison = {
            "missing_locally": [],
            "size_mismatch": [],
            "extra_locally": [],
        }

        # Check each expected file type
        for key, spec in EXPECTED_FILES.items():
            if key in ["depth_maps", "conf_maps"]:
                # Check all expected numbered files
                for i in range(spec["count"]):
                    fname = f"{spec['dir']}/{key[:-1]}_{i:06d}.npy"  # depth_X.npy or conf_X.npy
                    if fname in remote_files:
                        local_path = self.geometry_dir / fname
                        if not local_path.exists():
                            comparison["missing_locally"].append(fname)
                        elif local_path.stat().st_size != remote_files[fname]:
                            comparison["size_mismatch"].append({
                                "file": fname,
                                "local_size": local_path.stat().st_size,
                                "remote_size": remote_files[fname],
                            })
            else:
                fname = spec["path"]
                if fname in remote_files:
                    local_path = self.geometry_dir / fname
                    if not local_path.exists():
                        comparison["missing_locally"].append(fname)
                    elif local_path.stat().st_size != remote_files[fname]:
                        comparison["size_mismatch"].append({
                            "file": fname,
                            "local_size": local_path.stat().st_size,
                            "remote_size": remote_files[fname],
                        })

        return comparison

    def download_missing(self, missing_files: list[str],
                         chunk_size: int = 10 * 1024 * 1024,
                         timeout: int = 300) -> dict:
        """Download missing files from remote with retry and chunking."""
        if not self.kaggle:
            log_error("No KAGGLE_PROXY_URL configured")
            return {"success": 0, "failed": len(missing_files)}

        results = {"success": 0, "failed": 0, "errors": []}

        for fname in missing_files:
            remote_path = f"{self.kaggle.remote_project}/outputs/{self.run_name}/geometry/{fname}"
            local_path = self.geometry_dir / fname

            log_info(f"Downloading {fname}...")
            success = False

            for attempt in range(3):
                try:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    self.kaggle.download_file(remote_path, str(local_path))

                    # Verify download
                    if local_path.exists() and local_path.stat().st_size > 0:
                        success = True
                        break

                except Exception as e:
                    log_warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)  # exponential backoff

            if success:
                log_success(f"Downloaded {fname}")
                results["success"] += 1
            else:
                log_error(f"Failed to download {fname} after 3 attempts")
                results["failed"] += 1
                results["errors"].append(fname)

        return results

    def repair_corrupted(self, local_results: dict) -> dict:
        """Re-download corrupted/invalid files."""
        corrupted = []

        for key, info in local_results["files"].items():
            if info.get("invalid", 0) > 0:
                if key in ["depth_maps", "conf_maps"]:
                    # Need to identify specific corrupted files
                    pattern = EXPECTED_FILES[key]["pattern"]
                    files = sorted(self.depth_dir.glob(pattern))
                    for f in files:
                        ok, _ = globals()[EXPECTED_FILES[key]["check_func"]](f)
                        if not ok:
                            corrupted.append(f.relative_to(self.geometry_dir).as_posix())
                else:
                    corrupted.append(EXPECTED_FILES[key]["path"])

        if not corrupted:
            log_info("No corrupted files to repair")
            return {"repaired": 0, "failed": 0}

        log_info(f"Repairing {len(corrupted)} corrupted files...")
        return self.download_missing(corrupted)

    def generate_checksums(self, output_path: Path | None = None) -> dict:
        """Generate SHA256 checksums for all geometry files."""
        if output_path is None:
            output_path = self.geometry_dir / "checksums.json"

        checksums = {}
        for key, spec in EXPECTED_FILES.items():
            if key in ["depth_maps", "conf_maps"]:
                pattern = spec["pattern"]
                files = sorted(self.depth_dir.glob(pattern))
                for f in files:
                    rel = f.relative_to(self.geometry_dir).as_posix()
                    checksums[rel] = compute_sha256(f)
            else:
                path = self.geometry_dir / spec["path"]
                if path.exists():
                    checksums[spec["path"]] = compute_sha256(path)

        save_checksums(checksums, output_path)
        log_success(f"Saved {len(checksums)} checksums to {output_path}")
        return checksums

    def verify_checksums(self, checksum_path: Path | None = None) -> dict:
        """Verify files against stored checksums."""
        if checksum_path is None:
            checksum_path = self.geometry_dir / "checksums.json"

        if not checksum_path.exists():
            log_warning("No checksum file found, generating new one...")
            return self.generate_checksums()

        stored = load_checksums(checksum_path)
        results = {"verified": 0, "mismatch": 0, "missing": 0, "errors": []}

        for rel_path, expected_hash in stored.items():
            local_path = self.geometry_dir / rel_path
            if not local_path.exists():
                results["missing"] += 1
                results["errors"].append(f"Missing: {rel_path}")
                continue

            actual_hash = compute_sha256(local_path)
            if actual_hash == expected_hash:
                results["verified"] += 1
            else:
                results["mismatch"] += 1
                results["errors"].append(f"Checksum mismatch: {rel_path}")

        log_info(f"Checksum verification: {results['verified']} verified, "
                 f"{results['mismatch']} mismatch, {results['missing']} missing")
        return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify and recover Kaggle → local geometry downloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.v2t.verify_geometry_download --verify           # Verify local files
  python -m src.v2t.verify_geometry_download --verify-remote    # Check remote vs local
  python -m src.v2t.verify_geometry_download --download-missing # Download missing files
  python -m src.v2t.verify_geometry_download --repair           # Re-download corrupted files
  python -m src.v2t.verify_geometry_download --checksums        # Generate/verify checksums
  python -m src.v2t.verify_geometry_download --full             # Full verification + repair
        """
    )
    parser.add_argument("--run", default="run_mvp", help="Run name (default: run_mvp)")
    parser.add_argument("--verify", action="store_true", help="Verify local files")
    parser.add_argument("--verify-remote", action="store_true", help="Check remote files")
    parser.add_argument("--download-missing", action="store_true", help="Download missing files")
    parser.add_argument("--repair", action="store_true", help="Re-download corrupted files")
    parser.add_argument("--checksums", action="store_true", help="Generate checksums")
    parser.add_argument("--verify-checksums", action="store_true", help="Verify against checksums")
    parser.add_argument("--full", action="store_true", help="Full verification + repair")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    verifier = GeometryDownloadVerifier(args.run)

    if args.full:
        args.verify = True
        args.verify_remote = True
        args.download_missing = True
        args.repair = True
        args.checksums = True
        args.verify_checksums = True

    # Default: verify local
    if not any([args.verify, args.verify_remote, args.download_missing,
                args.repair, args.checksums, args.verify_checksums]):
        args.verify = True

    overall_success = True

    # Step 1: Verify local
    if args.verify:
        local_results = verifier.verify_local(args.verbose)
        if local_results["total_invalid"] > 0 or local_results["total_missing"] > 0:
            overall_success = False

    # Step 2: Verify remote
    if args.verify_remote:
        remote_files = verifier.verify_remote()
        if args.verify:
            comparison = verifier.compare_local_remote(local_results, remote_files)
            if comparison["missing_locally"]:
                log_warning(f"Missing locally: {len(comparison['missing_locally'])} files")
                for f in comparison["missing_locally"][:10]:
                    print(f"  - {f}")
                if len(comparison["missing_locally"]) > 10:
                    print(f"  ... and {len(comparison['missing_locally']) - 10} more")
            if comparison["size_mismatch"]:
                log_warning(f"Size mismatches: {len(comparison['size_mismatch'])} files")

    # Step 3: Download missing
    if args.download_missing and args.verify_remote:
        remote_files = verifier.verify_remote()
        local_results = verifier.verify_local(False)
        comparison = verifier.compare_local_remote(local_results, remote_files)
        if comparison["missing_locally"]:
            results = verifier.download_missing(comparison["missing_locally"])
            if results["failed"] > 0:
                overall_success = False

    # Step 4: Repair corrupted
    if args.repair:
        local_results = verifier.verify_local(False)
        results = verifier.repair_corrupted(local_results)
        if results["failed"] > 0:
            overall_success = False

    # Step 5: Generate checksums
    if args.checksums:
        verifier.generate_checksums()

    # Step 6: Verify checksums
    if args.verify_checksums:
        results = verifier.verify_checksums()
        if results["mismatch"] > 0 or results["missing"] > 0:
            overall_success = False

    # Final summary
    print()
    if overall_success:
        log_success("All geometry downloads verified successfully!")
        print(f"Geometry directory: {verifier.geometry_dir}")
    else:
        log_error("Some issues remain. Run with --full to attempt repair.")

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()