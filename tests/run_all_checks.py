#!/usr/bin/env python3
"""Comprehensive Verification Suite for Videos_to_3D (v3d).

Verifies:
1. Pose Conventions Regression Tests (tests/test_pose_conventions.py)
2. Quality Knobs Regression Tests (tests/test_quality_knobs.py)
3. Static Security & Cleanliness Scan:
   - Zero hardcoded API keys (nvapi-, tvly-, etc.) in src/v3d, scripts/, skills/
   - Zero hardcoded /Users/ paths in src/v3d, scripts/, skills/
4. Package Syntax & Import Verification:
   - Clean AST syntax on all Python files under src/v3d
   - Clean import of the v3d package and its core modules

Exits with code 0 if all checks pass, or 1 if any check fails.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Terminal formatting
USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, color_code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{color_code}m{text}\033[0m"


def green(text: str) -> str:
    return _c(text, "1;32")


def red(text: str) -> str:
    return _c(text, "1;31")


def yellow(text: str) -> str:
    return _c(text, "1;33")


def cyan(text: str) -> str:
    return _c(text, "1;36")


def bold(text: str) -> str:
    return _c(text, "1")


def dim(text: str) -> str:
    return _c(text, "2")


@dataclass
class CheckResult:
    name: str
    passed: bool
    duration_s: float
    summary: str
    details: List[str] = field(default_factory=list)


def run_subtest_script(script_path: Path, verbose: bool = False) -> CheckResult:
    """Run an existing test script as a subprocess and parse its output."""
    name = script_path.name
    t0 = time.perf_counter()

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        duration = time.perf_counter() - t0
        passed = proc.returncode == 0

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        pass_count = stdout.count("  pass  ")
        fail_count = stdout.count("  FAIL  ")

        summary_parts = []
        if pass_count > 0:
            summary_parts.append(f"{pass_count} assertions passed")
        if fail_count > 0:
            summary_parts.append(f"{fail_count} assertions FAILED")
        if not summary_parts:
            summary_parts.append(f"exit code {proc.returncode}")

        summary = ", ".join(summary_parts)

        details = []
        if verbose or not passed:
            if stdout:
                details.extend(stdout.splitlines())
            if stderr:
                details.append("--- STDERR ---")
                details.extend(stderr.splitlines())

        return CheckResult(
            name=f"Test Suite: {name}",
            passed=passed,
            duration_s=duration,
            summary=summary,
            details=details,
        )
    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - t0
        return CheckResult(
            name=f"Test Suite: {name}",
            passed=False,
            duration_s=duration,
            summary="Timed out after 120s",
            details=["Process killed due to timeout."],
        )
    except Exception as exc:
        duration = time.perf_counter() - t0
        return CheckResult(
            name=f"Test Suite: {name}",
            passed=False,
            duration_s=duration,
            summary=f"Exception: {type(exc).__name__}",
            details=[str(exc)],
        )


def run_static_security_scan(verbose: bool = False) -> CheckResult:
    """Scan src/v3d, scripts/, and skills/ for hardcoded keys and /Users/ paths."""
    t0 = time.perf_counter()
    target_dirs = ["src/v3d", "scripts", "skills"]

    # Patterns to detect hardcoded secrets or machine-specific paths
    patterns: List[Tuple[str, re.Pattern]] = [
        ("NVIDIA API Key", re.compile(r"nvapi-[A-Za-z0-9_-]{20,}")),
        ("Tavily API Key", re.compile(r"tvly-[A-Za-z0-9_-]{20,}")),
        ("GitHub Personal Token", re.compile(r"ghp_[A-Za-z0-9]{36}")),
        ("Google API Key", re.compile(r"AIza[0-9A-Za-z-_]{35}")),
        ("Generic Secret Key", re.compile(r"(?:api[_-]?key|secret)\s*[:=]\s*['\"][A-Za-z0-9_-]{25,}['\"]", re.IGNORECASE)),
        ("Hardcoded /Users/ path", re.compile(r"/Users/")),
    ]

    binary_or_asset_exts = {
        ".png", ".jpg", ".jpeg", ".mp4", ".glb", ".ply", ".obj",
        ".npy", ".onnx", ".pyc", ".tar", ".gz", ".zip", ".blend",
        ".bin", ".pt", ".pth",
    }

    violations: List[str] = []
    scanned_count = 0

    for rel_dir in target_dirs:
        dir_path = REPO_ROOT / rel_dir
        if not dir_path.exists():
            continue

        for p in dir_path.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in binary_or_asset_exts or "__pycache__" in p.parts:
                continue

            scanned_count += 1
            rel_file = p.relative_to(REPO_ROOT)

            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                violations.append(f"{rel_file}: could not read file ({e})")
                continue

            for line_no, line in enumerate(content.splitlines(), start=1):
                # Allow env var documentation or examples if needed, but flag real instances
                for pat_name, regex in patterns:
                    match = regex.search(line)
                    if match:
                        snippet = line.strip()
                        if len(snippet) > 80:
                            snippet = snippet[:77] + "..."
                        violations.append(
                            f"{rel_file}:{line_no} [{pat_name}]: {snippet}"
                        )

    duration = time.perf_counter() - t0
    passed = len(violations) == 0
    summary = f"{scanned_count} files scanned across {len(target_dirs)} directories; {len(violations)} violations"

    return CheckResult(
        name="Static Scan: API Keys & Hardcoded Paths",
        passed=passed,
        duration_s=duration,
        summary=summary,
        details=violations,
    )


def run_package_import_and_syntax_check(verbose: bool = False) -> CheckResult:
    """Validate AST syntax on all src/v3d files and test clean package imports."""
    t0 = time.perf_counter()
    v3d_root = REPO_ROOT / "src" / "v3d"

    syntax_errors: List[str] = []
    py_files = sorted(v3d_root.rglob("*.py"))

    # 1. AST syntax parsing
    for py_file in py_files:
        rel_path = py_file.relative_to(REPO_ROOT)
        try:
            source = py_file.read_text(encoding="utf-8")
            ast.parse(source, filename=str(py_file))
        except SyntaxError as syn_err:
            syntax_errors.append(
                f"{rel_path}:{syn_err.lineno}: SyntaxError: {syn_err.msg}"
            )
        except Exception as exc:
            syntax_errors.append(f"{rel_path}: {type(exc).__name__}: {exc}")

    # 2. Package import verification in an isolated subprocess
    import_errors: List[str] = []
    verify_code = """
import sys
from pathlib import Path

# Add src to path
src_dir = Path.cwd() / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import v3d
assert v3d is not None, "Failed to import root v3d"

# Core submodules that should import cleanly without special runtime dependencies
core_submodules = [
    "v3d.config",
    "v3d.manifest",
    "v3d.geometry",
    "v3d.geometry.quality",
    "v3d.geometry.pytorch3d_shim",
    "v3d.geometry.pgsr_build",
    "v3d.ingest",
    "v3d.ingest.poses",
    "v3d.ingest.frames",
    "v3d.scene",
    "v3d.scene.schema",
    "v3d.scene.decompose",
    "v3d.scene.from_holi",
    "v3d.validate",
    "v3d.validate.stage_j",
    "v3d.validate.gt_boxes",
    "v3d.export",
    "v3d.export.terrain",
    "v3d.export.glb",
    "v3d.export.visualize",
    "v3d.data",
    "v3d.data.scannetpp",
    "v3d.data.hf_tar",
    "v3d.paper_runner",
]

for mod in core_submodules:
    __import__(mod)
"""

    proc = subprocess.run(
        [sys.executable, "-c", verify_code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or proc.stdout.strip()
        import_errors.append(f"Import check failed (exit code {proc.returncode}):\n{err_msg}")

    duration = time.perf_counter() - t0
    total_issues = len(syntax_errors) + len(import_errors)
    passed = total_issues == 0

    summary = (
        f"{len(py_files)} Python files parsed cleanly; "
        f"v3d package and core modules imported successfully"
        if passed
        else f"{len(syntax_errors)} syntax errors, {len(import_errors)} import errors"
    )

    details = syntax_errors + import_errors
    return CheckResult(
        name="Package Syntax & Clean Import Validation",
        passed=passed,
        duration_s=duration,
        summary=summary,
        details=details,
    )


def print_banner() -> None:
    width = 72
    print("=" * width)
    print(bold("        Videos_to_3D (v3d) — Comprehensive Verification Suite").center(width))
    print("=" * width)
    print(f"Repository Root : {cyan(str(REPO_ROOT))}")
    print(f"Python Binary   : {cyan(sys.executable)}")
    print(f"Python Version  : {sys.version.split()[0]}")
    print("-" * width)


def print_results(results: List[CheckResult], total_time: float) -> int:
    width = 72
    print()
    print("=" * width)
    print(bold("                              VERIFICATION SUMMARY").center(width))
    print("=" * width)

    all_passed = True
    for res in results:
        status_tag = green("[ PASS ]") if res.passed else red("[ FAIL ]")
        time_tag = cyan(f"{res.duration_s:6.2f}s")
        print(f" {status_tag}  {time_tag}  {bold(res.name)}")
        print(f"           {dim(res.summary)}")

        if not res.passed and res.details:
            print()
            print(red(f"   >>> Failure Details for {res.name}:"))
            for line in res.details[:25]:
                print(f"       {line}")
            if len(res.details) > 25:
                print(f"       ... and {len(res.details) - 25} more items")
            print()

        if not res.passed:
            all_passed = False

    print("-" * width)
    total_passed = sum(1 for r in results if r.passed)
    total_checks = len(results)
    overall_status = green("ALL CHECKS PASSED") if all_passed else red("VERIFICATION FAILED")

    print(f"Status       : {bold(overall_status)}")
    print(f"Score        : {bold(f'{total_passed}/{total_checks}')} checks passed")
    print(f"Total Time   : {cyan(f'{total_time:.2f}s')}")
    print("=" * width)

    return 0 if all_passed else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run all v3d verification checks.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed outputs.")
    args = parser.parse_args(argv)

    print_banner()
    t_start = time.perf_counter()
    results: List[CheckResult] = []

    # 1. Pose conventions test suite
    print(f"[*] Running {bold('tests/test_pose_conventions.py')}...")
    res_poses = run_subtest_script(REPO_ROOT / "tests" / "test_pose_conventions.py", verbose=args.verbose)
    results.append(res_poses)
    print(f"    {'✓' if res_poses.passed else '✗'} {res_poses.summary} ({res_poses.duration_s:.2f}s)")

    # 2. Quality knobs test suite
    print(f"[*] Running {bold('tests/test_quality_knobs.py')}...")
    res_knobs = run_subtest_script(REPO_ROOT / "tests" / "test_quality_knobs.py", verbose=args.verbose)
    results.append(res_knobs)
    print(f"    {'✓' if res_knobs.passed else '✗'} {res_knobs.summary} ({res_knobs.duration_s:.2f}s)")

    # 3. Static security and path scan
    print(f"[*] Running static scan for API keys and /Users/ paths...")
    res_scan = run_static_security_scan(verbose=args.verbose)
    results.append(res_scan)
    print(f"    {'✓' if res_scan.passed else '✗'} {res_scan.summary} ({res_scan.duration_s:.2f}s)")

    # 4. Package syntax and clean import check
    print(f"[*] Validating src/v3d package imports and syntax...")
    res_import = run_package_import_and_syntax_check(verbose=args.verbose)
    results.append(res_import)
    print(f"    {'✓' if res_import.passed else '✗'} {res_import.summary} ({res_import.duration_s:.2f}s)")

    total_time = time.perf_counter() - t_start
    return print_results(results, total_time)


def test_all_checks():
    """Pytest hook for test discovery."""
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main())
