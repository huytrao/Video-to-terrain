#!/usr/bin/env python3
"""Relay a HuggingFace repo into the no-internet kernel, and verify it landed.

``kaggle_relay.py`` prints "DONE" as soon as its assemble step returns, whether
or not the assemble worked. It also opens the destination without creating the
parent directory, so relaying into a fresh folder fails on every file while
still reporting success — and leaves the uploaded parts behind, quietly
filling the 20 GB working disk.

This wrapper closes both holes: it creates the destination directory first,
then stats every file on the receiving kernel and compares against the size
HuggingFace advertises. A transfer that cannot be verified is a failure.

    export KAGGLE_INTERNET='https://.../proxy'
    export KAGGLE_WITHOUT_INTERNET='https://.../proxy'

    python scripts/kaggle/relay_hf_repo.py Holi-Spatial/HoliSpatial-2M-QA-Qwen3-VL-8B \\
        --dest /tmp/qwen3vl8b
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[2] / "kaggle-runtime-toolkit-agent-skills"
sys.path.insert(0, str(TOOLKIT))

from kaggle_executor import KaggleExecutor  # noqa: E402

HF_API = "https://huggingface.co/api"


def repo_files(repo: str, repo_type: str = "models") -> dict[str, int]:
    """Filename -> size in bytes, from the HF tree API."""
    url = f"{HF_API}/{repo_type}/{repo}/tree/main?recursive=1"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "v3d"}), timeout=60) as r:
        tree = json.loads(r.read())
    out: dict[str, int] = {}
    for node in tree:
        if node.get("type") != "file":
            continue
        name = node["path"]
        size = node.get("lfs", {}).get("size") or node.get("size", 0)
        out[name] = int(size)
    return out


def remote_sizes(ex: KaggleExecutor, kernel: str, dest: str) -> dict[str, int]:
    code = (
        "import os, json\n"
        f"d = {dest!r}\n"
        "out = {}\n"
        "for root, _, files in os.walk(d):\n"
        "    for f in files:\n"
        "        p = os.path.join(root, f)\n"
        "        out[os.path.relpath(p, d)] = os.path.getsize(p)\n"
        "print('V3D_SIZES=' + json.dumps(out))\n"
    )
    text = ex.execute(kernel, code, timeout=300) or ""
    for line in text.splitlines():
        if line.startswith("V3D_SIZES="):
            return json.loads(line[len("V3D_SIZES="):])
    return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Relay an HF repo and verify every file.")
    ap.add_argument("repo")
    ap.add_argument("--dest", required=True, help="destination directory on the offline kernel")
    ap.add_argument("--repo-type", default="models", choices=["models", "datasets"])
    ap.add_argument("--skip", nargs="*", default=["training_args.bin"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--subchunk", type=int, default=24)
    args = ap.parse_args(argv)

    src_proxy = os.environ.get("KAGGLE_INTERNET", "")
    dst_proxy = os.environ.get("KAGGLE_WITHOUT_INTERNET", "")
    if not src_proxy or not dst_proxy:
        sys.exit("ERROR: set KAGGLE_INTERNET and KAGGLE_WITHOUT_INTERNET")

    wanted = {
        name: size
        for name, size in repo_files(args.repo, args.repo_type).items()
        if name not in args.skip and not name.startswith(".")
    }
    print(f"{args.repo}: {len(wanted)} files, {sum(wanted.values())/1e9:.2f} GB\n")

    dst = KaggleExecutor(dst_proxy)
    dst_kernel = dst.get_or_create_kernel()
    dst.execute(dst_kernel, f"import os; os.makedirs({args.dest!r}, exist_ok=True); print('dest ready')", timeout=120)

    have = remote_sizes(dst, dst_kernel, args.dest)
    todo = [n for n, s in wanted.items() if have.get(n) != s]
    if not todo:
        print("everything already present and the right size.")
        return 0
    print(f"{len(todo)} file(s) to transfer\n")

    relay = str(TOOLKIT / "kaggle_relay.py")
    for i, name in enumerate(sorted(todo), 1):
        url = f"https://huggingface.co/{args.repo}/resolve/main/{name}"
        if args.repo_type == "datasets":
            url = f"https://huggingface.co/datasets/{args.repo}/resolve/main/{name}"
        print(f"[{i}/{len(todo)}] {name}  ({wanted[name]/1e6:.1f} MB)")
        proc = subprocess.run(
            [sys.executable, relay, url,
             "--dest", f"{args.dest}/{name}",
             "--workers", str(args.workers),
             "--subchunk", str(args.subchunk)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-1000:])
            return 1

    have = remote_sizes(dst, dst_kernel, args.dest)
    bad = {n: (wanted[n], have.get(n)) for n in wanted if have.get(n) != wanted[n]}
    for name in sorted(wanted):
        got = have.get(name)
        mark = "ok  " if got == wanted[name] else "BAD "
        print(f"  {mark} {name:<45} {wanted[name]:>12,} expected  {str(got):>14} on kernel")

    if bad:
        print(f"\nFAILED: {len(bad)} file(s) did not land intact")
        return 1
    print(f"\nverified {len(wanted)} files, {sum(wanted.values())/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
