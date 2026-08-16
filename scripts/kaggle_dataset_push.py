#!/usr/bin/env python3
"""Publish a directory as a Kaggle dataset, from the kernel that holds it.

Uploads run from a kernel with internet, because the Kaggle API needs one.
Anything sitting on the offline kernel has to be moved first — see
`scripts/relay_push.py` for that direction.

Credentials come from the environment or an existing `~/.kaggle/kaggle.json`,
never from this file. Pass them through the caller:

    KAGGLE_USERNAME=... KAGGLE_KEY=... python scripts/kaggle_dataset_push.py \\
        /kaggle/working/wheels --slug v3d-pgsr-wheels-sm120 \\
        --title "PGSR CUDA extensions for Blackwell sm_120"

The script is meant to run *on the kernel*, so it is normally injected through
`kaggle_executor.py exec` rather than run locally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"


def write_credentials(username: str, key: str) -> Path:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    path = KAGGLE_DIR / "kaggle.json"
    path.write_text(json.dumps({"username": username, "key": key}))
    path.chmod(0o600)
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    return path


def dataset_exists(dataset_id: str) -> bool:
    r = subprocess.run(
        ["kaggle", "datasets", "status", dataset_id], capture_output=True, text=True
    )
    return r.returncode == 0 and "404" not in (r.stdout + r.stderr)


def push(
    source: Path,
    slug: str,
    title: str,
    username: str,
    notes: str = "updated",
    public: bool = False,
) -> int:
    """Create the dataset, or add a version if it already exists."""
    if not source.is_dir():
        raise NotADirectoryError(f"{source} is not a directory")

    files = [p for p in source.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"{source}: {len(files)} files, {total/1e9:.2f} GB")
    if not files:
        raise ValueError(f"{source} is empty — nothing to upload")

    dataset_id = f"{username}/{slug}"
    meta = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (source / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    if dataset_exists(dataset_id):
        cmd = ["kaggle", "datasets", "version", "-p", str(source),
               "-m", notes, "--dir-mode", "zip"]
        print(f"adding a version to existing {dataset_id}")
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(source), "--dir-mode", "zip"]
        if public:
            cmd.append("--public")
        print(f"creating {dataset_id}")

    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.stderr.strip():
        print("stderr:", r.stderr.strip(), file=sys.stderr)
    if r.returncode == 0:
        print(f"https://www.kaggle.com/datasets/{dataset_id}")
    return r.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="directory to upload")
    ap.add_argument("--slug", required=True, help="dataset slug, lowercase with hyphens")
    ap.add_argument("--title", required=True)
    ap.add_argument("--notes", default="updated")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args(argv)

    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if username and key:
        write_credentials(username, key)
    elif (KAGGLE_DIR / "kaggle.json").exists():
        username = json.loads((KAGGLE_DIR / "kaggle.json").read_text())["username"]
    else:
        sys.exit("set KAGGLE_USERNAME and KAGGLE_KEY, or provide ~/.kaggle/kaggle.json")

    return push(args.source, args.slug, args.title, username,
                notes=args.notes, public=args.public)


if __name__ == "__main__":
    raise SystemExit(main())
