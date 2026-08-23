#!/usr/bin/env python3
"""Trim every video in visualize/ to its first N seconds.

    python scripts/tools/trim_videos.py --seconds 7

Re-encodes rather than stream-copying: `-c copy` can only cut on a keyframe, so
with a 2-second GOP a "7 second" clip lands anywhere between 6 and 8 seconds and
often opens on a grey frame. Re-encoding is slower but cuts exactly where asked.

Clips already at or under the limit are left untouched. Originals are replaced —
every one of these is reproducible from `make visualize`, `walk_headings.py` or
`visualize_objects.py`, so there is nothing here that cannot be rebuilt.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def duration(exe: str, path: Path) -> Optional[float]:
    out = subprocess.run([exe, "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def trim(exe: str, path: Path, seconds: float, crf: int) -> Optional[float]:
    """Cut in place via a temp file. Returns the new size in MB, or None on failure."""
    tmp = path.with_suffix(".trim.mp4")
    result = subprocess.run(
        [exe, "-y", "-i", str(path), "-t", str(seconds),
         "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)],
        capture_output=True, text=True)
    if result.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        print(f"    ffmpeg failed: {result.stderr.strip().splitlines()[-1:]}")
        return None
    shutil.move(str(tmp), str(path))
    return path.stat().st_size / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description="Trim visualize/ videos to their first N seconds")
    ap.add_argument("--root", type=Path, default=Path("visualize"))
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--crf", type=int, default=23, help="lower is better quality")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"{args.root} does not exist")

    exe = ffmpeg()
    videos: List[Path] = sorted(args.root.rglob("*.mp4"))
    print(f"{len(videos)} videos, target {args.seconds:g}s\n")

    before = after = 0.0
    trimmed = kept = 0
    for path in videos:
        size = path.stat().st_size / 1e6
        before += size
        secs = duration(exe, path)
        label = str(path.relative_to(args.root))

        if secs is None:
            print(f"  [?]    {label}  (unreadable, left alone)")
            after += size
            continue
        if secs <= args.seconds + 0.05:
            print(f"  [keep] {label:52s} {secs:5.1f}s  {size:6.1f} MB")
            after += size
            kept += 1
            continue
        if args.dry_run:
            print(f"  [cut]  {label:52s} {secs:5.1f}s -> {args.seconds:g}s")
            after += size
            trimmed += 1
            continue

        new_size = trim(exe, path, args.seconds, args.crf)
        if new_size is None:
            after += size
            continue
        print(f"  [cut]  {label:52s} {secs:5.1f}s -> {args.seconds:g}s   "
              f"{size:6.1f} -> {new_size:5.1f} MB")
        after += new_size
        trimmed += 1

    print(f"\n{trimmed} trimmed, {kept} already short enough")
    print(f"{before:.0f} MB -> {after:.0f} MB"
          + (f"  (saved {before - after:.0f} MB)" if after < before else ""))
    if not args.dry_run:
        print("\nrebuild the pages:  make video-index")


if __name__ == "__main__":
    main()
