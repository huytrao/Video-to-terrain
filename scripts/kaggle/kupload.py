"""Upload a local file to a Kaggle kernel in verified chunks.

`kaggle_executor.upload_file` sends one PUT carrying the whole file base64-encoded
(`kaggle_executor.py:250-262`). Kaggle's Contents API **silently truncates** a PUT
somewhere near 24 MB — `docs/reference/deb-technical.md` records this as the reason
`relay_push.py` must stay at `--subchunk 8`. Silently is the operative word: the
call returns 200, the executor prints the byte count it *sent*, and the file on
the kernel is short. A truncated tar of the paper repo then fails inside a GPU
stage forty minutes later, with a traceback that points at the paper's code.

So this uploads in parts and proves the result: the digest is computed on the
kernel, over the reassembled file, and compared to the digest computed here. A
mismatch raises. Nothing downstream has to trust that a transfer worked.

    python3 scripts/kaggle/kupload.py LOCAL REMOTE [--chunk-mb 8] [--extract-to DIR]

`REMOTE` is relative to `/kaggle/working`. `--extract-to` untars the result and
deletes the macOS AppleDouble files a `tar` made on this laptop leaves behind.

The proxy URL comes from the environment exactly as `scripts/kaggle/kx.sh` resolves it,
so `KX_KERNEL=run|build` selects the kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# scripts/kaggle/ -> scripts/ -> repo root
ROOT = HERE.parents[1]
KX = HERE / "kx.sh"


def kx(*args: str, timeout: str = "900") -> str:
    """Run one kaggle_executor command through kx.sh, returning its stdout.

    kx.sh is used rather than the executor directly because it resolves the proxy
    URL from .env and retries the handshake, which times out often enough that a
    single-shot call is not a reliable way to move twenty parts.
    """
    proc = subprocess.run([str(KX), *args], capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        sys.exit(
            f"kupload: `kx.sh {' '.join(args[:2])}` failed (rc {proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
        )
    return proc.stdout


def remote_exec(code: str, timeout: str = "900") -> str:
    return kx("exec", "--timeout", timeout, code)


def sha256_local(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def split(path: Path, out_dir: Path, chunk_bytes: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{path.name}.part*"):
        stale.unlink()
    parts: list[Path] = []
    with path.open("rb") as fh:
        i = 0
        while block := fh.read(chunk_bytes):
            part = out_dir / f"{path.name}.part{i:04d}"
            part.write_bytes(block)
            parts.append(part)
            i += 1
    return parts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("local", type=Path)
    ap.add_argument("remote", help="path under /kaggle/working, or an absolute path")
    ap.add_argument("--chunk-mb", type=float, default=8.0,
                    help="raw megabytes per PUT; base64 inflates this by 4/3, so "
                         "keep it well under the ~24 MB truncation point")
    ap.add_argument("--extract-to", help="untar the uploaded file into this directory")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args(argv)

    local: Path = args.local
    if not local.is_file():
        sys.exit(f"kupload: {local} is not a file")

    digest = sha256_local(local)
    size = local.stat().st_size
    chunk_bytes = int(args.chunk_mb * 1_000_000)
    parts = split(local, Path("/tmp/kupload") / local.name, chunk_bytes)
    print(f"kupload: {local.name}  {size:,} bytes  sha256 {digest[:16]}…  "
          f"{len(parts)} part(s) of <= {chunk_bytes / 1e6:.1f} MB")

    # Parts land in /kaggle/working even when the destination is /tmp: the
    # Contents API can only write under the working directory, so reassembly on
    # the kernel is what puts the file where it belongs.
    staging = f"kupload_parts/{local.name}"
    remote_exec(
        f"import os, shutil\n"
        f"shutil.rmtree({('/kaggle/working/' + staging)!r}, ignore_errors=True)\n"
        f"os.makedirs({('/kaggle/working/' + staging)!r}, exist_ok=True)\n"
        f"print('staging ready')\n",
        timeout="300",
    )

    for n, part in enumerate(parts, 1):
        kx("upload", str(part), f"{staging}/{part.name}", timeout="900")
        print(f"  part {n}/{len(parts)} sent ({part.stat().st_size:,} bytes)")

    dest = args.remote if args.remote.startswith("/") else f"/kaggle/working/{args.remote}"

    # Reassemble and verify on the kernel. The digest is recomputed there over the
    # file that actually exists, which is the only check that can catch a
    # truncated PUT — the local side has no way to know.
    verify = f'''
import hashlib, os, shutil, glob
stage = {('/kaggle/working/' + staging)!r}
dest  = {dest!r}
os.makedirs(os.path.dirname(dest) or "/", exist_ok=True)
parts = sorted(glob.glob(os.path.join(stage, "*.part*")))
print(f"reassembling {{len(parts)}} parts -> {{dest}}")
h = hashlib.sha256()
total = 0
with open(dest, "wb") as out:
    for p in parts:
        b = open(p, "rb").read()
        out.write(b); h.update(b); total += len(b)
got = h.hexdigest()
print(f"bytes {{total:,}}  sha256 {{got}}")
if total != {size} or got != {digest!r}:
    raise SystemExit(
        f"UPLOAD_CORRUPT: expected {size:,} bytes / {digest} , "
        f"got {{total:,}} / {{got}}. A PUT was truncated; re-run with a smaller "
        f"--chunk-mb rather than trusting this file."
    )
print("UPLOAD_VERIFIED", dest)
'''
    out = remote_exec(verify, timeout="1800")
    print(out.strip()[-2000:])
    if "UPLOAD_VERIFIED" not in out:
        sys.exit("kupload: verification did not report UPLOAD_VERIFIED; do not use this file")

    if args.extract_to:
        # macOS tar writes ._* AppleDouble members; left in place they shadow real
        # modules and produce import errors that look like a broken checkout.
        extract = f'''
import os, subprocess, tarfile
dest = {dest!r}
into = {args.extract_to!r}
os.makedirs(into, exist_ok=True)
with tarfile.open(dest) as tf:
    names = tf.getnames()
    tf.extractall(into)
subprocess.run(f"find {{into}} -name '._*' -delete", shell=True)
subprocess.run(f"find {{into}} -name '.DS_Store' -delete", shell=True)
print(f"extracted {{len(names)}} members into {{into}}")
tops = sorted({{n.split('/')[0] for n in names}})
print("top level:", tops[:12])
for t in tops[:12]:
    p = os.path.join(into, t)
    print("  ", t, "dir" if os.path.isdir(p) else "file",
          len(os.listdir(p)) if os.path.isdir(p) else os.path.getsize(p))
print("EXTRACT_OK")
'''
        out = remote_exec(extract, timeout="1800")
        print(out.strip()[-3000:])
        if "EXTRACT_OK" not in out:
            sys.exit("kupload: extraction failed")

    if not args.keep_parts:
        remote_exec(
            f"import shutil; shutil.rmtree({('/kaggle/working/' + staging)!r}, "
            f"ignore_errors=True); print('parts removed')",
            timeout="300",
        )
        for p in parts:
            p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
