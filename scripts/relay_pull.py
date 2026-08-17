#!/usr/bin/env python3
"""Pull a file off the no-internet notebook, onto one that has internet.

The mirror of `relay_push.py`, and the piece that was missing: results are
produced on the offline kernel, but publishing them needs a kernel that can
reach the network. Bytes never touch this machine, which matters because the
laptop link is 4.6 MB/s against 35 MB/s kernel-to-kernel.

Same reason it works in this direction too: B's Contents API is served by
Kaggle's platform rather than by B's kernel, so B's "Internet: off" toggle has
no say over an authenticated *read* issued from outside. B's kernel is never
asked to fetch anything.

    export KAGGLE_INTERNET='https://.../proxy'          # A, does the pulling
    export KAGGLE_WITHOUT_INTERNET='https://.../proxy'  # B, holds the file

    python scripts/relay_pull.py relay_out/results.tar.gz --dest /tmp/results.tar.gz

The source path is relative to B's /kaggle/working, because that is the only
tree its Contents API exposes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1] / "kaggle-runtime-toolkit-agent-skills"
sys.path.insert(0, str(TOOLKIT))

from kaggle_executor import KaggleExecutor  # noqa: E402

_DRIVER = r'''
import base64, os, time
from urllib.parse import quote
import requests

B_PROXY = "__B_PROXY__"
SRC     = "__SRC__"          # relative to B's /kaggle/working
DEST    = "__DEST__"         # absolute path on A
CHUNK   = __CHUNK__

base = B_PROXY.rstrip("/")
sess = requests.Session()
sess.get(base + "/api", timeout=30, allow_redirects=True).raise_for_status()

def contents_url(path):
    return base + "/api/contents/" + quote(path, safe="/")

# The Contents API returns whole files, so a multi-GB artefact must be split on
# B before it can be read. list the parts B wrote.
r = sess.get(contents_url(os.path.dirname(SRC) or ""), params={"content": 1}, timeout=120)
r.raise_for_status()
listing = r.json().get("content", [])
names = sorted(x["name"] for x in listing if x["name"].startswith(os.path.basename(SRC)))
if not names:
    raise SystemExit("no parts found for %r on B" % SRC)
print("parts on B:", len(names))

os.makedirs(os.path.dirname(DEST) or ".", exist_ok=True)
total = 0
t0 = time.time()
with open(DEST, "wb") as out:
    for i, name in enumerate(names):
        path = (os.path.dirname(SRC) + "/" + name).lstrip("/")
        for attempt in range(6):
            try:
                resp = sess.get(contents_url(path), params={"format": "base64"}, timeout=300)
                if resp.status_code >= 400:
                    raise RuntimeError("HTTP %d" % resp.status_code)
                blob = base64.b64decode(resp.json()["content"])
                break
            except Exception as e:
                wait = min(2 ** attempt, 20)
                print("  [retry %d for %s after %s, %ds]" % (attempt + 1, name, type(e).__name__, wait))
                time.sleep(wait)
        else:
            raise SystemExit("gave up on %s" % name)
        out.write(blob)
        total += len(blob)
        el = time.time() - t0
        print("  %d/%d  %.1f MB  %.1f MB/s" % (i + 1, len(names), total / 1e6, total / 1e6 / (el + 1e-9)))

print("PULLED", total, "bytes to", DEST)
print("RELAY_PULL_OK")
'''


def _env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        sys.exit(f"ERROR: {name} not set.")
    return v


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pull a file from notebook B to notebook A.")
    p.add_argument("src", help="path on B, relative to /kaggle/working; parts share this prefix")
    p.add_argument("--dest", required=True, help="absolute destination path on A")
    p.add_argument("--chunk", type=int, default=8, help="MB per Contents-API read")
    p.add_argument("--timeout", type=int, default=9000)
    args = p.parse_args(argv)

    driver = (
        _DRIVER
        .replace("__B_PROXY__", _env("KAGGLE_WITHOUT_INTERNET"))
        .replace("__SRC__", args.src)
        .replace("__DEST__", args.dest)
        .replace("__CHUNK__", str(args.chunk * 1024 * 1024))
    )

    tmp = Path(f"/tmp/_relay_pull_{os.getpid()}.py")
    tmp.write_text(driver)
    try:
        ex = KaggleExecutor(_env("KAGGLE_INTERNET"))
        remote = f"_relay/pull_{os.getpid()}.py"
        ex.upload_file(str(tmp), remote)
        print(f"Pulling {args.src} (on B) -> {args.dest} (on A)\n")
        output = ex.execute(
            ex.get_or_create_kernel(),
            f"exec(open('/kaggle/working/{remote}').read())",
            timeout=args.timeout,
        ) or ""
        if "RELAY_PULL_OK" not in output:
            sys.exit("relay_pull failed: transfer was not confirmed")
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
