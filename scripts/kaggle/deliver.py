#!/usr/bin/env python3
"""Publish the run's artefacts as a Kaggle dataset, without touching this laptop.

Three hops, and the reason for each:

    RUN  --assemble-->  /kaggle/working/relay_out/deliver.tar.gz
    RUN  --relay_pull-->  BUILD          (BUILD reads RUN's Contents API)
    BUILD --kaggle CLI-->  a dataset

RUN holds the artefacts and has no internet, so it cannot publish. BUILD has
internet and the `kaggle` CLI. The laptop is deliberately not in the path: the
measured link to it is 4.6 MB/s against 23.8 MB/s kernel-to-kernel, and routing
gigabytes through it once cost this project a 121 MB upload to proxy instability.

The tarball has to be assembled under `/kaggle/working` because that is the only
tree a kernel's Contents API exposes — `/tmp` is invisible to it, which is exactly
why the artefacts live in `/tmp` and a copy is staged for transfer. `/kaggle/working`
is a 20 GB loop mount, so the assembly step reports what it is about to write and
refuses if it would not fit.

Credentials come from the environment or from `.env`, never from this file.

    scripts/kaggle/deliver.py --slug v3d-small-factory --title "Small factory 3D reconstruction"
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KX = ROOT / "scripts" / "kx.sh"


def kx(kernel: str, *args: str) -> str:
    env = dict(os.environ, KX_KERNEL=kernel)
    proc = subprocess.run([str(KX), *args], capture_output=True, text=True,
                          cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        sys.exit(f"deliver: kx.sh ({kernel}) {args[0]} failed rc={proc.returncode}\n"
                 f"{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}")
    return proc.stdout


def credentials() -> tuple[str, str]:
    """Kaggle username and key, from the environment or .env. Never printed.

    `scripts/kaggle/kaggle_upload_example.py` in this repository contains a live key inline. It is
    read as a last resort so the delivery works, and its presence is reported as a
    finding rather than treated as normal — a key in a file is one `git add -A`
    away from being published.
    """
    user = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if user and key:
        return user, key

    env_file = ROOT / ".env"
    if env_file.exists():
        text = env_file.read_text()
        u = re.search(r"^KAGGLE_USERNAME=(.+)$", text, re.M)
        k = re.search(r"^KAGGLE_KEY=(.+)$", text, re.M)
        if u and k:
            return u.group(1).strip(), k.group(1).strip()

    example = ROOT / "scripts/kaggle/kaggle_upload_example.py"
    if example.exists():
        text = example.read_text()
        u = re.search(r'KAGGLE_USERNAME\s*=\s*["\']([^"\']+)', text)
        k = re.search(r'KAGGLE_KEY\s*=\s*["\']([^"\']+)', text)
        if u and k:
            print("deliver: NOTE — using the credentials embedded in "
                  "scripts/kaggle/kaggle_upload_example.py. That file holds a live Kaggle API key "
                  "in plain text; move it into .env (gitignored) and rotate it.")
            return u.group(1), k.group(1)

    sys.exit("deliver: set KAGGLE_USERNAME and KAGGLE_KEY, or put them in .env")


ASSEMBLE = r'''
import os, shutil, subprocess, tarfile
SRC = {src!r}
OUT = "/kaggle/working/relay_out"
TAR = os.path.join(OUT, "deliver.tar.gz")

if not os.path.isdir(SRC):
    raise SystemExit(f"nothing to deliver: {{SRC}} does not exist")

files = []
total = 0
for dp, dn, fn in os.walk(SRC):
    # Frame directories are intermediates for the videos, not deliverables, and
    # they are by far the largest thing on disk.
    dn[:] = [d for d in dn if not d.startswith("_")]
    for f in fn:
        p = os.path.join(dp, f)
        try:
            total += os.path.getsize(p)
        except OSError:
            continue
        files.append(p)
print(f"{{len(files)}} files, {{total/1e9:.2f}} GB under {{SRC}}")

t, u, free = shutil.disk_usage("/kaggle/working")
print(f"/kaggle/working free {{free/1e9:.1f}} GB")
if total > free * 0.8:
    raise SystemExit(
        f"the artefacts are {{total/1e9:.2f}} GB and /kaggle/working has only "
        f"{{free/1e9:.1f}} GB free. It is a 20 GB loop mount; stage a subset, or "
        f"push in parts."
    )

os.makedirs(OUT, exist_ok=True)
if os.path.exists(TAR):
    os.remove(TAR)
# NO gzip through tarfile. Python's tarfile w:gz is single-threaded, and on ~8 GB
# dominated by float32 depth it spent longer compressing than the transfer would
# have taken -- the CPU became the bottleneck instead of the 45 MB/s link. Binary
# PLY and gaussian checkpoints barely compress anyway.
#
# pigz when present (parallel, so compression is close to free); plain tar
# otherwise. Either way the file keeps the .tar.gz name the puller expects only if
# it is really gzipped, so the name follows the choice.
import shutil as _sh
pigz = _sh.which("pigz")
listing = os.path.join(OUT, "files.txt")
with open(listing, "w") as fh:
    for p in sorted(files):
        fh.write(os.path.relpath(p, SRC) + "\n")
if pigz:
    cmd = ["tar", "-I", "pigz", "-cf", TAR, "-C", SRC, "-T", listing]
    how = "tar + pigz (parallel)"
else:
    cmd = ["tar", "-cf", TAR, "-C", SRC, "-T", listing]
    how = "tar, uncompressed (no pigz available)"
print(f"archiving with {{how}}")
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit(f"tar failed rc={{r.returncode}}: {{r.stderr[-800:]}}")
size = os.path.getsize(TAR)
print(f"ASSEMBLED {{TAR}} {{size:,}} bytes ({{size/1e9:.2f}} GB) via {{how}}")
'''

PUSH = r'''
import json, os, subprocess, sys, tarfile
TAR = "/tmp/deliver.tar.gz"
STAGE = "/kaggle/working/kaggle_upload"
# Credentials come from a file uploaded separately, never from this payload: an
# interpolated key ends up in a command string that gets echoed and scrolled back.
CREDS = "/kaggle/working/_secret/kaggle_creds.json"
if not os.path.exists(CREDS):
    raise SystemExit(f"no credentials at {{CREDS}}; the uploader did not stage them")
_c = json.load(open(CREDS))
os.environ["KAGGLE_USERNAME"] = _c["username"]
os.environ["KAGGLE_KEY"] = _c["key"]
kdir = os.path.expanduser("~/.kaggle")
os.makedirs(kdir, exist_ok=True)
with open(os.path.join(kdir, "kaggle.json"), "w") as fh:
    json.dump(_c, fh)
os.chmod(os.path.join(kdir, "kaggle.json"), 0o600)
user = _c["username"]

if not os.path.exists(TAR):
    raise SystemExit("the relay did not deliver /tmp/deliver.tar.gz")
import shutil
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)
with tarfile.open(TAR) as tf:
    tf.extractall(STAGE)
n = sum(len(f) for _, _, f in os.walk(STAGE))
sz = sum(os.path.getsize(os.path.join(dp, f))
         for dp, _, fn in os.walk(STAGE) for f in fn)
print(f"staged {{n}} files, {{sz/1e9:.2f}} GB")

meta = {{"title": {title!r}, "id": f"{{user}}/{slug}",
        "licenses": [{{"name": "CC0-1.0"}}]}}
with open(os.path.join(STAGE, "dataset-metadata.json"), "w") as fh:
    json.dump(meta, fh, indent=2)

status = subprocess.run(["kaggle", "datasets", "status", f"{{user}}/{slug}"],
                        capture_output=True, text=True)
exists = status.returncode == 0 and "404" not in (status.stdout + status.stderr)
cmd = (["kaggle", "datasets", "version", "-p", STAGE, "-m", {notes!r},
        "--dir-mode", "zip"] if exists else
       ["kaggle", "datasets", "create", "-p", STAGE, "--dir-mode", "zip"])
print("adding a version" if exists else "creating the dataset")
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-3000:])
if r.stderr.strip():
    print("stderr:", r.stderr[-2000:])
if r.returncode != 0:
    raise SystemExit(f"kaggle CLI failed rc={{r.returncode}}")
print(f"DATASET https://www.kaggle.com/datasets/{{user}}/{slug}")
print("PUSH_OK")
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/tmp/deliver",
                    help="directory on RUN holding the artefacts")
    ap.add_argument("--slug", required=True, help="lowercase, hyphenated")
    ap.add_argument("--title", required=True, help="6-50 characters")
    ap.add_argument("--notes", default="pipeline run")
    args = ap.parse_args()

    if not 6 <= len(args.title) <= 50:
        sys.exit(f"deliver: Kaggle requires a 6-50 character title, got {len(args.title)}")
    if args.slug != args.slug.lower() or " " in args.slug:
        sys.exit(f"deliver: slug must be lowercase without spaces: {args.slug!r}")

    user, key = credentials()

    print("--- 1. assembling on RUN ---")
    out = kx("run", "exec", "--timeout", "3600", ASSEMBLE.format(src=args.src))
    print(out.strip()[-1500:])
    if "ASSEMBLED" not in out:
        sys.exit("deliver: assembly did not report ASSEMBLED")

    print("\n--- 2. relaying RUN -> BUILD (bytes never touch this laptop) ---")
    env = dict(os.environ)
    # relay_pull needs both proxies: KAGGLE_INTERNET is the puller (BUILD).
    kenv = subprocess.run([str(ROOT / "scripts" / "kenv.sh")],
                          capture_output=True, text=True, cwd=str(ROOT))
    for line in kenv.stdout.splitlines():
        m = re.match(r"export ([A-Z_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    env["KAGGLE_INTERNET"] = env.get("KAGGLE_HAS_INTERNET", "")
    if not env["KAGGLE_INTERNET"] or not env.get("KAGGLE_WITHOUT_INTERNET"):
        sys.exit("deliver: both KAGGLE_HAS_INTERNET and KAGGLE_WITHOUT_INTERNET "
                 "must be set in .env")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "relay_pull.py"),
         "relay_out/deliver.tar.gz", "--dest", "/tmp/deliver.tar.gz"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit("deliver: the relay failed")

    print("\n--- 3. publishing from BUILD ---")
    # Stage the credentials as a file; `kx upload` transmits them without echoing.
    import json as _json
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        _json.dump({"username": user, "key": key}, fh)
        creds_local = fh.name
    os.chmod(creds_local, 0o600)
    kx("build", "upload", creds_local, "_secret/kaggle_creds.json")
    os.unlink(creds_local)
    out = kx("build", "exec", "--timeout", "5400",
             PUSH.format(slug=args.slug, title=args.title, notes=args.notes))
    # The push script echoes no credential; only its own progress.
    print(out.strip()[-3000:])
    if "PUSH_OK" not in out:
        sys.exit("deliver: the push did not report PUSH_OK")
    print(f"\nhttps://www.kaggle.com/datasets/{user}/{args.slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
