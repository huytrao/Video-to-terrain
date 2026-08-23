#!/usr/bin/env bash
# Put the code the benchmark needs onto the RUN kernel, as two small tars.
#
# Why tars and not a file-by-file upload: the Contents API takes one base64 PUT
# per file, and the paper checkout plus src/v3d is several hundred files. Each PUT
# is a fresh chance for the proxy to time out, and a partial tree is worse than no
# tree because the failure surfaces later, as an ImportError in stage 3.
#
# Why the checkout is trimmed: .git (18 MB), assets/teaser.png (12 MB) and
# PGSR/submodules (21 MB of glm headers, needed only to *compile* the CUDA
# extensions, which are already installed as sm_120 wheels) are 96% of the bytes
# and no Python file references any of them -- verified by grep, not assumed.
#
# Usage:  scripts/kaggle/deploy_code.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

eval "$(scripts/kaggle/kenv.sh)"
: "${KAGGLE_WITHOUT_INTERNET:?not in .env}"
export KAGGLE_PROXY_URL="$KAGGLE_WITHOUT_INTERNET"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

tar --exclude='.git' --exclude='assets' --exclude='PGSR/submodules' \
    --exclude='__pycache__' --exclude='*.pyc' \
    -czf "$STAGE/paper.tgz" -C source-reference/holi-spatial .

tar --exclude='__pycache__' --exclude='*.pyc' \
    -czf "$STAGE/v3d.tgz" -C src v3d

for t in paper v3d; do
  printf '%s.tgz  %s bytes\n' "$t" "$(wc -c <"$STAGE/$t.tgz" | tr -d ' ')"
done

# Remote paths are relative: the Contents API is rooted at the notebook dir, so a
# leading /kaggle/working/ 404s.
scripts/kaggle/kx.sh upload "$STAGE/paper.tgz" _stage/paper.tgz
scripts/kaggle/kx.sh upload "$STAGE/v3d.tgz"   _stage/v3d.tgz

# `kx.sh run` takes a *local* path and uploads it to _run/<basename>, so the
# remote layout is unaffected by which scripts/ subdirectory this lives in.
scripts/kaggle/kx.sh run scripts/kernel/kernel_unpack_code.py --timeout 600
