#!/usr/bin/env bash
# Shell launcher for Multi-Agent AI Debate & Synthesis on NVIDIA NIM / Claude Code.
#
# Usage:
#   debate-nv "Kiến trúc hệ thống streaming video 3D thời gian thực"
#   debate-nv "Phương án tối ưu hóa pipeline" --rounds 4
#   debate-nv --mode combat "AI vs Human" --rounds 2
set -euo pipefail

# Resolve symlink to real directory
TARGET="${BASH_SOURCE[0]}"
while [ -h "$TARGET" ]; do
  DIR="$(cd -P "$(dirname "$TARGET")" && pwd)"
  TARGET="$(readlink "$TARGET")"
  [[ $TARGET != /* ]] && TARGET="$DIR/$TARGET"
done
SCRIPT_DIR="$(cd -P "$(dirname "$TARGET")" && pwd)"

PROXY_URL="${NV_PROXY_URL:-http://127.0.0.1:4011}"

# Ensure proxy is alive
if ! curl -fsS --max-time 3 "$PROXY_URL/health/liveliness" >/dev/null 2>&1; then
  echo "debate: LiteLLM proxy at $PROXY_URL not responding. Starting proxy..." >&2
  if [ -x "$HOME/.local/bin/nvclaudeproxy" ]; then
    "$HOME/.local/bin/nvclaudeproxy" || { echo "debate: could not start proxy." >&2; exit 1; }
  fi
fi

exec python3 "$SCRIPT_DIR/debate.py" "$@"
