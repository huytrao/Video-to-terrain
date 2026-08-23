#!/usr/bin/env bash
# kx.sh — Kaggle Execution Runner with Auto-Retry & Smart Proxy Resolution
# Part of kaggle-runtime-toolkit
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
EXEC="$HERE/kaggle_executor.py"
TRIES="${KX_TRIES:-6}"

# Ensure python3 is available
if command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif [ -f "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    echo "kx: python3 / python not found in PATH" >&2
    exit 1
fi

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

for try in $(seq 1 "$TRIES"); do
  if "$PY" "$EXEC" "$@" >"$log" 2>&1; then
    cat "$log"
    exit 0
  fi

  # HTTP 404 on /api handshake means proxy URL / JWT has expired
  if grep -q 'Cannot reach Jupyter API: HTTP 404' "$log"; then
    echo "kx: Proxy URL is expired (HTTP 404 on /api) — not a transient failure." >&2
    echo "    Reopen the notebook's 'Kaggle Jupyter Server' panel and update the URL in .env." >&2
    cat "$log" >&2
    exit 3
  fi

  # Transport failures worthy of backoff retry
  if grep -qE 'ReadTimeout|ConnectionError|HTTP 5[0-9][0-9]|Cannot reach Jupyter API|RemoteDisconnected|Connection aborted' "$log"; then
    reason="$(grep -oE 'ReadTimeout|ConnectionError|HTTP 5[0-9][0-9]|Cannot reach Jupyter API|RemoteDisconnected|Connection aborted' "$log" | tail -1)"
    echo "kx: attempt $try/$TRIES failed on transport ($reason), retrying in $((try * 4))s..." >&2
    sleep $((try * 4))
    continue
  fi

  cat "$log"
  echo "kx: failed with a non-transport error on attempt $try; not retrying." >&2
  exit 1
done

cat "$log" >&2
echo "kx: gave up after $TRIES attempts — please verify your network or update proxy URL in .env." >&2
exit 3
