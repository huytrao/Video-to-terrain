#!/usr/bin/env bash
# Retrying front-end to kaggle_executor.py.
#
# The Kaggle Jupyter proxy drops requests under no particular provocation: the
# `/api` handshake that every invocation opens with read-times-out at 30 s often
# enough that a single-shot call is not a reliable way to run anything. The
# earlier session lost a 121 MB upload to this and recorded it as "proxy
# instability"; the fix is not to babysit each call by hand.
#
# Exit status is the executor's, not the pipeline's -- `cmd | tail` reports
# tail's success and hides the failure, which is how a dead proxy read as
# "OK on attempt 1".
#
# Usage:
#   scripts/kx.sh <kaggle_executor subcommand and args...>
#
# Reads the proxy URL from $KAGGLE_PROXY_URL, or from the first line-like blob in
# server-connect.txt if that is unset. Retries $KX_TRIES times (default 6) with
# linear backoff, and treats "the kernel is busy" as a reason to wait rather
# than fail.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
EXEC="$ROOT/kaggle-runtime-toolkit-agent-skills/kaggle_executor.py"
TRIES="${KX_TRIES:-6}"

if [ -z "${KAGGLE_PROXY_URL:-}" ]; then
  # The URL is one long token with no spaces; pick the longest such line.
  KAGGLE_PROXY_URL="$(grep -o 'https://[^[:space:]]*proxy' "$ROOT/server-connect.txt" 2>/dev/null \
                      | awk '{ print length, $0 }' | sort -rn | head -1 | cut -d' ' -f2-)"
  export KAGGLE_PROXY_URL
fi
if [ -z "${KAGGLE_PROXY_URL:-}" ]; then
  echo "kx: no KAGGLE_PROXY_URL, and none found in $ROOT/server-connect.txt" >&2
  exit 2
fi

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

for try in $(seq 1 "$TRIES"); do
  if python3 "$EXEC" "$@" >"$log" 2>&1; then
    cat "$log"
    exit 0
  fi
  # A real error from the remote code is not worth retrying; a transport failure
  # is. Distinguish them, so a failing script does not take six attempts to say so.
  if grep -qE 'ReadTimeout|ConnectionError|HTTP 5[0-9][0-9]|Cannot reach Jupyter API|RemoteDisconnected|Connection aborted' "$log"; then
    reason="$(grep -oE 'ReadTimeout|ConnectionError|HTTP 5[0-9][0-9]|Cannot reach Jupyter API|RemoteDisconnected|Connection aborted' "$log" | tail -1)"
    echo "kx: attempt $try/$TRIES failed on transport ($reason), retrying in $((try * 6))s" >&2
    sleep $((try * 6))
    continue
  fi
  cat "$log"
  echo "kx: failed with a non-transport error on attempt $try; not retrying" >&2
  exit 1
done

cat "$log" >&2
echo "kx: gave up after $TRIES attempts -- the proxy URL is probably expired. Reopen the" >&2
echo "    notebook's 'Kaggle Jupyter Server' panel and update server-connect.txt." >&2
exit 3
