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
#   scripts/kaggle/kx.sh <kaggle_executor subcommand and args...>
#
# Reads the proxy URL from $KAGGLE_PROXY_URL, or from the first line-like blob in
# server-connect.txt if that is unset. Retries $KX_TRIES times (default 6) with
# linear backoff, and treats "the kernel is busy" as a reason to wait rather
# than fail.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/kaggle/ -> scripts/ -> repo root: two levels, not one.
ROOT="$(dirname "$(dirname "$HERE")")"
EXEC="$ROOT/kaggle-runtime-toolkit-agent-skills/kaggle_executor.py"
TRIES="${KX_TRIES:-6}"

if [ -z "${KAGGLE_PROXY_URL:-}" ]; then
  # .env first, because that is the file the expiry message tells the operator to
  # update -- and for a long time this script did not read it, so a freshly
  # pasted URL sat in .env while every call retried the dead one in
  # server-connect.txt and reported "expired". Read via kenv.sh so the file is
  # parsed rather than sourced; sourcing it once echoed a live credential.
  #
  # KX_KERNEL selects which kernel to talk to: the RUN kernel has no internet
  # (default), the BUILD/FETCH kernels do.
  case "${KX_KERNEL:-without}" in
    without|run)   want=KAGGLE_WITHOUT_INTERNET ;;
    has|build|fetch) want=KAGGLE_HAS_INTERNET ;;
    *) echo "kx: KX_KERNEL must be 'without' or 'has', got '${KX_KERNEL}'" >&2; exit 2 ;;
  esac
  KAGGLE_PROXY_URL="$(eval "$("$HERE/kenv.sh")"; printf '%s' "${!want:-${KAGGLE_PROXY_URL:-}}")"
  export KAGGLE_PROXY_URL
fi
if [ -z "${KAGGLE_PROXY_URL:-}" ]; then
  # The URL is one long token with no spaces; pick the longest such line.
  KAGGLE_PROXY_URL="$(grep -o 'https://[^[:space:]]*proxy' "$ROOT/server-connect.txt" 2>/dev/null \
                      | awk '{ print length, $0 }' | sort -rn | head -1 | cut -d' ' -f2-)"
  export KAGGLE_PROXY_URL
fi
if [ -z "${KAGGLE_PROXY_URL:-}" ]; then
  echo "kx: no KAGGLE_PROXY_URL; set it, or put KAGGLE_PROXY_URL (or ${want:-KAGGLE_WITHOUT_INTERNET}) in $ROOT/.env" >&2
  exit 2
fi

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

for try in $(seq 1 "$TRIES"); do
  if python3 "$EXEC" "$@" >"$log" 2>&1; then
    cat "$log"
    exit 0
  fi
  # A 404 on the handshake means the JWT in the URL has expired -- no number of
  # retries revives it, and waiting 126 s to be told so wastes the operator's
  # time. It has to be tested *before* the transport pattern below, because the
  # executor reports it as "Cannot reach Jupyter API: HTTP 404" and would
  # otherwise match as transient.
  if grep -q 'Cannot reach Jupyter API: HTTP 404' "$log"; then
    echo "kx: proxy URL is expired (HTTP 404 on /api) -- not a transient failure." >&2
    echo "    Reopen the notebook's 'Kaggle Jupyter Server' panel and update the URL" >&2
    echo "    in .env (KAGGLE_WITHOUT_INTERNET / KAGGLE_HAS_INTERNET)." >&2
    exit 3
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
