#!/usr/bin/env bash
# Run a read-only audit pass on NVIDIA Nemotron-3 Ultra -- the model the
# `claude-nv` shell function selects.
#
# Why a separate process rather than the Agent tool: model routing in Claude Code
# is process-wide environment (ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN /
# ANTHROPIC_MODEL), and the in-process Agent tool accepts only the first-party
# tiers (sonnet/opus/haiku/fable). So a subagent of a session talking to Anthropic
# cannot be pointed at the local proxy. A separate headless `claude -p` can, and
# that is the whole trick here.
#
# The proxy key is read with grep and never `source`d. Sourcing an env file
# *executes* every line; a key whose name contains a character that is illegal in
# a shell identifier makes the shell treat the line as a command and echo the
# value in its "command not found" message. That put a live credential into a
# terminal log once already in this project (docs/reference/bugs-found.md, C7).
# The file currently holds only identifier-safe names, but this script must not
# depend on that staying true.
#
# Usage:
#   scripts/tools/nv_audit.sh <prompt-file> [output-file]
#   scripts/tools/nv_audit.sh -            # read the prompt from stdin
#
# Tools are restricted to Read/Grep/Glob: an auditor that can edit is not an
# auditor. No Bash either, so it cannot reach the Kaggle kernels or read .env.
set -uo pipefail

PROXY_URL="${NV_PROXY_URL:-http://127.0.0.1:4011}"
KEY_FILE="${NV_KEY_FILE:-$HOME/.codex/litellm_nvidia.env}"
MODEL="${NV_MODEL:-nv-claude-nemotron}"
CLI="${CLAUDE_NV_CLI:-$HOME/.local/bin/claude}"

prompt_file="${1:-}"
out_file="${2:-}"
if [ -z "$prompt_file" ]; then
  echo "usage: $0 <prompt-file|-> [output-file]" >&2
  exit 2
fi

# Reachability, not authorisation: any HTTP answer proves the proxy is up.
if ! curl -fsS --max-time 5 "$PROXY_URL/health/liveliness" >/dev/null 2>&1; then
  echo "nv_audit: proxy at $PROXY_URL is not answering; starting it..." >&2
  if [ -x "$HOME/.local/bin/nvclaudeproxy" ]; then
    "$HOME/.local/bin/nvclaudeproxy" || { echo "nv_audit: could not start proxy." >&2; exit 1; }
  else
    echo "nv_audit: no nvclaudeproxy launcher found. Start it, then re-run." >&2
    exit 1
  fi
fi

if [ ! -r "$KEY_FILE" ]; then
  echo "nv_audit: cannot read $KEY_FILE -- the proxy key lives there." >&2
  exit 1
fi
# cut -f2- keeps '=' inside the value intact; -m1 takes the first definition.
KEY="$(grep -m1 '^LITELLM_MASTER_KEY=' "$KEY_FILE" | cut -d= -f2-)"
if [ -z "$KEY" ]; then
  echo "nv_audit: LITELLM_MASTER_KEY is absent from $KEY_FILE." >&2
  exit 1
fi

if [ "$prompt_file" = "-" ]; then
  prompt="$(cat)"
else
  [ -r "$prompt_file" ] || { echo "nv_audit: cannot read $prompt_file" >&2; exit 1; }
  prompt="$(cat "$prompt_file")"
fi

echo "nv_audit: model=$MODEL  proxy=$PROXY_URL  tools=Read,Grep,Glob" >&2

# ANTHROPIC_API_KEY is emptied deliberately: left set, Claude Code would prefer
# the Anthropic subscription credential and the run would silently NOT be on
# nemotron -- an audit attributed to the wrong model is worse than no audit.
run() {
  ANTHROPIC_BASE_URL="$PROXY_URL" \
  ANTHROPIC_AUTH_TOKEN="$KEY" \
  ANTHROPIC_API_KEY="" \
  ANTHROPIC_MODEL="$MODEL" \
  ANTHROPIC_SMALL_FAST_MODEL="$MODEL" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS="${NV_CTX:-1000000}" \
  MAX_THINKING_TOKENS="${NV_THINKING_TOKENS:-16384}" \
  command env -u CLAUDE_CODE_DISABLE_THINKING -u DISABLE_INTERLEAVED_THINKING \
    "$CLI" -p "$prompt" \
      --model "$MODEL" \
      --allowed-tools Read Grep Glob \
      --permission-mode acceptEdits
}

if [ -n "$out_file" ]; then
  run >"$out_file" 2>"${out_file}.err"
  rc=$?
  # An empty transcript with rc=0 means the model returned nothing -- report that
  # rather than letting a caller read silence as a clean bill of health.
  if [ "$rc" -eq 0 ] && [ ! -s "$out_file" ]; then
    echo "nv_audit: exit 0 but empty output; stderr follows" >&2
    tail -20 "${out_file}.err" >&2
    exit 1
  fi
  echo "nv_audit: wrote $out_file ($(wc -c <"$out_file" | tr -d ' ') bytes, rc=$rc)" >&2
  exit "$rc"
fi
run
