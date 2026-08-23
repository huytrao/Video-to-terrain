#!/usr/bin/env bash
# Export the Kaggle proxy URLs from .env without letting the shell execute the file.
#
# `. ./.env` runs every line as a command. A key whose name contains a dash --
# CLAUDEFARE-API-KEY -- is not a valid shell identifier, so zsh treats the line as
# a command invocation and echoes the *value* in the "command not found" error.
# That put a live credential into a terminal log. Parsing only the two keys we
# need, with grep, cannot do that no matter what else the file contains.
#
# Usage:  eval "$(scripts/kaggle/kenv.sh)"   then use $KAGGLE_WITHOUT_INTERNET etc.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for key in KAGGLE_WITHOUT_INTERNET KAGGLE_HAS_INTERNET HF_TOKEN; do
  line="$(grep -m1 "^${key}=" "$ROOT/.env" 2>/dev/null || true)"
  [ -n "$line" ] && printf 'export %s\n' "$line"
done
