#!/usr/bin/env bash
# Run a stage script on the RUN kernel as a detached process, and follow its log.
#
# Why detached. `kaggle_executor.py exec` holds a websocket for the duration of the
# cell, and that connection drops for no particular reason: this session already
# lost an eleven-minute wait to a hang while the kernel itself sat idle and the
# work had finished. A stage that takes two hours cannot be tied to that. Starting
# it with `start_new_session=True` means the job is a child of the kernel, not of
# the websocket, so a dropped connection costs nothing and the log is the interface.
#
# It also sidesteps a second problem: inside the IPython kernel, `open3d/__init__`
# detects Jupyter and calls `enable_webrtc()` on a half-initialised module, which
# fails. A detached subprocess is a plain interpreter and does not.
#
# Usage:
#   scripts/kaggle/kstage.sh <local-script.py> [start|follow|status|stop] [env KEY=VAL ...]
#
# `start` uploads and launches; `follow` tails until the process exits; `status`
# prints one line; `stop` kills it. `start` implies `follow`.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${1:?usage: kstage.sh <script.py> [start|follow|status|stop] [KEY=VAL ...]}"
ACTION="${2:-start}"
shift 2 2>/dev/null || shift 1
BASE="$(basename "$SCRIPT" .py)"
REMOTE="/kaggle/working/_run/${BASE}.py"
LOG="/tmp/run/logs/${BASE}.log"
PIDF="/tmp/run/logs/${BASE}.pid"

# Extra KEY=VAL arguments become the environment of the detached process. Passed as
# a dict literal rather than through the shell so a value with spaces survives.
ENVJSON="{"
for kv in "$@"; do
  k="${kv%%=*}"; v="${kv#*=}"
  ENVJSON="${ENVJSON}\"${k}\": \"${v}\", "
done
ENVJSON="${ENVJSON}}"

kx() { KX_KERNEL=run "$HERE/kx.sh" "$@"; }

case "$ACTION" in
  start)
    kx upload "$SCRIPT" "_run/${BASE}.py" >/dev/null || exit 1
    kx exec --timeout 300 "
import os, subprocess, sys
os.makedirs('/tmp/run/logs', exist_ok=True)
# Refuse to start a second copy: two GPU stages on one card is an OOM, and the
# first one's log would look like it simply stalled.
if os.path.exists('${PIDF}'):
    pid = open('${PIDF}').read().strip()
    if pid and os.path.exists(f'/proc/{pid}'):
        raise SystemExit(f'ALREADY_RUNNING pid={pid} — use follow, or stop it first')
env = dict(os.environ, PYTHONUNBUFFERED='1', **${ENVJSON})
p = subprocess.Popen([sys.executable, '${REMOTE}'],
                     stdout=open('${LOG}', 'w'), stderr=subprocess.STDOUT,
                     cwd='/kaggle/working', env=env, start_new_session=True)
open('${PIDF}', 'w').write(str(p.pid))
print('STARTED pid', p.pid, 'log ${LOG}')
" || exit 1
    exec "$0" "$SCRIPT" follow
    ;;

  follow)
    # Poll rather than stream: each call is short, so a dropped websocket costs one
    # poll instead of the whole wait. Only new bytes are fetched.
    off=0
    while :; do
      out="$(kx exec --timeout 180 "
import os
log, pidf = '${LOG}', '${PIDF}'
size = os.path.getsize(log) if os.path.exists(log) else 0
if size > ${off}:
    with open(log, 'rb') as fh:
        fh.seek(${off})
        print(fh.read().decode('utf-8', 'replace'), end='')
print(f'__OFFSET__{size}')
pid = open(pidf).read().strip() if os.path.exists(pidf) else ''
print('__ALIVE__' + ('1' if pid and os.path.exists(f'/proc/{pid}') else '0'))
" 2>&1)"
      body="$(printf '%s' "$out" | sed '/^__OFFSET__/d; /^__ALIVE__/d; /^↺ Reusing kernel/d; /^─\{10,\}$/d')"
      [ -n "$(printf '%s' "$body" | tr -d '[:space:]')" ] && printf '%s\n' "$body"
      new="$(printf '%s' "$out" | sed -n 's/^__OFFSET__//p' | tail -1)"
      [ -n "$new" ] && off="$new"
      alive="$(printf '%s' "$out" | sed -n 's/^__ALIVE__//p' | tail -1)"
      [ "$alive" = "0" ] && break
      sleep 20
    done
    echo "--- ${BASE} exited ---"
    ;;

  status)
    kx exec --timeout 120 "
import os
for p, what in (('${LOG}','log'), ('${PIDF}','pid')):
    print(what, p, os.path.getsize(p) if os.path.exists(p) else 'absent')
pid = open('${PIDF}').read().strip() if os.path.exists('${PIDF}') else ''
print('alive', bool(pid) and os.path.exists(f'/proc/{pid}'), 'pid', pid)
if os.path.exists('${LOG}'):
    print('--- tail ---'); print(''.join(open('${LOG}').readlines()[-25:]))
"
    ;;

  stop)
    kx exec --timeout 120 "
import os, signal
pid = open('${PIDF}').read().strip() if os.path.exists('${PIDF}') else ''
if pid and os.path.exists(f'/proc/{pid}'):
    os.killpg(os.getpgid(int(pid)), signal.SIGTERM); print('killed', pid)
else:
    print('not running')
os.path.exists('${PIDF}') and os.remove('${PIDF}')
"
    ;;

  *) echo "kstage: unknown action '$ACTION'" >&2; exit 2 ;;
esac
