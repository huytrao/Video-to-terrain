#!/usr/bin/env python3
"""Push a file that already lives on one Kaggle notebook into another.

``kaggle_relay.py`` relays an internet URL into a no-internet notebook. This is
the sibling case: the bytes are already on notebook A — wheels built there, a
dataset it assembled — and need to reach notebook B, which cannot fetch
anything itself.

Same mechanism, same reason it works: B's Contents API is served by Kaggle's
platform, not by B's kernel, so B's "Internet: off" toggle has no say over an
authenticated write pushed at it from outside. Bytes never touch this machine.

    export KAGGLE_INTERNET='https://.../proxy'          # notebook A, holds the file
    export KAGGLE_WITHOUT_INTERNET='https://.../proxy'  # notebook B, receives it

    python scripts/kaggle/relay_push.py /kaggle/working/deps.tar.gz \\
        --dest /tmp/deps.tar.gz --extract-to /tmp/deps
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[2] / "kaggle-runtime-toolkit-agent-skills"
sys.path.insert(0, str(TOOLKIT))

from kaggle_executor import KaggleExecutor  # noqa: E402

# Runs entirely on notebook A. Placeholders are substituted by str.replace
# because the payload is dense with braces.
_DRIVER = r'''
import base64, os, threading, time
from urllib.parse import urlsplit, quote
import requests

B_PROXY    = "__B_PROXY__"
SRC        = "__SRC__"
DEST       = "__DEST__"
EXTRACT_TO = "__EXTRACT_TO__"
PARTS_DIR  = "relay_push_parts"
SUBCHUNK   = __SUBCHUNK__
WORKERS    = __WORKERS__

TOTAL = os.path.getsize(SRC)
print("source:", SRC, TOTAL, "bytes")

base = B_PROXY.rstrip("/")
sess = requests.Session()
sess.get(base + "/api", timeout=30, allow_redirects=True).raise_for_status()

def xsrf():
    for c in sess.cookies:
        if c.name == "_xsrf":
            return {"X-XSRFToken": c.value}
    return {}

def contents_url(path):
    return base + "/api/contents/" + quote(path, safe="/")

r = sess.get(contents_url(PARTS_DIR), params={"content": 0}, timeout=30)
if r.status_code == 404:
    sess.put(contents_url(PARTS_DIR), headers=xsrf(), json={"type": "directory"}, timeout=30).raise_for_status()

def put_part(path, blob, retries=6):
    b64 = base64.b64encode(blob).decode("ascii")
    last = None
    for attempt in range(retries):
        try:
            resp = sess.put(contents_url(path), headers=xsrf(),
                            json={"type": "file", "format": "base64", "content": b64},
                            timeout=180)
            if resp.status_code >= 400:
                raise RuntimeError("HTTP %d: %s" % (resp.status_code, resp.text[:200]))
            return
        except Exception as e:
            last = e
            wait = min(2 ** attempt, 20)
            print("  [PUT retry %d/%d %s after %s, sleep %ds]" % (attempt+1, retries, path, type(e).__name__, wait))
            time.sleep(wait)
    raise last

lock = threading.Lock()
sent = 0
errors = []
parts = {i: [] for i in range(WORKERS)}

def worker(idx, start, end):
    global sent
    seq = 0
    with open(SRC, "rb") as f:
        f.seek(start)
        offset = start
        while offset < end:
            blob = f.read(min(SUBCHUNK, end - offset))
            if not blob:
                break
            path = "%s/part_%02d_%05d.bin" % (PARTS_DIR, idx, seq)
            try:
                put_part(path, blob)
            except Exception as e:
                errors.append("worker %d failed at %d: %s" % (idx, offset, e))
                return
            parts[idx].append(path)
            seq += 1
            offset += len(blob)
            with lock:
                sent += len(blob)
    print("worker %d: done, %d parts" % (idx, seq))

bounds = [round(TOTAL * i / WORKERS) for i in range(WORKERS + 1)]
threads = [threading.Thread(target=worker, args=(i, bounds[i], bounds[i+1])) for i in range(WORKERS)]
t0 = time.time()
for t in threads:
    t.start()

last_bucket = -1
while any(t.is_alive() for t in threads):
    time.sleep(5)
    with lock:
        s = sent
    pct = s / TOTAL * 100 if TOTAL else 0
    bucket = int(pct // 5)
    if bucket != last_bucket:
        last_bucket = bucket
        el = time.time() - t0
        print("%.1f MB / %.1f MB (%.1f%%)  %.2f MB/s  [%ds]" % (s/1e6, TOTAL/1e6, pct, s/1e6/(el+1e-9), el))

for t in threads:
    t.join()
if errors:
    print("FAILED:", errors)
    raise SystemExit(1)

ordered = []
for i in range(WORKERS):
    ordered.extend(parts[i])
print("uploaded %d parts, assembling on B..." % len(ordered))

import json, uuid, subprocess, sys as _sys
subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "websocket-client"], check=True)
import websocket

r = sess.post(base + "/api/kernels", headers=xsrf(), json={"name": "python3"}, timeout=30)
r.raise_for_status()
kid = r.json()["id"]
ws_url = base + "/api/kernels/%s/channels" % kid
ws_url = "wss://" + ws_url[len("https://"):] if ws_url.startswith("https://") else "ws://" + ws_url[len("http://"):]
origin = "%s://%s" % (urlsplit(base).scheme, urlsplit(base).netloc)
ws = websocket.create_connection(
    ws_url,
    cookie="; ".join("%s=%s" % (c.name, c.value) for c in sess.cookies),
    origin=origin, timeout=30)

def exec_on(code, timeout=1800):
    mid = str(uuid.uuid4())
    ws.send(json.dumps({
        "header": {"msg_id": mid, "username": "relay", "session": str(uuid.uuid4()),
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {}, "channel": "shell",
        "content": {"code": code, "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False},
    }))
    out = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        ws.settimeout(5.0)
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if raw is None:
            break
        msg = json.loads(raw)
        if msg.get("parent_header", {}).get("msg_id") != mid:
            continue
        mt = msg.get("header", {}).get("msg_type", "")
        c = msg.get("content", {})
        if mt == "stream":
            out.append(c.get("text", ""))
        elif mt == "error":
            out.append("ERROR: " + c.get("evalue", "") + "\n" + "\n".join(c.get("traceback", [])))
        elif mt == "status" and c.get("execution_state") == "idle":
            break
    return "".join(out)

assemble = """
import os, shutil, tarfile
parts = {parts!r}
dest = {dest!r}
expected = {total}
os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)

missing = [p for p in parts if not os.path.exists(os.path.join('/kaggle/working', p))]
if missing:
    raise SystemExit('MISSING %d part(s): %s' % (len(missing), missing[:5]))

with open(dest, 'wb') as out:
    for p in parts:
        with open(os.path.join('/kaggle/working', p), 'rb') as f:
            shutil.copyfileobj(f, out, length=16*1024*1024)

got = os.path.getsize(dest)
print('assembled', got, 'expected', expected)
if got != expected:
    raise SystemExit('SIZE MISMATCH: short by %d bytes across %d parts' % (expected - got, len(parts)))

shutil.rmtree(os.path.join('/kaggle/working', {parts_dir!r}), ignore_errors=True)
extract_to = {extract!r}
if extract_to:
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(dest) as tf:
        tf.extractall(extract_to)
    print('extracted to', extract_to, '->', len(os.listdir(extract_to)), 'entries')
print('RELAY_PUSH_OK')
""".format(parts=ordered, dest=DEST, total=TOTAL, parts_dir=PARTS_DIR, extract=EXTRACT_TO or None)

print(exec_on(assemble))
ws.close()
print("DONE in %ds" % (time.time() - t0))
'''


def _env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v and name in ("KAGGLE_INTERNET", "KAGGLE_HAS_INTERNET"):
        v = os.environ.get("KAGGLE_INTERNET", "") or os.environ.get("KAGGLE_HAS_INTERNET", "")
    if not v and name in ("KAGGLE_WITHOUT_INTERNET", "KAGGLE_PROXY_URL"):
        v = os.environ.get("KAGGLE_WITHOUT_INTERNET", "") or os.environ.get("KAGGLE_PROXY_URL", "")
    if not v:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        if env_file.exists():
            import re
            text = env_file.read_text()
            m = re.search(rf"^{name}=(.*)$", text, re.M)
            if m:
                v = m.group(1).strip()
            elif name in ("KAGGLE_INTERNET", "KAGGLE_HAS_INTERNET"):
                m2 = re.search(r"^KAGGLE_HAS_INTERNET=(.*)$", text, re.M) or re.search(r"^KAGGLE_INTERNET=(.*)$", text, re.M)
                if m2:
                    v = m2.group(1).strip()
            elif name in ("KAGGLE_WITHOUT_INTERNET", "KAGGLE_PROXY_URL"):
                m2 = re.search(r"^KAGGLE_WITHOUT_INTERNET=(.*)$", text, re.M) or re.search(r"^KAGGLE_PROXY_URL=(.*)$", text, re.M)
                if m2:
                    v = m2.group(1).strip()
    if not v:
        sys.exit(f"ERROR: {name} not set in environment or .env.")
    return v


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Push a file from Kaggle notebook A to notebook B.")
    p.add_argument("src", help="absolute path of the file on notebook A")
    p.add_argument("--dest", required=True, help="absolute destination path on notebook B")
    p.add_argument("--extract-to", default="", help="if set, untar the file into this directory on B")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--subchunk", type=int, default=8,
        help="MB per Contents-API PUT. Kaggle silently truncates PUTs around 24 MB, "
             "and this reads exact-size blocks, so keep it well under.",
    )
    p.add_argument("--timeout", type=int, default=9000)
    args = p.parse_args(argv)

    driver = (
        _DRIVER
        .replace("__B_PROXY__", _env("KAGGLE_WITHOUT_INTERNET"))
        .replace("__SRC__", args.src)
        .replace("__DEST__", args.dest)
        .replace("__EXTRACT_TO__", args.extract_to)
        .replace("__SUBCHUNK__", str(args.subchunk * 1024 * 1024))
        .replace("__WORKERS__", str(args.workers))
    )

    tmp = Path(f"/tmp/_relay_push_{os.getpid()}.py")
    tmp.write_text(driver)
    try:
        ex = KaggleExecutor(_env("KAGGLE_INTERNET"))
        remote = f"_relay/push_{os.getpid()}.py"
        ex.upload_file(str(tmp), remote)
        print(f"Pushing {args.src} -> {args.dest}\n")
        output = ex.execute(
            ex.get_or_create_kernel(),
            f"exec(open('/kaggle/working/{remote}').read())",
            timeout=args.timeout,
        ) or ""
        if "RELAY_PUSH_OK" not in output:
            sys.exit("relay_push failed: destination was not verified on the receiving kernel")
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
