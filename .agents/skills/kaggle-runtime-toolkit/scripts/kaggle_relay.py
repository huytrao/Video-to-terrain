#!/usr/bin/env python3
"""
kaggle_relay.py  —  kaggle-runtime-agent-skills
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pull a file from anywhere on the internet straight into a Kaggle notebook
that has *no* internet access — without the bytes ever passing through
your local machine.

Two running Kaggle notebooks are involved:
  • one with internet   — does the actual HTTP download
  • one without internet — receives the file

The file streams HTTP source -> notebook-with-internet -> notebook-without-
internet, in parallel byte-range workers, written on the target side via its
own Jupyter *Contents API* (no kernel execute_request round trips, which is
~40x slower for bulk transfer — see README). Your machine only ever submits
one small driver script to the internet-enabled notebook and watches short
progress lines; it never sees the file bytes.

Why a no-internet notebook can receive a push at all: the Contents API is
served by Kaggle's platform, not by code running inside the kernel. The
"Internet" toggle only blocks network calls the kernel's *own* process
initiates — it has no say over the platform accepting an authenticated file
write from another notebook.

Install:
    pip install requests websocket-client

Set env — two *different* running Kaggle notebooks, proxy URL from each
notebook's "Kaggle Jupyter Server" panel (⋮ More settings):
    export KAGGLE_INTERNET='https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/.../proxy'          # has internet
    export KAGGLE_WITHOUT_INTERNET='https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/.../proxy'  # internet OFF

Examples:
    python kaggle_relay.py https://huggingface.co/1038lab/sam3/resolve/main/sam3.safetensors

    python kaggle_relay.py https://huggingface.co/1038lab/sam3/resolve/main/sam3.safetensors \\
        --dest /kaggle/working/sam3.safetensors --workers 4 --subchunk 24
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_executor import KaggleExecutor  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Driver script — injected as a string, runs entirely on the internet-enabled
# notebook. __TOKEN__ placeholders are substituted before upload; the payload
# is otherwise plain Python (heavy on braces), so plain string.replace() is
# used instead of str.format().
# ─────────────────────────────────────────────────────────────────────────────

_DRIVER = r'''
import base64, threading, time
from urllib.parse import urlsplit, quote
import requests

B_PROXY    = "__B_PROXY__"
SRC_URL    = "__SRC_URL__"
DEST       = "__DEST__"
PARTS_DIR  = "relay_parts"
SUBCHUNK   = __SUBCHUNK__
WORKERS    = __WORKERS__

base = B_PROXY.rstrip("/")

sess = requests.Session()
r = sess.get(base + "/api", timeout=30, allow_redirects=True)
r.raise_for_status()

def xsrf():
    for c in sess.cookies:
        if c.name == "_xsrf":
            return {"X-XSRFToken": c.value}
    return {}

def contents_url(path):
    return base + "/api/contents/" + quote(path, safe="/")

def ensure_dir(remote_dir):
    r = sess.get(contents_url(remote_dir), params={"content": 0}, timeout=30)
    if r.status_code == 404:
        rr = sess.put(contents_url(remote_dir), headers=xsrf(), json={"type": "directory"}, timeout=30)
        rr.raise_for_status()

def put_part(path, content_b64, timeout=120, retries=6):
    url = contents_url(path)
    last_exc = None
    for attempt in range(retries):
        try:
            r = sess.put(url, headers=xsrf(),
                         json={"type": "file", "format": "base64", "content": content_b64},
                         timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            return
        except Exception as e:
            last_exc = e
            wait = min(2 ** attempt, 20)
            print(f"  [PUT retry {attempt+1}/{retries} for {path} after {type(e).__name__}, sleep {wait}s]")
            time.sleep(wait)
    raise last_exc

def probe_size(url):
    r = requests.head(url, allow_redirects=True, timeout=30)
    cl = r.headers.get("Content-Length")
    if cl:
        return int(cl)
    r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=30)
    cr = r.headers.get("Content-Range")
    r.close()
    if cr and "/" in cr:
        return int(cr.rsplit("/", 1)[1])
    raise RuntimeError("Could not determine remote file size (no Content-Length or Content-Range)")

TOTAL_SIZE = probe_size(SRC_URL)
print("source size:", TOTAL_SIZE, "bytes")

ensure_dir(PARTS_DIR)

progress_lock = threading.Lock()
sent_total = 0
errors = []
worker_parts = {i: [] for i in range(WORKERS)}

def worker(idx, start, end):
    global sent_total
    offset = start
    seq = 0
    hf_attempt = 0
    HF_RETRIES = 8
    while offset < end:
        try:
            resp = requests.get(SRC_URL, headers={"Range": f"bytes={offset}-{end-1}"}, stream=True, timeout=60)
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=SUBCHUNK):
                if not chunk:
                    continue
                path = f"{PARTS_DIR}/part_{idx:02d}_{seq:05d}.bin"
                b64 = base64.b64encode(chunk).decode("ascii")
                put_part(path, b64)
                worker_parts[idx].append(path)
                seq += 1
                offset += len(chunk)
                hf_attempt = 0
                with progress_lock:
                    sent_total += len(chunk)
        except (requests.exceptions.RequestException, ConnectionError, OSError) as e:
            hf_attempt += 1
            if hf_attempt > HF_RETRIES:
                errors.append(f"worker {idx}: gave up after {HF_RETRIES} source retries @ offset {offset}: {e}")
                print(errors[-1])
                return
            wait = min(2 ** hf_attempt, 30)
            print(f"worker {idx}: source stream broke @ {offset:,} ({type(e).__name__}), retry {hf_attempt}/{HF_RETRIES} in {wait}s")
            time.sleep(wait)
    print(f"worker {idx}: done, {seq} parts, {offset - start:,} bytes")

bounds = [round(TOTAL_SIZE * i / WORKERS) for i in range(WORKERS + 1)]
threads = [threading.Thread(target=worker, args=(i, bounds[i], bounds[i+1])) for i in range(WORKERS)]
t0 = time.time()
for t in threads:
    t.start()

last_pct_bucket = -1
while any(t.is_alive() for t in threads):
    time.sleep(5)
    with progress_lock:
        s = sent_total
    pct = s / TOTAL_SIZE * 100 if TOTAL_SIZE else 0
    bucket = int(pct // 5)
    if bucket != last_pct_bucket:
        last_pct_bucket = bucket
        elapsed = time.time() - t0
        mbps = s / 1e6 / (elapsed + 1e-9)
        print(f"{s/1e6:.1f} MB / {TOTAL_SIZE/1e6:.1f} MB  ({pct:.1f}%)  {mbps:.2f} MB/s aggregate  [{elapsed:.0f}s elapsed]")

for t in threads:
    t.join()

if errors:
    print("FAILED workers:", errors)
    raise SystemExit(1)

all_parts = []
for i in range(WORKERS):
    all_parts.extend(worker_parts[i])
print(f"Uploaded {len(all_parts)} parts. Assembling on the target notebook via one kernel exec...")

import json, uuid
import websocket
import subprocess, sys as _sys
subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "websocket-client"], check=True)

r = sess.post(base + "/api/kernels", headers=xsrf(), json={"name": "python3"}, timeout=30)
r.raise_for_status()
kid = r.json()["id"]
ws_url = base + f"/api/kernels/{kid}/channels"
ws_url = "wss://" + ws_url[len("https://"):] if ws_url.startswith("https://") else "ws://" + ws_url[len("http://"):]
origin = f"{urlsplit(base).scheme}://{urlsplit(base).netloc}"
cookie_header = "; ".join(f"{c.name}={c.value}" for c in sess.cookies)
ws = websocket.create_connection(ws_url, cookie=cookie_header, origin=origin, timeout=30)

def exec_on(code, timeout=900):
    msg_id = str(uuid.uuid4())
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "relay", "session": str(uuid.uuid4()),
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
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        mtype = msg.get("header", {}).get("msg_type", "")
        content = msg.get("content", {})
        if mtype == "stream":
            out.append(content.get("text", ""))
        elif mtype == "error":
            out.append("ERROR: " + content.get("evalue", "") + "\n" + "\n".join(content.get("traceback", [])))
        elif mtype == "status" and content.get("execution_state") == "idle":
            break
    return "".join(out)

assemble_code = f"""
import os, shutil
parts = {all_parts!r}
with open({DEST!r}, 'wb') as out:
    for p in parts:
        full = os.path.join('/kaggle/working', p)
        with open(full, 'rb') as f:
            shutil.copyfileobj(f, out, length=16*1024*1024)
size = os.path.getsize({DEST!r})
print('assembled size:', size, 'expected:', {TOTAL_SIZE})
shutil.rmtree(os.path.join('/kaggle/working', {PARTS_DIR!r}), ignore_errors=True)
print('parts dir removed')
"""
print(exec_on(assemble_code, timeout=900))
ws.close()
print(f"DONE in {time.time()-t0:.0f}s. Relay finished, local machine never touched the file bytes.")
'''


from kaggle_executor import KaggleExecutor, resolve_proxy_url  # noqa: E402


def _resolve_relay_proxy(name: str, fallbacks: list[str]) -> str:
    # Check env directly
    for k in [name] + fallbacks:
        v = resolve_proxy_url(env_var=k)
        if v:
            return v
    sys.exit(
        f"ERROR: {name} not set and could not be resolved from .env.\n\n"
        f"  export {name}='https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/.../proxy'\n"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="kaggle_relay",
        description="Pull a file into a no-internet Kaggle notebook via another notebook that has internet.",
    )
    p.add_argument("url", help="Source HTTPS URL to download (must support Range requests)")
    p.add_argument("--dest", default=None,
                    help="Destination path on the no-internet notebook (default: /kaggle/working/<basename of URL>)")
    p.add_argument("--src-proxy", default=None, help="Explicit internet-enabled proxy URL")
    p.add_argument("--dst-proxy", default=None, help="Explicit target proxy URL")
    p.add_argument("--workers", type=int, default=4, help="Parallel byte-range workers (default: 4)")
    p.add_argument("--subchunk", type=int, default=24, help="MB per Contents-API PUT (default: 24)")
    p.add_argument("--timeout", type=int, default=9000, help="Overall timeout in seconds (default: 9000)")
    args = p.parse_args(argv)

    src_proxy = args.src_proxy or _resolve_relay_proxy("KAGGLE_INTERNET", ["KAGGLE_HAS_INTERNET", "KAGGLE_PROXY_HAS_INTERNET_NO_GPU"])
    dst_proxy = args.dst_proxy or _resolve_relay_proxy("KAGGLE_WITHOUT_INTERNET", ["KAGGLE_PROXY_RTX6000_NO_INTERNET", "KAGGLE_PROXY_URL"])

    dest = args.dest or f"/kaggle/working/{Path(urlsplit(args.url).path).name}"

    driver = (
        _DRIVER
        .replace("__B_PROXY__", dst_proxy)
        .replace("__SRC_URL__", args.url)
        .replace("__DEST__", dest)
        .replace("__SUBCHUNK__", str(args.subchunk * 1024 * 1024))
        .replace("__WORKERS__", str(args.workers))
    )

    tmp = Path(f"/tmp/_kaggle_relay_driver_{os.getpid()}.py")
    tmp.write_text(driver)
    try:
        ex = KaggleExecutor(src_proxy)
        remote = f"_relay/driver_{os.getpid()}.py"
        ex.upload_file(str(tmp), remote)
        kid = ex.get_or_create_kernel()
        print(f"Relaying {args.url}\n  via   : notebook with internet (KAGGLE_INTERNET)\n  into  : {dest}  on notebook without internet (KAGGLE_WITHOUT_INTERNET)\n")
        ex.execute(kid, f"exec(open('/kaggle/working/{remote}').read())", timeout=args.timeout)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
