#!/usr/bin/env python3
"""
kaggle_executor.py  —  kaggle-runtime-agent-skills
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Execute arbitrary Python code on a live Kaggle GPU kernel via the
Jupyter WebSocket message protocol. Streams stdout in real-time,
downloads results when done.

No Kaggle API key required — just the proxy URL from a running notebook.

Install:
    pip install requests websocket-client

Set env:
    export KAGGLE_PROXY_URL='https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/.../proxy'

Examples:
    # Check GPU
    python kaggle_executor.py exec "import torch; print(torch.cuda.get_device_name(0))"

    # Run a local script (uploads + executes)
    python kaggle_executor.py run train.py

    # List active kernels
    python kaggle_executor.py kernels

    # Download a whole output folder
    python kaggle_executor.py download-dir output ./local_output --ext .glb .obj .pt
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests
import websocket


# ─────────────────────────────────────────────────────────────────────────────
# Core executor
# ─────────────────────────────────────────────────────────────────────────────

class KaggleExecutor:
    """
    Manages Jupyter kernels on a Kaggle proxy and executes code cells,
    streaming all output back to the local terminal in real-time.
    """

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self.http = requests.Session()
        parts = urlsplit(self.base)
        self.origin = f"{parts.scheme}://{parts.netloc}"
        r = self.http.get(self.url("/api"), timeout=30, allow_redirects=True)
        if r.status_code >= 400:
            raise RuntimeError(
                f"Cannot reach Jupyter API: HTTP {r.status_code}\n{r.text[:500]}"
            )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def url(self, path: str) -> str:
        return self.base + "/" + path.lstrip("/")

    def ws_url(self, path: str) -> str:
        u = self.url(path)
        return ("wss://" if u.startswith("https://") else "ws://") + u.split("://", 1)[1]

    def _xsrf(self) -> dict:
        for c in self.http.cookies:
            if c.name == "_xsrf":
                return {"X-XSRFToken": c.value}
        return {}

    def _cookies(self) -> str:
        return "; ".join(f"{c.name}={c.value}" for c in self.http.cookies)

    def _check(self, r: requests.Response) -> requests.Response:
        if r.status_code >= 400:
            raise RuntimeError(
                f"HTTP {r.status_code} {r.request.method} {r.url}\n{r.text[:2500]}"
            )
        return r

    # ── Kernel management ────────────────────────────────────────────────────

    def list_kernels(self) -> list:
        r = self._check(self.http.get(self.url("/api/kernels"), timeout=30))
        kernels = r.json()
        if not kernels:
            print("  (no active kernels)")
        for k in kernels:
            print(f"  {k['id']}  state={k['execution_state']}  name={k.get('name','')}")
        return kernels

    def get_or_create_kernel(self, name: str = "python3") -> str:
        """Return an idle kernel ID, creating one if necessary."""
        r = self._check(self.http.get(self.url("/api/kernels"), timeout=30))
        for k in r.json():
            if k.get("execution_state") in ("idle", "starting"):
                print(f"↺ Reusing kernel: {k['id']}")
                return k["id"]
        r = self._check(
            self.http.post(self.url("/api/kernels"), headers=self._xsrf(),
                           json={"name": name}, timeout=30)
        )
        kid = r.json()["id"]
        print(f"✚ New kernel: {kid}")
        return kid

    # ── Code execution ───────────────────────────────────────────────────────

    def execute(
        self,
        kernel_id: str,
        code: str,
        timeout: int = 1800,
        silent: bool = False,
    ) -> str:
        """
        Send an execute_request and stream all output until the kernel goes idle.

        Args:
            kernel_id: Jupyter kernel ID.
            code:      Python source to execute.
            timeout:   Max seconds to wait (default 30 min).
            silent:    Suppress local echo of output.

        Returns:
            Combined stdout/stderr string.
        """
        ws = websocket.create_connection(
            self.ws_url(f"/api/kernels/{kernel_id}/channels"),
            cookie=self._cookies() or None,
            origin=self.origin,
            timeout=30,
        )

        msg_id = str(uuid.uuid4())
        ws.send(json.dumps({
            "header": {
                "msg_id": msg_id,
                "username": "kaggle-runtime-agent-skills",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
            "channel": "shell",
        }))

        if not silent:
            print(f"\n{'─'*60}")

        collected: list[str] = []
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                ws.settimeout(5.0)
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:
                print(f"\n[WS error: {exc}]", file=sys.stderr)
                break

            if raw is None:
                break

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            mtype = msg.get("header", {}).get("msg_type", "")
            content = msg.get("content", {})

            if mtype == "stream":
                text = content.get("text", "")
                if not silent:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                collected.append(text)

            elif mtype in ("display_data", "execute_result"):
                text = content.get("data", {}).get("text/plain", "")
                if text:
                    if not silent:
                        print(text)
                    collected.append(text)

            elif mtype == "error":
                err = f"\n❌ {content.get('ename')}: {content.get('evalue')}\n"
                tb  = "\n".join(content.get("traceback", []))
                sys.stderr.write(err + tb + "\n")
                collected.append(err)

            elif mtype == "status" and content.get("execution_state") == "idle":
                break

        ws.close()
        if not silent:
            print(f"{'─'*60}\n")
        return "".join(collected)

    # ── File helpers ─────────────────────────────────────────────────────────

    def ensure_dir(self, remote_dir: str) -> None:
        remote_dir = remote_dir.strip("/")
        current: list[str] = []
        for part in remote_dir.split("/"):
            current.append(part)
            path = "/".join(current)
            url = self.url("/api/contents/" + quote(path, safe="/"))
            r = self.http.get(url, params={"content": 0}, timeout=30)
            if r.status_code == 404:
                self._check(
                    self.http.put(url, headers=self._xsrf(),
                                  json={"type": "directory"}, timeout=30)
                )

    def upload_file(self, local_path: str, remote_path: str) -> None:
        src = Path(local_path)
        if not src.is_file():
            raise FileNotFoundError(src)
        remote_path = remote_path.strip("/")
        parent = str(Path(remote_path).parent).replace("\\", "/")
        if parent != ".":
            self.ensure_dir(parent)
        raw = src.read_bytes()
        self._check(
            self.http.put(
                self.url("/api/contents/" + quote(remote_path, safe="/")),
                headers=self._xsrf(),
                json={
                    "type": "file",
                    "format": "base64",
                    "content": base64.b64encode(raw).decode("ascii"),
                },
                timeout=max(60, len(raw) / 500_000),
            )
        )
        print(f"↑ {src} → {remote_path}  ({len(raw):,} bytes)")

    def download_file(self, remote_path: str, local_path: str) -> bool:
        remote_path = remote_path.strip("/")
        r = self.http.get(
            self.url("/api/contents/" + quote(remote_path, safe="/")),
            params={"content": 1}, timeout=120,
        )
        if r.status_code >= 400:
            print(f"  [skip] {remote_path}  HTTP {r.status_code}")
            return False
        model = r.json()
        if model.get("type") != "file":
            return False
        raw = (
            base64.b64decode(model["content"])
            if model.get("format") == "base64"
            else (model.get("content") or "").encode()
        )
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)
        print(f"↓ {remote_path} → {dst}  ({len(raw):,} bytes)")
        return True

    def list_path(self, remote_path: str = "") -> list:
        remote_path = remote_path.strip("/")
        suffix = "/" + quote(remote_path, safe="/") if remote_path else ""
        r = self.http.get(
            self.url("/api/contents" + suffix), params={"content": 1}, timeout=30,
        )
        if r.status_code >= 400:
            return []
        model = r.json()
        items = []
        if model.get("type") == "directory":
            for item in model.get("content") or []:
                mark = "/" if item["type"] == "directory" else ""
                sz = f"  ({item['size']:,} B)" if item.get("size") else ""
                print(f"  {item['path']}{mark}{sz}")
                items.append(item)
        return items

    def download_dir(
        self,
        remote_dir: str,
        local_dir: str,
        extensions: tuple[str, ...] = (),
    ) -> int:
        """Recursively download a remote directory. Returns file count."""
        items = self.list_path(remote_dir)
        count = 0
        for item in items:
            if item["type"] == "directory":
                count += self.download_dir(item["path"], local_dir, extensions)
            elif item["type"] == "file":
                if extensions and not any(item["path"].endswith(e) for e in extensions):
                    continue
                local_path = Path(local_dir) / Path(item["path"]).name
                if self.download_file(item["path"], str(local_path)):
                    count += 1
        return count


def resolve_proxy_url(explicit: str | None = None, env_var: str = "KAGGLE_PROXY_URL") -> str:
    """
    Resolves proxy URL with fallback hierarchy:
    1. Explicit parameter / CLI arg
    2. Specified env var (e.g. KAGGLE_PROXY_URL)
    3. Standard fallback env keys
    4. Upward search for .env files
    """
    if explicit and explicit.strip():
        return explicit.strip()
    if os.environ.get(env_var):
        return os.environ[env_var].strip()

    fallback_keys = [
        "KAGGLE_PROXY_URL",
        "KAGGLE_PROXY_RTX6000_NO_INTERNET",
        "KAGGLE_WITHOUT_INTERNET",
        "KAGGLE_T4X2_PROXY",
        "KAGGLE_PROXY_HAS_INTERNET_NO_GPU",
        "KAGGLE_HAS_INTERNET",
        "KAGGLE_INTERNET",
    ]
    for key in fallback_keys:
        val = os.environ.get(key)
        if val and val.startswith("http"):
            return val.strip()

    search_dirs = [
        Path.cwd(),
        *Path.cwd().parents,
        Path(__file__).resolve().parent,
        *Path(__file__).resolve().parents,
    ]
    seen_dirs = set()
    for d in search_dirs:
        try:
            d_resolved = d.resolve()
        except Exception:
            continue
        if d_resolved in seen_dirs:
            continue
        seen_dirs.add(d_resolved)
        env_file = d_resolved / ".env"
        if env_file.is_file():
            try:
                content = env_file.read_text(encoding="utf-8", errors="ignore")
                env_dict = {}
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if v.startswith("http"):
                            env_dict[k] = v
                if env_var in env_dict:
                    return env_dict[env_var]
                for k in fallback_keys:
                    if k in env_dict:
                        return env_dict[k]
            except Exception:
                pass
    return ""


def _executor(proxy_url: str | None = None) -> KaggleExecutor:
    url = resolve_proxy_url(proxy_url)
    if not url:
        sys.exit(
            "ERROR: KAGGLE_PROXY_URL not set and could not be resolved from .env.\n\n"
            "  export KAGGLE_PROXY_URL='https://kkb-production.../proxy'\n"
            "  or pass --proxy-url 'https://...'\n"
        )
    return KaggleExecutor(url)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="kaggle_executor",
        description="Kaggle GPU kernel executor — run any Python on Kaggle, stream output live",
    )
    p.add_argument(
        "--proxy-url",
        dest="proxy_url",
        default=None,
        help="Explicit Kaggle Jupyter proxy URL (overrides .env / env vars)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("kernels", help="List active Jupyter kernels")

    ec = sub.add_parser("exec", help="Execute an inline Python snippet")
    ec.add_argument("code", help="Python code string")
    ec.add_argument("--timeout", type=int, default=600)

    ru = sub.add_parser("run", help="Upload + execute a local .py file")
    ru.add_argument("script", help="Local Python script path")
    ru.add_argument("--timeout", type=int, default=1800)

    up = sub.add_parser("upload", help="Upload a file")
    up.add_argument("local")
    up.add_argument("remote")

    dl = sub.add_parser("download", help="Download a file")
    dl.add_argument("remote")
    dl.add_argument("local")

    dd = sub.add_parser("download-dir", help="Download a remote directory")
    dd.add_argument("remote_dir")
    dd.add_argument("local_dir")
    dd.add_argument("--ext", nargs="*", metavar="EXT",
                    help="Filter by extensions, e.g. --ext .pt .glb .csv")

    ls = sub.add_parser("ls", help="List remote path")
    ls.add_argument("remote", nargs="?", default="")

    args = p.parse_args(argv)
    ex = _executor(args.proxy_url)

    match args.cmd:
        case "kernels":
            ex.list_kernels()

        case "exec":
            kid = ex.get_or_create_kernel()
            ex.execute(kid, args.code, timeout=args.timeout)

        case "run":
            script = Path(args.script)
            if not script.is_file():
                sys.exit(f"ERROR: {script} not found")
            remote = f"_run/{script.name}"
            ex.upload_file(str(script), remote)
            kid = ex.get_or_create_kernel()
            ex.execute(
                kid,
                f"exec(open('/kaggle/working/{remote}').read())",
                timeout=args.timeout,
            )

        case "upload":
            ex.upload_file(args.local, args.remote)

        case "download":
            # download_file returns False on HTTP 4xx/5xx and on a non-file
            # model. Discarding that made a 404 exit 0: two videos "downloaded"
            # and the retry wrapper reported success on a transfer that moved no
            # bytes. The Contents API is rooted at the notebook directory, so an
            # absolute /kaggle/working/... path 404s where the relative one works
            # -- a mistake that has to be loud or it is invisible.
            if not ex.download_file(args.remote, args.local):
                sys.exit(
                    f"ERROR: download failed: {args.remote}\n"
                    f"       paths are relative to the notebook dir "
                    f"(/kaggle/working), so drop any leading /kaggle/working/"
                )

        case "download-dir":
            ext = tuple(args.ext) if args.ext else ()
            n = ex.download_dir(args.remote_dir, args.local_dir, ext)
            print(f"\n✓ Downloaded {n} file(s) → {args.local_dir}")
            if n == 0:
                sys.exit(f"ERROR: no files downloaded from {args.remote_dir}")

        case "ls":
            ex.list_path(args.remote)


if __name__ == "__main__":
    main()
