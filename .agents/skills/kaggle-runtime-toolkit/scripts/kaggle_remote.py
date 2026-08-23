#!/usr/bin/env python3
"""
kaggle_remote.py  —  kaggle-runtime-agent-skills
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use a live Kaggle Jupyter proxy URL as:
  • an interactive remote terminal (PTY over WebSocket)
  • a bidirectional file transport (Jupyter Contents API)

Works on macOS and Linux. No Kaggle API key required.

Install:
    pip install requests websocket-client

Set env:
    export KAGGLE_PROXY_URL='https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/.../proxy'

Examples:
    python kaggle_remote.py ping
    python kaggle_remote.py shell
    python kaggle_remote.py ls
    python kaggle_remote.py upload ./model.py scripts/model.py
    python kaggle_remote.py upload-dir ./project remote_project
    python kaggle_remote.py download output/result.glb ./result.glb
    python kaggle_remote.py terminals
    python kaggle_remote.py attach 1
    python kaggle_remote.py kill 1
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import sys
import termios
import threading
import tty
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests
import websocket


# ─────────────────────────────────────────────────────────────────────────────
# Core client
# ─────────────────────────────────────────────────────────────────────────────

class KaggleProxy:
    """
    Low-level client for the Jupyter/Kaggle proxy.

    Exposes:
      • File operations  – upload_file, upload_dir, download_file, list_path
      • Terminal ops     – create_terminal, delete_terminal, interactive (PTY)
    """

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self.http = requests.Session()
        parts = urlsplit(self.base)
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self._bootstrap()

    # ── URL helpers ──────────────────────────────────────────────────────────

    def url(self, path: str) -> str:
        return self.base + "/" + path.lstrip("/")

    def ws_url(self, path: str) -> str:
        u = self.url(path)
        if u.startswith("https://"):
            return "wss://" + u[len("https://"):]
        if u.startswith("http://"):
            return "ws://" + u[len("http://"):]
        raise RuntimeError("KAGGLE_PROXY_URL must start with http:// or https://")

    # ── Session bootstrap ────────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        """GET /api to warm up the session and collect cookies."""
        r = self.http.get(self.url("/api"), timeout=20, allow_redirects=True)
        if r.status_code >= 400:
            raise RuntimeError(
                f"Cannot reach Jupyter API: HTTP {r.status_code}\n{r.text[:1500]}"
            )

    def _xsrf_headers(self) -> dict:
        for cookie in self.http.cookies:
            if cookie.name == "_xsrf":
                return {"X-XSRFToken": cookie.value}
        return {}

    @staticmethod
    def _check(r: requests.Response) -> requests.Response:
        if r.status_code >= 400:
            raise RuntimeError(
                f"HTTP {r.status_code} {r.request.method} {r.url}\n{r.text[:2500]}"
            )
        return r

    # ── File operations ──────────────────────────────────────────────────────

    def ensure_dir(self, remote_dir: str) -> None:
        """Recursively create remote directories (no-op if they exist)."""
        remote_dir = remote_dir.strip("/")
        if not remote_dir:
            return
        current: list[str] = []
        for part in remote_dir.split("/"):
            current.append(part)
            path = "/".join(current)
            url = self.url("/api/contents/" + quote(path, safe="/"))
            r = self.http.get(url, params={"content": 0}, timeout=30)
            if r.status_code == 404:
                self._check(
                    self.http.put(url, headers=self._xsrf_headers(),
                                  json={"type": "directory"}, timeout=30)
                )
            elif r.status_code >= 400:
                self._check(r)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file to the Kaggle working directory."""
        src = Path(local_path)
        if not src.is_file():
            raise FileNotFoundError(src)
        remote_path = remote_path.strip("/")
        parent = str(Path(remote_path).parent).replace("\\", "/")
        if parent != ".":
            self.ensure_dir(parent)
        raw = src.read_bytes()
        url = self.url("/api/contents/" + quote(remote_path, safe="/"))
        self._check(
            self.http.put(
                url,
                headers=self._xsrf_headers(),
                json={
                    "type": "file",
                    "format": "base64",
                    "content": base64.b64encode(raw).decode("ascii"),
                },
                timeout=max(60, len(raw) / 500_000),
            )
        )
        print(f"↑ {src} → {remote_path}  ({len(raw):,} bytes)")

    def upload_dir(self, local_dir: str, remote_dir: str) -> None:
        """Upload a directory tree, skipping .git / __pycache__ / node_modules."""
        root = Path(local_dir).resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules", ".DS_Store"}
        uploaded = 0
        for path in root.rglob("*"):
            if any(p in SKIP for p in path.relative_to(root).parts):
                continue
            if not path.is_file():
                continue
            dst = f"{remote_dir.strip('/')}/{path.relative_to(root).as_posix()}"
            self.upload_file(str(path), dst.strip("/"))
            uploaded += 1
        print(f"\nUploaded {uploaded} files.")

    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a single file from the Kaggle working directory."""
        remote_path = remote_path.strip("/")
        url = self.url("/api/contents/" + quote(remote_path, safe="/"))
        r = self._check(self.http.get(url, params={"content": 1}, timeout=120))
        model = r.json()
        if model.get("type") != "file":
            raise RuntimeError(f"{remote_path!r} is not a file")
        raw = (
            base64.b64decode(model["content"])
            if model.get("format") == "base64"
            else (model.get("content") or "").encode()
        )
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)
        print(f"↓ {remote_path} → {dst}  ({len(raw):,} bytes)")

    def list_path(self, remote_path: str = "") -> None:
        """List a remote path (file or directory)."""
        remote_path = remote_path.strip("/")
        suffix = "/" + quote(remote_path, safe="/") if remote_path else ""
        r = self._check(
            self.http.get(self.url("/api/contents" + suffix),
                          params={"content": 1}, timeout=30)
        )
        model = r.json()
        if model.get("type") == "directory":
            for item in model.get("content") or []:
                mark = "/" if item["type"] == "directory" else ""
                print(f"  {item['path']}{mark}")
        else:
            print(json.dumps(model, indent=2)[:5000])

    # ── Terminal / PTY ───────────────────────────────────────────────────────

    def terminals(self) -> list:
        r = self._check(self.http.get(self.url("/api/terminals"), timeout=30))
        return r.json()

    def create_terminal(self) -> str:
        r = self._check(
            self.http.post(self.url("/api/terminals"),
                           headers=self._xsrf_headers(), timeout=30)
        )
        return str(r.json()["name"])

    def delete_terminal(self, name: str) -> None:
        r = self.http.delete(
            self.url(f"/api/terminals/{quote(str(name))}"),
            headers=self._xsrf_headers(), timeout=30,
        )
        if r.status_code not in (204, 404):
            self._check(r)

    def _cookie_header(self) -> str:
        return "; ".join(f"{c.name}={c.value}" for c in self.http.cookies)

    def open_ws(self, name: str) -> websocket.WebSocket:
        return websocket.create_connection(
            self.ws_url(f"/terminals/websocket/{quote(str(name))}"),
            cookie=self._cookie_header() or None,
            origin=self.origin,
            timeout=None,
            enable_multithread=True,
        )

    def interactive(self, name: str, initial_command: str | None = None) -> None:
        """
        Attach local stdin/stdout to the Kaggle PTY.
        Press Ctrl+] to detach without killing the remote terminal.
        """
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive shell requires a real local terminal (TTY).")

        ws = self.open_ws(name)
        stop = threading.Event()

        def resize(*_):
            try:
                sz = shutil.get_terminal_size((120, 30))
                ws.send(json.dumps(["set_size", sz.lines, sz.columns, 0, 0]))
            except Exception:
                pass

        def receiver():
            try:
                while not stop.is_set():
                    raw = ws.recv()
                    if raw is None:
                        break
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "stdout":
                        os.write(sys.stdout.fileno(),
                                 msg[1].encode("utf-8", errors="replace"))
            except Exception as exc:
                if not stop.is_set():
                    os.write(sys.stderr.fileno(),
                             f"\n[WebSocket closed: {exc}]\n".encode())
            finally:
                stop.set()

        t = threading.Thread(target=receiver, daemon=True)
        t.start()

        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)
        old_winch = signal.getsignal(signal.SIGWINCH)
        signal.signal(signal.SIGWINCH, resize)

        print(f"\nConnected to Kaggle terminal '{name}'.")
        print("Ctrl+] = detach  (terminal stays alive on Kaggle)\n")

        try:
            tty.setraw(fd)
            resize()
            if initial_command:
                ws.send(json.dumps(["stdin", initial_command.rstrip("\r\n") + "\r"]))
            while not stop.is_set():
                data = os.read(fd, 4096)
                if not data or b"\x1d" in data:   # Ctrl+]
                    break
                ws.send(json.dumps(["stdin", data.decode("utf-8", errors="ignore")]))
        finally:
            stop.set()
            try:
                ws.close()
            except Exception:
                pass
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
            signal.signal(signal.SIGWINCH, old_winch)
            print(f"\nDetached. Reattach: python kaggle_remote.py attach {name}\n")


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


def _client(proxy_url: str | None = None) -> KaggleProxy:
    url = resolve_proxy_url(proxy_url)
    if not url:
        sys.exit(
            "ERROR: KAGGLE_PROXY_URL not set and could not be resolved from .env.\n\n"
            "  export KAGGLE_PROXY_URL='https://kkb-production.../proxy'\n"
            "  or pass --proxy-url 'https://...'\n"
        )
    return KaggleProxy(url)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="kaggle_remote",
        description="Kaggle GPU remote — file manager + interactive terminal",
    )
    p.add_argument(
        "--proxy-url",
        dest="proxy_url",
        default=None,
        help="Explicit Kaggle Jupyter proxy URL (overrides .env / env vars)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping",      help="Check connectivity")
    sub.add_parser("terminals", help="List active terminal sessions")

    sh = sub.add_parser("shell", help="Open a new interactive shell")
    sh.add_argument("--cwd", default="/kaggle/working")

    at = sub.add_parser("attach", help="Reattach to an existing terminal")
    at.add_argument("name")

    kl = sub.add_parser("kill", help="Kill a terminal")
    kl.add_argument("name")

    ls = sub.add_parser("ls", help="List a remote path")
    ls.add_argument("remote", nargs="?", default="")

    up = sub.add_parser("upload", help="Upload a file")
    up.add_argument("local")
    up.add_argument("remote")

    ud = sub.add_parser("upload-dir", help="Upload a directory tree")
    ud.add_argument("local_dir")
    ud.add_argument("remote_dir")

    dl = sub.add_parser("download", help="Download a file")
    dl.add_argument("remote")
    dl.add_argument("local")

    args = p.parse_args(argv)
    k = _client(args.proxy_url)

    match args.cmd:
        case "ping":
            print("✓ Kaggle Jupyter API reachable")
        case "terminals":
            print(json.dumps(k.terminals(), indent=2))
        case "shell":
            name = k.create_terminal()
            print(f"Created terminal: {name}")
            k.interactive(name, initial_command=f"cd {args.cwd}")
        case "attach":
            k.interactive(args.name)
        case "kill":
            k.delete_terminal(args.name)
            print(f"Terminal '{args.name}' deleted.")
        case "ls":
            k.list_path(args.remote)
        case "upload":
            k.upload_file(args.local, args.remote)
        case "upload-dir":
            k.upload_dir(args.local_dir, args.remote_dir)
        case "download":
            k.download_file(args.remote, args.local)


if __name__ == "__main__":
    main()
