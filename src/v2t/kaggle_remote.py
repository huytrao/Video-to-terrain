#!/usr/bin/env python3
"""
kaggle_remote.py

Use a Kaggle Jupyter proxy URL as:
- an interactive remote terminal over HTTPS/WSS
- a file upload/download transport via Jupyter Contents API

macOS/Linux client.

Dependencies:
    pip install requests websocket-client

Environment:
    export KAGGLE_PROXY_URL='https://.../proxy'

Examples:
    python kaggle_remote.py ping
    python kaggle_remote.py shell
    python kaggle_remote.py terminals
    python kaggle_remote.py attach 1

    python kaggle_remote.py upload train.py train.py
    python kaggle_remote.py upload-dir ./project project
    python kaggle_remote.py download model.pt ./model.pt
    python kaggle_remote.py ls
    python kaggle_remote.py list-dir /kaggle/working
"""

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


class KaggleProxy:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.http = requests.Session()

        parts = urlsplit(self.base)
        self.origin = f"{parts.scheme}://{parts.netloc}"

        self._bootstrap()

    def url(self, path: str) -> str:
        return self.base + "/" + path.lstrip("/")

    def ws_url(self, path: str) -> str:
        url = self.url(path)

        if url.startswith("https://"):
            return "wss://" + url[len("https://"):]

        if url.startswith("http://"):
            return "ws://" + url[len("http://"):]

        raise RuntimeError("KAGGLE_PROXY_URL must start with http:// or https://")

    def _bootstrap(self):
        """
        Hit the API once so requests.Session collects any cookies
        issued by the Kaggle/Jupyter proxy.
        """
        r = self.http.get(
            self.url("/api"),
            timeout=20,
            allow_redirects=True,
        )

        if r.status_code >= 400:
            raise RuntimeError(
                f"Cannot reach Jupyter API: HTTP {r.status_code}\n"
                f"{r.text[:1500]}"
            )

    def _xsrf_headers(self):
        """
        Jupyter may protect modifying REST calls with an _xsrf cookie.
        If present, reflect it in X-XSRFToken.
        """
        xsrf = None

        for cookie in self.http.cookies:
            if cookie.name == "_xsrf":
                xsrf = cookie.value

        if xsrf:
            return {"X-XSRFToken": xsrf}

        return {}

    @staticmethod
    def _check(r):
        if r.status_code >= 400:
            raise RuntimeError(
                f"HTTP {r.status_code} {r.request.method} {r.url}\n"
                f"{r.text[:2500]}"
            )

        return r

    # ============================================================
    # Files - Jupyter Contents API
    # ============================================================

    def ensure_dir(self, remote_dir: str):
        remote_dir = remote_dir.strip("/")

        if not remote_dir:
            return

        current = []

        for part in remote_dir.split("/"):
            current.append(part)
            path = "/".join(current)

            url = self.url(
                "/api/contents/" + quote(path, safe="/")
            )

            r = self.http.get(
                url,
                params={"content": 0},
                timeout=30,
            )

            if r.status_code == 404:
                r = self.http.put(
                    url,
                    headers=self._xsrf_headers(),
                    json={"type": "directory"},
                    timeout=30,
                )
                self._check(r)
            else:
                self._check(r)

    def upload_file(self, local_path: str, remote_path: str):
        src = Path(local_path)

        if not src.is_file():
            raise FileNotFoundError(src)

        remote_path = remote_path.strip("/")

        parent = str(Path(remote_path).parent).replace("\\", "/")

        if parent != ".":
            self.ensure_dir(parent)

        raw = src.read_bytes()

        payload = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(raw).decode("ascii"),
        }

        url = self.url(
            "/api/contents/" + quote(remote_path, safe="/")
        )

        r = self.http.put(
            url,
            headers=self._xsrf_headers(),
            json=payload,
            timeout=max(60, len(raw) / 500_000),
        )

        self._check(r)

        print(
            f"[UPLOAD] {src} -> {remote_path} "
            f"({len(raw):,} bytes)"
        )

    def upload_dir(self, local_dir: str, remote_dir: str, skip=None):
        """
        Upload a local directory tree to the remote runtime.

        Args:
            local_dir: source directory on local filesystem
            remote_dir: destination directory under the Jupyter home
            skip: optional iterable of directory/file names to skip
        """
        root = Path(local_dir).resolve()

        if not root.is_dir():
            raise NotADirectoryError(root)

        if skip is None:
            skip = {
                ".git",
                ".venv",
                "venv",
                "__pycache__",
                "node_modules",
                ".DS_Store",
                "outputs",
                ".claude",
                ".mypy_cache",
                ".pytest_cache",
            }

        uploaded = 0
        skipped = 0

        for path in sorted(root.rglob("*")):
            relative_parts = path.relative_to(root).parts

            if any(part in skip for part in relative_parts):
                skipped += 1
                continue

            if not path.is_file():
                continue

            rel = path.relative_to(root).as_posix()

            dst = (
                f"{remote_dir.strip('/')}/{rel}"
            ).strip("/")

            self.upload_file(str(path), dst)
            uploaded += 1

        print(f"\n[UPLOAD] {uploaded} files uploaded, {skipped} skipped.")
        return uploaded

    def download_file(self, remote_path: str, local_path: str):
        remote_path = remote_path.strip("/")

        url = self.url(
            "/api/contents/" + quote(remote_path, safe="/")
        )

        r = self.http.get(
            url,
            params={"content": 1},
            timeout=60,
        )

        self._check(r)

        model = r.json()

        if model.get("type") != "file":
            raise RuntimeError(
                f"{remote_path!r} is not a file"
            )

        if model.get("format") == "base64":
            raw = base64.b64decode(model["content"])
        else:
            raw = (
                model.get("content") or ""
            ).encode("utf-8")

        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)

        print(
            f"[DOWNLOAD] {remote_path} -> {dst} "
            f"({len(raw):,} bytes)"
        )

    def download_dir(self, remote_dir: str, local_dir: str, pattern: str | None = None):
        """
        Recursively download a remote directory tree to local disk.

        Args:
            remote_dir: source directory in Jupyter (e.g. /kaggle/working/foo)
            local_dir: destination directory on local filesystem
            pattern: optional glob pattern (e.g. '*.npy') to filter filenames
        """
        remote_dir = remote_dir.strip("/")
        local_root = Path(local_dir)
        local_root.mkdir(parents=True, exist_ok=True)

        items = self._list_dir_recursive(remote_dir)

        downloaded = 0

        for item in items:
            if item.get("type") != "file":
                continue

            rel_path = item["path"]
            if rel_path.startswith(remote_dir + "/"):
                rel = rel_path[len(remote_dir) + 1:]
            elif rel_path == remote_dir:
                continue
            else:
                rel = rel_path

            if pattern and not Path(rel).match(pattern):
                continue

            local_path = local_root / rel
            try:
                self.download_file(rel_path, str(local_path))
                downloaded += 1
            except Exception as exc:
                print(f"[DOWNLOAD] Failed: {rel_path} ({exc})")

        print(f"\n[DOWNLOAD] {downloaded} files downloaded to {local_root}")
        return downloaded

    def _list_dir_recursive(self, remote_dir: str) -> list:
        """Walk a remote directory tree and return metadata for every entry."""
        remote_dir = remote_dir.strip("/")
        collected = []

        def _walk(path: str):
            url = self.url(
                "/api/contents/" + quote(path, safe="/")
            )
            r = self.http.get(url, params={"content": 1}, timeout=30)
            self._check(r)
            model = r.json()
            if model.get("type") == "directory":
                for item in model.get("content") or []:
                    collected.append(item)
                    if item.get("type") == "directory":
                        _walk(item["path"])
            else:
                collected.append(model)

        _walk(remote_dir)
        return collected

    def list_path(self, remote_path: str = "", recursive: bool = False):
        remote_path = remote_path.strip("/")

        suffix = (
            "/" + quote(remote_path, safe="/")
            if remote_path
            else ""
        )

        r = self.http.get(
            self.url("/api/contents" + suffix),
            params={"content": 1},
            timeout=30,
        )

        self._check(r)

        model = r.json()

        if model.get("type") == "directory":
            items = model.get("content") or []
            for item in items:
                mark = "/" if item["type"] == "directory" else ""
                print(item["path"] + mark)
                if recursive and item.get("type") == "directory":
                    self.list_path(item["path"], recursive=True)
        else:
            print(json.dumps(model, indent=2)[:5000])

    # ============================================================
    # Terminal - Jupyter REST + Terminado WebSocket
    # ============================================================

    def terminals(self):
        r = self.http.get(
            self.url("/api/terminals"),
            timeout=30,
        )

        self._check(r)
        return r.json()

    def create_terminal(self):
        r = self.http.post(
            self.url("/api/terminals"),
            headers=self._xsrf_headers(),
            timeout=30,
        )

        self._check(r)

        data = r.json()
        return str(data["name"])

    def delete_terminal(self, name: str):
        r = self.http.delete(
            self.url(
                f"/api/terminals/{quote(str(name))}"
            ),
            headers=self._xsrf_headers(),
            timeout=30,
        )

        if r.status_code not in (204, 404):
            self._check(r)

    def _cookie_header(self):
        return "; ".join(
            f"{cookie.name}={cookie.value}"
            for cookie in self.http.cookies
        )

    def open_ws(self, name: str):
        ws = websocket.create_connection(
            self.ws_url(
                f"/terminals/websocket/{quote(str(name))}"
            ),
            cookie=self._cookie_header() or None,
            origin=self.origin,
            timeout=None,
            enable_multithread=True,
        )

        return ws

    def run_command(self, command: str, cwd: str | None = None,
                    timeout: float = 30.0) -> str:
        """
        Run a single shell command on a new Kaggle terminal and return its
        stdout. Intended for short, non-interactive operations.

        Args:
            command: shell command string
            cwd: remote working directory (default /kaggle/working)
            timeout: seconds to wait for completion
        """
        if cwd is None:
            cwd = "/kaggle/working"

        name = self.create_terminal()
        ws = self.open_ws(name)
        captured: list[str] = []
        stop = threading.Event()

        def _recv():
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
                    if (
                        isinstance(msg, list)
                        and len(msg) >= 2
                        and msg[0] == "stdout"
                    ):
                        captured.append(msg[1])
            except Exception:
                pass
            finally:
                stop.set()

        t = threading.Thread(target=_recv, daemon=True)
        t.start()

        prefix = f"cd {cwd} && " if cwd else ""
        ws.send(json.dumps(["stdin", f"{prefix}{command}\r"]))
        t.join(timeout=timeout)
        stop.set()

        try:
            ws.close()
        except Exception:
            pass
        try:
            self.delete_terminal(name)
        except Exception:
            pass

        return "".join(captured)

    def interactive(
        self,
        name: str,
        initial_command: str | None = None,
    ):
        """
        Attach local stdin/stdout directly to the Jupyter PTY.
        Ctrl+] detaches locally without deleting the remote terminal.
        """
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Interactive shell requires a real local terminal."
            )

        ws = self.open_ws(name)
        stop = threading.Event()

        def resize(*_):
            try:
                size = shutil.get_terminal_size((120, 30))

                ws.send(
                    json.dumps([
                        "set_size",
                        size.lines,
                        size.columns,
                        0,
                        0,
                    ])
                )
            except Exception:
                pass

        def receiver():
            try:
                while not stop.is_set():
                    raw = ws.recv()

                    if raw is None:
                        break

                    if isinstance(raw, bytes):
                        raw = raw.decode(
                            "utf-8",
                            errors="replace",
                        )

                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    if (
                        isinstance(msg, list)
                        and len(msg) >= 2
                        and msg[0] == "stdout"
                    ):
                        text = msg[1]

                        os.write(
                            sys.stdout.fileno(),
                            text.encode(
                                "utf-8",
                                errors="replace",
                            ),
                        )

            except Exception as exc:
                if not stop.is_set():
                    os.write(
                        sys.stderr.fileno(),
                        (
                            f"\n[WebSocket closed: {exc}]\n"
                        ).encode(),
                    )

            finally:
                stop.set()

        receiver_thread = threading.Thread(
            target=receiver,
            daemon=True,
        )

        receiver_thread.start()

        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)
        old_winch = signal.getsignal(signal.SIGWINCH)

        signal.signal(signal.SIGWINCH, resize)

        print(
            f"\nConnected to Kaggle terminal {name}.\n"
            "Ctrl+] = detach without killing the remote terminal.\n"
        )

        try:
            tty.setraw(fd)

            resize()

            if initial_command:
                ws.send(
                    json.dumps([
                        "stdin",
                        initial_command.rstrip("\r\n") + "\r",
                    ])
                )

            while not stop.is_set():
                data = os.read(fd, 4096)

                if not data:
                    break

                # Ctrl+]
                if b"\x1d" in data:
                    break

                ws.send(
                    json.dumps([
                        "stdin",
                        data.decode(
                            "utf-8",
                            errors="ignore",
                        ),
                    ])
                )

        finally:
            stop.set()

            try:
                ws.close()
            except Exception:
                pass

            termios.tcsetattr(
                fd,
                termios.TCSADRAIN,
                old_term,
            )

            signal.signal(
                signal.SIGWINCH,
                old_winch,
            )

            print(
                f"\nDetached from Kaggle terminal {name}.\n"
                f"Reattach:\n"
                f"  python kaggle_remote.py attach {name}\n"
            )


def make_client():
    base = os.environ.get("KAGGLE_PROXY_URL")

    if not base:
        print(
            "KAGGLE_PROXY_URL is not set.\n\n"
            "Example:\n"
            "  export KAGGLE_PROXY_URL='https://.../proxy'\n",
            file=sys.stderr,
        )
        sys.exit(2)

    return KaggleProxy(base)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Kaggle runtime shell over "
            "Jupyter HTTPS/WebSocket proxy"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser("ping")
    sub.add_parser("terminals")

    shell = sub.add_parser("shell")
    shell.add_argument(
        "--cwd",
        default="/kaggle/working",
    )

    attach = sub.add_parser("attach")
    attach.add_argument("name")

    kill = sub.add_parser("kill")
    kill.add_argument("name")

    ls = sub.add_parser("ls")
    ls.add_argument(
        "remote",
        nargs="?",
        default="",
    )
    ls.add_argument(
        "-R", "--recursive",
        action="store_true",
        help="Recurse into subdirectories",
    )

    list_dir = sub.add_parser("list-dir")
    list_dir.add_argument("remote")
    list_dir.add_argument(
        "--local",
        default=None,
        help="Optional local directory to mirror listing into",
    )

    upload = sub.add_parser("upload")
    upload.add_argument("local")
    upload.add_argument("remote")

    upload_dir = sub.add_parser("upload-dir")
    upload_dir.add_argument("local_dir")
    upload_dir.add_argument("remote_dir")

    download = sub.add_parser("download")
    download.add_argument("remote")
    download.add_argument("local")

    download_dir = sub.add_parser("download-dir")
    download_dir.add_argument("remote_dir")
    download_dir.add_argument("local_dir")
    download_dir.add_argument(
        "--pattern",
        default=None,
        help="Optional filename glob to filter downloads",
    )

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    kaggle = make_client()

    if args.command == "ping":
        print("OK: Kaggle Jupyter API reachable")

    elif args.command == "terminals":
        print(
            json.dumps(
                kaggle.terminals(),
                indent=2,
            )
        )

    elif args.command == "shell":
        name = kaggle.create_terminal()

        print(f"Created remote terminal: {name}")

        kaggle.interactive(
            name,
            initial_command=f"cd {args.cwd}",
        )

    elif args.command == "attach":
        kaggle.interactive(args.name)

    elif args.command == "kill":
        kaggle.delete_terminal(args.name)
        print(f"Deleted terminal {args.name}")

    elif args.command == "ls":
        kaggle.list_path(args.remote, recursive=args.recursive)

    elif args.command == "list-dir":
        if args.local:
            kaggle.download_dir(args.remote, args.local)
        else:
            kaggle.list_path(args.remote, recursive=True)

    elif args.command == "upload":
        kaggle.upload_file(
            args.local,
            args.remote,
        )

    elif args.command == "upload-dir":
        kaggle.upload_dir(
            args.local_dir,
            args.remote_dir,
        )

    elif args.command == "download":
        kaggle.download_file(
            args.remote,
            args.local,
        )

    elif args.command == "download-dir":
        kaggle.download_dir(
            args.remote_dir,
            args.local_dir,
            pattern=args.pattern,
        )

    elif args.command == "run":
        if not args.command:
            parser.error("run requires a command")
        cmd = " ".join(args.command)
        out = kaggle.run_command(cmd)
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
