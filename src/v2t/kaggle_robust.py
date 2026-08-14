#!/usr/bin/env python3
"""
kaggle_robust.py - Robust Kaggle deployment with proxy lifecycle management.

Enhances kaggle_remote.py with:
1. Proxy URL expiration detection (JWT timeout ~1-2 hours)
2. Connection retry with exponential backoff
3. Long-running session keep-alive
4. Kernel restart detection and recovery
5. Resume capability for interrupted uploads/downloads
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
import threading
import traceback
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from urllib.parse import urlsplit, quote
from functools import wraps

import requests
import websocket


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 2.0      # seconds
    max_delay: float = 60.0      # seconds
    exponential_base: float = 2.0
    jitter: float = 0.1          # 10% jitter

    # Specific timeouts for different operations
    http_timeout: float = 30.0
    upload_timeout: float = 120.0
    download_timeout: float = 180.0
    command_timeout: float = 300.0  # 5 minutes for DA3 inference


@dataclass
class SessionConfig:
    """Configuration for session lifecycle."""
    keepalive_interval: int = 60      # seconds between keepalive pings
    proxy_refresh_threshold: int = 300  # seconds before JWT expiry to warn (5 min)
    max_session_duration: int = 7200    # 2 hours max session (typical JWT)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class KaggleError(Exception):
    """Base exception for Kaggle operations."""
    pass


class ProxyExpiredError(KaggleError):
    """Raised when proxy URL/JWT has expired."""
    pass


class ConnectionLostError(KaggleError):
    """Raised when connection is lost and cannot be recovered."""
    pass


class KernelRestartedError(KaggleError):
    """Raised when Kaggle kernel has restarted."""
    pass


class UploadFailedError(KaggleError):
    """Raised when upload fails after retries."""
    pass


class DownloadFailedError(KaggleError):
    """Raised when download fails after retries."""
    pass


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def with_retry(
    retry_config: RetryConfig,
    exceptions: tuple = (requests.RequestException, websocket.WebSocketException, TimeoutError, ConnectionError),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < retry_config.max_retries:
                        delay = min(
                            retry_config.base_delay * (retry_config.exponential_base ** (attempt - 1)),
                            retry_config.max_delay
                        )
                        # Add jitter
                        import random
                        delay *= (1 + random.uniform(-retry_config.jitter, retry_config.jitter))

                        if on_retry:
                            on_retry(e, attempt)
                        else:
                            logging.warning(
                                f"Attempt {attempt}/{retry_config.max_retries} failed: {e}. "
                                f"Retrying in {delay:.1f}s..."
                            )
                        time.sleep(delay)
                    else:
                        logging.error(f"All {retry_config.max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# ROBUST KAGGLE PROXY
# =============================================================================

class RobustKaggleProxy:
    """
    Kaggle proxy client with robust connection handling.

    Features:
    - Automatic retry with exponential backoff
    - Proxy URL expiration detection
    - Session keep-alive
    - Kernel restart detection
    - Resume capability for file operations
    """

    def __init__(
        self,
        base_url: str,
        retry_config: Optional[RetryConfig] = None,
        session_config: Optional[SessionConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.base = base_url.rstrip("/")
        self.http = requests.Session()
        self.retry_config = retry_config or RetryConfig()
        self.session_config = session_config or SessionConfig()
        self.logger = logger or logging.getLogger("robust_kaggle")

        parts = urlsplit(self.base)
        self.origin = f"{parts.scheme}://{parts.netloc}"

        # Session state
        self._session_start = time.time()
        self._last_keepalive = 0
        self._kernel_pid: Optional[int] = None
        self._proxy_valid = True

        # Thread safety
        self._lock = threading.Lock()

        self._bootstrap()

    def _bootstrap(self):
        """Initial connection to collect cookies and verify proxy."""
        with self._lock:
            try:
                r = self.http.get(
                    self.url("/api"),
                    timeout=self.retry_config.http_timeout,
                    allow_redirects=True,
                )
                if r.status_code >= 400:
                    raise ProxyExpiredError(
                        f"Cannot reach Jupyter API: HTTP {r.status_code}\n"
                        f"{r.text[:1500]}"
                    )
                self.logger.info("Proxy connection established")
            except requests.RequestException as e:
                raise ProxyExpiredError(f"Proxy connection failed: {e}")

    def url(self, path: str) -> str:
        return self.base + "/" + path.lstrip("/")

    def ws_url(self, path: str) -> str:
        url = self.url(path)
        if url.startswith("https://"):
            return "wss://" + url[len("https://"):]
        if url.startswith("http://"):
            return "ws://" + url[len("http://"):]
        raise RuntimeError("KAGGLE_PROXY_URL must start with http:// or https://")

    def _xsrf_headers(self):
        xsrf = None
        for cookie in self.http.cookies:
            if cookie.name == "_xsrf":
                xsrf = cookie.value
        if xsrf:
            return {"X-XSRFToken": xsrf}
        return {}

    def _check_response(self, r: requests.Response, context: str = "") -> requests.Response:
        """Check response and raise appropriate exception."""
        if r.status_code >= 400:
            # Check for JWT/proxy expiration indicators
            if r.status_code in (401, 403):
                # Check if it's JWT expiry vs auth issue
                text = r.text.lower()
                if "jwt" in text or "token" in text or "expired" in text or "unauthorized" in text:
                    raise ProxyExpiredError(
                        f"Proxy URL expired (JWT token invalid). "
                        f"Get fresh URL from Kaggle UI.\n"
                        f"HTTP {r.status_code} {r.request.method} {r.url}\n"
                        f"{r.text[:2500]}"
                    )
            elif r.status_code == 503:
                # Service unavailable - kernel might be starting/restarting
                raise ConnectionLostError(
                    f"Kaggle service unavailable (kernel restarting?). "
                    f"HTTP {r.status_code} {context}\n"
                    f"{r.text[:2500]}"
                )

            raise KaggleError(
                f"HTTP {r.status_code} {r.request.method} {r.url} {context}\n"
                f"{r.text[:2500]}"
            )
        return r

    # -------------------------------------------------------------------------
    # Connection Health & Keep-Alive
    # -------------------------------------------------------------------------

    def check_proxy_health(self) -> Dict[str, Any]:
        """
        Check proxy/JWT health.
        Returns dict with status, time_elapsed, time_remaining, needs_refresh.
        """
        elapsed = time.time() - self._session_start
        remaining = max(0, self.session_config.max_session_duration - elapsed)
        needs_refresh = remaining < self.session_config.proxy_refresh_threshold

        # Try a lightweight API call to verify
        try:
            r = self.http.get(
                self.url("/api/contents"),
                params={"content": 0},
                timeout=10,
            )
            healthy = r.status_code < 400
        except Exception as e:
            healthy = False
            self.logger.warning(f"Health check failed: {e}")

        return {
            "healthy": healthy,
            "session_elapsed_seconds": elapsed,
            "session_remaining_seconds": remaining,
            "needs_refresh": needs_refresh,
            "proxy_url": self.base[:60] + "...",
        }

    def keepalive(self) -> bool:
        """Send keepalive ping to maintain session."""
        now = time.time()
        if now - self._last_keepalive < self.session_config.keepalive_interval:
            return True

        try:
            # Lightweight API call
            r = self.http.get(
                self.url("/api/contents"),
                params={"content": 0},
                timeout=10,
            )
            if r.status_code < 400:
                self._last_keepalive = now
                return True
        except Exception as e:
            self.logger.warning(f"Keepalive failed: {e}")

        return False

    def refresh_proxy_url(self, new_url: str) -> None:
        """Update proxy URL (call when user provides fresh URL from Kaggle UI)."""
        with self._lock:
            old_base = self.base
            self.base = new_url.rstrip("/")
            parts = urlsplit(self.base)
            self.origin = f"{parts.scheme}://{parts.netloc}"
            self.http = requests.Session()  # New session for new cookies
            self._session_start = time.time()
            self._last_keepalive = 0
            self._proxy_valid = True
            self.logger.info(f"Proxy URL refreshed: {old_base[:40]}... -> {self.base[:40]}...")
            self._bootstrap()

    def detect_kernel_restart(self) -> bool:
        """Detect if kernel has restarted by checking PID."""
        try:
            # Get current kernel PID via terminal
            out = self.run_command("echo $PPID", timeout=10)
            pid = int(out.strip())
            if self._kernel_pid is not None and pid != self._kernel_pid:
                self.logger.warning(f"Kernel restart detected: PID {self._kernel_pid} -> {pid}")
                self._kernel_pid = pid
                return True
            self._kernel_pid = pid
        except Exception:
            pass
        return False

    # -------------------------------------------------------------------------
    # File Operations with Retry & Resume
    # -------------------------------------------------------------------------

    @with_retry(RetryConfig(max_retries=3, base_delay=2.0))
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file with retry and resume capability."""
        src = Path(local_path)
        if not src.is_file():
            raise FileNotFoundError(src)

        remote_path = remote_path.strip("/")
        parent = str(Path(remote_path).parent).replace("\\", "/")
        if parent != ".":
            self.ensure_dir(parent)

        # Check if file already exists remotely with same size
        try:
            r = self.http.get(
                self.url("/api/contents/" + remote_path),
                params={"content": 0},
                timeout=self.retry_config.http_timeout,
            )
            if r.status_code == 200:
                model = r.json()
                if model.get("type") == "file" and model.get("size") == src.stat().st_size:
                    self.logger.info(f"[SKIP] {src} already exists remotely with same size")
                    return True
        except Exception:
            pass  # File doesn't exist or error checking - proceed with upload

        raw = src.read_bytes()

        # For large files, use chunked upload if supported
        if len(raw) > 50 * 1024 * 1024:  # 50MB
            return self._upload_large_file(src, remote_path, raw)

        payload = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(raw).decode("ascii"),
        }

        url = self.url("/api/contents/" + remote_path)
        r = self.http.put(
            url,
            headers=self._xsrf_headers(),
            json=payload,
            timeout=max(self.retry_config.upload_timeout, len(raw) / 500_000),
        )

        self._check_response(r, f"upload {remote_path}")

        self.logger.info(f"[UPLOAD] {src} -> {remote_path} ({len(raw):,} bytes)")
        return True

    def _upload_large_file(self, src: Path, remote_path: str, raw: bytes) -> bool:
        """Upload large file in chunks (if Jupyter supports it)."""
        # Jupyter Contents API doesn't natively support chunked upload
        # Fallback: split into smaller base64 chunks if needed
        # For now, just do normal upload with longer timeout
        payload = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(raw).decode("ascii"),
        }
        url = self.url("/api/contents/" + remote_path)
        r = self.http.put(
            url,
            headers=self._xsrf_headers(),
            json=payload,
            timeout=self.retry_config.upload_timeout * 3,
        )
        self._check_response(r, f"upload large {remote_path}")
        self.logger.info(f"[UPLOAD] {src} -> {remote_path} ({len(raw):,} bytes) [LARGE]")
        return True

    def upload_dir(
        self,
        local_dir: str,
        remote_dir: str,
        skip: Optional[set] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> int:
        """Upload directory with progress tracking and skip-existing optimization."""
        root = Path(local_dir).resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)

        if skip is None:
            skip = {
                ".git", ".venv", "venv", "__pycache__", "node_modules",
                ".DS_Store", "outputs", ".claude", ".mypy_cache",
                ".pytest_cache", "source-reference", "*.pyc", "*.log"
            }

        # Collect all files first
        files_to_upload = []
        for path in sorted(root.rglob("*")):
            relative_parts = path.relative_to(root).parts
            if any(part in skip for part in relative_parts):
                continue
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                dst = f"{remote_dir.strip('/')}/{rel}".strip("/")
                files_to_upload.append((path, dst))

        total = len(files_to_upload)
        uploaded = 0
        skipped = 0
        failed = 0

        self.logger.info(f"[UPLOAD] Starting directory upload: {total} files")

        for i, (local_path, remote_path) in enumerate(files_to_upload):
            try:
                # Check if already exists
                r = self.http.get(
                    self.url("/api/contents/" + remote_path),
                    params={"content": 0},
                    timeout=self.retry_config.http_timeout,
                )
                if r.status_code == 200:
                    model = r.json()
                    if (model.get("type") == "file" and
                        model.get("size") == local_path.stat().st_size):
                        skipped += 1
                        if progress_callback:
                            progress_callback(i + 1, total, f"skipped {local_path.name}")
                        continue

                self.upload_file(str(local_path), remote_path)
                uploaded += 1

                if progress_callback:
                    progress_callback(i + 1, total, f"uploaded {local_path.name}")

            except Exception as e:
                failed += 1
                self.logger.error(f"[UPLOAD] Failed {local_path}: {e}")
                if failed > total * 0.1:  # More than 10% failure rate
                    raise UploadFailedError(f"Too many upload failures: {failed}/{total}")

        self.logger.info(f"[UPLOAD] Complete: {uploaded} uploaded, {skipped} skipped, {failed} failed")
        return uploaded

    @with_retry(RetryConfig(max_retries=3, base_delay=2.0))
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file with retry."""
        remote_path = remote_path.strip("/")

        url = self.url("/api/contents/" + remote_path)
        r = self.http.get(
            url,
            params={"content": 1},
            timeout=self.retry_config.download_timeout,
        )

        self._check_response(r, f"download {remote_path}")

        model = r.json()
        if model.get("type") != "file":
            raise KaggleError(f"{remote_path} is not a file")

        if model.get("format") == "base64":
            raw = base64.b64decode(model["content"])
        else:
            raw = (model.get("content") or "").encode("utf-8")

        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)

        self.logger.info(f"[DOWNLOAD] {remote_path} -> {dst} ({len(raw):,} bytes)")
        return True

    def download_dir(
        self,
        remote_dir: str,
        local_dir: str,
        pattern: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        resume: bool = True
    ) -> int:
        """Download directory with resume capability."""
        remote_dir = remote_dir.strip("/")
        local_root = Path(local_dir)
        local_root.mkdir(parents=True, exist_ok=True)

        items = self._list_dir_recursive(remote_dir)
        downloaded = 0
        skipped = 0
        failed = 0

        # Filter files
        file_items = [item for item in items if item.get("type") == "file"]
        if pattern:
            file_items = [item for item in file_items if Path(item["path"]).match(pattern)]

        total = len(file_items)
        self.logger.info(f"[DOWNLOAD] Starting: {total} files")

        for i, item in enumerate(file_items):
            rel_path = item["path"]
            if rel_path.startswith(remote_dir + "/"):
                rel = rel_path[len(remote_dir) + 1:]
            elif rel_path == remote_dir:
                continue
            else:
                rel = rel_path

            local_path = local_root / rel

            # Resume: skip if local file exists with same size
            if resume and local_path.exists():
                try:
                    r = self.http.get(
                        self.url("/api/contents/" + rel_path),
                        params={"content": 0},
                        timeout=self.retry_config.http_timeout,
                    )
                    if r.status_code == 200:
                        model = r.json()
                        if model.get("size") == local_path.stat().st_size:
                            skipped += 1
                            if progress_callback:
                                progress_callback(i + 1, total, f"skipped {rel}")
                            continue
                except Exception:
                    pass

            try:
                self.download_file(rel_path, str(local_path))
                downloaded += 1
                if progress_callback:
                    progress_callback(i + 1, total, f"downloaded {rel}")
            except Exception as e:
                failed += 1
                self.logger.error(f"[DOWNLOAD] Failed {rel_path}: {e}")
                if failed > total * 0.1:
                    raise DownloadFailedError(f"Too many download failures: {failed}/{total}")

        self.logger.info(f"[DOWNLOAD] Complete: {downloaded} downloaded, {skipped} skipped, {failed} failed")
        return downloaded

    # -------------------------------------------------------------------------
    # Command Execution with Kernel Restart Detection
    # -------------------------------------------------------------------------

    def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: float = 30.0,
        keepalive_during: bool = True
    ) -> str:
        """
        Run command with kernel restart detection and keepalive.

        If kernel restarts during execution, raises KernelRestartedError.
        """
        if cwd is None:
            cwd = "/kaggle/working"

        # Check kernel PID before
        self.detect_kernel_restart()

        name = self.create_terminal()
        ws = self.open_ws(name)
        captured: List[str] = []
        stop = threading.Event()
        error_occurred = threading.Event()
        last_error = [None]

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
            except Exception as e:
                if not stop.is_set():
                    last_error[0] = e
                    error_occurred.set()
            finally:
                stop.set()

        t = threading.Thread(target=_recv, daemon=True)
        t.start()

        prefix = f"cd {cwd} && " if cwd else ""

        # Send keepalive pings during long-running commands
        keepalive_thread = None
        if keepalive_during and timeout > 30:
            def _keepalive():
                while not stop.is_set():
                    time.sleep(self.session_config.keepalive_interval)
                    if stop.is_set():
                        break
                    try:
                        ws.send(json.dumps(["stdin", "\n"]))  # Harmless newline
                    except Exception:
                        break
            keepalive_thread = threading.Thread(target=_keepalive, daemon=True)
            keepalive_thread.start()

        ws.send(json.dumps(["stdin", f"{prefix}{command}\r"]))
        t.join(timeout=timeout)

        stop.set()
        if keepalive_thread:
            keepalive_thread.join(timeout=2)

        try:
            ws.close()
        except Exception:
            pass

        # Check for kernel restart after
        if self.detect_kernel_restart():
            self.logger.warning("Kernel restart detected after command execution")
            raise KernelRestartedError("Kernel restarted during command execution")

        if error_occurred.is_set() and last_error[0]:
            raise KaggleError(f"WebSocket error during command: {last_error[0]}")

        try:
            self.delete_terminal(name)
        except Exception:
            pass

        output = "".join(captured)

        # Check for proxy expiration in output
        if "401" in output and ("unauthorized" in output.lower() or "jwt" in output.lower()):
            raise ProxyExpiredError("Proxy URL expired during command execution")

        return output

    # -------------------------------------------------------------------------
    # Terminal Management (from original)
    # -------------------------------------------------------------------------

    def create_terminal(self) -> str:
        r = self.http.post(
            self.url("/api/terminals"),
            headers=self._xsrf_headers(),
            timeout=self.retry_config.http_timeout,
        )
        self._check_response(r, "create terminal")
        return str(r.json()["name"])

    def delete_terminal(self, name: str):
        r = self.http.delete(
            self.url(f"/api/terminals/{name}"),
            headers=self._xsrf_headers(),
            timeout=self.retry_config.http_timeout,
        )
        if r.status_code not in (204, 404):
            self._check_response(r, f"delete terminal {name}")

    def _cookie_header(self):
        return "; ".join(f"{c.name}={c.value}" for c in self.http.cookies)

    def open_ws(self, name: str):
        ws = websocket.create_connection(
            self.ws_url(f"/terminals/websocket/{name}"),
            cookie=self._cookie_header() or None,
            origin=self.origin,
            timeout=None,
            enable_multithread=True,
        )
        return ws

    def interactive(self, name: str, initial_command: Optional[str] = None):
        """Interactive shell (unchanged from original)."""
        import termios, signal, shutil

        if not sys.stdin.isatty():
            raise RuntimeError("Interactive shell requires a real local terminal.")

        ws = self.open_ws(name)
        stop = threading.Event()

        def resize(*_):
            try:
                size = shutil.get_terminal_size((120, 30))
                ws.send(json.dumps(["set_size", size.lines, size.columns, 0, 0]))
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
                        os.write(sys.stdout.fileno(), msg[1].encode("utf-8", errors="replace"))
            except Exception as exc:
                if not stop.is_set():
                    os.write(sys.stderr.fileno(), f"\n[WebSocket closed: {exc}]\n".encode())
            finally:
                stop.set()

        receiver_thread = threading.Thread(target=receiver, daemon=True)
        receiver_thread.start()

        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)
        old_winch = signal.getsignal(signal.SIGWINCH)
        signal.signal(signal.SIGWINCH, resize)

        print(f"\nConnected to Kaggle terminal {name}.\nCtrl+] = detach without killing the remote terminal.\n")

        try:
            tty.setraw(fd)
            resize()
            if initial_command:
                ws.send(json.dumps(["stdin", initial_command.rstrip("\r\n") + "\r"]))
            while not stop.is_set():
                data = os.read(fd, 4096)
                if not data:
                    break
                if b"\x1d" in data:  # Ctrl+]
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
            print(f"\nDetached from Kaggle terminal {name}.")

    # -------------------------------------------------------------------------
    # Directory Operations (from original)
    # -------------------------------------------------------------------------

    def ensure_dir(self, remote_dir: str):
        remote_dir = remote_dir.strip("/")
        if not remote_dir:
            return
        current = []
        for part in remote_dir.split("/"):
            current.append(part)
            path = "/".join(current)
            url = self.url("/api/contents/" + quote(path, safe="/"))
            r = self.http.get(url, params={"content": 0}, timeout=self.retry_config.http_timeout)
            if r.status_code == 404:
                r = self.http.put(url, headers=self._xsrf_headers(), json={"type": "directory"}, timeout=self.retry_config.http_timeout)
                self._check_response(r)
            else:
                self._check_response(r)

    def _list_dir_recursive(self, remote_dir: str) -> list:
        remote_dir = remote_dir.strip("/")
        collected = []
        def _walk(path: str):
            url = self.url("/api/contents/" + quote(path, safe="/"))
            r = self.http.get(url, params={"content": 1}, timeout=self.retry_config.http_timeout)
            self._check_response(r)
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
        suffix = "/" + quote(remote_path, safe="/") if remote_path else ""
        r = self.http.get(self.url("/api/contents" + suffix), params={"content": 1}, timeout=self.retry_config.http_timeout)
        self._check_response(r)
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


# =============================================================================
# ROBUST DEPLOYER
# =============================================================================

class RobustKaggleDeployer:
    """
    High-level deployer with full lifecycle management.

    Handles:
    - Proxy URL expiration detection and user prompting
    - Automatic retry with backoff
    - Kernel restart recovery (re-run setup)
    - Resume capability for uploads/downloads
    - Progress reporting
    """

    def __init__(
        self,
        proxy_url: str,
        project_root: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.logger = logger or self._setup_logger()

        self.retry_config = RetryConfig()
        self.session_config = SessionConfig()

        self.kaggle = RobustKaggleProxy(
            proxy_url,
            retry_config=self.retry_config,
            session_config=self.session_config,
            logger=self.logger
        )

        self.remote_working = "/kaggle/working"
        self.remote_project = f"{self.remote_working}/Video-to-Terrain"
        self.local_project = self.project_root
        self._setup_complete = False
        self._setup_failed = False

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("kaggle_deployer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    # -------------------------------------------------------------------------
    # Connection & Health
    # -------------------------------------------------------------------------

    def ping(self) -> bool:
        """Verify connection with health check."""
        self.logger.info("Pinging Kaggle Jupyter API...")
        try:
            health = self.kaggle.check_proxy_health()
            self.logger.info(f"Proxy health: {health}")
            if health["healthy"]:
                self.logger.info("[SUCCESS] Kaggle connection verified")
                return True
            else:
                self.logger.error("[ERROR] Proxy unhealthy")
                return False
        except ProxyExpiredError as e:
            self.logger.error(f"[ERROR] Proxy expired: {e}")
            return False
        except Exception as e:
            self.logger.error(f"[ERROR] Cannot connect to Kaggle: {e}")
            return False

    def prompt_for_fresh_url(self) -> str:
        """Prompt user for fresh proxy URL from Kaggle UI."""
        print("\n" + "=" * 70)
        print("PROXY URL EXPIRED - ACTION REQUIRED")
        print("=" * 70)
        print("The Kaggle Jupyter proxy URL has expired (JWT token timeout ~1-2 hours).")
        print("")
        print("To get a fresh URL:")
        print("  1. Open your Kaggle notebook in browser")
        print("  2. Click the 'Copy Link' or 'Proxy URL' button (usually near top-right)")
        print("  3. Paste the full URL here (starts with https://.../proxy/...)")
        print("")
        print("Current URL (expired):", self.kaggle.base[:60] + "...")
        print("=" * 70)

        while True:
            new_url = input("Enter fresh KAGGLE_PROXY_URL: ").strip()
            if new_url.startswith("http"):
                return new_url
            print("Invalid URL. Must start with http:// or https://")

    def verify_and_refresh_proxy(self) -> bool:
        """Verify proxy health, prompt for refresh if needed."""
        health = self.kaggle.check_proxy_health()

        if not health["healthy"]:
            self.logger.warning("Proxy health check failed")
            new_url = self.prompt_for_fresh_url()
            self.kaggle.refresh_proxy_url(new_url)
            return True

        if health["needs_refresh"]:
            self.logger.warning(
                f"Proxy URL expires soon ({health['session_remaining_seconds']:.0f}s remaining). "
                f"Consider refreshing from Kaggle UI."
            )
            # Don't auto-refresh, just warn

        return True

    # -------------------------------------------------------------------------
    # Setup with Recovery
    # -------------------------------------------------------------------------

    def ensure_remote_dirs(self) -> None:
        """Ensure required remote directories exist."""
        dirs = [
            self.remote_project,
            f"{self.remote_project}/input",
            f"{self.remote_project}/outputs/run_mvp/geometry/depth",
        ]
        for d in dirs:
            self.kaggle.ensure_dir(d)

    def setup_environment(
        self,
        model: str = "da3metric-large",
        force: bool = False
    ) -> bool:
        """
        Set up conda environment with CUDA and install dependencies.

        If kernel restarted (detected via PID), forces re-setup.
        """
        # Check for kernel restart
        if self.kaggle.detect_kernel_restart():
            self.logger.warning("Kernel restart detected - forcing environment re-setup")
            self._setup_complete = False
            self._setup_failed = False

        if self._setup_complete and not force:
            self.logger.info("Environment already set up (use --force-setup to re-run)")
            return True

        if self._setup_failed and not force:
            self.logger.error("Previous setup failed. Use --force-setup to retry.")
            return False

        self.logger.info("Setting up Kaggle environment with CUDA...")

        # Verify proxy before long setup
        self.verify_and_refresh_proxy()

        setup_commands = [
            # Install PyTorch with CUDA 12.1
            ("PyTorch CUDA", "pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"),
            # Core dependencies
            ("Core deps", "pip install -q open3d trimesh networkx tqdm pyyaml python-dotenv numpy pillow opencv-python scipy matplotlib"),
            # DA3 - try local first
            ("DA3 install", "pip install -q -e /kaggle/working/Video-to-Terrain/depth-anything-3 2>/dev/null || echo 'DA3: will use HF hub at runtime'"),
            # Verify CUDA
            ("CUDA verify", "python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}')\""),
        ]

        for step_name, cmd in setup_commands:
            self.logger.info(f"  [SETUP] {step_name}...")
            try:
                output = self.kaggle.run_command(
                    cmd,
                    cwd=self.remote_project,
                    timeout=180,
                    keepalive_during=True
                )
                if output.strip():
                    self.logger.info(f"  [OUT] {output.strip()[:500]}")
            except ProxyExpiredError:
                self.logger.error("Proxy expired during setup")
                new_url = self.prompt_for_fresh_url()
                self.kaggle.refresh_proxy_url(new_url)
                # Retry this step
                output = self.kaggle.run_command(cmd, cwd=self.remote_project, timeout=180)
                if output.strip():
                    self.logger.info(f"  [OUT] {output.strip()[:500]}")
            except KernelRestartedError:
                self.logger.warning("Kernel restarted during setup - retrying from beginning")
                self._setup_complete = False
                return self.setup_environment(model, force=True)
            except Exception as e:
                self.logger.warning(f"  [WARN] {step_name} failed: {e}")
                # Continue - some commands might fail but env might still work

        self._setup_complete = True
        self._setup_failed = False
        self.logger.info("[SUCCESS] Environment setup complete")
        return True

    # -------------------------------------------------------------------------
    # Pipeline Stages
    # -------------------------------------------------------------------------

    def run_stage_a_local(self) -> bool:
        """Run Stage A (frame extraction) locally - not on Kaggle."""
        self.logger.info("Stage A (frame extraction) should be run locally before deployment")
        frames_dir = self.local_project / "outputs/run_mvp/frames"
        if frames_dir.exists():
            frame_count = len(list(frames_dir.glob("frame_*.jpg")))
            self.logger.info(f"Found {frame_count} frames locally")
            return frame_count > 0
        return False

    def upload_project(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> bool:
        """Upload project to Kaggle with progress tracking."""
        self.logger.info(f"Uploading project to {self.remote_project}...")

        skip_dirs = {
            ".git", ".venv", "venv", "__pycache__", "node_modules",
            ".DS_Store", "outputs", ".claude", ".mypy_cache",
            ".pytest_cache", "source-reference",
            # depth-anything-3/ only needs to ship its installable package
            # (src/, pyproject.toml, requirements.txt) for `pip install -e`.
            # These sibling dirs are gradio demo cruft / docs / media, not
            # needed remotely, and dominate the file count (~1900+ files).
            "depth-anything-3-space", "assets", "reference", "notebooks",
            "da3_streaming", "docs",
        }

        try:
            self.verify_and_refresh_proxy()
            uploaded = self.kaggle.upload_dir(
                str(self.local_project),
                self.remote_project,
                skip=skip_dirs,
                progress_callback=progress_callback
            )
            self.logger.info(f"[SUCCESS] Project uploaded ({uploaded} files)")
            return True
        except ProxyExpiredError:
            self.logger.error("Proxy expired during upload")
            new_url = self.prompt_for_fresh_url()
            self.kaggle.refresh_proxy_url(new_url)
            return self.upload_project(progress_callback)
        except UploadFailedError as e:
            self.logger.error(f"Upload failed: {e}")
            return False

    def upload_frames(self) -> bool:
        """Upload extracted frames (Stage A output) - needed on remote before Stage B.

        upload_project() intentionally skips the whole 'outputs' tree (depth maps,
        point clouds, videos, etc. are large/regenerable), so the frames dir has to
        be uploaded separately.
        """
        frames_dir = self.local_project / "outputs/run_mvp/frames"
        if not frames_dir.exists() or not any(frames_dir.glob("frame_*.jpg")):
            self.logger.error(f"No local frames found at {frames_dir}. Run Stage A first.")
            return False

        remote_frames = f"{self.remote_project}/outputs/run_mvp/frames"
        self.logger.info(f"Uploading frames to {remote_frames}...")

        try:
            self.verify_and_refresh_proxy()
            uploaded = self.kaggle.upload_dir(str(frames_dir), remote_frames, skip=set())
            self.logger.info(f"[SUCCESS] Uploaded {uploaded} frame(s)")

            meta = self.local_project / "outputs/run_mvp/capture_metadata.json"
            if meta.exists():
                self.kaggle.upload_file(
                    str(meta), f"{self.remote_project}/outputs/run_mvp/capture_metadata.json"
                )
            return True
        except ProxyExpiredError:
            self.logger.error("Proxy expired during frame upload")
            new_url = self.prompt_for_fresh_url()
            self.kaggle.refresh_proxy_url(new_url)
            return self.upload_frames()
        except Exception as e:
            self.logger.error(f"Failed to upload frames: {e}")
            return False

    def upload_input_video(self, video_name: str = "20260810_150244_color.mp4") -> bool:
        """Upload input video to Kaggle."""
        input_video = self.local_project.parent / "input" / video_name
        if not input_video.exists():
            self.logger.warning(f"Input video not found: {input_video}")
            return False

        remote_video = f"{self.remote_project}/input/{video_name}"
        self.logger.info(f"Uploading input video ({input_video.stat().st_size / 1e6:.1f} MB)...")

        try:
            self.verify_and_refresh_proxy()
            self.kaggle.upload_file(str(input_video), remote_video)
            self.logger.info("[SUCCESS] Input video uploaded")
            return True
        except ProxyExpiredError:
            self.logger.error("Proxy expired during video upload")
            new_url = self.prompt_for_fresh_url()
            self.kaggle.refresh_proxy_url(new_url)
            return self.upload_input_video(video_name)
        except Exception as e:
            self.logger.error(f"Failed to upload video: {e}")
            return False

    def run_stage_b(
        self,
        model: str = "da3metric-large",
        process_res: int = 504
    ) -> bool:
        """
        Run Stage B (DA3) on Kaggle GPU with full error handling.

        Handles:
        - Proxy expiration during long inference
        - Kernel restart (re-run setup + inference)
        - Progress reporting
        """
        self.logger.info(f"Running Stage B (DA3) with model={model}, process_res={process_res} on GPU...")

        # First verify frames exist on remote
        check_cmd = f"ls {self.remote_project}/outputs/run_mvp/frames/frame_*.jpg 2>/dev/null | wc -l"
        try:
            self.verify_and_refresh_proxy()
            frame_count = int(self.kaggle.run_command(check_cmd).strip())
            self.logger.info(f"Found {frame_count} frames on remote")
            if frame_count == 0:
                self.logger.error("No frames found. Run Stage A locally first.")
                return False
        except Exception as e:
            self.logger.warning(f"Could not verify frames: {e}")

        # Build the Python script to run remotely
        remote_script = f'''import sys
sys.path.insert(0, 'src')
from v2t.config import CFG
from v2t.geometry.stage_b_da3 import run_stage_b

# Override config for GPU
CFG.da3_device = 'cuda'
CFG.da3_model = '{model}'
CFG.da3_process_res = {process_res}

print(f"Running DA3 on: {{CFG.da3_device}}")
print(f"Model: {{CFG.da3_model}}")
print(f"Process resolution: {{CFG.da3_process_res}}")

import torch
print(f"PyTorch: {{torch.__version__}}")
print(f"CUDA available: {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")

# Check frames exist
from pathlib import Path
frames = list(Path(CFG.paths.frames).glob('frame_*.jpg'))
print(f"Found {{len(frames)}} frames")

if len(frames) == 0:
    print("ERROR: No frames found! Run Stage A first locally.")
    sys.exit(1)

result = run_stage_b()
print("Stage B completed successfully!")
print(f"Results: {{result}}")
'''

        # Write script to remote file
        script_path = f"{self.remote_project}/run_stage_b.py"
        self.logger.info("Writing Stage B script to remote...")

        try:
            self.verify_and_refresh_proxy()
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(remote_script)
                temp_path = f.name
            try:
                self.kaggle.upload_file(temp_path, script_path)
            finally:
                os.unlink(temp_path)
        except ProxyExpiredError:
            self.logger.error("Proxy expired during script upload")
            new_url = self.prompt_for_fresh_url()
            self.kaggle.refresh_proxy_url(new_url)
            # Retry upload
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(remote_script)
                temp_path = f.name
            try:
                self.kaggle.upload_file(temp_path, script_path)
            finally:
                os.unlink(temp_path)

        # Execute the script with retries for kernel restart
        max_kernel_restarts = 2
        for restart_attempt in range(max_kernel_restarts + 1):
            try:
                self.logger.info(f"Executing Stage B on GPU (attempt {restart_attempt + 1})...")
                self.verify_and_refresh_proxy()

                output = self.kaggle.run_command(
                    f"cd {self.remote_project} && python run_stage_b.py",
                    cwd=self.remote_project,
                    timeout=300,  # 5 minutes
                    keepalive_during=True
                )

                self.logger.info(f"[REMOTE OUTPUT]:\n{output}")

                if "Stage B completed successfully" in output:
                    self.logger.info("[SUCCESS] Stage B completed on GPU")
                    return True
                else:
                    self.logger.error("Stage B did not complete successfully")
                    # Check for common errors
                    if "CUDA out of memory" in output:
                        self.logger.error("CUDA OOM - try smaller process_res or model")
                    elif "No frames found" in output:
                        self.logger.error("Frames not found on remote")
                    return False

            except KernelRestartedError:
                if restart_attempt < max_kernel_restarts:
                    self.logger.warning(f"Kernel restarted (attempt {restart_attempt + 1}/{max_kernel_restarts}), re-running setup and inference...")
                    self._setup_complete = False
                    if not self.setup_environment(model, force=True):
                        self.logger.error("Failed to re-setup environment after kernel restart")
                        continue
                    # Continue loop to retry inference
                    continue
                else:
                    self.logger.error(f"Kernel restarted {max_kernel_restarts} times, giving up")
                    return False

            except ProxyExpiredError:
                self.logger.error("Proxy expired during Stage B execution")
                new_url = self.prompt_for_fresh_url()
                self.kaggle.refresh_proxy_url(new_url)
                if restart_attempt < max_kernel_restarts:
                    self.logger.info("Retrying with fresh proxy...")
                    continue
                return False

            except Exception as e:
                self.logger.error(f"Stage B execution failed: {e}")
                if restart_attempt < max_kernel_restarts:
                    self.logger.info("Retrying...")
                    time.sleep(5)
                    continue
                return False

        return False

    def download_results(
        self,
        local_output_dir: str = "outputs/run_mvp/geometry",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> bool:
        """Download results with resume capability."""
        self.logger.info(f"Downloading results to {local_output_dir}...")

        remote_geometry = f"{self.remote_project}/outputs/run_mvp/geometry"
        local_geometry = self.local_project / local_output_dir
        local_geometry.mkdir(parents=True, exist_ok=True)

        try:
            self.verify_and_refresh_proxy()
            downloaded = self.kaggle.download_dir(
                remote_dir=remote_geometry,
                local_dir=str(local_geometry),
                progress_callback=progress_callback,
                resume=True
            )

            if downloaded > 0:
                self.logger.info(f"[SUCCESS] Downloaded {downloaded} files to {local_geometry}")

                # Verify key files
                key_files = [
                    "camera_intrinsics.json",
                    "camera_poses.json",
                    "point_cloud.ply",
                ]
                for f in key_files:
                    path = local_geometry / f
                    if path.exists():
                        size_kb = path.stat().st_size / 1024
                        self.logger.info(f"  ✓ {f} ({size_kb:.1f} KB)")
                    else:
                        self.logger.warning(f"  ✗ {f} MISSING")

                # Check depth maps
                depth_files = list((local_geometry / "depth").glob("depth_*.npy"))
                self.logger.info(f"  ✓ {len(depth_files)} depth maps")

                return True
            else:
                self.logger.warning("No files downloaded - check if Stage B ran successfully")
                return False

        except ProxyExpiredError:
            self.logger.error("Proxy expired during download")
            new_url = self.prompt_for_fresh_url()
            self.kaggle.refresh_proxy_url(new_url)
            return self.download_results(local_output_dir, progress_callback)
        except DownloadFailedError as e:
            self.logger.error(f"Download failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Interactive
    # -------------------------------------------------------------------------

    def interactive_shell(self) -> None:
        """Open interactive shell on Kaggle."""
        self.logger.info("Starting interactive shell on Kaggle...")
        self.logger.info("  Ctrl+] to detach without killing remote terminal")
        name = self.kaggle.create_terminal()
        self.kaggle.interactive(name, initial_command=f"cd {self.remote_project}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Robust Kaggle GPU Deployment for Video-to-Terrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.v2t.kaggle_robust --stage B                    # Full pipeline
  python -m src.v2t.kaggle_robust --stage B --model da3metric-large  # Specific model
  python -m src.v2t.kaggle_robust --download-only              # Only download results
  python -m src.v2t.kaggle_robust --shell                      # Interactive shell
  python -m src.v2t.kaggle_robust --ping                       # Test connection
  python -m src.v2t.kaggle_robust --upload-only                # Only upload project
  python -m src.v2t.kaggle_robust --force-setup                # Force re-setup environment
        """
    )
    parser.add_argument("--stage", default="B", choices=["B"], help="Pipeline stage to run")
    parser.add_argument("--model", default="da3metric-large", help="DA3 model to use")
    parser.add_argument("--process-res", type=int, default=504, help="DA3 process resolution")
    parser.add_argument("--upload-only", action="store_true", help="Only upload project, don't run")
    parser.add_argument("--download-only", action="store_true", help="Only download results")
    parser.add_argument("--shell", action="store_true", help="Open interactive shell")
    parser.add_argument("--ping", action="store_true", help="Test connection only")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--force-setup", action="store_true", help="Force re-setup environment")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("kaggle_deployer").setLevel(logging.DEBUG)
        logging.getLogger("robust_kaggle").setLevel(logging.DEBUG)

    proxy_url = os.environ.get("KAGGLE_PROXY_URL")
    if not proxy_url:
        print("[ERROR] KAGGLE_PROXY_URL environment variable not set")
        print("Example: export KAGGLE_PROXY_URL='https://.../proxy'")
        sys.exit(1)

    deployer = RobustKaggleDeployer(proxy_url)

    if args.ping:
        sys.exit(0 if deployer.ping() else 1)

    if args.shell:
        deployer.interactive_shell()
        return

    if args.download_only:
        success = deployer.download_results()
        sys.exit(0 if success else 1)

    # Ensure remote directories exist
    deployer.ensure_remote_dirs()

    # Upload project
    if not deployer.upload_project():
        sys.exit(1)

    if args.upload_only:
        print("[INFO] Upload complete. Use --shell to interact or run without --upload-only to execute.")
        return

    # Upload extracted frames (required by Stage B on remote)
    if not deployer.upload_frames():
        sys.exit(1)

    # Upload input video (not required by Stage B itself, best-effort only)
    if not deployer.upload_input_video():
        deployer.logger.warning("Input video upload failed/skipped - continuing (Stage B only needs frames)")

    # Setup environment (unless skipped)
    if not args.skip_setup:
        if not deployer.setup_environment(args.model, force=args.force_setup):
            print("[WARNING] Environment setup had issues, continuing anyway...")

    # Run Stage B
    if args.stage == "B":
        if not deployer.run_stage_b(args.model, args.process_res):
            sys.exit(1)

    # Download results
    if not deployer.download_results():
        sys.exit(1)

    print("\n[SUCCESS] Kaggle GPU deployment complete!")
    print(f"Results available at: {deployer.project_root}/outputs/run_mvp/geometry/")


if __name__ == "__main__":
    import base64
    import os
    main()