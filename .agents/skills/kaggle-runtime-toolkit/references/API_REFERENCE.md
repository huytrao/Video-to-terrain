# Kaggle Runtime Toolkit — API & Architecture Reference

This document provides deep technical details on how the Kaggle Runtime Toolkit communicates with remote Kaggle Jupyter sessions without requiring Kaggle API keys or Docker containers.

---

## 1. Overview of Protocols

The toolkit interacts with two primary Jupyter subsystems exposed via Kaggle's reverse proxy:

```
+-------------------+        WebSocket (JSON RPC)        +----------------------+
|  kaggle_executor  | =================================> | Jupyter Kernel (IPy) |
|  (Execute / Stream)                                   | (PyTorch, CUDA GPU)  |
+-------------------+                                    +----------------------+
                                                                    ^
                                                                    | Local FS
+-------------------+        HTTP REST (Base64 PUT/GET)  +----------------------+
|   kaggle_remote   | ---------------------------------> | Jupyter Contents API |
|   kaggle_relay    |                                    | (/kaggle/working)    |
+-------------------+                                    +----------------------+
          |                                                         ^
          |                  WebSocket (PTY I/O)                    |
          +=================================================> Terminado / bash
```

---

## 2. Kernel Execution Protocol (`kaggle_executor.py`)

### 2.1 Handshake & Session Warmup
1. `GET /api` warm-up call collects cookies and the `_xsrf` token.
2. `GET /api/kernels` lists active kernels. If none exist, `POST /api/kernels` spawns a new `python3` kernel.
3. A WebSocket connection is established to `wss://<proxy_domain>/.../api/kernels/<kernel_id>/channels`.

### 2.2 Message Types
The Jupyter message spec (v5.3) is used:
- `execute_request`: Sends code payload to the kernel with a unique `msg_id`.
- `stream`: Captures live `stdout` and `stderr` chunks in real-time.
- `display_data` / `execute_result`: Captures rich outputs (JSON, text, HTML).
- `error`: Captures stack traces (`ename`, `evalue`, `traceback`).
- `status`: Indicates `idle` or `busy`. When `status == "idle"` for the matching `parent_header.msg_id`, execution is complete.

---

## 3. Contents API Protocol (`kaggle_remote.py`, `kaggle_relay.py`)

Kaggle exposes the standard Jupyter Contents API under `/api/contents/<path>`:

### 3.1 Uploading Files (`PUT /api/contents/<path>`)
- Payload structure:
  ```json
  {
    "type": "file",
    "format": "base64",
    "content": "<base64_encoded_data>"
  }
  ```
- **Important Limit**: Uploading files larger than ~24 MB in a single request can be truncated by the proxy without returning an HTTP error. The toolkit chunks transfers automatically (via `kaggle_relay.py` or chunked multipart upload).

### 3.2 Downloading Files (`GET /api/contents/<path>`)
- Returns JSON containing file metadata and base64-encoded file content.
- For directories, `GET /api/contents/<path>` returns the directory listing with item types and timestamps.

---

## 4. Interactive Terminal Protocol (`kaggle_remote.py`)

- `POST /api/terminals` creates a new PTY session and returns its ID (e.g. `"1"`).
- WebSocket connection to `wss://<proxy_domain>/.../terminals/websocket/<id>`.
- Client runs in raw terminal mode (`tty.setraw`), piping stdin/stdout bidirectionally and sending JSON `["stdin", "..."]` messages.
- Disconnecting (`Ctrl+]`) leaves the shell alive on Kaggle so long-running jobs continue uninterrupted.
