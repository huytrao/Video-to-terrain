#!/usr/bin/env python3
"""Serve the local Qwen3-VL as an OpenAI-compatible endpoint, for one caller.

`3d_bounding_instance_gs_region.py:184` verifies a borderline instance by POSTing
three images and a prompt to `http://127.0.0.1:8000/v1/chat/completions`, the
vLLM API. That server cannot run here: the RUN kernel has no internet, and
standing up vLLM alongside a 3DGS job would fight it for the GPU.

The stage's quality logic depends on that call. `3d_bounding_instance_gs_region.py`
accepts an instance outright at score >= 0.9, drops it below 0.75, and asks the
VLM about everything in between — so with no server every borderline instance is
silently rejected, and the box set is quietly biased toward whatever SAM3 happened
to be confident about. Serving the same API locally keeps the paper's script
unmodified while making its verification path actually run.

Scope on purpose: one model, one endpoint, one request at a time. It is not a
general vLLM replacement and does not try to be.

    python3 qwen_openai_shim.py --model /tmp/qwen3vl8b --port 8000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

_MODEL = None
_PROCESSOR = None
_LOCK = threading.Lock()          # generation is not re-entrant; serialise it


def load(model_path: str, device: str = "cuda:0"):
    global _MODEL, _PROCESSOR
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logging.info("loading %s", model_path)
    _PROCESSOR = AutoProcessor.from_pretrained(model_path)
    _MODEL = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device
    )
    _MODEL.eval()
    logging.info("model ready")


def _decode(url: str) -> Image.Image:
    """`data:image/jpeg;base64,...` -> PIL image."""
    if "," in url:
        url = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB")


def generate(messages: list[dict], max_tokens: int = 2048) -> str:
    """Run one chat completion, in the same shape the caller expects back."""
    images: list[Image.Image] = []
    parts: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
            continue
        for item in content or []:
            if item.get("type") == "image_url":
                images.append(_decode(item["image_url"]["url"]))
                parts.append({"type": "image"})
            elif item.get("type") == "text":
                parts.append({"type": "text", "text": item["text"]})

    chat = [{"role": "user", "content": parts}]
    prompt = _PROCESSOR.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = _PROCESSOR(text=[prompt], images=images or None, return_tensors="pt")
    inputs = {k: v.to(_MODEL.device) for k, v in inputs.items()}

    import torch

    with torch.no_grad():
        out = _MODEL.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    return _PROCESSOR.decode(trimmed, skip_special_tokens=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # keep the stage log readable
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("health"):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if "chat/completions" not in self.path:
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        try:
            with _LOCK:
                text = generate(req.get("messages", []),
                                int(req.get("max_tokens", 2048)))
        except Exception as exc:                       # noqa: BLE001
            logging.exception("generation failed")
            # A 500 makes the caller's `raise_for_status()` fail loudly, which is
            # correct: a silent False would look identical to a real REJECT and
            # would quietly drop a real object.
            self._send(500, {"error": str(exc)})
            return
        self._send(200, {"choices": [{"message": {"role": "assistant",
                                                  "content": text}}]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/qwen3vl8b")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    load(args.model, args.device)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    logging.info("serving on http://127.0.0.1:%d/v1/chat/completions", args.port)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
