"""Pull individual members out of a remote .tar without downloading it.

The Holi-Spatial releases ship one 14.4 GB tar per shard, and we only ever need
a few files from one scene. A tar is a flat sequence of 512-byte headers each
followed by its payload, so the archive can be walked with HTTP range requests
and a single member fetched by byte offset.

No dependencies beyond the standard library — this has to run on a Kaggle
kernel before anything is installed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

BLOCK = 512
_UA = {"User-Agent": "v3d/1.0"}


@dataclass(frozen=True)
class TarMember:
    name: str
    size: int
    offset: int  # byte offset of the payload, not of the header

    @property
    def end(self) -> int:
        return self.offset + self.size - 1


class RemoteTar:
    """Random access to a tar served over HTTP range requests."""

    def __init__(self, url: str, token: str | None = None, chunk: int = 4 << 20):
        self.url = url
        self.chunk = chunk
        self.headers = dict(_UA)
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self._size: int | None = None

    # ---- raw http -------------------------------------------------------

    def size(self) -> int:
        if self._size is None:
            req = urllib.request.Request(self.url, method="HEAD", headers=self.headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                self._size = int(r.headers["Content-Length"])
        return self._size

    def read_range(self, start: int, end: int) -> bytes:
        """Inclusive byte range, as HTTP defines it."""
        h = dict(self.headers, Range=f"bytes={start}-{end}")
        req = urllib.request.Request(self.url, headers=h)
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read()

    # ---- tar walking ----------------------------------------------------

    def members(self, stop_after: int | None = None) -> Iterator[TarMember]:
        """Walk headers front to back, yielding every member.

        Reads in ``chunk``-sized windows so a run of small files costs one
        request rather than one per header.
        """
        total = self.size()
        pos = 0
        buf = b""
        buf_start = 0
        pax_name: str | None = None
        yielded = 0

        while pos + BLOCK <= total:
            if not (buf_start <= pos and pos + BLOCK <= buf_start + len(buf)):
                end = min(pos + self.chunk, total) - 1
                buf = self.read_range(pos, end)
                buf_start = pos
                if not buf:
                    return

            head = buf[pos - buf_start : pos - buf_start + BLOCK]
            if head[:1] == b"\x00":  # end-of-archive padding
                return

            name = head[0:100].rstrip(b"\x00").decode("utf-8", "replace")
            try:
                size = int(head[124:136].rstrip(b"\x00 ").decode() or "0", 8)
            except ValueError:
                return
            if not name:
                return

            payload = pos + BLOCK
            advance = BLOCK + ((size + BLOCK - 1) // BLOCK) * BLOCK

            if name.endswith("@PaxHeader"):
                # The real path for the *next* entry lives in this record.
                pax = self.read_range(payload, payload + size - 1).decode("utf-8", "replace")
                pax_name = _pax_path(pax)
            else:
                yield TarMember(name=pax_name or name, size=size, offset=payload)
                pax_name = None
                yielded += 1
                if stop_after is not None and yielded >= stop_after:
                    return

            pos += advance

    def index(self, cache: str | Path | None = None) -> list[TarMember]:
        """Full member list, cached to disk because walking a 14 GB tar is slow."""
        if cache is not None:
            p = Path(cache)
            if p.exists():
                return [TarMember(**m) for m in json.loads(p.read_text())]
        members = list(self.members())
        if cache is not None:
            p = Path(cache)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps([m.__dict__ for m in members]))
        return members

    def extract(self, member: TarMember, dest: str | Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if member.size == 0:
            dest.write_bytes(b"")
            return dest
        dest.write_bytes(self.read_range(member.offset, member.end))
        return dest

    def read_member(self, member: TarMember) -> bytes:
        return b"" if member.size == 0 else self.read_range(member.offset, member.end)


def _pax_path(record: str) -> str | None:
    """PAX records are 'LEN key=value\\n'; we only care about the path key."""
    for line in record.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].startswith("path="):
            return parts[1][len("path=") :]
    return None


def hf_url(repo: str, path: str, repo_type: str = "datasets") -> str:
    return f"https://huggingface.co/{repo_type}/{repo}/resolve/main/{path}"


def find_scene_shard(
    repo: str,
    shard_paths: list[str],
    scene_id: str,
    token: str | None = None,
) -> tuple[str, list[TarMember]] | None:
    """Locate which shard holds ``scene_id`` and return its members.

    Shards are grouped by scene, so the first header is enough to tell whether
    a shard is the right one — we bail out after one member per shard.
    """
    for path in shard_paths:
        tar = RemoteTar(hf_url(repo, path), token=token)
        try:
            head = list(tar.members(stop_after=4))
        except urllib.error.HTTPError:
            continue
        if any(m.name.split("/")[0] == scene_id for m in head):
            return path, [m for m in tar.members() if m.name.startswith(scene_id)]
    return None
