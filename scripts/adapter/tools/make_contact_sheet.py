#!/usr/bin/env python3
"""Build a before/after review page for the Adapter A crops.

Puts the old union-mask crop next to the OBB-anchored one for every object, with
the numbers the gate decided on, so a whole scene can be eyeballed in one pass.

    python scripts/adapter/tools/make_contact_sheet.py \
        --old outputs-full-factory/delivery/OUTPUT_D \
        --new outputs-full-factory/delivery/OUTPUT_D_v2 \
        --c1 outputs-full-factory/delivery/OUTPUT_C1 \
        --c3 outputs-full-factory/delivery/OUTPUT_C3 \
        --out outputs-full-factory/delivery/OUTPUT_D_v2/contact_sheet.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

THUMB = 300


def encode(image: Optional[np.ndarray], size: int = THUMB) -> str:
    """Fit an image into a square checkerboard tile and return it as a data URI."""
    if image is None:
        return ""

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[..., 3:4].astype(np.float32) / 255.0
        board = np.zeros(image.shape[:2] + (3,), np.uint8)
        cell = max(8, min(image.shape[:2]) // 16)
        yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        board[...] = np.where((((yy // cell) + (xx // cell)) % 2)[..., None], 235, 205)
        image = (image[..., :3] * alpha + board * (1 - alpha)).astype(np.uint8)

    h, w = image.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    tile = np.full((size, size, 3), 24, np.uint8)
    oy, ox = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    tile[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized[..., :3]

    ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}" if ok else ""


def overlay(rgb: Optional[np.ndarray], mask: Optional[np.ndarray],
            colour) -> Optional[np.ndarray]:
    """Tint the masked region and crop to it so small objects stay visible."""
    if rgb is None or mask is None or not mask.any():
        return None
    vis = rgb.copy()
    vis[mask] = (0.45 * vis[mask] + 0.55 * np.array(colour)).astype(np.uint8)
    ys, xs = np.where(mask)
    pad = int(max(60, 0.3 * max(int(ys.max() - ys.min()), int(xs.max() - xs.min()))))
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
    return vis[y0:y1, x0:x1]


def read(path: Path, unchanged: bool = False) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR)


def read_mask(path: Path) -> Optional[np.ndarray]:
    raw = read(path)
    if raw is None:
        return None
    grey = raw if raw.ndim == 2 else cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    return grey > 127


def find_view(obj_dir: Path, view_id: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg", ".JPG"):
        candidate = obj_dir / "views" / f"{view_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def build(old: Path, new: Path, c1: Path, c3: Path, out: Path) -> None:
    report: Dict[str, Dict] = {}
    report_path = c3 / "resegment_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    cards: List[str] = []
    stats = {"objects": 0, "grew": 0, "shrank": 0, "multiview": 0, "kept_old_mask": 0}

    for meta_path in sorted((new / "adapter_a").glob("*/adapter_a_meta.json")):
        obj_id = meta_path.parent.name
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        view_id = meta["source_frame"]
        entry = report.get(obj_id, {}).get("views", {}).get(view_id, {})
        stats["objects"] += 1
        if meta["mode"] == "multiview":
            stats["multiview"] += 1
        if entry.get("variant") == "holi_original":
            stats["kept_old_mask"] += 1

        rgb = read(find_view(c1 / obj_id, view_id)) if find_view(c1 / obj_id, view_id) else None
        old_mask = read_mask(c1 / obj_id / "masks" / f"{view_id}.png")
        new_mask = read_mask(c3 / obj_id / "masks_v2" / f"{view_id}.png")

        old_area = int(old_mask.sum()) if old_mask is not None else 0
        new_area = int(new_mask.sum()) if new_mask is not None else 0
        if new_area > old_area * 1.05:
            stats["grew"] += 1
        elif new_area < old_area * 0.95:
            stats["shrank"] += 1

        tiles = [
            ("mask cũ (Holi)", encode(overlay(rgb, old_mask, (0, 0, 255)))),
            ("mask mới (OBB+SAM3)", encode(overlay(rgb, new_mask, (0, 255, 0)))),
            ("input cũ", encode(read(old / "adapter_a" / obj_id / "hunyuan_input.png", True))),
            ("input mới (RGBA)", encode(read(meta_path.parent / "hunyuan_input.png", True))),
        ]
        cells = "".join(
            f'<figure><img src="{src}" alt="{label}" loading="lazy">'
            f'<figcaption>{label}</figcaption></figure>' if src
            else f'<figure class="missing"><div>không có</div>'
                 f'<figcaption>{label}</figcaption></figure>'
            for label, src in tiles
        )

        delta = "" if not old_area else f"{new_area / old_area:.2f}×"
        badges = " ".join([
            f'<span class="b">{meta["semantic_class"]}</span>',
            f'<span class="b">{meta["mode"]}</span>',
            f'<span class="b">slots: {", ".join(meta["available_mv_slots"]) or "—"}</span>'
            if meta.get("available_mv_slots") else
            f'<span class="b">slots: {", ".join(meta["canonical_multiview_slots"]) or "—"}</span>',
            f'<span class="b">fill {entry.get("fill", 0):.2f}</span>',
            f'<span class="b {"warn" if entry.get("leak", 0) > 0.05 else ""}">'
            f'leak {entry.get("leak", 0):.3f}</span>',
            f'<span class="b">{entry.get("variant", "—")}</span>',
            f'<span class="b {"good" if new_area >= old_area else "warn"}">'
            f'{old_area:,} → {new_area:,} px {delta}</span>',
        ])

        cards.append(
            f'<article><header><h2>{obj_id}</h2><div class="badges">{badges}</div>'
            f'<div class="view">{view_id}</div></header>'
            f'<div class="row">{cells}</div></article>'
        )

    summary = (
        f'<p class="sum"><b>{stats["objects"]}</b> objects · '
        f'<b>{stats["grew"]}</b> mask lớn hơn · '
        f'<b>{stats["shrank"]}</b> nhỏ hơn · '
        f'<b>{stats["kept_old_mask"]}</b> giữ mask cũ (fallback thắng) · '
        f'<b>{stats["multiview"]}</b> đủ ≥2 slot cho 2mv</p>'
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("{{SUMMARY}}", summary).replace("{{CARDS}}", "\n".join(cards)),
                   encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {stats['objects']} objects)")


PAGE = """<title>Adapter A Crop Review</title>
<style>
  :root{--bg:#fbfaf8;--fg:#1a1917;--mut:#6b6660;--card:#fff;--line:#e3ded6;
        --good:#1a7f4b;--warn:#a8410f;--chip:#f0ece5;}
  :root:not([data-theme="light"]){}
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --bg:#16151a;--fg:#eceaf0;--mut:#9b96a3;--card:#1f1e25;--line:#33313b;
    --good:#5fd39b;--warn:#f0a06a;--chip:#2b2933;}}
  :root[data-theme="dark"]{--bg:#16151a;--fg:#eceaf0;--mut:#9b96a3;--card:#1f1e25;
    --line:#33313b;--good:#5fd39b;--warn:#f0a06a;--chip:#2b2933;}
  *{box-sizing:border-box}
  body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
       font:15px/1.5 ui-sans-serif,-apple-system,"Segoe UI",sans-serif;}
  h1{font-size:1.55rem;margin:0 0 .4rem}
  .sum{color:var(--mut);margin:0 0 2rem;font-size:.95rem}
  .sum b{color:var(--fg)}
  article{background:var(--card);border:1px solid var(--line);border-radius:12px;
          padding:1rem;margin-bottom:1.1rem;}
  header{display:flex;flex-wrap:wrap;gap:.5rem .9rem;align-items:baseline;margin-bottom:.75rem}
  h2{font-size:1rem;margin:0;font-family:ui-monospace,monospace}
  .view{color:var(--mut);font-family:ui-monospace,monospace;font-size:.8rem;margin-left:auto}
  .badges{display:flex;flex-wrap:wrap;gap:.35rem}
  .b{background:var(--chip);border-radius:999px;padding:.12rem .55rem;font-size:.75rem;
     font-family:ui-monospace,monospace;color:var(--mut)}
  .b.good{color:var(--good)} .b.warn{color:var(--warn)}
  .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.6rem;
       overflow-x:auto}
  figure{margin:0}
  img{width:100%;max-width:100%;display:block;border-radius:7px;background:#111}
  figcaption{color:var(--mut);font-size:.72rem;text-align:center;margin-top:.3rem}
  .missing div{aspect-ratio:1;display:grid;place-items:center;border-radius:7px;
               border:1px dashed var(--line);color:var(--mut);font-size:.75rem}
</style>
<h1>Adapter A — cắt ảnh trước / sau</h1>
{{SUMMARY}}
{{CARDS}}
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Adapter A before/after contact sheet")
    p.add_argument("--old", type=Path, default=Path("outputs-full-factory/delivery/OUTPUT_D"))
    p.add_argument("--new", type=Path, default=Path("outputs-full-factory/delivery/OUTPUT_D_v2"))
    p.add_argument("--c1", type=Path, default=Path("outputs-full-factory/delivery/OUTPUT_C1"))
    p.add_argument("--c3", type=Path, default=Path("outputs-full-factory/delivery/OUTPUT_C3"))
    p.add_argument("--out", type=Path,
                   default=Path("outputs-full-factory/delivery/OUTPUT_D_v2/contact_sheet.html"))
    a = p.parse_args()
    build(a.old, a.new, a.c1, a.c3, a.out)


if __name__ == "__main__":
    main()
