#!/usr/bin/env python3
"""Build a single page listing every video in visualize/, with playback and verdicts.

    python scripts/tools/make_video_index.py

The main index mixes videos with stills and reports; this is the videos alone,
in one grid, each with its duration and — for the policy runs — what actually
happened in it, read from the JSON reports rather than restated by hand.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# folder -> (section title, blurb)
SECTIONS = {
    "1_reconstruction": ("Reconstruction",
                         "What the 4K video became, before any robot was involved."),
    "5_policy": ("Policy in the room",
                 "The trained policy driving the robot around the reconstructed factory."),
    "6_headings": ("Straight walks, eight headings",
                   "Turn on the spot to a compass heading, then walk straight at "
                   "0.5 m/s for 12 s. Only one heading has open floor ahead of it."),
    "7_objects_3d": ("Object turntables",
                     "Completed Hunyuan3D meshes, orbited. Grey because the meshes "
                     "carry uniform grey vertex colour — shape without appearance."),
}

CAPTIONS = {
    "01_mesh_turntable": "Reconstructed mesh, full turn",
    "02_mesh_with_gt_boxes": "Mesh with the 223 ground-truth boxes",
    "03_3dgs_renders": "3D Gaussian splat renders",
    "04_source_photos": "The source photographs",
    "05_normals": "Surface normals",
    "06_depth": "Depth",
    "07_cutaway_interior": "Cutaway through the interior",
    "walk_4s": "Walking, 4 s — the clean stretch before the furniture",
    "walk_tour": "Attempted waypoint tour — the room is only 8% passable",
}


def duration(path: Path) -> Optional[float]:
    """Seconds, from ffprobe; None if it cannot be read."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    try:
        out = subprocess.run(
            [exe, "-i", str(path)], capture_output=True, text=True, timeout=30).stderr
        for line in out.splitlines():
            if "Duration:" in line:
                hms = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = hms.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return None
    return None


def heading_facts(root: Path) -> Dict[str, Dict]:
    """Map heading video filename -> its measured outcome."""
    report = root / "6_headings" / "headings_report.json"
    if not report.is_file():
        return {}
    data = json.loads(report.read_text(encoding="utf-8"))
    best = data.get("best_heading_deg")
    out = {}
    for run in data.get("runs", []):
        name = f"heading_{int(round(run['heading_deg'])):03d}.mp4"
        if run.get("fell"):
            verdict, tone = "fell", "bad"
        elif not run.get("clean", True):
            verdict, tone = f"lurched, {run['free_speed_mps']:.2f} m/s", "warn"
        elif run.get("stopped_at_s") is not None:
            verdict, tone = f"blocked at {run['stopped_at_s']}s", "warn"
        else:
            verdict, tone = "ran clear", "good"
        out[name] = {
            "caption": f"Heading {run['heading_deg']:+.0f}°",
            "facts": f"{run['distance_m']:.2f} m · {run['free_speed_mps']:.3f} m/s · "
                     f"upright {run['upright']:.2f}",
            "verdict": verdict, "tone": tone,
            "best": run["heading_deg"] == best,
        }
    return out


def collect(root: Path) -> List[Dict]:
    facts = heading_facts(root)
    sections: List[Dict] = []

    for folder, (title, blurb) in SECTIONS.items():
        directory = root / folder
        if not directory.is_dir():
            continue
        videos = []
        for path in sorted(directory.glob("*.mp4")):
            stem = path.stem
            info = facts.get(path.name, {})
            caption = info.get("caption") or CAPTIONS.get(stem) or \
                stem.replace("_turntable", "").replace("_", " ")
            secs = duration(path)
            videos.append({
                "src": f"{folder}/{path.name}",
                "file": path.name,
                "caption": caption,
                "facts": info.get("facts", ""),
                "verdict": info.get("verdict", ""),
                "tone": info.get("tone", ""),
                "best": info.get("best", False),
                "mb": round(path.stat().st_size / 1e6, 1),
                "secs": None if secs is None else round(secs, 1),
            })
        if videos:
            sections.append({"title": title, "blurb": blurb, "videos": videos})
            print(f"[{folder}] {len(videos)} videos")
    return sections


def build(root: Path, sections: List[Dict]) -> Path:
    blocks = []
    for section in sections:
        cards = []
        for v in section["videos"]:
            badge = ('<span class="badge best">best</span>' if v["best"] else "")
            verdict = (f'<span class="verdict {v["tone"]}">{v["verdict"]}</span>'
                       if v["verdict"] else "")
            facts = f'<div class="facts">{v["facts"]}</div>' if v["facts"] else ""
            meta = " · ".join(filter(None, [
                f'{v["secs"]:g}s' if v["secs"] else None, f'{v["mb"]} MB']))
            cards.append(f"""<figure>
  <video src="{v['src']}" controls preload="metadata" playsinline muted loop></video>
  <figcaption>
    <div class="row"><b>{v['caption']}</b>{badge}{verdict}</div>
    {facts}
    <div class="meta">{v['file']} · {meta}</div>
  </figcaption>
</figure>""")
        blocks.append(f'<section><h2>{section["title"]}</h2>'
                      f'<p class="blurb">{section["blurb"]}</p>'
                      f'<div class="grid">{"".join(cards)}</div></section>')

    count = sum(len(s["videos"]) for s in sections)
    total = sum(v["mb"] for s in sections for v in s["videos"])
    page = PAGE.replace("{{SECTIONS}}", "\n".join(blocks))
    page = page.replace("{{SUMMARY}}", f"{count} videos · {total:.0f} MB")
    path = root / "videos.html"
    path.write_text(page, encoding="utf-8")
    return path


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Video index</title>
<style>
  :root{--bg:#faf9f7;--fg:#1b1a18;--mut:#6c6660;--card:#fff;--line:#e4dfd7;
        --good:#1a7f4b;--warn:#a8410f;--bad:#a01f1f;--chip:#f1ede6;}
  @media (prefers-color-scheme:dark){:root{
    --bg:#151419;--fg:#ecebf0;--mut:#9a95a2;--card:#1e1d24;--line:#33313b;
    --good:#5fd39b;--warn:#f0a06a;--bad:#f07a7a;--chip:#2a2832;}}
  *{box-sizing:border-box}
  body{margin:0;padding:2.2rem 1.25rem 5rem;background:var(--bg);color:var(--fg);
       font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif;}
  .wrap{max-width:1280px;margin:0 auto}
  h1{font-size:1.55rem;margin:0 0 .3rem}
  .sub{color:var(--mut);margin:0 0 .4rem;font-family:ui-monospace,monospace;font-size:.85rem}
  .back{display:inline-block;margin-bottom:2rem;color:var(--mut);font-size:.85rem}
  section{margin-bottom:2.6rem}
  h2{font-size:1.06rem;margin:0 0 .3rem;padding-bottom:.45rem;border-bottom:1px solid var(--line)}
  .blurb{color:var(--mut);margin:.5rem 0 1rem;font-size:.9rem;max-width:76ch}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:1rem}
  figure{margin:0;background:var(--card);border:1px solid var(--line);
         border-radius:12px;padding:.6rem;display:flex;flex-direction:column}
  video{width:100%;display:block;border-radius:8px;background:#0d0d0f}
  figcaption{margin-top:.55rem;font-size:.82rem}
  .row{display:flex;align-items:center;gap:.45rem;flex-wrap:wrap}
  .facts{color:var(--mut);font-family:ui-monospace,monospace;font-size:.75rem;margin-top:.25rem}
  .meta{color:var(--mut);font-family:ui-monospace,monospace;font-size:.7rem;
        margin-top:.3rem;opacity:.75}
  .badge,.verdict{border-radius:99px;padding:.06rem .5rem;font-size:.7rem;
                  font-family:ui-monospace,monospace;background:var(--chip)}
  .badge.best{color:var(--good);font-weight:600}
  .verdict.good{color:var(--good)} .verdict.warn{color:var(--warn)}
  .verdict.bad{color:var(--bad)}
</style>
<div class="wrap">
  <h1>Video index</h1>
  <p class="sub">{{SUMMARY}}</p>
  <a class="back" href="index.html">← everything else (stills, 3D, reports)</a>
  {{SECTIONS}}
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Index every video in visualize/")
    ap.add_argument("--root", type=Path, default=Path("visualize"))
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"{args.root} does not exist — run `make visualize` first")

    sections = collect(args.root)
    if not sections:
        raise SystemExit("no videos found")
    page = build(args.root, sections)
    count = sum(len(s["videos"]) for s in sections)
    print(f"\n[done] {page}  ({count} videos)")


if __name__ == "__main__":
    main()
