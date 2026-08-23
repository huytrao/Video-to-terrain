#!/usr/bin/env python3
"""Gather every visualizable artefact in the pipeline into one browsable folder.

    python scripts/tools/collect_visualize.py

Outputs are scattered across OUTPUT_B1 (reconstruction), OUTPUT_D_v2 (object
crops), OUTPUT_I (assembled scene) and OUTPUT_ROOM (robot scene), each in its own
layout. This copies them into `visualize/` grouped by pipeline stage and writes an
index page, so the whole run can be reviewed without knowing where anything lives.

Files are copied, not moved or symlinked: the folder has to survive a rebuild of
any stage, and several of these were already lost once to an `rm -rf` of their
source directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

DELIVERY = Path("outputs-full-factory/delivery")
ROOM = DELIVERY / "OUTPUT_ROOM"

# stage folder -> (title, blurb, [(source, renamed-to, caption)])
PLAN: List[Dict] = [
    {
        "dir": "1_reconstruction",
        "title": "1 · Reconstruction (Holi-Spatial)",
        "blurb": "What the 4K video became: mesh, depth, normals, and the "
                 "ground-truth boxes the later stages depend on.",
        "items": [
            (DELIVERY / "OUTPUT_B1/videos/01_mesh_turntable.mp4", None, "Mesh turntable"),
            (DELIVERY / "OUTPUT_B1/videos/02_mesh_with_gt_boxes.mp4", None, "Mesh with the 223 ground-truth OBBs"),
            (DELIVERY / "OUTPUT_B1/videos/03_3dgs_renders.mp4", None, "3D Gaussian splat renders"),
            (DELIVERY / "OUTPUT_B1/videos/04_source_photos.mp4", None, "Source photos"),
            (DELIVERY / "OUTPUT_B1/videos/05_normals.mp4", None, "Surface normals"),
            (DELIVERY / "OUTPUT_B1/videos/06_depth.mp4", None, "Depth"),
            (DELIVERY / "OUTPUT_B1/videos/07_cutaway_interior.mp4", None, "Cutaway interior"),
            (DELIVERY / "OUTPUT_B1/visualizations/render_vs_photo.png", None, "Render vs photo — the reconstruction's honesty check"),
            (DELIVERY / "OUTPUT_B1/visualizations/floorplan.png", None, "Floor plan"),
            (DELIVERY / "OUTPUT_B1/visualizations/depth_maps.png", None, "Depth maps"),
            (DELIVERY / "OUTPUT_B1/visualizations/boxes_00.png", None, "Ground-truth boxes"),
            (DELIVERY / "OUTPUT_B1/visualizations/cutaway_00.png", None, "Cutaway"),
            (DELIVERY / "OUTPUT_B1/visualizations/orbit_02.png", None, "Orbit"),
        ],
    },
    {
        "dir": "2_object_crops",
        "title": "2 · Object crops (OBB-anchored SAM 3)",
        "blurb": "Before/after for every object cut out for Hunyuan3D. 29 of 38 "
                 "masks grew, none shrank. Open the contact sheet to review the lot.",
        "items": [(DELIVERY / "OUTPUT_D_v2/contact_sheet.html", None,
                   "Contact sheet — old mask, new mask, old crop, new crop, per object")],
    },
    {
        "dir": "3_assembled_scene",
        "title": "3 · Assembled scene",
        "blurb": "The 38 completed object meshes placed back into the room.",
        "items": [
            (DELIVERY / "OUTPUT_I/factory_scene_overview.png", None, "Scene overview"),
            (DELIVERY / "OUTPUT_I/factory_full_scene_render_4views.png", None, "Full scene, four views"),
            (DELIVERY / "OUTPUT_I/factory_photorealistic_textured_4views.png", None, "Textured, four views"),
            (DELIVERY / "OUTPUT_I/key_objects_textured_closeup.png", None, "Key objects close up"),
            (DELIVERY / "OUTPUT_I/robotics_manipulation_layout.png", None, "Manipulation layout"),
        ],
    },
    {
        "dir": "4_robot_scene",
        "title": "4 · Robot scene (MuJoCo)",
        "blurb": "The room turned into physics: gravity fixed, a primitive collider, "
                 "79 furniture boxes, and a spawn point with measured clearance.",
        "items": [
            (ROOM / "room_robot_check.png", None,
             "Scene build check — robot at t=0 and t=3s, collider shell, top down"),
        ],
    },
    {
        "dir": "5_policy",
        "title": "5 · Policy in the room",
        "blurb": "policy_12000.onnx driving the robot. Standing holds indefinitely; "
                 "walking covers 0.385 m/s on open floor and stops when it meets furniture.",
        "items": [
            (ROOM / "walk_4s.mp4", None, "Walking, 4 s — 1.20 m at 0.30 m/s, before it reaches the furniture"),
            (ROOM / "walk_tour.mp4", None, "Attempted tour — the room is only 8% passable, see the report"),
        ],
    },
]

# Stages 6 and 7 are produced in place by their own scripts, so the collector
# links them rather than copying hundreds of MB a second time.
GENERATED = [
    {"dir": "6_headings", "title": "6 \u00b7 Straight walks, eight headings",
     "blurb": "The policy turned to each of eight compass headings and told to walk "
              "straight at 0.5 m/s for 12 s. Best clean run: +315\u00b0, 2.99 m, ran the "
              "full 12 s upright. Two headings end in a fall and one is a lurch, "
              "not a walk \u2014 see headings_report.json.",
     "regen": "python scripts/sim/walk_headings.py --headings 8 --seconds 12"},
    {"dir": "7_objects_3d", "title": "7 \u00b7 Completed objects in 3D",
     "blurb": "Four Hunyuan3D meshes aligned back into the room: orbit them live, "
              "or watch the turntable. They render grey because the meshes carry "
              "uniform grey vertex colour \u2014 Hunyuan3D generates shape, not appearance.",
     "regen": "python scripts/tools/visualize_objects.py"},
]

REPORTS = [
    (ROOM / "room_scene_report.json", "Scene build: orientation, collider, spawn, both physics checks"),
    (ROOM / "policy_play_report.json", "Policy run: distance, speed, uprightness, what it hit"),
    (ROOM / "walk_tour_report.json", "Tour: waypoints reached, path length"),
    (Path("visualize/6_headings/headings_report.json"),
     "Eight headings: distance, speed, uprightness, what each run touched"),
    (Path("visualize/7_objects_3d/objects_report.json"),
     "Object meshes: size, face count, volume, watertightness"),
]


def copy_into(items, dest: Path) -> List[Dict]:
    dest.mkdir(parents=True, exist_ok=True)
    out: List[Dict] = []
    for src, rename, caption in items:
        if not src.is_file():
            print(f"  [skip] {src} (missing)")
            continue
        target = dest / (rename or src.name)
        shutil.copy2(src, target)
        out.append({"file": target.name, "caption": caption,
                    "mb": round(target.stat().st_size / 1e6, 2),
                    "kind": target.suffix.lower()})
        print(f"  {target.name}  {out[-1]['mb']} MB")
    return out


def render_policy_stills(out_dir: Path, policy: Path, scene: Path) -> Optional[Dict]:
    """Four moments of the policy run, in case an earlier render was lost."""
    import sys
    sys.path.insert(0, str(Path("scripts/sim").resolve()))
    try:
        from play_policy import PolicyRunner, render
    except Exception as exc:
        print(f"  [skip] policy stills: {exc}")
        return None

    tag = policy.stem.replace("policy_", "")
    deploy = policy.with_name(f"deploy_{tag}.yaml")
    if not deploy.is_file() or not scene.is_file() or not policy.is_file():
        print("  [skip] policy stills: policy, deploy or scene missing")
        return None

    runner = PolicyRunner(scene, policy, deploy, command=[0.0, 0.0, 0.0])
    png = render(runner, out_dir / "policy_stand_4moments.png", 10.0)
    print(f"  {png.name}  {round(png.stat().st_size / 1e6, 2)} MB")
    return {"file": png.name, "caption": "Standing, four moments over 10 s",
            "mb": round(png.stat().st_size / 1e6, 2), "kind": ".png"}


def figure(stage_dir: str, item: Dict) -> str:
    src = f"{stage_dir}/{item['file']}"
    if item["kind"] == ".mp4":
        media = (f'<video src="{src}" controls preload="metadata" '
                 f'playsinline muted loop></video>')
    elif item["kind"] == ".html":
        media = (f'<a class="doc" href="{src}"><span>{item["file"]}</span>'
                 f'<em>open page</em></a>')
    else:
        media = f'<a href="{src}"><img src="{src}" alt="{item["caption"]}" loading="lazy"></a>'
    return (f'<figure>{media}<figcaption>{item["caption"]}'
            f'<span class="sz">{item["mb"]} MB</span></figcaption></figure>')


def build_index(root: Path, stages: List[Dict], reports: List[Dict]) -> Path:
    sections = []
    for stage in stages:
        if not stage["items"]:
            continue
        cards = "".join(figure(stage["dir"], i) for i in stage["items"])
        sections.append(
            f'<section><h2>{stage["title"]}</h2>'
            f'<p class="blurb">{stage["blurb"]}</p>'
            f'<div class="grid">{cards}</div></section>'
        )

    rows = "".join(
        f'<li><a href="reports/{r["file"]}">{r["file"]}</a>'
        f'<span>{r["caption"]}</span></li>' for r in reports
    )
    if rows:
        sections.append(f'<section><h2>Reports</h2><ul class="reports">{rows}</ul></section>')

    total = sum(i["mb"] for s in stages for i in s["items"])
    count = sum(len(s["items"]) for s in stages)
    page = PAGE.replace("{{SECTIONS}}", "\n".join(sections))
    page = page.replace("{{SUMMARY}}",
                        f"{count} artefacts · {total:.0f} MB · "
                        f"{len([s for s in stages if s['items']])} stages")
    path = root / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Video-to-3D — visual output</title>
<style>
  :root{--bg:#faf9f7;--fg:#1b1a18;--mut:#6c6660;--card:#fff;--line:#e4dfd7;--chip:#f1ede6;}
  @media (prefers-color-scheme:dark){:root{
    --bg:#151419;--fg:#ecebf0;--mut:#9a95a2;--card:#1e1d24;--line:#33313b;--chip:#2a2832;}}
  *{box-sizing:border-box}
  body{margin:0;padding:2.2rem 1.25rem 5rem;background:var(--bg);color:var(--fg);
       font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif;}
  .wrap{max-width:1240px;margin:0 auto}
  h1{font-size:1.6rem;margin:0 0 .3rem}
  .sub{color:var(--mut);margin:0 0 .6rem;font-family:ui-monospace,monospace;font-size:.85rem}
  .videos{display:inline-block;margin-bottom:2.2rem;padding:.35rem .8rem;
          background:var(--chip);border-radius:99px;text-decoration:none;
          color:inherit;font-size:.84rem}
  section{margin-bottom:2.8rem}
  h2{font-size:1.08rem;margin:0 0 .3rem;padding-bottom:.45rem;border-bottom:1px solid var(--line)}
  .blurb{color:var(--mut);margin:.5rem 0 1rem;font-size:.9rem;max-width:74ch}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:1rem}
  figure{margin:0;background:var(--card);border:1px solid var(--line);
         border-radius:12px;padding:.6rem;display:flex;flex-direction:column}
  img,video{width:100%;display:block;border-radius:8px;background:#0d0d0f}
  img{cursor:zoom-in}
  .doc{display:flex;flex-direction:column;align-items:center;justify-content:center;
       gap:.4rem;aspect-ratio:16/10;border:1px dashed var(--line);border-radius:8px;
       text-decoration:none;color:var(--fg);background:var(--chip)}
  .doc span{font-family:ui-monospace,monospace;font-size:.82rem}
  .doc em{color:var(--mut);font-style:normal;font-size:.76rem}
  figcaption{color:var(--mut);font-size:.79rem;margin-top:.5rem;display:flex;
             justify-content:space-between;gap:.6rem;align-items:baseline}
  .sz{font-family:ui-monospace,monospace;font-size:.7rem;white-space:nowrap;opacity:.7}
  ul.reports{list-style:none;padding:0;margin:.6rem 0 0}
  ul.reports li{display:flex;gap:1rem;padding:.5rem 0;border-bottom:1px solid var(--line);
                align-items:baseline;flex-wrap:wrap}
  ul.reports a{font-family:ui-monospace,monospace;font-size:.84rem;color:inherit}
  ul.reports span{color:var(--mut);font-size:.84rem}
</style>
<div class="wrap">
  <h1>Video-to-3D — visual output</h1>
  <p class="sub">{{SUMMARY}}</p>
  <a class="videos" href="videos.html">▶ all videos on one page →</a>
  {{SECTIONS}}
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect every visualizable output")
    ap.add_argument("--out", type=Path, default=Path("visualize"))
    ap.add_argument("--policy", type=Path,
                    default=Path("/Users/traoanhuy/source_code/Robot_code/"
                                 "vinrobotics-vr_h3_2/logs/rsl_rl/exported/policy_12000.onnx"))
    ap.add_argument("--trim", type=float, default=7.0,
                    help="cut every video to its first N seconds; 0 keeps them whole")
    ap.add_argument("--no-render", action="store_true",
                    help="do not regenerate stills that are missing")
    args = ap.parse_args()

    root = args.out
    root.mkdir(parents=True, exist_ok=True)

    stages: List[Dict] = []
    for spec in PLAN:
        print(f"\n[{spec['dir']}]")
        items = copy_into(spec["items"], root / spec["dir"])

        # The policy stills get lost whenever OUTPUT_ROOM is rebuilt, so make them
        # again rather than shipping a gap.
        if spec["dir"] == "5_policy" and not args.no_render:
            still = render_policy_stills(root / spec["dir"], args.policy,
                                         ROOM / "room_scene.xml")
            if still:
                items.insert(0, still)

        stages.append({**spec, "items": items})

    print("\n[reports]")
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for src, caption in REPORTS:
        if not src.is_file():
            print(f"  [skip] {src.name}")
            continue
        shutil.copy2(src, report_dir / src.name)
        reports.append({"file": src.name, "caption": caption})
        print(f"  {src.name}")

    # Fold in anything the dedicated scripts already wrote into visualize/.
    for spec in GENERATED:
        folder = root / spec["dir"]
        if not folder.is_dir():
            print(f"\n[{spec['dir']}] not built yet — run: {spec['regen']}")
            continue
        print(f"\n[{spec['dir']}] linking in place")
        items = []
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in (".mp4", ".png", ".html"):
                continue
            if f.name == "index.html":
                items.insert(0, {"file": f.name, "caption": "Gallery page",
                                 "mb": round(f.stat().st_size / 1e6, 2), "kind": ".html"})
                continue
            items.append({"file": f.name, "caption": f.stem.replace("_", " "),
                          "mb": round(f.stat().st_size / 1e6, 2),
                          "kind": f.suffix.lower()})
        stages.append({**spec, "items": items})
        print(f"  {len(items)} items")

    # Trim before indexing. The collector copies pristine originals out of
    # outputs-full-factory/, which silently undoes any earlier trim — so the cut
    # has to happen after the copy, and the index has to be written after that or
    # it reports the pre-trim durations.
    if args.trim:
        import subprocess as _sp
        import sys as _sys
        print(f"\n[trim] cutting to {args.trim:g}s")
        _sp.run([_sys.executable, "scripts/tools/trim_videos.py",
                 "--root", str(root), "--seconds", str(args.trim)], check=False)

    index = build_index(root, stages, reports)

    # The video page indexes whatever ended up in visualize/, so it has to be
    # written after everything else has landed.
    try:
        import subprocess, sys as _sys
        subprocess.run([_sys.executable, "scripts/tools/make_video_index.py",
                        "--root", str(root)], check=True, capture_output=True)
        print(f"[videos] {root}/videos.html")
    except Exception as exc:
        print(f"[videos] skipped: {exc}")

    total = sum(i["mb"] for s in stages for i in s["items"])
    print(f"\n[done] {index}  ({sum(len(s['items']) for s in stages)} artefacts, "
          f"{total:.0f} MB)")
    print(f"       open {index}")


if __name__ == "__main__":
    main()
