#!/usr/bin/env python3
"""Render completed object meshes as turntables and a browsable 3D gallery.

    python scripts/tools/visualize_objects.py \
        obj_0052_machine obj_0179_workbench obj_0202_blade obj_0203_trash_can

For each object: a four-view still, a turntable MP4, and a copy of the GLB behind
a `<model-viewer>` so it can be orbited in the browser. The gallery page falls
back to the turntable when the viewer script is unavailable, because these files
are meant to be openable offline.

Rendering goes through MuJoCo's offscreen renderer rather than trimesh's, which
needs pyglet — not installed, and not worth a new dependency when MuJoCo is
already required and proven here.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh

OUTPUT_G = Path("outputs-full-factory/delivery/OUTPUT_G")


def load(path: Path) -> Optional[trimesh.Trimesh]:
    if not path.is_file():
        return None
    scene = trimesh.load(str(path))
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene
    return mesh if isinstance(mesh, trimesh.Trimesh) else None


def render_views(mesh: trimesh.Trimesh, angles, size=(640, 640)) -> List[np.ndarray]:
    """Offscreen renders around the object, via MuJoCo rather than trimesh.

    trimesh's `save_image` needs pyglet, which is not installed and would be a new
    dependency for rendering alone. MuJoCo is already a hard requirement and its
    offscreen renderer is proven in this repo, so the mesh is wrapped in a
    throwaway MJCF and orbited there. Vertex colours survive the round trip
    through PLY, which is why the objects keep their real appearance here even
    though the physics scene renders flat.
    """
    import tempfile

    import mujoco

    centred = mesh.copy()
    centred.apply_translation(-centred.bounds.mean(axis=0))
    # Half the bounding-sphere diameter is what the camera has to clear; MuJoCo's
    # default 45-degree vertical FOV then needs distance = radius / tan(22.5deg).
    # The earlier factor left the object a speck in the middle of the frame.
    radius = float(np.linalg.norm(centred.extents)) / 2.0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # MuJoCo decodes OBJ and STL, not PLY. OBJ over STL because STL has no
        # place to put the vertex colours the reconstruction carries.
        centred.export(str(tmp / "m.obj"))
        (tmp / "s.xml").write_text(f"""<mujoco>
  <compiler assetdir="." angle="radian"/>
  <visual>
    <global offwidth="{max(size[0], 640)}" offheight="{max(size[1], 640)}"/>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.42 0.42 0.42" specular="0.15 0.15 0.15"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.20 0.22 0.27" rgb2="0.05 0.05 0.07" width="128" height="128"/>
    <mesh name="obj" file="m.obj"/>
  </asset>
  <worldbody>
    <light pos="0 0 {radius * 2:.3f}" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom type="mesh" mesh="obj" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>
""", encoding="utf-8")

        model = mujoco.MjModel.from_xml_path(str(tmp / "s.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, size[1], size[0])
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, 0.0, 0.0]
        # 1.25 gives a small margin around the object without cropping it.
        camera.distance = radius / math.tan(math.radians(22.5)) * 1.25

        import cv2
        frames = []
        for azimuth, elevation in angles:
            camera.azimuth, camera.elevation = azimuth, -elevation
            renderer.update_scene(data, camera)
            frames.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        renderer.close()
        return frames


def stats(mesh: trimesh.Trimesh) -> Dict:
    extents = np.round(mesh.extents, 3)
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "extent_m": extents.tolist(),
        "longest_m": round(float(extents.max()), 2),
        "volume_m3": round(float(abs(mesh.volume)), 4),
        "watertight": bool(mesh.is_watertight),
    }


def process(name: str, out_dir: Path, turntable_frames: int, fps: int) -> Optional[Dict]:
    import cv2

    src = OUTPUT_G / name / "mesh_aligned.glb"
    mesh = load(src)
    if mesh is None:
        print(f"  [skip] {name}: no mesh at {src}")
        return None

    info = stats(mesh)
    shutil.copy2(src, out_dir / f"{name}.glb")
    info["glb"] = f"{name}.glb"
    info["glb_mb"] = round((out_dir / f"{name}.glb").stat().st_size / 1e6, 2)

    # Four views: front, side, back, and a raised three-quarter.
    quad = render_views(mesh, [(0, 10), (90, 10), (180, 10), (45, 40)])
    if quad:
        row1 = np.hstack(quad[:2])
        row2 = np.hstack(quad[2:])
        grid = np.vstack([row1, row2])
        cv2.rectangle(grid, (0, 0), (grid.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(grid, f"{name}   {info['extent_m']} m   {info['faces']:,} faces",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        path = out_dir / f"{name}_4views.png"
        cv2.imwrite(str(path), grid)
        info["quad"] = path.name
        print(f"  {path.name}")

    frames = render_views(mesh, [(a * 360.0 / turntable_frames, 15)
                                 for a in range(turntable_frames)], size=(560, 560))
    if frames:
        import imageio.v2 as imageio
        path = out_dir / f"{name}_turntable.mp4"
        writer = imageio.get_writer(path, fps=fps, quality=8, macro_block_size=1)
        for f in frames:
            writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        writer.close()
        info["turntable"] = path.name
        print(f"  {path.name}  {round(path.stat().st_size / 1e6, 2)} MB")

    if not quad and not frames:
        print(f"  [warn] {name}: no GL context, GLB copied but nothing rendered")

    info["name"] = name
    return info


def build_page(out_dir: Path, objects: List[Dict]) -> Path:
    cards = []
    for o in objects:
        viewer = (f'<model-viewer src="{o["glb"]}" camera-controls auto-rotate '
                  f'shadow-intensity="1" exposure="1.1" '
                  f'alt="{o["name"]}"></model-viewer>')
        fallback = (f'<video src="{o["turntable"]}" autoplay muted loop playsinline></video>'
                    if o.get("turntable") else
                    f'<div class="none">no render — open the GLB</div>')
        quad = (f'<a class="quad" href="{o["quad"]}"><img src="{o["quad"]}" '
                f'alt="four views" loading="lazy"></a>' if o.get("quad") else "")
        cards.append(f"""<article>
  <h2>{o["name"]}</h2>
  <div class="pair">
    <div class="live">{viewer}<span class="hint">drag to orbit</span></div>
    <div class="fallback">{fallback}<span class="hint">turntable</span></div>
  </div>
  {quad}
  <dl>
    <div><dt>size</dt><dd>{" × ".join(f"{v:g}" for v in o["extent_m"])} m</dd></div>
    <div><dt>faces</dt><dd>{o["faces"]:,}</dd></div>
    <div><dt>volume</dt><dd>{o["volume_m3"]} m³</dd></div>
    <div><dt>watertight</dt><dd class="{'ok' if o['watertight'] else 'warn'}">{o["watertight"]}</dd></div>
    <div><dt>file</dt><dd><a href="{o["glb"]}">{o["glb"]}</a> · {o["glb_mb"]} MB</dd></div>
  </dl>
</article>""")

    page = PAGE.replace("{{CARDS}}", "\n".join(cards))
    page = page.replace("{{SUMMARY}}", f"{len(objects)} objects · "
                        f"{sum(o['glb_mb'] for o in objects):.0f} MB of GLB")
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Completed objects — 3D</title>
<script type="module"
  src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
<style>
  :root{--bg:#faf9f7;--fg:#1b1a18;--mut:#6c6660;--card:#fff;--line:#e4dfd7;
        --ok:#1a7f4b;--warn:#a8410f;--chip:#f1ede6;}
  @media (prefers-color-scheme:dark){:root{
    --bg:#151419;--fg:#ecebf0;--mut:#9a95a2;--card:#1e1d24;--line:#33313b;
    --ok:#5fd39b;--warn:#f0a06a;--chip:#2a2832;}}
  *{box-sizing:border-box}
  body{margin:0;padding:2.2rem 1.25rem 5rem;background:var(--bg);color:var(--fg);
       font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif;}
  .wrap{max-width:1180px;margin:0 auto}
  h1{font-size:1.55rem;margin:0 0 .3rem}
  .sub{color:var(--mut);margin:0 0 2rem;font-family:ui-monospace,monospace;font-size:.85rem}
  article{background:var(--card);border:1px solid var(--line);border-radius:14px;
          padding:1rem;margin-bottom:1.3rem}
  h2{font-size:1rem;margin:0 0 .8rem;font-family:ui-monospace,monospace}
  .pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:.8rem}
  .live,.fallback{position:relative}
  model-viewer{width:100%;aspect-ratio:1;border-radius:10px;background:#101014;
               --poster-color:transparent}
  video{width:100%;aspect-ratio:1;object-fit:cover;border-radius:10px;background:#101014}
  .none{aspect-ratio:1;display:grid;place-items:center;border-radius:10px;
        border:1px dashed var(--line);color:var(--mut);font-size:.82rem}
  .hint{position:absolute;left:.6rem;bottom:.6rem;background:rgba(0,0,0,.62);
        color:#fff;border-radius:99px;padding:.1rem .55rem;font-size:.7rem;
        font-family:ui-monospace,monospace}
  .quad{display:block;margin-top:.8rem}
  .quad img{width:100%;border-radius:10px;display:block}
  dl{display:flex;flex-wrap:wrap;gap:.4rem .6rem;margin:.9rem 0 0}
  dl div{background:var(--chip);border-radius:99px;padding:.16rem .6rem;
         display:flex;gap:.4rem;font-size:.78rem;font-family:ui-monospace,monospace}
  dt{color:var(--mut)} dd{margin:0}
  dd.ok{color:var(--ok)} dd.warn{color:var(--warn)}
  a{color:inherit}
</style>
<div class="wrap">
  <h1>Completed objects — 3D</h1>
  <p class="sub">{{SUMMARY}} · Hunyuan3D meshes aligned back into the room's frame</p>
  {{CARDS}}
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Render completed object meshes")
    ap.add_argument("objects", nargs="*", default=[
        "obj_0179_workbench", "obj_0202_blade", "obj_0203_trash_can"])
    ap.add_argument("--out", type=Path, default=Path("visualize/7_objects_3d"))
    ap.add_argument("--frames", type=int, default=72, help="turntable frames")
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rendered: List[Dict] = []
    for name in args.objects:
        print(f"[{name}]")
        info = process(name, args.out, args.frames, args.fps)
        if info:
            rendered.append(info)

    if not rendered:
        raise SystemExit("nothing rendered")

    page = build_page(args.out, rendered)
    (args.out / "objects_report.json").write_text(
        json.dumps(rendered, indent=2), encoding="utf-8")
    print(f"\n[done] {page}  ({len(rendered)} objects)")


if __name__ == "__main__":
    main()
