#!/usr/bin/env python3
"""Render the policy run to video and serve it on a local port.

    python scripts/sim/serve_room.py --port 8080

MuJoCo's own viewer is a native window, so there is nothing to point a browser
at. This renders the same simulation `play_policy.py` measures — three camera
angles of it — writes them as MP4, and puts them behind a page next to the
numbers, so what you watch and what the report claims come from one run.

Existing videos are reused unless `--rerender` is passed, which keeps reloading
the page cheap.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import socketserver
import threading
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# azimuth, elevation, distance, and whether the camera tracks the robot
VIEWS = [
    ("robot", 135, -10, 3.6, True),
    ("room", 135, -22, 11.0, False),
    ("topdown", 215, -80, 12.0, False),
]


def render_run(runner, out_dir: Path, seconds: float, fps: int = 30,
               width: int = 960, height: int = 640) -> List[Dict]:
    """One simulation, three cameras, one MP4 each."""
    import imageio.v2 as imageio
    import mujoco

    frames_per_shot = max(1, int(round(1.0 / (fps * runner.step_dt))))
    written: List[Dict] = []

    for name, azimuth, elevation, distance, track in VIEWS:
        runner.reset()
        renderer = mujoco.Renderer(runner.model, height, width)
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        anchor = runner.data.xpos[runner.pelvis].copy()
        anchor[2] = 1.2

        path = out_dir / f"policy_{name}.mp4"
        writer = imageio.get_writer(path, fps=fps, quality=8, macro_block_size=1)
        try:
            for step in range(int(seconds / runner.step_dt)):
                runner.step()
                if step % frames_per_shot:
                    continue
                camera.lookat[:] = runner.data.xpos[runner.pelvis] if track else anchor
                renderer.update_scene(runner.data, camera)
                writer.append_data(renderer.render())
        finally:
            writer.close()
        renderer.close()

        written.append({"name": name, "file": path.name,
                        "size_mb": round(path.stat().st_size / 1e6, 2)})
        print(f"[video] {path.name}  {written[-1]['size_mb']} MB", flush=True)

    return written


def build_page(out_dir: Path, videos: List[Dict], report: Dict, scene: Path) -> Path:
    def row(label: str, value: str, good: Optional[bool] = None) -> str:
        cls = "" if good is None else (" ok" if good else " bad")
        return f'<div class="row"><dt>{label}</dt><dd class="mono{cls}">{value}</dd></div>'

    standing = bool(report.get("standing"))
    stats = "".join([
        row("verdict", "STANDING" if standing else "FELL", standing),
        row("command", " ".join(f"{v:g}" for v in report.get("command", []))),
        row("pelvis height",
            f'{report.get("start_pelvis_z")} → {report.get("final_pelvis_z")} m'),
        row("lowest reached", f'{report.get("min_pelvis_z")} m'),
        row("uprightness",
            f'{report.get("final_uprightness")} (min {report.get("min_uprightness")})',
            float(report.get("final_uprightness", 0)) > 0.85),
        row("drift, last second", f'{report.get("last_second_drift")} m',
            float(report.get("last_second_drift", 1)) < 0.05),
        row("simulated", f'{report.get("seconds")} s'),
        row("policy", Path(str(report.get("policy", ""))).name),
    ])

    players = "".join(
        f'<figure><video src="{v["file"]}" controls autoplay muted loop playsinline></video>'
        f'<figcaption>{v["name"]} · {v["size_mb"]} MB</figcaption></figure>'
        for v in videos
    )

    page = PAGE.replace("{{STATS}}", stats).replace("{{PLAYERS}}", players)
    page = page.replace("{{SCENE}}", scene.name)
    page = page.replace("{{BADGE}}", "standing" if standing else "fell")
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Room scene — policy run</title>
<style>
  :root{--bg:#faf9f7;--fg:#1b1a18;--mut:#6c6660;--card:#fff;--line:#e4dfd7;
        --ok:#1a7f4b;--bad:#a8410f;--chip:#f1ede6;}
  @media (prefers-color-scheme:dark){:root{
    --bg:#151419;--fg:#ecebf0;--mut:#9a95a2;--card:#1e1d24;--line:#33313b;
    --ok:#5fd39b;--bad:#f0a06a;--chip:#2a2832;}}
  *{box-sizing:border-box}
  body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
       font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif;}
  .wrap{max-width:1180px;margin:0 auto}
  h1{font-size:1.5rem;margin:0 0 .25rem}
  .sub{color:var(--mut);margin:0 0 1.6rem;font-size:.92rem}
  .badge{display:inline-block;padding:.15rem .6rem;border-radius:999px;
         font:600 .78rem ui-monospace,monospace;background:var(--chip)}
  .badge.standing{color:var(--ok)} .badge.fell{color:var(--bad)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}
  figure{margin:0;background:var(--card);border:1px solid var(--line);
         border-radius:12px;padding:.6rem}
  video{width:100%;max-width:100%;display:block;border-radius:8px;background:#000}
  figcaption{color:var(--mut);font-size:.76rem;text-align:center;margin-top:.45rem;
             font-family:ui-monospace,monospace}
  dl{background:var(--card);border:1px solid var(--line);border-radius:12px;
     padding:.35rem .9rem;margin:0 0 1.4rem}
  .row{display:flex;justify-content:space-between;gap:1rem;padding:.42rem 0;
       border-bottom:1px solid var(--line)}
  .row:last-child{border-bottom:0}
  dt{color:var(--mut);font-size:.87rem} dd{margin:0}
  .mono{font-family:ui-monospace,monospace;font-size:.87rem}
  .mono.ok{color:var(--ok)} .mono.bad{color:var(--bad)}
  footer{color:var(--mut);font-size:.8rem;margin-top:2rem;
         font-family:ui-monospace,monospace}
</style>
<div class="wrap">
  <h1>Room scene — policy run <span class="badge {{BADGE}}">{{BADGE}}</span></h1>
  <p class="sub">Cùng một lần mô phỏng với báo cáo bên dưới, quay từ ba góc.</p>
  <dl>{{STATS}}</dl>
  <div class="grid">{{PLAYERS}}</div>
  <footer>{{SCENE}} · mở tương tác: <code>make room-view</code></footer>
</div>
"""


def serve(directory: Path, port: int, open_browser: bool) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))

    class Reusable(socketserver.TCPServer):
        allow_reuse_address = True

    try:
        httpd = Reusable(("127.0.0.1", port), handler)
    except OSError as exc:
        raise SystemExit(f"cannot bind port {port}: {exc}")

    url = f"http://127.0.0.1:{port}/"
    print(f"\n  {url}\n  Ctrl-C to stop\n", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from play_policy import PolicyRunner, run

    ap = argparse.ArgumentParser(description="Render the policy run and serve it")
    ap.add_argument("--scene", type=Path,
                    default=Path("outputs-full-factory/delivery/OUTPUT_ROOM/room_scene.xml"))
    default_policy = os.environ.get("ROBOT_POLICY") or os.environ.get("POLICY")
    if not default_policy:
        for candidate in (
            Path("policies/policy.onnx"),
            Path("policy.onnx"),
        ):
            if candidate.is_file():
                default_policy = str(candidate)
                break
    ap.add_argument("--policy", type=Path,
                    default=Path(default_policy or os.environ.get("ROBOT_POLICY", "policy.onnx")))
    ap.add_argument("--deploy", type=Path, default=None)
    ap.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--rerender", action="store_true",
                    help="re-render even when the MP4s already exist")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    out_dir = args.scene.parent
    deploy = args.deploy
    if deploy is None:
        tag = args.policy.stem.replace("policy_", "")
        for candidate in (args.policy.with_name(f"deploy_{tag}.yaml"),
                          args.policy.with_name("deploy.yaml")):
            if candidate.is_file():
                deploy = candidate
                break
    if deploy is None or not deploy.is_file():
        raise SystemExit("no deploy YAML found; pass --deploy")

    existing = [out_dir / f"policy_{name}.mp4" for name, *_ in VIEWS]
    report_path = out_dir / "policy_play_report.json"

    if args.rerender or not all(p.is_file() for p in existing) or not report_path.is_file():
        runner = PolicyRunner(args.scene, args.policy, deploy, command=args.command)
        print(f"[policy] {args.policy.name}  obs={runner.obs_dim}  "
              f"{'STAND' if runner.standing else 'WALK'}", flush=True)
        result = run(runner, args.seconds)
        print(f"[result] {'STANDING' if result.get('standing') else 'FELL'}  "
              f"pelvis {result.get('start_pelvis_z')} -> {result.get('final_pelvis_z')} m",
              flush=True)
        report_path.write_text(json.dumps({
            "policy": str(args.policy), "deploy": str(deploy),
            "command": list(args.command), **result}, indent=2), encoding="utf-8")
        videos = render_run(runner, out_dir, args.seconds)
    else:
        print("[video] reusing existing MP4s (--rerender to redo)")
        videos = [{"name": n, "file": p.name,
                   "size_mb": round(p.stat().st_size / 1e6, 2)}
                  for (n, *_), p in zip(VIEWS, existing)]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    page = build_page(out_dir, videos, report, args.scene)
    print(f"[page] {page}")
    serve(out_dir, args.port, not args.no_open)


if __name__ == "__main__":
    main()
