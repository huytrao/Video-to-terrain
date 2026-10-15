#!/usr/bin/env python3
"""Walk the policy straight out from the spawn on several headings and compare.

    python scripts/sim/walk_headings.py --headings 8 --seconds 12

Each run turns on the spot to the target heading, then commands a constant
forward velocity and lets it walk until the time runs out or it stops making
progress. One MP4 per heading plus a contact sheet, so the directions can be
compared side by side rather than one at a time.

Turning first matters: `ang_vel_z` is capped at 0.5 rad/s, so a policy asked to
arc onto a new heading covers several metres before it gets there, and the run
would measure the arc rather than the direction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

YAW_TOLERANCE = 0.12      # rad; close enough to start walking
STALL_WINDOW = 2.0        # s of no progress before calling it stopped
STALL_DISTANCE = 0.10     # m of progress required within that window


def yaw_of(runner) -> float:
    q = runner.data.qpos[runner.root_qpos + 3:runner.root_qpos + 7]
    m = np.zeros(9)
    runner.mujoco.mju_quat2Mat(m, q)
    return math.atan2(m.reshape(3, 3)[1, 0], m.reshape(3, 3)[0, 0])


def obstacle_geoms(runner) -> set:
    mj = runner.mujoco
    return {g for g in range(runner.model.ngeom)
            if (mj.mj_id2name(runner.model, mj.mjtObj.mjOBJ_GEOM, g) or "").startswith("obs_")}


def walk_one(runner, heading: float, seconds: float, speed: float,
             writer=None, renderer=None, camera=None, fps: int = 40) -> Dict:
    """Turn to `heading`, then walk forward until time runs out or progress stops."""
    import cv2

    runner.reset()
    obstacles = obstacle_geoms(runner)
    start = runner.data.xpos[runner.pelvis][:2].copy()
    every = max(1, int(round(1.0 / (fps * runner.step_dt))))
    stall_steps = int(STALL_WINDOW / runner.step_dt)

    history: List[float] = []
    hits: set = set()
    turning = True
    stopped_at: Optional[float] = None

    for k in range(int(seconds / runner.step_dt)):
        error = (heading - yaw_of(runner) + math.pi) % (2 * math.pi) - math.pi
        if turning and abs(error) < YAW_TOLERANCE:
            turning = False

        if turning:
            command = np.array([0.0, 0.0, float(np.clip(1.5 * error, -0.5, 0.5))], np.float32)
        else:
            # Small yaw correction keeps it on the line; the policy drifts otherwise.
            command = np.array([speed, 0.0, float(np.clip(0.8 * error, -0.3, 0.3))], np.float32)

        runner.command = command
        runner.standing = bool(np.linalg.norm(command) < 0.1)
        runner.step()

        here = runner.data.xpos[runner.pelvis][:2]
        distance = float(np.linalg.norm(here - start))
        history.append(distance)

        for c in range(runner.data.ncon):
            for g in (runner.data.contact[c].geom1, runner.data.contact[c].geom2):
                if g in obstacles:
                    hits.add(runner.mujoco.mj_id2name(
                        runner.model, runner.mujoco.mjtObj.mjOBJ_GEOM, g))

        if writer is not None and k % every == 0:
            camera.lookat[:] = runner.data.xpos[runner.pelvis]
            renderer.update_scene(runner.data, camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            w = frame.shape[1]
            cv2.rectangle(frame, (0, 0), (w, 46), (0, 0, 0), -1)
            state = "turning" if turning else ("blocked" if stopped_at else "walking")
            cv2.putText(frame,
                        f"heading {math.degrees(heading):+4.0f}deg   t={k * runner.step_dt:5.1f}s   "
                        f"{distance:5.2f} m   {state}",
                        (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Progress is judged over a window: a walking gait rocks, so an
        # instantaneous velocity near zero happens every stride.
        if stopped_at is None and not turning and len(history) > stall_steps:
            if history[-1] - history[-stall_steps] < STALL_DISTANCE:
                stopped_at = k * runner.step_dt

    final = history[-1] if history else 0.0
    upright = float(-runner._projected_gravity()[2])
    free_run = stopped_at if stopped_at else seconds
    return {
        "heading_deg": round(math.degrees(heading), 1),
        "distance_m": round(final, 3),
        "free_speed_mps": round(final / max(free_run, 1e-6), 3),
        "stopped_at_s": None if stopped_at is None else round(stopped_at, 1),
        "upright": round(upright, 3),
        "fell": bool(upright < 0.85),
        "touched": sorted(hits),
    }


def main() -> None:
    from play_policy import PolicyRunner

    ap = argparse.ArgumentParser(description="Straight-line walks on several headings")
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
    ap.add_argument("--headings", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("visualize/6_headings"))
    ap.add_argument("--fps", type=int, default=40)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    deploy = args.deploy
    if deploy is None:
        tag = args.policy.stem.replace("policy_", "")
        for c in (args.policy.with_name(f"deploy_{tag}.yaml"),
                  args.policy.with_name("deploy.yaml")):
            if c.is_file():
                deploy = c
                break
    if deploy is None:
        raise SystemExit("no deploy YAML; pass --deploy")

    args.out.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []

    print(f"[plan] {args.headings} headings, {args.seconds}s each at {args.speed} m/s")
    for i in range(args.headings):
        heading = 2 * math.pi * i / args.headings
        label = f"{int(round(math.degrees(heading))):03d}"

        runner = PolicyRunner(args.scene, args.policy, deploy, command=[args.speed, 0, 0])
        writer = renderer = camera = None
        if not args.no_video:
            import imageio.v2 as imageio
            import mujoco
            path = args.out / f"heading_{label}.mp4"
            writer = imageio.get_writer(path, fps=args.fps, quality=7, macro_block_size=1)
            renderer = mujoco.Renderer(runner.model, 640, 1024)
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 4.5, 150, -14

        r = walk_one(runner, heading, args.seconds, args.speed,
                     writer, renderer, camera, args.fps)
        if writer is not None:
            writer.close()
            renderer.close()
            r["video"] = f"heading_{label}.mp4"
        results.append(r)

        stop = "ran clear" if r["stopped_at_s"] is None else f"stopped {r['stopped_at_s']}s"
        print(f"  {r['heading_deg']:+6.0f}deg  {r['distance_m']:5.2f} m  "
              f"{r['free_speed_mps']:5.3f} m/s  upright {r['upright']:.2f}  {stop}"
              f"{'  FELL' if r['fell'] else ''}")

    # Distance alone picks the wrong run. Measured here: the 270deg run covered
    # 3.48 m at 1.281 m/s against a 0.5 m/s command, with uprightness 0.87 and a
    # stall at 2.7 s — a lurch forward, not walking. A clean run has to stay
    # upright and move at roughly the speed it was told to.
    def clean(r):
        return (not r["fell"]
                and r["upright"] > 0.95
                and r["free_speed_mps"] < 1.8 * args.speed)

    good = [r for r in results if clean(r)]
    for r in results:
        r["clean"] = clean(r)
    best = max(good or results, key=lambda r: r["distance_m"])

    print(f"\n[best] heading {best['heading_deg']:+.0f}deg — {best['distance_m']} m "
          f"at {best['free_speed_mps']} m/s, upright {best['upright']}")
    for r in results:
        if r["clean"]:
            continue
        why = ("fell" if r["fell"]
               else f"{r['free_speed_mps']:.2f} m/s vs {args.speed} commanded"
               if r["free_speed_mps"] >= 1.8 * args.speed
               else f"upright {r['upright']:.2f}")
        print(f"       rejected {r['heading_deg']:+.0f}deg — {why}")

    (args.out / "headings_report.json").write_text(json.dumps({
        "seconds": args.seconds, "speed": args.speed,
        "best_heading_deg": best["heading_deg"],
        "clean_runs": len(good), "runs": results,
    }, indent=2), encoding="utf-8")
    print(f"[done] {args.out}/headings_report.json")


if __name__ == "__main__":
    main()
