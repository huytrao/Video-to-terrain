#!/usr/bin/env python3
"""Steer the velocity policy on a tour of the room instead of into a wall.

    python scripts/sim/walk_tour.py --seconds 30 --video walk_tour.mp4

The policy only takes (vx, vy, wz) and has no obstacle sensing, so left on a
constant command it walks straight into the furniture. This adds the missing
layer: a clearance map, A* through the free cells, and pure pursuit that turns
the path into commands.

The binding constraint is `ang_vel_z` = +/-0.5 rad/s — twelve seconds for a full
turn — so the controller turns on the spot when badly aimed rather than trying to
arc, which at that yaw rate would carry it into whatever it was avoiding.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CELL = 0.15          # m per grid cell
ROBOT_RADIUS = 0.45  # m of clearance the body needs
REACHED = 0.35       # m; waypoint counts as visited
LOOKAHEAD = 0.60     # m along the path to aim at
ALIGN_FIRST = 0.7    # rad; beyond this, stop and turn on the spot


def clearance_grid(bounds, obstacles, stand_height: float = 1.8):
    """Distance in plan from every cell to the nearest wall or obstacle."""
    (x0, y0, _), (x1, y1, _) = bounds
    nx, ny = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    xs = x0 + (np.arange(nx) + 0.5) * CELL
    ys = y0 + (np.arange(ny) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    clear = np.minimum.reduce([pts[:, 0] - x0, x1 - pts[:, 0],
                               pts[:, 1] - y0, y1 - pts[:, 1]])
    for o in obstacles:
        c = np.asarray(o["pos"], float)
        h = np.asarray(o["size"], float)
        if c[2] - h[2] > stand_height or c[2] + h[2] < 0.05:
            continue
        w, x, y, z = np.asarray(o["quat"], float)
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z)],
        ])
        local = (pts - c[:2]) @ rot
        foot = np.abs(rot.T) @ h[:2]
        clear = np.minimum(clear, np.linalg.norm(np.maximum(np.abs(local) - foot, 0), axis=1))

    return clear.reshape(nx, ny), (x0, y0), (nx, ny)


def to_cell(xy, origin) -> Tuple[int, int]:
    return (int((xy[0] - origin[0]) / CELL), int((xy[1] - origin[1]) / CELL))


def to_world(cell, origin) -> np.ndarray:
    return np.array([origin[0] + (cell[0] + 0.5) * CELL,
                     origin[1] + (cell[1] + 0.5) * CELL])


def astar(grid, start, goal) -> List[Tuple[int, int]]:
    """Shortest path over cells with enough clearance, preferring roomy ones."""
    nx, ny = grid.shape
    free = grid > ROBOT_RADIUS
    if not free[start] or not free[goal]:
        return []

    def h(c):
        return math.dist(c, goal)

    open_set = [(h(start), 0.0, start)]
    came: Dict = {}
    best = {start: 0.0}
    steps = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]

    while open_set:
        _, g, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for dx, dy in steps:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or not free[nxt]:
                continue
            # Hugging a wall is legal but unpleasant; bias toward open floor.
            penalty = max(0.0, 1.2 - grid[nxt]) * 0.8
            step = math.hypot(dx, dy) * CELL * (1.0 + penalty)
            cand = g + step
            if cand < best.get(nxt, 1e18):
                best[nxt] = cand
                came[nxt] = cur
                heapq.heappush(open_set, (cand + h(nxt) * CELL, cand, nxt))
    return []


def pick_waypoints(grid, origin, start_xy, count: int = 4) -> List[np.ndarray]:
    """Roomy, well-separated spots, ordered as a loop back to the start."""
    nx, ny = grid.shape
    cells = [(i, j) for i in range(nx) for j in range(ny) if grid[i, j] > ROBOT_RADIUS + 0.35]
    if not cells:
        return []
    cells.sort(key=lambda c: -grid[c])

    chosen: List[np.ndarray] = []
    for c in cells:
        p = to_world(c, origin)
        if np.linalg.norm(p - start_xy) < 1.2:
            continue
        if all(np.linalg.norm(p - q) > 1.8 for q in chosen):
            chosen.append(p)
        if len(chosen) == count:
            break

    # Nearest-neighbour ordering keeps the tour from criss-crossing the room.
    tour, here, left = [], np.asarray(start_xy, float), list(chosen)
    while left:
        i = int(np.argmin([np.linalg.norm(p - here) for p in left]))
        here = left.pop(i)
        tour.append(here)
    tour.append(np.asarray(start_xy, float))
    return tour


def pursue(pose_xy, yaw, path_world, limits) -> Tuple[np.ndarray, int]:
    """Pure pursuit: aim a lookahead point, split into forward and yaw command."""
    if not len(path_world):
        return np.zeros(3, np.float32), 0

    d = np.linalg.norm(path_world - pose_xy, axis=1)
    i = int(d.argmin())
    while i < len(path_world) - 1 and d[i] < LOOKAHEAD:
        i += 1
    target = path_world[i]

    delta = target - pose_xy
    err = math.atan2(delta[1], delta[0]) - yaw
    err = (err + math.pi) % (2 * math.pi) - math.pi

    wz = float(np.clip(1.5 * err, limits["wz"][0], limits["wz"][1]))
    if abs(err) > ALIGN_FIRST:
        # Too far off to arc: at 0.5 rad/s an arc would sweep into the obstacle.
        vx = 0.0
    else:
        vx = float(np.clip(0.5 * (1.0 - abs(err) / ALIGN_FIRST),
                           0.0, min(0.5, limits["vx"][1])))
    return np.array([vx, 0.0, wz], np.float32), i


def yaw_of(runner) -> float:
    q = runner.data.qpos[runner.root_qpos + 3:runner.root_qpos + 7]
    m = np.zeros(9)
    runner.mujoco.mju_quat2Mat(m, q)
    m = m.reshape(3, 3)
    return math.atan2(m[1, 0], m[0, 0])


def main() -> None:
    from play_policy import PolicyRunner

    ap = argparse.ArgumentParser(description="Walk the policy on a tour of the room")
    ap.add_argument("--scene", type=Path,
                    default=Path("outputs-full-factory/delivery/OUTPUT_ROOM/room_scene.xml"))
    ap.add_argument("--policy", type=Path,
                    default=Path("/Users/traoanhuy/source_code/Robot_code/"
                                 "vinrobotics-vr_h3_2/logs/rsl_rl/exported/policy_12000.onnx"))
    ap.add_argument("--deploy", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--waypoints", type=int, default=4)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=50)
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

    report = json.loads((args.scene.parent / "room_scene_report.json").read_text())
    bounds = report["orientation"]["bounds_after"]
    obstacles = report["obstacles"]

    grid, origin, _ = clearance_grid(bounds, obstacles)
    runner = PolicyRunner(args.scene, args.policy, deploy, command=[0.4, 0.0, 0.0])
    runner.reset()
    start_xy = runner.data.xpos[runner.pelvis][:2].copy()

    waypoints = pick_waypoints(grid, origin, start_xy, args.waypoints)
    print(f"[map] {grid.shape[0]}x{grid.shape[1]} cells, "
          f"{int((grid > ROBOT_RADIUS).sum())} passable at {ROBOT_RADIUS} m clearance")
    print(f"[tour] {len(waypoints)} waypoints: "
          f"{[np.round(w, 1).tolist() for w in waypoints]}")

    limits = {"vx": (-0.5, 1.4), "vy": (-0.8, 0.8), "wz": (-0.5, 0.5)}
    writer = renderer = camera = None
    if args.video:
        import imageio.v2 as imageio
        import mujoco
        writer = imageio.get_writer(args.video, fps=args.fps, quality=8,
                                    macro_block_size=1)
        renderer = mujoco.Renderer(runner.model, 720, 1280)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 5.0, 150, -20

    obstacle_geoms = {
        g for g in range(runner.model.ngeom)
        if (runner.mujoco.mj_id2name(runner.model, runner.mujoco.mjtObj.mjOBJ_GEOM, g)
            or "").startswith("obs_")
    }

    leg = 0
    path_world = np.empty((0, 2))
    travelled, hits = 0.0, set()
    prev = start_xy.copy()
    every = max(1, int(round(1.0 / (args.fps * runner.step_dt))))
    reached = 0

    for k in range(int(args.seconds / runner.step_dt)):
        here = runner.data.xpos[runner.pelvis][:2].copy()

        if leg < len(waypoints):
            if not len(path_world) or np.linalg.norm(here - waypoints[leg]) < REACHED:
                if len(path_world) and np.linalg.norm(here - waypoints[leg]) < REACHED:
                    reached += 1
                    print(f"  t={k * runner.step_dt:5.1f}s  reached waypoint {leg + 1}"
                          f"/{len(waypoints)}")
                    leg += 1
                if leg < len(waypoints):
                    cells = astar(grid, to_cell(here, origin),
                                  to_cell(waypoints[leg], origin))
                    path_world = (np.array([to_world(c, origin) for c in cells])
                                  if cells else np.empty((0, 2)))
                    if not len(path_world):
                        print(f"  t={k * runner.step_dt:5.1f}s  no route to waypoint "
                              f"{leg + 1}; skipping")
                        leg += 1
                        continue

        runner.command, _ = (pursue(here, yaw_of(runner), path_world, limits)
                             if leg < len(waypoints) else (np.zeros(3, np.float32), 0))
        runner.standing = bool(np.linalg.norm(runner.command) < 0.1)
        runner.step()

        travelled += float(np.linalg.norm(runner.data.xpos[runner.pelvis][:2] - prev))
        prev = runner.data.xpos[runner.pelvis][:2].copy()
        for c in range(runner.data.ncon):
            for g in (runner.data.contact[c].geom1, runner.data.contact[c].geom2):
                if g in obstacle_geoms:
                    hits.add(runner.mujoco.mj_id2name(
                        runner.model, runner.mujoco.mjtObj.mjOBJ_GEOM, g))

        if writer is not None and k % every == 0:
            import cv2
            camera.lookat[:] = runner.data.xpos[runner.pelvis]
            renderer.update_scene(runner.data, camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (0, 0), (1280, 46), (0, 0, 0), -1)
            cv2.putText(frame,
                        f"t={k * runner.step_dt:5.1f}s  waypoint {min(leg + 1, len(waypoints))}"
                        f"/{len(waypoints)}  path {travelled:5.2f} m  "
                        f"cmd vx={runner.command[0]:.2f} wz={runner.command[2]:+.2f}",
                        (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if writer is not None:
        writer.close()
        renderer.close()
        print(f"[video] {args.video}  {args.video.stat().st_size / 1e6:.1f} MB")

    upright = float(-runner._projected_gravity()[2])
    print(f"[result] path {travelled:.2f} m, {reached}/{len(waypoints)} waypoints, "
          f"upright {upright:.3f}, obstacles touched {len(hits)}")
    (args.scene.parent / "walk_tour_report.json").write_text(json.dumps({
        "seconds": args.seconds, "waypoints": [w.tolist() for w in waypoints],
        "reached": reached, "path_length_m": round(travelled, 3),
        "final_uprightness": round(upright, 4), "touched": sorted(hits),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
