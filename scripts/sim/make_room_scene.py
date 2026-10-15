#!/usr/bin/env python3
"""Turn the reconstructed room mesh into a MuJoCo scene a robot can stand in.

Two things make a room mesh unusable as physics geometry, and both are silent:

1. **Z points down in the reconstruction.** Ground truth puts `floor` at
   z = [-0.204, 0.132] and `ceiling` at z = [-3.731, -2.922]; doors run from the
   floor *up* to z = -2.46. MuJoCo wants +Z up, so the mesh has to be turned over
   — by a 180 deg rotation about X, not by negating Z. Negating one axis is a
   reflection (det = -1): it mirrors the world, and a right-handed robot comes out
   left-handed. The rotation keeps handedness.

2. **MuJoCo collides mesh geoms as convex hulls.** The convex hull of a room is a
   solid block, so a robot dropped into it is pushed *outside*. The room can be
   the visual, but the collider has to be built from primitives that enclose a
   volume the robot can be inside: a plane for the floor, boxes for the walls and
   ceiling.

The scene ships with a control: a sphere dropped from mid-room. If it does not
come to rest on the floor, the collider is wrong and nothing about the robot's
behaviour means anything.

    python scripts/sim/make_room_scene.py \
        --room outputs-full-factory/delivery/OUTPUT_B1/mesh/room_300k.glb \
        --gt   outputs-full-factory/delivery/OUTPUT_B1/metadata/ground_truth_boxes.json \
        --out  outputs-full-factory/delivery/OUTPUT_ROOM
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

# 180 deg about X: (x, y, z) -> (x, -y, -z). det = +1, so handedness survives.
FLIP_X180 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

WALL_THICKNESS = 0.10


def load_room(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(path))
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene
    if not isinstance(mesh, trimesh.Trimesh):
        raise SystemExit(f"{path} did not load as a single mesh")
    return mesh


def gt_label_bounds(gt_path: Path, label: str) -> Optional[np.ndarray]:
    """AABB of the largest ground-truth instance carrying a label, or None."""
    entries = json.loads(gt_path.read_text(encoding="utf-8"))
    best, best_span = None, -1.0
    for entry in entries:
        if label not in entry.get("label", "").lower():
            continue
        corners = np.array([[p["x"], p["y"], p["z"]] for p in entry["bounding_box"]])
        span = float(np.prod(corners.max(0) - corners.min(0)))
        if span > best_span:
            best, best_span = corners, span
    if best is None:
        return None
    return np.stack([best.min(0), best.max(0)])


def apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (homogeneous @ transform.T)[:, :3]


def orient(mesh: trimesh.Trimesh, gt_path: Optional[Path]) -> Tuple[trimesh.Trimesh, Dict]:
    """Rotate the room upright and drop the floor onto z = 0.

    The floor height comes from ground truth when it is available, because the
    lowest vertex of a reconstructed room is usually a stray, not the floor.
    """
    mesh = mesh.copy()
    mesh.apply_transform(FLIP_X180)

    floor_z: Optional[float] = None
    ceiling_z: Optional[float] = None
    source = "vertex_percentile"

    if gt_path is not None and gt_path.is_file():
        floor_box = gt_label_bounds(gt_path, "floor")
        ceiling_box = gt_label_bounds(gt_path, "ceiling")
        if floor_box is not None:
            floor_z = float(apply(FLIP_X180, floor_box)[:, 2].mean())
            source = "ground_truth"
        if ceiling_box is not None:
            ceiling_z = float(apply(FLIP_X180, ceiling_box)[:, 2].mean())

    if floor_z is None:
        # 1st percentile, not the minimum: reconstruction noise hangs below the floor.
        floor_z = float(np.percentile(mesh.vertices[:, 2], 1.0))

    mesh.apply_translation([0.0, 0.0, -floor_z])
    if ceiling_z is not None:
        ceiling_z -= floor_z

    bounds = mesh.bounds
    info = {
        "floor_source": source,
        "floor_z_before": round(floor_z, 4),
        "ceiling_z_after": None if ceiling_z is None else round(ceiling_z, 4),
        "bounds_after": np.round(bounds, 4).tolist(),
        "height": round(float(bounds[1][2] - bounds[0][2]), 4),
    }

    # The whole point of the rotation is that up is now +Z. If the ceiling did not
    # end up above the floor, the transform is wrong and every number downstream
    # — gravity, body height, contact forces — would be wrong with it.
    if ceiling_z is not None and ceiling_z <= 0.5:
        raise SystemExit(
            f"ceiling landed at z={ceiling_z:.3f} after orienting; expected well above 0. "
            "The room is upside down — check the flip before running any physics."
        )
    if bounds[1][2] <= 0.5:
        raise SystemExit(f"room top is z={bounds[1][2]:.3f}; nothing to stand in.")

    return mesh, info


def matrix_to_quat(rot: np.ndarray) -> List[float]:
    """Rotation matrix -> MuJoCo's w-first quaternion."""
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x, y, z = (rot[2, 1] - rot[1, 2]) * s, (rot[0, 2] - rot[2, 0]) * s, (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w, x = (rot[2, 1] - rot[1, 2]) / s, 0.25 * s
        y, z = (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w, x = (rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s
        y, z = 0.25 * s, (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w, x = (rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s
        y, z = (rot[1, 2] + rot[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return (q / np.linalg.norm(q)).tolist()


# Labels that are surface markings or fixtures rather than things a robot can walk
# into. Keeping them would litter the floor with invisible obstacles.
NON_OBSTACLE_LABELS = {
    "sticker", "label", "paper", "sign", "light", "glass", "window",
    "electrical outlet", "socket", "poster", "screw", "hinge", "wire",
    "floor", "ceiling", "wall", "door",
}
MIN_OBSTACLE_SIZE = 0.20
MIN_OBSTACLE_VOLUME = 0.01


def furniture_colliders(gt_path: Optional[Path], world_from_gt: np.ndarray,
                        floor_z: float) -> List[Dict]:
    """Box colliders for the furniture, taken from the ground-truth OBBs.

    The room mesh cannot supply these. Its 300k triangles are a single connected
    component spanning the whole room — walls, floor and every machine fused into
    one surface — so component splitting yields nothing to hull, and hulling the
    whole thing gives the solid block the shell primitives already avoid.

    The annotation, on the other hand, is exactly a list of oriented boxes in
    metres. Each becomes one MuJoCo box geom, which is also the right fidelity: a
    robot needs to not walk through a workbench, not to reach into its drawers.
    """
    if gt_path is None or not gt_path.is_file():
        return []

    out: List[Dict] = []
    for entry in json.loads(gt_path.read_text(encoding="utf-8")):
        label = entry.get("label", "").strip().lower()
        if label in NON_OBSTACLE_LABELS:
            continue
        extent = np.asarray(entry.get("obb_extents", []), dtype=float)
        transform = np.asarray(entry.get("obb_transform", []), dtype=float)
        if extent.shape != (3,) or transform.shape != (4, 4):
            continue
        if extent.max() < MIN_OBSTACLE_SIZE or float(np.prod(extent)) < MIN_OBSTACLE_VOLUME:
            continue

        # Ground truth lives in the reconstruction's frame; the scene has been
        # turned upright and dropped onto z = 0, so the boxes must follow.
        placed = world_from_gt @ transform
        centre = placed[:3, 3].copy()
        centre[2] -= floor_z
        rot = placed[:3, :3]
        # Orthonormalise: the annotation's rotation carries scale in places, and a
        # non-unit quaternion silently distorts the geom.
        u, _, vt = np.linalg.svd(rot)
        rot = u @ vt
        if np.linalg.det(rot) < 0:
            rot = u @ np.diag([1.0, 1.0, -1.0]) @ vt

        # Anything whose top is at or below the floor is annotation noise.
        if centre[2] + extent[2] / 2 <= 0.02:
            continue

        out.append({
            "name": f"obs_{entry['ins_id']}_{re.sub(r'[^a-z0-9]+', '_', label) or 'thing'}",
            "type": "box",
            "pos": centre.tolist(),
            "quat": matrix_to_quat(rot),
            "size": (extent / 2.0).tolist(),
            "label": label,
        })
    return out


def find_spawn(bounds: np.ndarray, obstacles: List[Dict], stand_height: float = 1.8,
               grid: int = 120) -> Dict:
    """The clearest patch of floor, and how much room it actually has.

    The centre of a room is not free floor. Spawning there put the robot inside a
    machine: 88 contacts at up to 0.57 m of penetration, the solver ejected it,
    and it read as "the policy cannot stand" when nothing had been asked of the
    policy yet. A spawn point has to be found and measured, not assumed.

    Clearance is the distance in plan to the nearest obstacle that occupies the
    band the robot's body passes through, and to the walls.
    """
    (x0, y0, _), (x1, y1, _) = bounds
    margin = 0.35
    xs = np.linspace(x0 + margin, x1 - margin, grid)
    ys = np.linspace(y0 + margin, y1 - margin, grid)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])

    # Start from distance to the walls, then cut it down obstacle by obstacle.
    clearance = np.minimum.reduce([
        points[:, 0] - x0, x1 - points[:, 0],
        points[:, 1] - y0, y1 - points[:, 1],
    ])

    for obstacle in obstacles:
        centre = np.asarray(obstacle["pos"], dtype=float)
        half = np.asarray(obstacle["size"], dtype=float)
        # Only things the robot would actually walk into: a floor-level cable tray
        # or a roof beam is not in the way of its torso.
        if centre[2] - half[2] > stand_height or centre[2] + half[2] < 0.05:
            continue

        quat = np.asarray(obstacle["quat"], dtype=float)
        w, x, y, z = quat
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        # Distance in plan to a rotated rectangle: work in the box's own frame,
        # where the rectangle is axis-aligned and the distance is separable.
        local = (points - centre[:2]) @ rot[:2, :2]
        footprint = np.abs(rot[:2, :2].T) @ half[:2]
        outside = np.maximum(np.abs(local) - footprint, 0.0)
        clearance = np.minimum(clearance, np.linalg.norm(outside, axis=1))

    best = int(clearance.argmax())
    return {
        "xy": points[best].tolist(),
        "clearance": round(float(clearance[best]), 3),
        "considered": len(obstacles),
    }


def mesh_collider(mesh: trimesh.Trimesh, out_dir: Path, assets: str,
                  max_hulls: int = 64, threshold: float = 0.06,
                  decimate_to: int = 40_000) -> List[Dict]:
    """Convex-decompose the reconstruction itself into MuJoCo mesh colliders.

    This is the room's real geometry as physics, not a stand-in for it. A single
    mesh geom collides as its convex hull — a solid block enclosing the whole room
    — but a *set* of convex parts reproduces the actual surfaces, so the robot can
    stand on the real floor and walk into the real machines.

    The earlier version approximated all of this with six boxes plus one box per
    ground-truth annotation. That threw away the reconstruction, and it showed:
    axis-aligned boxes around hoses and ceiling beams walled off the floor until
    only 8% of it was passable.

    Decimated first because CoACD on 300k faces is slow and a collider does not
    need that resolution; the visual mesh stays at full detail.
    """
    import coacd

    coacd.set_log_level("error")
    reduced = (mesh.simplify_quadric_decimation(face_count=decimate_to)
               if len(mesh.faces) > decimate_to else mesh)

    parts = coacd.run_coacd(
        coacd.Mesh(reduced.vertices, reduced.faces),
        threshold=threshold, max_convex_hull=max_hulls, preprocess_mode="off")

    asset_dir = out_dir / assets
    asset_dir.mkdir(parents=True, exist_ok=True)
    for stale in asset_dir.glob("collider_*.obj"):
        stale.unlink()

    geoms: List[Dict] = []
    for i, (vertices, faces) in enumerate(parts):
        hull = trimesh.Trimesh(vertices=np.asarray(vertices),
                               faces=np.asarray(faces), process=False)
        if hull.volume < 1e-6:
            continue
        name = f"collider_{i:03d}"
        hull.export(str(asset_dir / f"{name}.obj"))
        geoms.append({"name": name, "file": f"{name}.obj",
                      "faces": int(len(hull.faces))})
    return geoms


def room_shell(bounds: np.ndarray, ceiling_z: Optional[float]) -> List[Dict]:
    """Floor plane plus wall and ceiling boxes enclosing the room's footprint.

    Primitives, not the mesh: a convex hull of the room would be a solid block
    with the robot locked outside it.
    """
    (x0, y0, _), (x1, y1, z1) = bounds
    top = ceiling_z if ceiling_z else float(z1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
    ht = WALL_THICKNESS / 2

    geoms = [{
        "name": "floor", "type": "plane",
        "pos": [cx, cy, 0.0], "size": [hx + 1.0, hy + 1.0, 0.05],
    }]
    for name, pos, size in [
        ("wall_xmin", [x0 - ht, cy, top / 2], [ht, hy + WALL_THICKNESS, top / 2]),
        ("wall_xmax", [x1 + ht, cy, top / 2], [ht, hy + WALL_THICKNESS, top / 2]),
        ("wall_ymin", [cx, y0 - ht, top / 2], [hx + WALL_THICKNESS, ht, top / 2]),
        ("wall_ymax", [cx, y1 + ht, top / 2], [hx + WALL_THICKNESS, ht, top / 2]),
        ("ceiling", [cx, cy, top + ht], [hx + WALL_THICKNESS, hy + WALL_THICKNESS, ht]),
    ]:
        geoms.append({"name": name, "type": "box", "pos": pos, "size": size})
    return geoms


def stage_robot(robot_dir: Path, out_dir: Path, assets: str) -> Dict:
    """Copy a Unitree robot into the scene directory as a self-contained include.

    Three things have to be handled or the scene will not compile:

    * The robot's own `scene.xml` carries a `floor` plane. Including it would put
      a second floor at z=0 fighting the room's, so only the model, sensors and
      actuators come across.
    * Each robot XML declares `<compiler meshdir="../meshes/">`, resolved against
      whichever file MuJoCo loaded. Once the file lives somewhere else that path
      is wrong, so the directive is stripped and the top-level scene owns meshdir.
    * The `home` keyframe puts the pelvis at the world origin. It is re-emitted at
      the room's centre, otherwise the robot spawns in a corner or inside a wall.
    """
    import re
    import shutil

    origin = robot_dir / "origin"
    name = robot_dir.name
    parts = [f"{name}_rl.xml", f"{name}_sensors.xml", f"{name}_actuators.xml"]
    # Some robot models ship their files under a shortened stem.
    if not (origin / parts[0]).is_file():
        stems = {p.stem.rsplit("_rl", 1)[0] for p in origin.glob("*_rl.xml")}
        if not stems:
            raise SystemExit(f"no *_rl.xml under {origin}")
        stem = sorted(stems)[0]
        parts = [f"{stem}_rl.xml", f"{stem}_sensors.xml", f"{stem}_actuators.xml"]

    asset_dir = out_dir / assets
    asset_dir.mkdir(parents=True, exist_ok=True)
    for mesh_file in (robot_dir / "meshes").iterdir():
        if mesh_file.is_file():
            shutil.copy2(mesh_file, asset_dir / mesh_file.name)

    copied: List[str] = []
    for part in parts:
        src = origin / part
        if not src.is_file():
            continue
        text = re.sub(r"<compiler\b[^>]*/>\s*", "", src.read_text(encoding="utf-8"))
        (out_dir / part).write_text(text, encoding="utf-8")
        copied.append(part)

    home: Optional[List[float]] = None
    for candidate in origin.glob("*_rl_scene.xml"):
        match = re.search(r'<key\s+name="home"\s+qpos="([^"]+)"', candidate.read_text(encoding="utf-8"))
        if match:
            home = [float(v) for v in match.group(1).split()]
            break

    return {"name": name, "includes": copied, "home_qpos": home,
            "stand_height": None if not home else home[2]}


def write_terrain(out_dir: Path, visual_mesh: str, assets: str,
                  shell: List[Dict], obstacles: List[Dict], hulls: List[Dict],
                  spawn: Dict) -> Path:
    """The room on its own, as a terrain another harness can import.

    mjlab's `TerrainEntityCfg` takes `spec_fn: Callable[[], mujoco.MjSpec]`, so a
    trainer or a play script can swap its generated terrain for this room. That
    file must contain the ground and nothing else — no robot, no control sphere,
    no keyframe — because the harness adds its own and a second free body would
    shift every qpos address it computes.
    """
    def fmt(values) -> str:
        return " ".join(f"{v:.5f}" for v in values)

    hull_assets = "".join(
        f'    <mesh name="{h["name"]}" file="{h["file"]}"/>\n' for h in hulls)

    geoms = [
        f'    <geom name="{h["name"]}_geom" type="mesh" mesh="{h["name"]}" '
        f'group="0" friction="1.0 0.02 0.001" rgba="0.4 0.5 0.65 0"/>'
        for h in hulls
    ]
    for geom in shell:
        is_plane = geom["type"] == "plane"
        geoms.append(
            f'    <geom name="{geom["name"]}" type="{geom["type"]}" '
            f'pos="{fmt(geom["pos"])}" size="{fmt(geom["size"])}" '
            f'group="0" friction="1.0 0.02 0.001" '
            f'rgba="0.55 0.55 0.58 {1 if is_plane else 0.18}"/>'
        )
    for o in obstacles:
        geoms.append(
            f'    <geom name="{o["name"]}" type="box" pos="{fmt(o["pos"])}" '
            f'quat="{fmt(o["quat"])}" size="{fmt(o["size"])}" group="0" '
            f'friction="1.0 0.02 0.001" rgba="0.35 0.45 0.62 0.35"/>'
        )

    # group 0 for everything collidable: mjlab's terrain_scan and foot_height
    # sensors filter on `include_geom_groups=(0,)`, so a collider outside that
    # group is invisible to them even though the robot can walk into it.
    xml = f"""<mujoco model="room_terrain">
  <compiler angle="radian" assetdir="{assets}" autolimits="true"/>

  <asset>
    <texture name="gridtex" type="2d" builtin="checker" rgb1="0.22 0.23 0.25"
             rgb2="0.28 0.29 0.32" width="512" height="512"/>
    <material name="grid" texture="gridtex" texrepeat="12 12" reflectance="0.05"/>
    <mesh name="room_visual" file="{visual_mesh}"/>
{hull_assets}  </asset>

  <worldbody>
{chr(10).join(geoms)}
    <geom name="room_visual_geom" type="mesh" mesh="room_visual"
          contype="0" conaffinity="0" group="2" rgba="0.82 0.80 0.77 1"/>
  </worldbody>
  <!-- clearest floor: {fmt(spawn["xy"])} with {spawn["clearance"]} m to anything -->
</mujoco>
"""
    path = out_dir / "room_terrain.xml"
    path.write_text(xml, encoding="utf-8")

    loader = f'''"""Room terrain for mjlab, as a `spec_fn` for `TerrainEntityCfg`.

    from mjlab.terrains import TerrainEntityCfg
    from room_terrain import spec_fn, SPAWN_XY

    env_cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="plane",          # 'generator' would rebuild its own ground
        spec_fn=spec_fn,
    )
    env_cfg.scene.entities["robot"].init_state.pos = (*SPAWN_XY, {0.841!r})

`SPAWN_XY` is the clearest patch of floor, {spawn["clearance"]} m from the nearest
obstacle. The room's centre is inside a machine, so spawning there puts the robot
in a solid box: 88 contacts at up to 0.57 m of penetration, which reads as the
policy failing to balance when it has not been asked anything yet.
"""

from pathlib import Path

import mujoco

TERRAIN_XML = Path(__file__).resolve().parent / "room_terrain.xml"
SPAWN_XY = ({spawn["xy"][0]:.4f}, {spawn["xy"][1]:.4f})
SPAWN_CLEARANCE = {spawn["clearance"]}


def spec_fn() -> mujoco.MjSpec:
    """Parse the room terrain. Paths inside it are relative to this directory."""
    return mujoco.MjSpec.from_file(str(TERRAIN_XML))
'''
    (out_dir / "room_terrain.py").write_text(loader, encoding="utf-8")
    return path


def write_mjcf(
    out_dir: Path,
    visual_mesh: str,
    assets: str,
    shell: List[Dict],
    obstacles: List[Dict],
    hulls: List[Dict],
    drop_from: List[float],
    robot: Optional[Dict],
    robot_xy: List[float],
    keyframe_xml: str = "",
) -> Path:
    def fmt(values) -> str:
        return " ".join(f"{v:.5f}" for v in values)

    include = ""
    robot_note = "  <!-- no robot staged; scene holds the control sphere only -->\n"
    if robot:
        include = "".join(f'  <include file="{f}"/>\n' for f in robot["includes"])
        robot_note = f'  <!-- robot: {robot["name"]}, spawned at room centre -->\n'
    keyframe = keyframe_xml or ""


    # The reconstruction's own surfaces, as convex parts. group 0 so mjlab's
    # terrain sensors see them; invisible because the full-detail visual mesh is
    # drawn over the top of exactly the same geometry.
    hull_assets = "".join(
        f'    <mesh name="{h["name"]}" file="{h["file"]}"/>\n' for h in hulls)
    hull_xml = [
        f'      <geom name="{h["name"]}_geom" type="mesh" mesh="{h["name"]}" '
        f'group="0" friction="1.0 0.02 0.001" rgba="0.4 0.5 0.65 0"/>'
        for h in hulls
    ]

    # Furniture: visible, collidable, and in their own group so the viewer can
    # switch them off independently of the shell.
    obstacle_xml = [
        f'      <geom name="{o["name"]}" type="box" pos="{fmt(o["pos"])}" '
        f'quat="{fmt(o["quat"])}" size="{fmt(o["size"])}" group="4" '
        f'friction="1.0 0.02 0.001" rgba="0.35 0.45 0.62 0.35"/>'
        for o in obstacles
    ]

    shell_xml = []
    for geom in shell:
        is_plane = geom["type"] == "plane"
        material = 'material="grid" ' if is_plane else ""
        alpha = "1" if is_plane else "0.18"
        shell_xml.append(
            f'      <geom name="{geom["name"]}" type="{geom["type"]}" '
            f'pos="{fmt(geom["pos"])}" size="{fmt(geom["size"])}" '
            f'{material}friction="1.0 0.02 0.001" '
            f'rgba="0.55 0.55 0.58 {alpha}"/>'
        )

    xml = f"""<mujoco model="room_scene">
  <compiler angle="radian" assetdir="{assets}" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

  <!-- the offscreen buffer defaults to 640x480; renders larger than that fail -->
  <visual>
    <global offwidth="1920" offheight="1080" azimuth="135" elevation="-15"/>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.53 0.6 0.7" rgb2="0.1 0.12 0.16" width="256" height="256"/>
    <texture name="gridtex" type="2d" builtin="checker" rgb1="0.22 0.23 0.25"
             rgb2="0.28 0.29 0.32" width="512" height="512"/>
    <material name="grid" texture="gridtex" texrepeat="12 12" reflectance="0.05"/>
    <!-- visual only: never referenced by a collider -->
    <mesh name="room_visual" file="{visual_mesh}"/>
{hull_assets}  </asset>

  <worldbody>
    <light pos="{fmt([drop_from[0], drop_from[1], drop_from[2] + 1.5])}" dir="0 0 -1"
           directional="true" diffuse="0.7 0.7 0.7"/>

    <body name="room_shell" pos="0 0 0">
{chr(10).join(shell_xml)}
{chr(10).join(hull_xml)}
{chr(10).join(obstacle_xml)}
      <geom name="room_visual_geom" type="mesh" mesh="room_visual"
            contype="0" conaffinity="0" group="2" rgba="0.82 0.80 0.77 1"/>
    </body>

    <!-- Control. If this does not settle on the floor, the collider is broken
         and nothing the robot does afterwards is evidence of anything. -->
    <body name="control_sphere" pos="{fmt(drop_from)}">
      <freejoint/>
      <geom name="control_sphere_geom" type="sphere" size="0.10" density="800"
            rgba="0.9 0.35 0.2 1"/>
    </body>
  </worldbody>
{robot_note}{include}{keyframe}</mujoco>
"""
    path = out_dir / "room_scene.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def drop_test(xml_path: Path, seconds: float = 3.0) -> Dict:
    """Simulate the control sphere and report where it ended up."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    sphere = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "control_sphere")
    # xpos is stale until the first kinematics pass; without this the start height
    # reads 0 and the drop looks like it never happened.
    mujoco.mj_forward(model, data)
    start_z = float(data.xpos[sphere][2]) if sphere >= 0 else float("nan")

    steps = int(seconds / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)

    end_z = float(data.xpos[sphere][2])
    speed = float(np.linalg.norm(data.cvel[sphere][3:]))
    radius = 0.10

    # With the mesh as the collider the sphere lands on whatever is under it — a
    # bench top, a machine housing — not necessarily the floor, and a sloped
    # surface keeps it rolling. Requiring "resting at exactly the floor" was a
    # test written for a flat box shell; against real geometry it fails a working
    # collider. What the control is actually for is: something caught it, and it
    # did not sink through.
    supported = bool(data.ncon > 0 and end_z > radius * 0.5)
    on_floor = abs(end_z - radius) < 0.05

    return {
        "start_z": round(start_z, 4),
        "end_z": round(end_z, 4),
        "speed": round(speed, 5),
        "contacts": int(data.ncon),
        "expected_resting_z": radius,
        "resting_on_floor": bool(on_floor),
        "settled_on_floor": supported,
        "fell_through": bool(end_z < -0.5),
    }


def build_keyframe(xml_path: Path, robot_home: List[float], robot_xy: List[float]) -> str:
    """Compile the scene once, then write the home key against the real qpos layout.

    Keyframes are addressed positionally, and this scene has two free joints: the
    control sphere and the robot's pelvis. Whichever body is declared first owns
    qpos[0:7]. Splicing the robot's 35-value home vector in by hand put the robot's
    standing pose onto the *sphere* and fed the pelvis a slice of joint angles —
    silently, because the vector was simply short. Reading `jnt_qposadr` from the
    compiled model removes the guess.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos = np.array(model.qpos0, dtype=float).copy()

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    free = [j for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and model.jnt_bodyid[j] == pelvis]
    if pelvis < 0 or not free:
        raise SystemExit("no free joint on a body named 'pelvis'; cannot place the robot")

    root = int(model.jnt_qposadr[free[0]])
    qpos[root:root + 7] = robot_home[:7]
    qpos[root], qpos[root + 1] = robot_xy[0], robot_xy[1]

    # The remaining home values are the joint angles, in the robot's own order.
    joints = robot_home[7:]
    tail = qpos[root + 7:]
    if len(joints) != len(tail):
        raise SystemExit(
            f"home pose has {len(joints)} joint values but the model expects {len(tail)}")
    qpos[root + 7:] = joints

    return ('  <keyframe>\n    <key name="home" qpos="'
            + " ".join(f"{v:g}" for v in qpos) + '"/>\n  </keyframe>\n')


def stand_test(xml_path: Path, seconds: float = 3.0) -> Dict:
    """Hold the robot at its home pose and check the room can carry it.

    This deliberately does not ask the robot to balance — with no controller a
    humanoid falls over, and that says nothing about the room. What it does ask
    is whether the *floor* holds: feet above the floor, pelvis at roughly its
    standing height, and contact forces that stay finite. A robot sinking through
    the floor or launched across the room is a collision bug, not a control one.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key < 0:
        return {"ran": False, "reason": "no home keyframe"}
    mujoco.mj_resetDataKeyframe(model, data, key)

    # Hold the joints at the home pose so the legs do not simply collapse; the
    # question under test is the floor, not the absence of a walking policy.
    # Each actuator's target is read through its own transmission, because the
    # scene's qpos layout interleaves the control sphere with the robot.
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        if 0 <= joint < model.njnt:
            data.ctrl[a] = data.qpos[model.jnt_qposadr[joint]]

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis < 0:
        return {"ran": False, "reason": "no body named 'pelvis'"}

    mujoco.mj_forward(model, data)
    start_z = float(data.xpos[pelvis][2])

    lowest = float("inf")
    max_force = 0.0
    diverged = False
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            diverged = True
            break
        lowest = min(lowest, float(data.xpos[:, 2].min()))
        if data.ncon:
            max_force = max(max_force, float(np.abs(data.efc_force).max()) if data.nefc else 0.0)

    end_z = float(data.xpos[pelvis][2])
    return {
        "ran": True,
        "start_pelvis_z": round(start_z, 4),
        "end_pelvis_z": round(end_z, 4),
        "lowest_body_z": round(lowest, 4),
        "max_constraint_force": round(max_force, 1),
        "contacts": int(data.ncon),
        "diverged": diverged,
        # The room passes if nothing sank through the floor and the sim stayed finite.
        "supported": bool(not diverged and lowest > -0.05 and np.isfinite(end_z)),
    }


def render_check(xml_path: Path, out_png: Path, width: int = 1140, height: int = 760) -> Optional[Path]:
    """Four views of the finished scene: the robot before and after, plus the shell.

    A scene that compiles and passes both physics checks can still be wrong in a
    way only the eye catches — a robot facing a wall, a room inside out, a texture
    mapped to the wrong chart. Three separate bugs in this project were found by
    opening the picture after every automated check had passed.
    """
    import mujoco
    try:
        import cv2
    except ImportError:
        return None

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if model.nkey == 0:
        return None
    renderer = mujoco.Renderer(model, height, width)

    def simulate(seconds: float):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        for a in range(model.nu):
            joint = int(model.actuator_trnid[a, 0])
            if 0 <= joint < model.njnt:
                data.ctrl[a] = data.qpos[model.jnt_qposadr[joint]]
        for _ in range(int(seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        return data

    def shot(data, azimuth, elevation, distance, lookat, label):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = lookat
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        renderer.update_scene(data, camera)
        image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (0, 0), (image.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(image, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        return image

    start, settled = simulate(0.0), simulate(3.0)
    focus = start.xpos[pelvis].copy() if pelvis >= 0 else np.zeros(3)
    room_centre = [focus[0], focus[1], 1.2]

    grid = np.vstack([
        np.hstack([
            shot(start, 135, -12, 4.5, focus, f"t=0    pelvis z={start.xpos[pelvis][2]:.3f} m"),
            shot(settled, 135, -12, 4.5, focus, f"t=3s   pelvis z={settled.xpos[pelvis][2]:.3f} m"),
        ]),
        np.hstack([
            shot(settled, 135, -22, 11.0, room_centre, "room + collision shell"),
            shot(settled, 215, -80, 12.0, room_centre, "top down"),
        ]),
    ])
    cv2.imwrite(str(out_png), grid)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description="Room mesh -> MuJoCo scene with real physics")
    ap.add_argument("--room", type=Path,
                    default=Path("outputs-full-factory/delivery/OUTPUT_B1/mesh/room_300k.glb"))
    ap.add_argument("--gt", type=Path,
                    default=Path("outputs-full-factory/delivery/OUTPUT_B1/metadata/ground_truth_boxes.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs-full-factory/delivery/OUTPUT_ROOM"))
    ap.add_argument("--robot", type=Path, default=None,
                    help="a unitree_robots/<name> directory holding origin/ and meshes/")
    ap.add_argument("--assets", default="assets", help="subdirectory for meshes")
    ap.add_argument("--box-collider", action="store_true",
                    help="fall back to the old box shell instead of the mesh itself")
    ap.add_argument("--max-hulls", type=int, default=64)
    ap.add_argument("--hull-threshold", type=float, default=0.06,
                    help="CoACD concavity threshold; lower is a tighter fit")
    ap.add_argument("--no-furniture", action="store_true",
                    help="shell only, no furniture colliders; the ground-truth file is "
                         "still read for the floor height, which is a separate question")
    ap.add_argument("--no-test", action="store_true", help="skip the physics checks")
    ap.add_argument("--no-render", action="store_true", help="skip the inspection render")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / args.assets).mkdir(parents=True, exist_ok=True)

    mesh = load_room(args.room)
    print(f"[load] {args.room.name}  {len(mesh.vertices):,} verts  "
          f"watertight={mesh.is_watertight}  bounds={np.round(mesh.bounds, 3).tolist()}")

    upright, info = orient(mesh, args.gt)
    print(f"[orient] floor from {info['floor_source']} at z={info['floor_z_before']} "
          f"-> 0;  ceiling now {info['ceiling_z_after']};  height {info['height']} m")

    visual_path = args.out / args.assets / "room_visual.obj"
    upright.export(str(visual_path))
    print(f"[visual] {args.assets}/{visual_path.name}  (contype=0, never collides)")

    # The reconstruction is the collider. A ground plane stays under it as a
    # backstop — the mesh floor has holes where the scan did not reach, and
    # dropping through one of them looks exactly like a physics failure.
    hulls = [] if args.box_collider else mesh_collider(
        upright, args.out, args.assets,
        max_hulls=args.max_hulls, threshold=args.hull_threshold)

    if hulls:
        shell = [g for g in room_shell(upright.bounds, info["ceiling_z_after"])
                 if g["type"] == "plane"]
        obstacles = []
        print(f"[collider] {len(hulls)} convex hulls from the mesh itself "
              f"({sum(h['faces'] for h in hulls):,} faces) + 1 backstop plane")
    else:
        shell = room_shell(upright.bounds, info["ceiling_z_after"])
        obstacles = [] if args.no_furniture else furniture_colliders(
            args.gt if args.gt.is_file() else None, FLIP_X180, info["floor_z_before"])
        print(f"[collider] 1 plane + {len(shell) - 1} shell boxes "
              f"+ {len(obstacles)} furniture boxes")

    robot = None
    if args.robot:
        robot = stage_robot(args.robot, args.out, args.assets)
        print(f"[robot] {robot['name']}  includes={robot['includes']}  "
              f"stand height={robot['stand_height']} m")

    spawn = find_spawn(upright.bounds, obstacles)
    centre = spawn["xy"]
    print(f"[spawn] robot at {np.round(centre, 2).tolist()} "
          f"with {spawn['clearance']} m clearance "
          f"(room centre is not free floor)")
    drop_from = [centre[0], centre[1], 1.5]
    xml_path = write_mjcf(args.out, visual_path.name, args.assets, shell, obstacles, hulls,
                          drop_from, robot, centre)
    if robot and robot["home_qpos"]:
        # Second pass: the keyframe can only be written once the model compiles,
        # because it is addressed by the scene's actual qpos layout.
        keyframe = build_keyframe(xml_path, robot["home_qpos"], centre)
        xml_path = write_mjcf(args.out, visual_path.name, args.assets, shell, obstacles, hulls,
                              drop_from, robot, centre, keyframe_xml=keyframe)
    terrain_path = write_terrain(args.out, visual_path.name, args.assets,
                                 shell, obstacles, hulls, spawn)
    print(f"[terrain] {terrain_path.name} + room_terrain.py  "
          f"(spec_fn for mjlab TerrainEntityCfg, no robot inside)")
    print(f"[mjcf] {xml_path}")

    report = {"room": str(args.room), "orientation": info,
              "shell": shell, "obstacles": obstacles, "hulls": hulls,
              "spawn": spawn, "drop_from": drop_from, "robot": robot}

    if not args.no_test:
        control = drop_test(xml_path)
        report["control_drop"] = control
        where = ("on the floor" if control["resting_on_floor"]
                 else f"on geometry at z={control['end_z']}")
        print(f"[control] sphere {control['start_z']} -> {control['end_z']} m, "
              f"caught {where}, {control['contacts']} contacts  "
              f"{'PASS' if control['settled_on_floor'] else 'FAIL'}")
        if control["fell_through"]:
            print("[control] it fell through the floor: the collider is wrong, "
                  "so nothing below this line means anything.")

        if robot:
            stand = stand_test(xml_path)
            report["robot_stand"] = stand
            if stand.get("ran"):
                print(f"[robot] pelvis {stand['start_pelvis_z']} -> {stand['end_pelvis_z']} m, "
                      f"lowest body z={stand['lowest_body_z']}, contacts={stand['contacts']}, "
                      f"{'PASS' if stand['supported'] else 'FAIL'}")
            else:
                print(f"[robot] stand test skipped: {stand['reason']}")

    if not args.no_render:
        png = render_check(xml_path, args.out / "room_robot_check.png")
        if png:
            report["render"] = png.name
            print(f"[render] {png}")

    (args.out / "room_scene_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] {args.out / 'room_scene_report.json'}")
    print(f"       view it:  {'{}'} -m mujoco.viewer --mjcf={xml_path}".format(".venv/bin/python"))


if __name__ == "__main__":
    main()
