"""Stage G: Simulation geometry compiler + physics metadata.

Reads structured_scene.json, produces:
- CFG.paths.visual/scene_visual.obj (copy from geometry if exists, else skip)
- CFG.paths.collision/{floor_plane.json, walls/wall_*.box.json, objects/obj_*.box.json}
- CFG.paths.simulation/physics.json (dict keyed by object_id, each with 'source' field)
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any
import numpy as np

from src.v2t.config import CFG


def load_structured_scene() -> dict[str, Any] | None:
    """Load structured_scene.json from CFG.paths.scene."""
    scene_path = CFG.paths.scene / "structured_scene.json"
    if not scene_path.exists():
        print(f"WARNING: {scene_path} not found. Producing stub outputs.")
        return None
    with open(scene_path) as f:
        return json.load(f)


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[1, 2] + R[2, 1]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=float)


def quat_from_xyzw(xyzw: list[float]) -> list[float]:
    """Convert quaternion from [x, y, z, w] to [w, x, y, z] format."""
    if len(xyzw) == 4:
        return [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    return [1.0, 0.0, 0.0, 0.0]


def obb_to_box_json(center: list[float], size: list[float], rotation: list[float] | None, category: str, source_id: str) -> dict:
    """Convert center, size, rotation to box collision JSON."""
    center_arr = np.array(center, dtype=float)
    size_arr = np.array(size, dtype=float)
    if rotation is not None:
        if len(rotation) == 4:
            quat = rotation
        elif len(rotation) == 9:
            R = np.array(rotation, dtype=float).reshape(3, 3)
            quat = quat_from_matrix(R).tolist()
        else:
            quat = [1.0, 0.0, 0.0, 0.0]
    else:
        quat = [1.0, 0.0, 0.0, 0.0]
    return {
        "type": "box",
        "position": center_arr.tolist(),
        "size": size_arr.tolist(),
        "rotation": quat,
        "category": category,
        "source_object_id": source_id,
    }


def create_floor_plane_from_plane(plane: list[float], category: str, source_id: str) -> dict:
    """Create a floor plane collision from plane equation [nx, ny, nz, d]."""
    normal = np.array(plane[:3], dtype=float)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    return {
        "type": "plane",
        "position": [0.0, 0.0, 0.0],
        "normal": normal.tolist(),
        "category": category,
        "source_object_id": source_id,
    }


def create_wall_box_from_plane_bounds(plane: list[float], bounds: list[list[float]], height: list[float], normal: list[float], category: str, source_id: str) -> dict:
    """Create a wall box from plane equation, 2D bounds, and height range."""
    x_min, y_min = bounds[0]
    x_max, y_max = bounds[1]
    z_min, z_max = height

    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    size = [x_max - x_min, y_max - y_min, z_max - z_min]

    normal_arr = np.array(normal, dtype=float)
    norm = np.linalg.norm(normal_arr)
    if norm > 0:
        normal_arr = normal_arr / norm

    # Find rotation that aligns [0,0,1] with normal
    if abs(normal_arr[2]) > 0.9:
        quat = [1.0, 0.0, 0.0, 0.0]
    elif abs(normal_arr[0]) > 0.9:
        quat = [0.0, 0.0, 0.7071, 0.7071]
    elif abs(normal_arr[1]) > 0.9:
        quat = [0.0, 0.7071, 0.0, 0.7071]
    else:
        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(z_axis, normal_arr)
        s = np.linalg.norm(v)
        if s > 1e-6:
            c = np.dot(z_axis, normal_arr)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
            quat = quat_from_matrix(R).tolist()
        else:
            quat = [1.0, 0.0, 0.0, 0.0] if normal_arr[2] > 0 else [0.0, 1.0, 0.0, 0.0]

    return {
        "type": "box",
        "position": center,
        "size": size,
        "rotation": quat,
        "category": category,
        "source_object_id": source_id,
    }


def create_stacked_boxes_stairs(obb: dict, category: str, source_id: str, num_steps: int = 5) -> list[dict]:
    """Create stacked boxes approximating stairs from an OBB."""
    center = np.array(obb.get("center", [0, 0, 0]), dtype=float)
    size = np.array(obb.get("size", [1, 1, 1]), dtype=float)
    rotation = obb.get("rotation")
    if rotation is not None:
        R = np.array(rotation, dtype=float).reshape(3, 3)
        quat = quat_from_matrix(R).tolist()
    else:
        quat = [1.0, 0.0, 0.0, 0.0]

    step_height = size[2] / num_steps
    step_depth = size[1] / num_steps
    boxes = []
    for i in range(num_steps):
        step_center = center.copy()
        step_center[2] = center[2] - size[2] / 2 + (i + 0.5) * step_height
        step_center[1] = center[1] - size[1] / 2 + (i + 0.5) * step_depth
        step_size = np.array([size[0], step_depth, step_height])
        boxes.append({
            "type": "box",
            "position": step_center.tolist(),
            "size": step_size.tolist(),
            "rotation": quat,
            "category": category,
            "source_object_id": f"{source_id}_step{i}",
        })
    return boxes


def copy_visual_mesh() -> bool:
    """Copy scene mesh from geometry to visual output."""
    src_objs = list(CFG.paths.geometry.glob("*.obj"))
    if not src_objs:
        print("WARNING: No .obj found in geometry output. Skipping visual copy.")
        return False
    src = src_objs[0]
    CFG.paths.visual.mkdir(parents=True, exist_ok=True)
    dst = CFG.paths.visual / "scene_visual.obj"
    shutil.copy2(src, dst)
    print(f"Copied visual mesh: {src} -> {dst}")
    return True


def write_collision_json(output_dir: Path, filename: str, data: dict | list):
    """Write collision JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote: {out_path}")


def build_physics_objects(collision_objects: list[dict], scene_data: dict | None) -> dict[str, dict]:
    """Build physics object dict for physics.json, keyed by object_id.

    Returns a dict where each key is an object_id and each value is a physics object
    with a 'source' field (required by validation).
    """
    physics_objects = {}
    profiles = CFG.physics_profiles

    for obj in collision_objects:
        category = obj.get("category", "default")
        profile = profiles.get(category, profiles["default"])
        obj_id = obj.get("source_object_id", f"obj_{len(physics_objects)}")

        physics_obj = {
            "id": obj_id,
            "category": category,
            "static": profile.get("static", True),
            "friction": profile.get("friction", 0.5),
            "restitution": profile.get("restitution", 0.1),
            "center_of_mass": [0.0, 0.0, 0.0],
            "inertia_diag": [1.0, 1.0, 1.0],
            "inertia_off": [0.0, 0.0, 0.0],
            "source": profile.get("source", "default_profile:generic"),
        }
        if not profile.get("static", True):
            physics_obj["mass"] = profile.get("mass", 1.0)
        else:
            physics_obj["mass"] = 0.0

        physics_objects[obj_id] = physics_obj

    return physics_objects


def run_stage_g() -> int:
    """Run Stage G: collision geometry + physics metadata."""
    print("=== Stage G: Collision Geometry + Physics ===")

    CFG.paths.collision.mkdir(parents=True, exist_ok=True)
    CFG.paths.simulation.mkdir(parents=True, exist_ok=True)
    CFG.paths.visual.mkdir(parents=True, exist_ok=True)
    (CFG.paths.collision / "walls").mkdir(parents=True, exist_ok=True)
    (CFG.paths.collision / "objects").mkdir(parents=True, exist_ok=True)
    (CFG.paths.collision / "stairs").mkdir(parents=True, exist_ok=True)

    scene_data = load_structured_scene()

    # Copy visual mesh
    copy_visual_mesh()

    collision_objects = []

    if scene_data is None:
        # Produce stub outputs for downstream
        print("Producing stub collision/physics outputs.")
        floor_plane = create_floor_plane_from_plane([0, 0, 1, 0], "floor", "stub_floor")
        write_collision_json(CFG.paths.collision, "floor_plane.json", floor_plane)
        collision_objects.append(floor_plane)

        physics = build_physics_objects(collision_objects, None)
        with open(CFG.paths.simulation / "physics.json", "w") as f:
            json.dump(physics, f, indent=2)
        print(f"Wrote: {CFG.paths.simulation / 'physics.json'}")
        return 0

    # ---- Process room structure (floor, walls, ceiling) ----
    room = scene_data.get("room", {})
    wall_idx = 0

    # Process floor from room.floor (plane + bounds)
    floor = room.get("floor")
    if floor is not None:
        plane = floor.get("plane")
        source_id = floor.get("id", "floor_0")
        if plane is not None:
            floor_plane = create_floor_plane_from_plane(plane, "floor", source_id)
            write_collision_json(CFG.paths.collision, "floor_plane.json", floor_plane)
            collision_objects.append(floor_plane)

    # Process walls from room.walls (plane + bounds + height + normal)
    walls = room.get("walls", [])
    for wall in walls:
        plane = wall.get("plane")
        bounds = wall.get("bounds")
        height = wall.get("height")
        normal = wall.get("normal")
        source_id = wall.get("id", f"wall_{wall_idx}")
        if plane is not None and bounds is not None and height is not None and normal is not None:
            box_json = create_wall_box_from_plane_bounds(plane, bounds, height, normal, "wall", source_id)
            write_collision_json(CFG.paths.collision / "walls", f"wall_{wall_idx}.box.json", box_json)
            collision_objects.append(box_json)
            wall_idx += 1

    # Process ceiling from room.ceiling (plane)
    ceiling = room.get("ceiling")
    if ceiling is not None:
        plane = ceiling.get("plane")
        source_id = ceiling.get("id", "ceiling_0")
        if plane is not None:
            ceiling_plane = create_floor_plane_from_plane(plane, "ceiling", source_id)
            write_collision_json(CFG.paths.collision / "walls", f"wall_{wall_idx}.box.json", ceiling_plane)
            collision_objects.append(ceiling_plane)
            wall_idx += 1

    # ---- Process objects array (position, rotation, size) ----
    # Skip structural elements (floor, wall, ceiling) — room shell handles them
    STRUCTURAL_CATEGORIES = {"floor", "wall", "ceiling", "door"}

    objects = scene_data.get("objects", [])
    obj_idx = 0

    for obj in objects:
        category = obj.get("category", "default")
        classified_type = obj.get("classified_type", category)
        source_id = obj.get("id", f"obj_{obj_idx}")
        position = obj.get("position")
        rotation = obj.get("rotation")
        size = obj.get("size")

        if position is None or size is None:
            continue

        # Skip structural categories — use room.floor / room.walls instead
        if category in STRUCTURAL_CATEGORIES or classified_type in STRUCTURAL_CATEGORIES:
            print(f"  Skipping structural object: {source_id} (category={category}, type={classified_type})")
            obj_idx += 1
            continue

        quat = quat_from_xyzw(rotation) if rotation is not None else None
        box_json = obb_to_box_json(position, size, quat, category, source_id)

        if category == "stairs":
            obb_dict = {"center": position, "size": size, "rotation": rotation}
            boxes = create_stacked_boxes_stairs(obb_dict, category, source_id)
            for i, box in enumerate(boxes):
                write_collision_json(CFG.paths.collision / "stairs", f"stair_{obj_idx}_step{i}.box.json", box)
                collision_objects.append(box)
            obj_idx += 1
            continue

        # Furniture and other objects
        write_collision_json(CFG.paths.collision / "objects", f"obj_{obj_idx}_{category}.box.json", box_json)
        collision_objects.append(box_json)
        obj_idx += 1

    # Write physics.json as dict keyed by object_id (each entry has 'source' field)
    physics = build_physics_objects(collision_objects, scene_data)
    with open(CFG.paths.simulation / "physics.json", "w") as f:
        json.dump(physics, f, indent=2)
    print(f"Wrote: {CFG.paths.simulation / 'physics.json'}")

    print(f"Stage G complete. Collision objects: {len(collision_objects)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_stage_g())