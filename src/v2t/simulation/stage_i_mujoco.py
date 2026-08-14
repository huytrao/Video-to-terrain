"""Stage I: Compile MJCF + MJLab config for MuJoCo simulation."""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Any

import mujoco

from src.v2t.config import CFG


# Robot XML path
ROBOT_XML_PATH = Path(
    "/Users/traoanhuy/source_code/vinrobotics_mjlab/src/assets/robots/vr_m3_1/xmls/vr_m3_1.xml"
)


def load_structured_scene() -> dict[str, Any] | None:
    """Load structured_scene.json from scene directory."""
    scene_path = CFG.paths.scene / "structured_scene.json"
    if scene_path.exists():
        with open(scene_path) as f:
            return json.load(f)
    return None


def load_visual_meshes() -> list[dict[str, Any]]:
    """Load visual mesh data from simulation/visual directory."""
    visual_dir = CFG.paths.visual
    meshes = []
    for mesh_file in visual_dir.glob("*.json"):
        with open(mesh_file) as f:
            meshes.append(json.load(f))
    return meshes


def load_collision_meshes() -> list[dict[str, Any]]:
    """Load collision mesh data from simulation/collision directory."""
    collision_dir = CFG.paths.collision
    meshes = []
    for mesh_file in collision_dir.glob("*.json"):
        with open(mesh_file) as f:
            meshes.append(json.load(f))
    return meshes


def load_physics_config() -> dict[str, Any]:
    """Load physics config from simulation/physics.json."""
    physics_path = CFG.paths.simulation / "physics.json"
    if physics_path.exists():
        with open(physics_path) as f:
            return json.load(f)
    return CFG.physics_profiles


def build_mujoco_spec() -> mujoco.MjSpec:
    """Build full MJCF spec using MjSpec API."""
    scene_data = load_structured_scene()
    physics = load_physics_config()

    # Load robot spec
    if ROBOT_XML_PATH.exists():
        spec = mujoco.MjSpec.from_file(str(ROBOT_XML_PATH))
        print(f"Loaded robot spec: {ROBOT_XML_PATH}")
    else:
        warnings.warn(f"Robot XML not found at {ROBOT_XML_PATH}, creating empty spec")
        spec = mujoco.MjSpec()

    # Set compiler options - use absolute path to MJLab robot assets
    robot_assets_dir = ROBOT_XML_PATH.parent / "assets"
    spec.compiler.degree = False  # False = radians
    spec.compiler.meshdir = str(robot_assets_dir)
    spec.compiler.autolimits = True

    # Set simulation options
    spec.option.timestep = 0.005
    spec.option.gravity = [0, 0, -9.81]

    # Add default geom settings
    spec.default.geom.solref = [0.02, 1]
    spec.default.geom.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]

    # Set robot spawn height - MjSpec API doesn't expose qpos0 on MjsJoint directly
    # The robot will spawn at its default position; we rely on the free joint in the XML
    spawn_height = CFG.robot_spawn_height
    robot_root_body = spec.body("pelvis")
    if robot_root_body is not None:
        print(f"Robot root body 'pelvis' found, spawn height will be handled at runtime: z={spawn_height}")

    # Add floor plane
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        name="floor",
        pos=[0, 0, 0],
        size=[10, 10, 0.1],
        friction=[0.8, 0.02, 0.0001],
        rgba=[0.5, 0.5, 0.5, 1],
    )

    # Add walls
    wall_profile = physics.get("wall", physics.get("default", {}))
    wall_friction = wall_profile.get("friction", 0.6)
    room_size = 5.0
    wall_height = 2.5
    wall_thickness = 0.1

    walls = [
        ("wall_north", [0, room_size / 2, wall_height / 2],
         [room_size / 2, wall_thickness / 2, wall_height / 2]),
        ("wall_south", [0, -room_size / 2, wall_height / 2],
         [room_size / 2, wall_thickness / 2, wall_height / 2]),
        ("wall_east", [room_size / 2, 0, wall_height / 2],
         [wall_thickness / 2, room_size / 2, wall_height / 2]),
        ("wall_west", [-room_size / 2, 0, wall_height / 2],
         [wall_thickness / 2, room_size / 2, wall_height / 2]),
    ]

    for name, pos, size in walls:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            name=name,
            pos=pos,
            size=size,
            friction=[wall_friction, 0.02, 0.0001],
            rgba=[0.7, 0.7, 0.7, 1],
        )

    # Add objects from structured scene
    if scene_data:
        objects = scene_data.get("objects", [])
        for i, obj in enumerate(objects):
            obj_type = obj.get("type", "default")
            profile = physics.get(obj_type, physics.get("default", {}))
            friction = profile.get("friction", 0.5)

            name = obj.get("name", f"obj_{i}")
            pos = obj.get("position", [0, 0, 0])

            if "size" in obj:
                size = obj["size"]
                half_extents = [s / 2 for s in size]
                static = profile.get("static", True)

                contype = 0 if static else 1
                conaffinity = 1

                body = spec.worldbody.add_body(name=name, pos=pos)
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    name=f"{name}_geom",
                    size=half_extents,
                    friction=[friction, 0.02, 0.0001],
                    contype=contype,
                    conaffinity=conaffinity,
                    rgba=[0.8, 0.6, 0.4, 1],
                )

    return spec


def save_mujoco_scene(spec: mujoco.MjSpec, output_path: Path) -> bool:
    """Save MJCF spec to file and verify it compiles."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compile and save
    try:
        model = spec.compile()
        print(f"MJCF compiled successfully: {model.nq} DOFs, {model.nbody} bodies, {model.ngeom} geoms")

        # Save XML
        spec.to_file(str(output_path))
        print(f"Saved MJCF to {output_path}")
        return True
    except Exception as e:
        print(f"ERROR: MJCF compilation failed: {e}")
        # Try to save minimal valid scene
        save_minimal_scene(output_path)
        return False


def save_minimal_scene(output_path: Path) -> None:
    """Save a minimal valid scene with just floor and robot."""
    if ROBOT_XML_PATH.exists():
        spec = mujoco.MjSpec.from_file(str(ROBOT_XML_PATH))
        # Set robot spawn height for the loaded robot
        robot_root_body = spec.body("pelvis")
        if robot_root_body is not None:
            free_joint = None
            for j in robot_root_body.joints:
                if j.name == "base_joint":
                    free_joint = j
                    break
            if free_joint is not None:
                free_joint.qpos0 = [0.0, 0.0, CFG.robot_spawn_height, 1.0, 0.0, 0.0, 0.0]
                print(f"Set robot spawn height (minimal): z={CFG.robot_spawn_height}")
    else:
        spec = mujoco.MjSpec()
        # Add simple capsule robot
        robot_body = spec.worldbody.add_body(name="pelvis", pos=[0, 0, CFG.robot_spawn_height])
        robot_body.add_joint(name="base_joint", type=mujoco.mjtJoint.mjJNT_FREE)
        robot_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            name="robot_body",
            size=[0.15, 0.5],
            pos=[0, 0, 0.5],
            friction=[0.5, 0.02, 0.0001],
        )

    # Set compiler options - use absolute path to MJLab robot assets
    robot_assets_dir = ROBOT_XML_PATH.parent / "assets"
    spec.compiler.degree = False  # False = radians
    spec.compiler.meshdir = str(robot_assets_dir)
    spec.compiler.autolimits = True

    # Set simulation options
    spec.option.timestep = 0.005
    spec.option.gravity = [0, 0, -9.81]

    # Add floor
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        name="floor",
        pos=[0, 0, 0],
        size=[10, 10, 0.1],
        friction=[0.8, 0.02, 0.0001],
    )

    # Compile and save
    try:
        model = spec.compile()
        print(f"Minimal scene compiled: {model.nq} DOFs, {model.nbody} bodies, {model.ngeom} geoms")
        spec.to_file(str(output_path))
        print(f"Saved minimal MJCF to {output_path}")
    except Exception as e:
        print(f"ERROR: Even minimal scene failed: {e}")


def write_mjlab_config(output_path: Path) -> None:
    """Write MJLab scene configuration Python file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_content = '''"""MJLab scene configuration for VR-M3-1 in reconstructed environment."""
from __future__ import annotations

# Try to import mjlab - if not available, this serves as a template
try:
    from mjlab import SceneCfg, TerrainEntityCfg, RobotEntityCfg
    MJLAB_AVAILABLE = True
except ImportError:
    MJLAB_AVAILABLE = False
    # Minimal stubs for type hints
    class SceneCfg:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class TerrainEntityCfg:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class RobotEntityCfg:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# Robot entity reference for vinrobotics_mjlab:
#   from vinrobotics_mjlab.robots.vr_m3_1.config import get_vr_m3_1_implicit_actuator_robot_cfg
#   robot_cfg = get_vr_m3_1_implicit_actuator_robot_cfg(spawn_pos=(0.0, 0.0, 1.0))


class SceneConfig:
    """Scene configuration container."""

    def __init__(self):
        self.terrain = TerrainEntityCfg(
            terrain_type="plane",
            plane_cfg=dict(
                size=(10.0, 10.0),
                friction=(0.8, 0.02, 0.0001),
            ),
        )
        self.entities = {
            # Robot entity - use vinrobotics_mjlab function when available:
            #   "robot": get_vr_m3_1_implicit_actuator_robot_cfg(spawn_pos=(0.0, 0.0, 1.0)),
            "robot": RobotEntityCfg(
                robot_cfg="vinrobotics_mjlab.robots.vr_m3_1.config:get_vr_m3_1_implicit_actuator_robot_cfg",
                spawn_pos=(0.0, 0.0, 1.0),
            ) if MJLAB_AVAILABLE else None,
        }


# Exported scene configuration
scene_cfg = SceneConfig()


if __name__ == "__main__":
    print("MJLab scene config loaded")
    print(f"Terrain type: {getattr(scene_cfg.terrain, 'terrain_type', 'plane')}")
    print(f"Entities: {list(scene_cfg.entities.keys())}")
    if not MJLAB_AVAILABLE:
        print("WARNING: mjlab not installed - this is a template only")
'''

    with open(output_path, "w") as f:
        f.write(config_content)
    print(f"Saved MJLab config to {output_path}")


def run_stage_i() -> bool:
    """Run Stage I: Compile MJCF + MJLab config."""
    print("=" * 60)
    print("Stage I: Compile MJCF + MJLab config")
    print("=" * 60)

    # Ensure output directories exist
    CFG.paths.mujoco.mkdir(parents=True, exist_ok=True)
    CFG.paths.mjlab.mkdir(parents=True, exist_ok=True)

    # Check available scene data
    scene_data = load_structured_scene()
    if scene_data is None:
        print("WARNING: No structured_scene.json found, using placeholder geometry")
    else:
        n_obj = len(scene_data.get("objects", []))
        print(f"Loaded structured scene with {n_obj} objects")

    # Check for visual/collision mesh data
    visual_meshes = load_visual_meshes()
    collision_meshes = load_collision_meshes()
    print(f"Visual meshes: {len(visual_meshes)}, Collision meshes: {len(collision_meshes)}")

    # Build MuJoCo scene spec
    spec = build_mujoco_spec()

    # Save and verify
    mujoco_path = CFG.paths.mujoco / "scene.xml"
    success = save_mujoco_scene(spec, mujoco_path)

    # Write MJLab config
    mjlab_path = CFG.paths.mjlab / "scene_cfg.py"
    write_mjlab_config(mjlab_path)

    if success:
        print("\nStage I completed successfully!")
    else:
        print("\nStage I completed with warnings (used minimal scene)")

    return success


if __name__ == "__main__":
    run_stage_i()