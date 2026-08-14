"""Stage H: Robot Policy Configuration.

Loads MJLab robot config for vr_m3_1 and generates policy/control configuration.
Saves policy config to simulation/policy/ for downstream MJCF/MJLab stages.

Production: Supports real trained policy checkpoints, terrain-aware configs,
and multiple controller types (tracking, velocity, rough terrain).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

from src.v2t.config import CFG


# MJLab robot config import path
VR_M3_1_ROBOT_CONFIG_IMPORT = os.getenv(
    "VR_M3_1_ROBOT_CONFIG",
    "vinrobotics_mjlab.assets.robots.vr_m3_1.vr_m3_1_constants:get_vr_m3_1_implicit_actuator_robot_cfg"
)

# Available task configs in vinrobotics_mjlab
TASK_CONFIGS = {
    "tracking": "vinrobotics_mjlab.tasks.tracking.config.vr_m3_1.env_cfgs:vr_m3_1_flat_tracking_env_cfg",
    "velocity": "vinrobotics_mjlab.tasks.velocity.config.vr_m3_1.env_cfgs:vr_m3_1_velocity_env_cfg",
    "rough": "vinrobotics_mjlab.tasks.tracking.config.vr_m3_1.env_cfgs:vr_m3_1_rough_tracking_env_cfg",
}

# Default policy type for MVP
DEFAULT_POLICY_TYPE = os.getenv("ROBOT_POLICY_TYPE", "tracking")

# Production: Path to trained policy checkpoints (optional)
POLICY_CHECKPOINT_DIR = os.getenv("POLICY_CHECKPOINT_DIR")  # e.g., /path/to/checkpoints


def load_structured_scene() -> dict[str, Any] | None:
    """Load structured_scene.json from scene directory."""
    scene_path = CFG.paths.scene / "structured_scene.json"
    if scene_path.exists():
        with open(scene_path) as f:
            return json.load(f)
    return None


def get_robot_spawn_info() -> dict[str, Any]:
    """Get robot spawn configuration from pipeline config."""
    return {
        "robot_name": CFG.robot_name,
        "spawn_height": CFG.robot_spawn_height,
        "spawn_position": [0.0, 0.0, CFG.robot_spawn_height],
        "spawn_orientation": [1.0, 0.0, 0.0, 0.0],  # quaternion [w, x, y, z]
    }


def get_actuator_config() -> dict[str, Any]:
    """Get VR M3.1 actuator configuration from MJLab constants.

    This mirrors the action_scale computation from vr_m3_1_constants.py.
    """
    # These match VR_M3_1_ACTION_SCALE computation in vinrobotics_mjlab
    # action_scale = 0.25 * effort_limit / stiffness
    action_scales = {
        "hip_pitch": 0.25 * 360.0 / 150.0,    # 0.6
        "hip_roll": 0.25 * 360.0 / 150.0,     # 0.6
        "hip_yaw": 0.25 * 130.0 / 120.0,      # ~0.27
        "knee_pitch": 0.25 * 360.0 / 200.0,   # 0.45
        "ankle_pitch": 0.25 * 120.0 / 200.0,  # 0.15 (effort_limit * 2 / stiffness)
        "ankle_roll": 0.25 * 120.0 / 200.0,   # 0.15
        "waist_yaw": 0.25 * 102.0 / 367.0,    # ~0.069
        "shoulder_pitch": 0.25 * 66.0 / 140.0, # ~0.118
        "shoulder_roll": 0.25 * 66.0 / 140.0,  # ~0.118
        "shoulder_yaw": 0.25 * 34.0 / 40.0,    # 0.2125
        "elbow_pitch": 0.25 * 34.0 / 40.0,     # 0.2125
        "wrist_yaw": 0.25 * 11.0 / 30.0,       # ~0.092
        "wrist_roll": 0.25 * 11.0 / 30.0,      # ~0.092
        "wrist_pitch": 0.25 * 11.0 / 30.0,     # ~0.092
    }

    # Joint name patterns (from vr_m3_1_constants.py)
    joint_patterns = {
        "left_hip_pitch_joint": "hip_pitch",
        "right_hip_pitch_joint": "hip_pitch",
        "left_hip_roll_joint": "hip_roll",
        "right_hip_roll_joint": "hip_roll",
        "left_hip_yaw_joint": "hip_yaw",
        "right_hip_yaw_joint": "hip_yaw",
        "left_knee_pitch_joint": "knee_pitch",
        "right_knee_pitch_joint": "knee_pitch",
        "left_ankle_pitch_joint": "ankle_pitch",
        "right_ankle_pitch_joint": "ankle_pitch",
        "left_ankle_roll_joint": "ankle_roll",
        "right_ankle_roll_joint": "ankle_roll",
        "waist_yaw_joint": "waist_yaw",
        "left_shoulder_pitch_joint": "shoulder_pitch",
        "right_shoulder_pitch_joint": "shoulder_pitch",
        "left_shoulder_roll_joint": "shoulder_roll",
        "right_shoulder_roll_joint": "shoulder_roll",
        "left_shoulder_yaw_joint": "shoulder_yaw",
        "right_shoulder_yaw_joint": "shoulder_yaw",
        "left_elbow_pitch_joint": "elbow_pitch",
        "right_elbow_pitch_joint": "elbow_pitch",
        "left_wrist_yaw_joint": "wrist_yaw",
        "right_wrist_yaw_joint": "wrist_yaw",
        "left_wrist_roll_joint": "wrist_roll",
        "right_wrist_roll_joint": "wrist_roll",
        "left_wrist_pitch_joint": "wrist_pitch",
        "right_wrist_pitch_joint": "wrist_pitch",
    }

    return {
        "action_scales": action_scales,
        "joint_patterns": joint_patterns,
        "control_type": "position",
        "decimation": 4,  # control frequency = sim_freq / decimation
    }


def get_tracking_policy_config(terrain_type: str = "plane") -> dict[str, Any]:
    """Get motion tracking policy configuration.

    Args:
        terrain_type: "plane" for flat, "rough" for rough terrain
    """
    tracked_bodies = [
        "pelvis",
        "left_hip_roll_link",
        "left_knee_pitch_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "left_shoulder_roll_link",
        "left_elbow_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_pitch_link",
        "right_wrist_yaw_link",
    ]

    end_effector_bodies = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]

    # Adjust rewards for rough terrain
    foot_clearance_weight = 0.1 if terrain_type == "rough" else 0.05
    feet_air_time_weight = 0.1 if terrain_type == "rough" else 0.0
    bad_orientation_limit = 1.0 if terrain_type == "rough" else 0.785

    return {
        "policy_type": "tracking",
        "task_config": TASK_CONFIGS["rough"] if terrain_type == "rough" else TASK_CONFIGS["tracking"],
        "root_body": "pelvis",
        "torso_body": "waist_yaw_link",
        "tracked_bodies": tracked_bodies,
        "end_effector_bodies": end_effector_bodies,
        "foot_geoms_pattern": r"^(left|right)_ankle_roll_link_collision_\d+$",
        "motion_command": {
            "anchor_body_name": "waist_yaw_link",
            "body_names": tracked_bodies,
            "sampling_mode": "start",  # for play/eval
            "pose_range": {},
            "velocity_range": {},
        },
        "rewards": {
            "body_pos_tracking": {"weight": 1.0},
            "body_vel_tracking": {"weight": 0.1},
            "body_orientation_tracking": {"weight": 0.5},
            "joint_pos_regularization": {"weight": 0.01},
            "action_rate": {"weight": 0.005},
            "foot_clearance": {"weight": foot_clearance_weight},
            "feet_air_time": {"weight": feet_air_time_weight},
        },
        "terminations": {
            "ee_body_pos": {"body_names": end_effector_bodies},
            "bad_orientation": {"limit_angle": bad_orientation_limit},
            "height": {"min_height": 0.3, "max_height": 2.0},
        },
    }


def get_velocity_policy_config() -> dict[str, Any]:
    """Get velocity/locomotion policy configuration."""
    return {
        "policy_type": "velocity",
        "task_config": TASK_CONFIGS["velocity"],
        "root_body": "pelvis",
        "torso_body": "waist_yaw_link",
        "command_ranges": {
            "lin_vel_x": [-1.0, 1.0],
            "lin_vel_y": [-0.5, 0.5],
            "ang_vel_z": [-1.0, 1.0],
        },
        "rewards": {
            "tracking_lin_vel": {"weight": 1.0},
            "tracking_ang_vel": {"weight": 0.5},
            "lin_vel_z": {"weight": -2.0},
            "ang_vel_xy": {"weight": -0.05},
            "dof_pos_limits": {"weight": -1.0},
            "dof_vel": {"weight": -1e-3},
            "dof_torque": {"weight": -1e-4},
            "dof_acc": {"weight": -2.5e-7},
            "action_rate": {"weight": -0.01},
            "feet_air_time": {"weight": 1.0},
            "feet_slide": {"weight": -0.25},
        },
    }


def build_policy_config(policy_type: str = DEFAULT_POLICY_TYPE) -> dict[str, Any]:
    """Build complete policy configuration with production features."""
    scene_data = load_structured_scene()

    # Determine terrain type from scene
    terrain_type = "plane"
    terrain_friction = 0.8
    if scene_data:
        has_stairs = any(o.get("category") == "stairs" for o in scene_data.get("objects", []))
        has_rough = any(o.get("category") in ["gravel", "grass", "dirt", "uneven"] for o in scene_data.get("objects", []))
        if has_stairs or policy_type == "rough":
            terrain_type = "rough"
            terrain_friction = 0.9
        elif has_rough:
            terrain_type = "rough"
            terrain_friction = 0.7

    config = {
        "pipeline": {
            "run_name": CFG.run_name,
            "stage": "H",
            "description": "Robot policy configuration for MJLab scene",
            "policy_type": policy_type,
            "terrain_type": terrain_type,
        },
        "robot": {
            "config_import": VR_M3_1_ROBOT_CONFIG_IMPORT,
            "spawn": get_robot_spawn_info(),
        },
        "actuators": get_actuator_config(),
        "simulation": {
            "dt": 0.005,
            "decimation": 4,
            "control_freq_hz": 50,  # 1 / (0.005 * 4)
        },
        "terrain": {
            "type": terrain_type,
            "friction": terrain_friction,
        },
        "production": {
            "checkpoint_dir": POLICY_CHECKPOINT_DIR,
            "checkpoint_available": bool(POLICY_CHECKPOINT_DIR and Path(POLICY_CHECKPOINT_DIR).exists()),
        },
    }

    # Add task-specific policy config
    if policy_type == "tracking":
        config["policy"] = get_tracking_policy_config(terrain_type)
    elif policy_type == "velocity":
        config["policy"] = get_velocity_policy_config()
    elif policy_type == "rough":
        config["policy"] = get_rough_terrain_policy_config()
    else:
        config["policy"] = get_tracking_policy_config(terrain_type)  # default

    # Add environment info from scene if available
    if scene_data:
        config["environment"] = {
            "room_dims": scene_data.get("room", {}).get("dimensions", [10, 10, 3]),
            "floor_height": 0.0,
            "has_stairs": any(o.get("category") == "stairs" for o in scene_data.get("objects", [])),
            "objects": [
                {
                    "name": o.get("name", o.get("id", "unknown")),
                    "category": o.get("category", "default"),
                    "position": o.get("position", [0, 0, 0]),
                }
                for o in scene_data.get("objects", [])
            ],
        }

    # Add trained policy checkpoint info if available
    if POLICY_CHECKPOINT_DIR and Path(POLICY_CHECKPOINT_DIR).exists():
        checkpoint_files = list(Path(POLICY_CHECKPOINT_DIR).glob("*.pt")) + list(Path(POLICY_CHECKPOINT_DIR).glob("*.pth"))
        if checkpoint_files:
            config["policy"]["checkpoint"] = str(checkpoint_files[0])
            config["policy"]["checkpoint_available"] = True

    return config


def get_rough_terrain_policy_config() -> dict[str, Any]:
    """Get rough terrain tracking policy configuration."""
    tracked_bodies = [
        "pelvis",
        "left_hip_roll_link", "left_knee_pitch_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_pitch_link", "right_ankle_roll_link",
        "waist_yaw_link",
        "left_shoulder_roll_link", "left_elbow_pitch_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_pitch_link", "right_wrist_yaw_link",
    ]

    end_effector_bodies = [
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
    ]

    return {
        "policy_type": "rough",
        "task_config": TASK_CONFIGS["rough"],
        "root_body": "pelvis",
        "torso_body": "waist_yaw_link",
        "tracked_bodies": tracked_bodies,
        "end_effector_bodies": end_effector_bodies,
        "foot_geoms_pattern": r"^(left|right)_ankle_roll_link_collision_\d+$",
        "motion_command": {
            "anchor_body_name": "waist_yaw_link",
            "body_names": tracked_bodies,
            "sampling_mode": "start",
            "pose_range": {},
            "velocity_range": {},
        },
        "rewards": {
            "body_pos_tracking": {"weight": 1.0},
            "body_vel_tracking": {"weight": 0.1},
            "body_orientation_tracking": {"weight": 0.5},
            "joint_pos_regularization": {"weight": 0.01},
            "action_rate": {"weight": 0.005},
            "foot_clearance": {"weight": 0.1},  # Higher for rough terrain
            "feet_air_time": {"weight": 0.1},   # Encourage foot lifting
        },
        "terminations": {
            "ee_body_pos": {"body_names": end_effector_bodies},
            "bad_orientation": {"limit_angle": 1.0},  # More tolerant for rough terrain
            "height": {"min_height": 0.3, "max_height": 2.0},
        },
    }


def write_policy_config(config: dict[str, Any], output_dir: Path) -> Path:
    """Write policy configuration to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "policy_config.json"

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote policy config: {output_path}")
    return output_path


def write_mjlab_policy_py(config: dict[str, Any], output_dir: Path) -> Path:
    """Write MJLab-compatible Python policy config file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "policy_cfg.py"

    robot_config = config["robot"]
    actuator_config = config["actuators"]
    policy_config = config["policy"]
    sim_config = config["simulation"]
    pipeline_config = config["pipeline"]
    production_config = config.get("production", {})

    # Pre-compute values that need to be inserted
    action_scales_str = str(actuator_config["action_scales"])
    joint_patterns_str = str(actuator_config["joint_patterns"])
    tracked_bodies_str = str(policy_config.get("tracked_bodies", []))
    end_effector_bodies_str = str(policy_config.get("end_effector_bodies", []))
    foot_geoms_pattern = policy_config.get("foot_geoms_pattern", r"^(left|right)_ankle_roll_link_collision_\d+$")

    # Checkpoint info
    checkpoint_path = policy_config.get("checkpoint", "")
    checkpoint_available = policy_config.get("checkpoint_available", False)

    content = f'''"""MJLab policy configuration for VR-M3-1 robot.

Auto-generated by Stage H (Robot Policy Config).
Run: {pipeline_config["run_name"]}
Policy type: {policy_config["policy_type"]}
Terrain type: {pipeline_config.get("terrain_type", "plane")}
Checkpoint available: {checkpoint_available}
"""
from __future__ import annotations

# MJLab imports (available when running in MJLab environment)
try:
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    MJLAB_AVAILABLE = True
except ImportError:
    MJLAB_AVAILABLE = False
    # Stubs for type hints
    class ManagerBasedRlEnvCfg:
        pass
    class JointPositionActionCfg:
        pass
    class MotionCommandCfg:
        pass
    def make_tracking_env_cfg():
        pass
    class ContactMatch:
        pass
    class ContactSensorCfg:
        pass


# Robot configuration reference
ROBOT_CONFIG_IMPORT = "{robot_config["config_import"]}"


# Actuator configuration
ACTION_SCALES = {action_scales_str}

JOINT_PATTERNS = {joint_patterns_str}

CONTROL_TYPE = "{actuator_config["control_type"]}"
DECIMATION = {actuator_config["decimation"]}


# Simulation settings
SIM_DT = {sim_config["dt"]}
SIM_DECIMATION = {sim_config["decimation"]}
CONTROL_FREQ_HZ = {sim_config["control_freq_hz"]}


# Terrain settings
TERRAIN_TYPE = "{config["terrain"]["type"]}"
TERRAIN_FRICTION = {config["terrain"]["friction"]}


# Policy-specific configuration
POLICY_TYPE = "{policy_config["policy_type"]}"
TASK_CONFIG_IMPORT = "{policy_config["task_config"]}"

ROOT_BODY = "{policy_config.get("root_body", "pelvis")}"
TORSO_BODY = "{policy_config.get("torso_body", "waist_yaw_link")}"

# Production: Trained policy checkpoint (if available)
CHECKPOINT_PATH = "{checkpoint_path}"
CHECKPOINT_AVAILABLE = {checkpoint_available}


def get_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create environment configuration for the selected policy type.

    This function mirrors the structure in vinrobotics_mjlab task configs.
    """
    if not MJLAB_AVAILABLE:
        raise ImportError("mjlab not available. Install mjlab to use this config.")

    if POLICY_TYPE == "tracking":
        return _get_tracking_env_cfg(play)
    elif POLICY_TYPE == "rough":
        return _get_rough_terrain_env_cfg(play)
    elif POLICY_TYPE == "velocity":
        return _get_velocity_env_cfg(play)
    else:
        return _get_tracking_env_cfg(play)


def _get_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create tracking environment configuration (flat or rough terrain)."""
    cfg = make_tracking_env_cfg()

    # Robot entity - will be populated by MJLab scene config
    # cfg.scene.entities = {{"robot": get_vr_m3_1_implicit_actuator_robot_cfg()}}

    # Self-collision sensor
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    # Action scale
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = ACTION_SCALES

    # Motion command
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = TORSO_BODY
    motion_cmd.body_names = {tracked_bodies_str}

    # Event configurations
    foot_geoms = r"{foot_geoms_pattern}"
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
    cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

    # Terminations
    ee_bodies = {end_effector_bodies_str}
    cfg.terminations["ee_body_pos"].params["body_names"] = ee_bodies

    # Viewer
    cfg.viewer.body_name = TORSO_BODY

    # Simulation sizing for full-body robot
    cfg.sim.nconmax = 256
    cfg.sim.njmax = 700
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.mujoco.ccd_iterations = 50

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion_cmd.pose_range = {{}}
        motion_cmd.velocity_range = {{}}
        motion_cmd.sampling_mode = "start"

    return cfg


def _get_rough_terrain_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create rough terrain tracking environment configuration."""
    # Re-use tracking config but with rough terrain adjustments
    cfg = _get_tracking_env_cfg(play)

    # Adjust terrain curriculum for rough terrain
    if hasattr(cfg, "terrain"):
        cfg.terrain.terrain_type = "rough"
        cfg.terrain.terrain_proportions = [0.0, 0.2, 0.3, 0.2, 0.2, 0.1]  # Rougher mix

    return cfg


def _get_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create velocity environment configuration.

    Note: This requires the velocity task config from vinrobotics_mjlab.
    """
    if not MJLAB_AVAILABLE:
        raise ImportError("mjlab not available.")

    # Import velocity task config dynamically
    # from vinrobotics_mjlab.tasks.velocity.config.vr_m3_1.env_cfgs import vr_m3_1_velocity_env_cfg
    # return vr_m3_1_velocity_env_cfg(play=play)
    raise NotImplementedError("Velocity policy config not yet implemented in auto-generated config.")


# Policy loading for inference (production)
def load_policy_for_inference(device: str = "cpu"):
    """Load trained policy for inference (production use).

    Requires CHECKPOINT_AVAILABLE = True and MJLab installed.
    Returns policy actor network ready for inference.
    """
    if not CHECKPOINT_AVAILABLE:
        raise RuntimeError("No trained policy checkpoint available. Set POLICY_CHECKPOINT_DIR.")

    if not MJLAB_AVAILABLE:
        raise ImportError("mjlab not available. Install mjlab to load policy.")

    import torch
    from mjlab.runners import OnPolicyRunner

    # Load runner config
    runner_cfg = get_runner_cfg()

    # Create runner and load checkpoint
    runner = OnPolicyRunner(
        env=None,  # Will be set by caller
        train_cfg=runner_cfg,
        device=device,
    )
    runner.load(CHECKPOINT_PATH)
    return runner.alg.actor


# RL Runner configuration (for training)
def get_runner_cfg() -> dict:
    """Get RL runner configuration for training."""
    if POLICY_TYPE in ("tracking", "rough"):
        exp_name = "vr_m3_1_rough" if POLICY_TYPE == "rough" else "vr_m3_1_tracking"
        return {{
            "actor": {{
                "hidden_dims": (512, 256, 128),
                "activation": "elu",
                "obs_normalization": True,
                "distribution_cfg": {{
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                }},
            }},
            "critic": {{
                "hidden_dims": (512, 256, 128),
                "activation": "elu",
                "obs_normalization": True,
            }},
            "algorithm": {{
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.005,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 1.0e-3,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
            }},
            "experiment_name": exp_name,
            "logger": "tensorboard",
            "save_interval": 500,
            "num_steps_per_env": 24,
            "max_iterations": 30000,
        }}
    else:
        return {{}}


if __name__ == "__main__":
    print("MJLab policy config loaded")
    print(f"Policy type: {{POLICY_TYPE}}")
    print(f"Robot config: {{ROBOT_CONFIG_IMPORT}}")
    print(f"Task config: {{TASK_CONFIG_IMPORT}}")
    print(f"Action scales: {{len(ACTION_SCALES)}} joints")
    if not MJLAB_AVAILABLE:
        print("WARNING: mjlab not installed - this is a template only")
'''

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Wrote MJLab policy config: {output_path}")
    return output_path


def run_stage_h(policy_type: str = DEFAULT_POLICY_TYPE) -> int:
    """Run Stage H: Generate robot policy configuration.

    Args:
        policy_type: "tracking" or "velocity"

    Returns:
        0 on success
    """
    print("=" * 60)
    print(f"Stage H: Robot Policy Config ({policy_type})")
    print("=" * 60)

    # Ensure output directories exist
    policy_dir = CFG.paths.simulation / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Also create robot/policy output for direct access
    robot_policy_dir = Path(CFG.paths.run_dir) / "robot" / "policy"
    robot_policy_dir.mkdir(parents=True, exist_ok=True)

    # Build configuration
    config = build_policy_config(policy_type)

    # Write JSON config (for programmatic access)
    json_path = write_policy_config(config, policy_dir)
    write_policy_config(config, robot_policy_dir)

    # Write Python config (for MJLab integration)
    py_path = write_mjlab_policy_py(config, policy_dir)
    write_mjlab_policy_py(config, robot_policy_dir)

    print(f"\nStage H completed successfully!")
    print(f"  Policy type: {policy_type}")
    print(f"  JSON config: {json_path}")
    print(f"  Python config: {py_path}")

    return 0


if __name__ == "__main__":
    import sys
    policy_type = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_POLICY_TYPE
    raise SystemExit(run_stage_h(policy_type))