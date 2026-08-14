"""Stage H: Robot Policy Configuration.

Reads structured_scene.json, physics.json, and MJCF outputs, produces:
- CFG.paths.simulation/policy_config.json (robot policy/skill configuration)
- CFG.paths.simulation/skills/ (individual skill configurations)
- CFG.paths.simulation/tasks/ (task definitions)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from src.v2t.config import CFG


def load_structured_scene() -> dict[str, Any] | None:
    """Load structured_scene.json from scene directory."""
    scene_path = CFG.paths.scene / "structured_scene.json"
    if not scene_path.exists():
        print(f"WARNING: {scene_path} not found. Producing stub outputs.")
        return None
    with open(scene_path) as f:
        return json.load(f)


def load_physics_config() -> dict[str, Any]:
    """Load physics config from simulation/physics.json."""
    physics_path = CFG.paths.simulation / "physics.json"
    if physics_path.exists():
        with open(physics_path) as f:
            return json.load(f)
    return CFG.physics_profiles


def load_mujoco_scene() -> dict[str, Any] | None:
    """Load MJCF scene info if available."""
    mujoco_path = CFG.paths.mujoco / "scene.xml"
    if not mujoco_path.exists():
        return None
    # Could parse XML here, but for MVP return basic info
    return {
        "path": str(mujoco_path),
        "robot": CFG.robot_name,
        "spawn_height": CFG.robot_spawn_height,
    }


def build_navigation_skills(scene_data: dict | None) -> list[dict]:
    """Build navigation skill configurations based on scene structure."""
    skills = []

    # Basic locomotion skill
    skills.append({
        "id": "nav_walk",
        "name": "Walking Locomotion",
        "type": "locomotion",
        "description": "Basic forward/backward/strafe walking on flat terrain",
        "policy_path": "policies/nav_walk.onnx",
        "action_space": {
            "type": "continuous",
            "dim": 12,  # VR-M3-1 has 12 DoF legs
            "bounds": [-1.0, 1.0],
        },
        "observation_space": {
            "type": "continuous",
            "dim": 48,  # base + joint states
            "bounds": [-10.0, 10.0],
        },
        "reward_config": {
            "forward_velocity": 1.0,
            "lateral_velocity": -0.1,
            "yaw_rate": -0.05,
            "energy": -0.01,
            "height": 0.5,
        },
        "termination": {
            "height_below": 0.3,
            "height_above": 2.0,
            "roll_exceed": 0.5,
            "pitch_exceed": 0.5,
        },
        "applicable_terrains": ["flat", "rough", "sloped"],
        "max_episode_length": 1000,
    })

    # Stair climbing skill
    skills.append({
        "id": "nav_stairs",
        "name": "Stair Climbing",
        "type": "locomotion",
        "description": "Ascending/descending stairs",
        "policy_path": "policies/nav_stairs.onnx",
        "action_space": {
            "type": "continuous",
            "dim": 12,
            "bounds": [-1.0, 1.0],
        },
        "observation_space": {
            "type": "continuous",
            "dim": 52,  # base + joint states + terrain scan
            "bounds": [-10.0, 10.0],
        },
        "reward_config": {
            "forward_velocity": 1.0,
            "height_progress": 2.0,
            "foot_clearance": 0.5,
            "energy": -0.01,
        },
        "termination": {
            "height_below": 0.3,
            "stuck_steps": 50,
        },
        "applicable_terrains": ["stairs"],
        "max_episode_length": 500,
    })

    # Navigation with obstacle avoidance
    skills.append({
        "id": "nav_avoid",
        "name": "Obstacle Avoidance Navigation",
        "type": "navigation",
        "description": "Navigate to goal while avoiding dynamic/static obstacles",
        "policy_path": "policies/nav_avoid.onnx",
        "action_space": {
            "type": "continuous",
            "dim": 12,
            "bounds": [-1.0, 1.0],
        },
        "observation_space": {
            "type": "continuous",
            "dim": 80,  # base + joint + lidar/scan
            "bounds": [-10.0, 10.0],
        },
        "reward_config": {
            "goal_progress": 2.0,
            "collision": -10.0,
            "energy": -0.01,
            "smoothness": -0.1,
        },
        "termination": {
            "collision": True,
            "goal_reached_radius": 0.5,
            "timeout": 300,
        },
        "applicable_terrains": ["flat", "cluttered", "indoor"],
        "max_episode_length": 1500,
    })

    return skills


def build_manipulation_skills(scene_data: dict | None) -> list[dict]:
    """Build manipulation skill configurations based on scene objects."""
    skills = []

    if not scene_data:
        return skills

    objects = scene_data.get("objects", [])

    # Detect manipulable objects
    manipulable = [obj for obj in objects if obj.get("category") in
                   ("table", "chair", "cabinet", "shelf", "door", "drawer")]

    if not manipulable:
        return skills

    # Pick and place skill
    skills.append({
        "id": "manip_pick_place",
        "name": "Pick and Place",
        "type": "manipulation",
        "description": "Pick up object and place at target location",
        "policy_path": "policies/manip_pick_place.onnx",
        "action_space": {
            "type": "continuous",
            "dim": 7,  # 6D arm + gripper
            "bounds": [-1.0, 1.0],
        },
        "observation_space": {
            "type": "continuous",
            "dim": 35,  # arm state + object pose + target
            "bounds": [-10.0, 10.0],
        },
        "reward_config": {
            "reach_reward": 1.0,
            "grasp_reward": 2.0,
            "place_reward": 3.0,
            "energy": -0.01,
        },
        "termination": {
            "success": True,
            "timeout": 200,
            "drop_object": True,
        },
        "target_objects": [obj.get("category") for obj in manipulable],
        "max_episode_length": 300,
    })

    # Door opening skill
    door_objs = [obj for obj in objects if obj.get("category") == "door"]
    if door_objs:
        skills.append({
            "id": "manip_open_door",
            "name": "Door Opening",
            "type": "manipulation",
            "description": "Open hinged or sliding door",
            "policy_path": "policies/manip_open_door.onnx",
            "action_space": {
                "type": "continuous",
                "dim": 7,
                "bounds": [-1.0, 1.0],
            },
            "observation_space": {
                "type": "continuous",
                "dim": 40,
                "bounds": [-10.0, 10.0],
            },
            "reward_config": {
                "handle_reach": 1.0,
                "open_progress": 2.0,
                "energy": -0.01,
            },
            "termination": {
                "fully_open": True,
                "timeout": 150,
            },
            "target_objects": ["door"],
            "max_episode_length": 200,
        })

    return skills


def build_task_definitions(scene_data: dict | None, skills: list[dict]) -> list[dict]:
    """Build high-level task definitions combining skills."""
    tasks = []

    if not scene_data:
        return tasks

    # Inspection task
    tasks.append({
        "id": "task_inspect",
        "name": "Environment Inspection",
        "description": "Navigate through environment and inspect key areas",
        "type": "inspection",
        "skills": ["nav_walk", "nav_avoid"],
        "objectives": [
            {"type": "visit_regions", "regions": ["nav_0"], "order": "any"},
            {"type": "avoid_collisions"},
        ],
        "success_criteria": {
            "coverage_percent": 80,
            "max_collisions": 0,
        },
        "timeout": 600,
    })

    # Object interaction task
    objects = scene_data.get("objects", [])
    manipulable = [obj for obj in objects if obj.get("category") in
                   ("table", "chair", "cabinet", "shelf", "door")]

    if manipulable:
        tasks.append({
            "id": "task_rearrange",
            "name": "Object Rearrangement",
            "description": "Pick and place objects to target locations",
            "type": "manipulation",
            "skills": ["nav_walk", "nav_avoid", "manip_pick_place"],
            "objectives": [
                {"type": "rearrange_objects", "objects": [obj.get("id") for obj in manipulable]},
            ],
            "success_criteria": {
                "objects_placed": len(manipulable),
                "max_collisions": 2,
            },
            "timeout": 1200,
        })

    # Point-to-point navigation
    tasks.append({
        "id": "task_goto",
        "name": "Go To Position",
        "description": "Navigate from current position to target position",
        "type": "navigation",
        "skills": ["nav_walk", "nav_avoid"],
        "objectives": [
            {"type": "reach_position", "target": [0.0, 0.0, 0.0], "tolerance": 0.5},
        ],
        "success_criteria": {
            "position_error": 0.5,
        },
        "timeout": 300,
    })

    return tasks


def build_policy_config(scene_data: dict | None, physics: dict, mujoco_scene: dict | None) -> dict:
    """Build complete policy configuration."""
    navigation_skills = build_navigation_skills(scene_data)
    manipulation_skills = build_manipulation_skills(scene_data)
    all_skills = navigation_skills + manipulation_skills

    tasks = build_task_definitions(scene_data, all_skills)

    config = {
        "robot": {
            "name": CFG.robot_name,
            "spawn_height": CFG.robot_spawn_height,
            "mjcf_path": mujoco_scene.get("path") if mujoco_scene else None,
        },
        "simulation": {
            "timestep": 0.005,
            "gravity": [0, 0, -9.81],
            "physics_profiles": physics.get("profiles", {}),
        },
        "skills": {skill["id"]: skill for skill in all_skills},
        "skill_order": [skill["id"] for skill in all_skills],
        "tasks": {task["id"]: task for task in tasks},
        "task_order": [task["id"] for task in tasks],
        "defaults": {
            "default_locomotion_skill": "nav_walk",
            "default_navigation_skill": "nav_avoid",
            "default_manipulation_skill": "manip_pick_place" if manipulation_skills else None,
        },
        "provenance": {
            "structured_scene": "scene/structured_scene.json",
            "physics": "simulation/physics.json",
            "mujoco": "mujoco/scene.xml",
        },
    }

    return config


def run_stage_h() -> dict:
    """Run Stage H: Robot Policy Configuration."""
    print("=" * 60)
    print("Stage H: Robot Policy Configuration")
    print("=" * 60)

    CFG.paths.simulation.mkdir(parents=True, exist_ok=True)
    (CFG.paths.simulation / "skills").mkdir(parents=True, exist_ok=True)
    (CFG.paths.simulation / "tasks").mkdir(parents=True, exist_ok=True)

    # Load inputs
    scene_data = load_structured_scene()
    physics = load_physics_config()
    mujoco_scene = load_mujoco_scene()

    # Build policy config
    policy_config = build_policy_config(scene_data, physics, mujoco_scene)

    # Write policy_config.json
    config_path = CFG.paths.simulation / "policy_config.json"
    with open(config_path, "w") as f:
        json.dump(policy_config, f, indent=2)
    print(f"Wrote: {config_path}")

    # Write individual skill files
    skills_dir = CFG.paths.simulation / "skills"
    for skill_id, skill in policy_config["skills"].items():
        skill_path = skills_dir / f"{skill_id}.json"
        with open(skill_path, "w") as f:
            json.dump(skill, f, indent=2)
        print(f"Wrote: {skill_path}")

    # Write individual task files
    tasks_dir = CFG.paths.simulation / "tasks"
    for task_id, task in policy_config["tasks"].items():
        task_path = tasks_dir / f"{task_id}.json"
        with open(task_path, "w") as f:
            json.dump(task, f, indent=2)
        print(f"Wrote: {task_path}")

    print(f"Stage H complete. Skills: {len(policy_config['skills'])}, Tasks: {len(policy_config['tasks'])}")
    return {
        "policy_config": str(config_path),
        "skills_dir": str(skills_dir),
        "tasks_dir": str(tasks_dir),
        "num_skills": len(policy_config["skills"]),
        "num_tasks": len(policy_config["tasks"]),
    }


if __name__ == "__main__":
    run_stage_h()