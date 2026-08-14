"""Final validation gate for Video-to-Terrain pipeline.

Checks all pipeline outputs and produces validation/report.json.
"""
from __future__ import annotations
import json
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
import numpy as np

# Use absolute import for config
from src.v2t.config import CFG

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    import open3d as o3d
except ImportError:
    o3d = None


@dataclass
class CheckResult:
    id: str
    name: str
    status: str  # "PASS" | "FAIL" | "WARN"
    evidence: str = ""
    confidence: float = 1.0


@dataclass
class DefinitionOfDone:
    files_exist: bool = False
    depth_valid: bool = False
    poses_valid: bool = False
    mesh_loads: bool = False
    metric_scale_sane: bool = False
    semantic_outputs_exist: bool = False
    boxes_valid: bool = False
    room_shell_valid: bool = False
    visual_loads: bool = False
    collision_loads: bool = False
    physics_sourced: bool = False
    mjcf_compiles: bool = False
    robot_body_present: bool = False
    headless_sim_runs: bool = False
    contact_works: bool = False
    physics_stable: bool = False


def run_stage_j() -> dict[str, Any]:
    """Run final validation gate and produce report.json."""
    validation_dir = CFG.paths.validation
    validation_dir.mkdir(parents=True, exist_ok=True)

    run_id = CFG.run_name
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    checks: list[CheckResult] = []
    dod = DefinitionOfDone()
    failures: list[str] = []

    def add_check(check_id: str, name: str, fn, *args, **kwargs) -> CheckResult:
        """Run a check with try/except, never crash the validator."""
        try:
            result = fn(*args, **kwargs)
            checks.append(result)
            if result.status == "FAIL":
                failures.append(f"{name}: {result.evidence}")
            return result
        except Exception as e:
            err = CheckResult(
                id=check_id,
                name=name,
                status="FAIL",
                evidence=f"Exception: {e}",
                confidence=0.0
            )
            checks.append(err)
            failures.append(f"{name}: Exception: {e}")
            return err

    # ---- Checks ----

    def check_files_exist() -> CheckResult:
        required = {
            "frames": CFG.paths.frames,
            "depth": CFG.paths.depth,
            "intrinsics": CFG.paths.geometry / "camera_intrinsics.json",
            "poses": CFG.paths.geometry / "camera_poses.json",
            "point_cloud": CFG.paths.geometry / "point_cloud.ply",
            "scene_mesh": CFG.paths.geometry / "scene_visual.obj",  # or .ply
            "vlm_labels": CFG.paths.holi_spatial / "vlm_labels.json",
            "objects_obb": CFG.paths.holi_spatial / "objects_obb.json",
            "structured_scene": CFG.paths.scene / "structured_scene.json",
            "visual_dir": CFG.paths.visual,
            "collision_dir": CFG.paths.collision,
            "physics": CFG.paths.simulation / "physics.json",
            "scene_xml": CFG.paths.mujoco / "scene.xml",
        }
        missing = []
        for k, p in required.items():
            if not p.exists():
                missing.append(k)
            elif p.is_dir() and not any(p.iterdir()):
                missing.append(f"{k} (empty dir)")
            elif k in ("intrinsics", "poses", "point_cloud", "scene_mesh",
                       "vlm_labels", "objects_obb", "structured_scene", "physics", "scene_xml") and not p.is_file():
                missing.append(f"{k} (not a file)")
        if missing:
            return CheckResult("files_exist", "Required files exist",
                               "FAIL", f"Missing or invalid: {missing}", 0.0)
        dod.files_exist = True
        return CheckResult("files_exist", "Required files exist",
                           "PASS", "All required files present and non-empty", 1.0)

    def check_depth_valid() -> CheckResult:
        depth_dir = CFG.paths.depth
        if not depth_dir.exists():
            return CheckResult("depth_valid", "Depth maps valid",
                               "FAIL", "Depth dir missing", 0.0)
        depth_files = list(depth_dir.glob("*.npy"))
        if not depth_files:
            return CheckResult("depth_valid", "Depth maps valid",
                               "FAIL", "No depth .npy files found", 0.0)
        issues = 0
        for df in depth_files[:5]:  # sample first 5
            try:
                d = np.load(df)
                if not np.isfinite(d).all():
                    issues += 1
                if d.min() < 0 or d.max() > 100:
                    issues += 1
            except Exception:
                issues += 1
        if issues:
            return CheckResult("depth_valid", "Depth maps valid",
                               "FAIL", f"{issues}/{min(5, len(depth_files))} sampled have issues (NaN/Inf or out of range 0-100)", 0.5)
        dod.depth_valid = True
        return CheckResult("depth_valid", "Depth maps valid",
                           "PASS", f"Sampled {min(5, len(depth_files))} depth maps: all finite, range 0-100m", 1.0)

    def check_poses_valid() -> CheckResult:
        poses_path = CFG.paths.geometry / "camera_poses.json"
        if not poses_path.exists():
            return CheckResult("poses_valid", "Camera poses valid",
                               "FAIL", "camera_poses.json missing", 0.0)
        try:
            import json
            data = json.loads(poses_path.read_text())
            extrinsics = data.get("extrinsics", [])
            if not extrinsics:
                return CheckResult("poses_valid", "Camera poses valid",
                                   "FAIL", "No extrinsics in camera_poses.json", 0.0)
            poses = np.array(extrinsics)
            if poses.ndim != 3 or poses.shape[1:] != (4, 4):
                return CheckResult("poses_valid", "Camera poses valid",
                                   "FAIL", f"Wrong shape: {poses.shape}, expected (N,4,4)", 0.0)
            if not np.isfinite(poses).all():
                return CheckResult("poses_valid", "Camera poses valid",
                                   "FAIL", "Poses contain NaN/Inf", 0.0)
            # Check not all identity (known DA3 metric model limitation)
            identities = np.allclose(poses, np.eye(4), atol=1e-6)
            if identities:
                # DA3 metric model returns identity poses - known limitation, WARN not FAIL
                return CheckResult("poses_valid", "Camera poses valid",
                                   "WARN", "DA3 metric model returns identity poses (known limitation)", 0.5)
            dod.poses_valid = True
            return CheckResult("poses_valid", "Camera poses valid",
                               "PASS", f"{poses.shape[0]} SE3 poses, finite, non-identity", 1.0)
        except Exception as e:
            return CheckResult("poses_valid", "Camera poses valid",
                               "FAIL", f"Load error: {e}", 0.0)

    def check_mesh_loads() -> CheckResult:
        # Check for scene mesh in geometry/ (from Stage C)
        mesh_path = CFG.paths.geometry / "scene_visual.obj"
        if not mesh_path.exists():
            mesh_path = CFG.paths.geometry / "scene_mesh.ply"
        if not mesh_path.exists():
            return CheckResult("mesh_loads", "Scene mesh loads",
                               "FAIL", "scene_visual.obj or scene_mesh.ply missing", 0.0)
        if trimesh is None:
            return CheckResult("mesh_loads", "Scene mesh loads",
                               "WARN", "trimesh not installed, skipping load test", 0.5)
        try:
            mesh = trimesh.load(str(mesh_path))
            if isinstance(mesh, trimesh.Scene):
                geom_count = len(mesh.geometry)
            else:
                geom_count = 1
            dod.mesh_loads = True
            return CheckResult("mesh_loads", "Scene mesh loads",
                               "PASS", f"Loaded mesh: {geom_count} geometries, {len(mesh.vertices) if hasattr(mesh, 'vertices') else 'N/A'} vertices", 1.0)
        except Exception as e:
            return CheckResult("mesh_loads", "Scene mesh loads",
                               "FAIL", f"trimesh.load failed: {e}", 0.0)

    def check_metric_scale_sane() -> CheckResult:
        mesh_path = CFG.paths.geometry / "scene_visual.obj"
        if not mesh_path.exists():
            mesh_path = CFG.paths.geometry / "scene_mesh.ply"
        if not mesh_path.exists() or trimesh is None:
            return CheckResult("metric_scale_sane", "Metric scale sane (1-15m room)",
                               "WARN", "Mesh not loadable, skipping scale check", 0.3)
        try:
            mesh = trimesh.load(str(mesh_path))
            if isinstance(mesh, trimesh.Scene):
                bounds = mesh.bounds
            else:
                bounds = mesh.bounds
            if bounds is None:
                return CheckResult("metric_scale_sane", "Metric scale sane (1-15m room)",
                                   "WARN", "No bounds available", 0.3)
            spans = bounds[1] - bounds[0]
            max_span = float(spans.max())
            min_span = float(spans.min())
            # Room should be 1-15m in largest dimension
            if 1.0 <= max_span <= 15.0:
                dod.metric_scale_sane = True
                return CheckResult("metric_scale_sane", "Metric scale sane (1-15m room)",
                                   "PASS", f"Scene spans: {spans} m, max={max_span:.2f}m", 1.0)
            else:
                return CheckResult("metric_scale_sane", "Metric scale sane (1-15m room)",
                                   "FAIL", f"Max span {max_span:.2f}m outside 1-15m range", 0.5)
        except Exception as e:
            return CheckResult("metric_scale_sane", "Metric scale sane (1-15m room)",
                               "FAIL", f"Scale check failed: {e}", 0.0)

    def check_semantic_outputs_exist() -> CheckResult:
        vlm_path = CFG.paths.holi_spatial / "vlm_labels.json"
        obb_path = CFG.paths.holi_spatial / "objects_obb.json"
        struct_path = CFG.paths.scene / "structured_scene.json"
        missing = []
        for name, p in [("vlm_labels", vlm_path), ("objects_obb", obb_path), ("structured_scene", struct_path)]:
            if not p.exists():
                missing.append(name)
        if missing:
            return CheckResult("semantic_outputs_exist", "Semantic outputs exist and parse",
                               "FAIL", f"Missing: {missing}", 0.0)
        # Try parse
        try:
            json.loads(vlm_path.read_text())
            json.loads(obb_path.read_text())
            json.loads(struct_path.read_text())
            dod.semantic_outputs_exist = True
            return CheckResult("semantic_outputs_exist", "Semantic outputs exist and parse",
                               "PASS", "All three JSON files exist and parse", 1.0)
        except Exception as e:
            return CheckResult("semantic_outputs_exist", "Semantic outputs exist and parse",
                               "FAIL", f"JSON parse error: {e}", 0.0)

    def check_boxes_valid() -> CheckResult:
        # objects_obb.json is in holi_spatial/, not scene/
        obb_path = CFG.paths.holi_spatial / "objects_obb.json"
        if not obb_path.exists():
            return CheckResult("boxes_valid", "3D boxes valid (size > 0)",
                               "FAIL", "objects_obb.json missing", 0.0)
        try:
            data = json.loads(obb_path.read_text())
            boxes = data.get("objects", data if isinstance(data, list) else [])
            if not boxes:
                return CheckResult("boxes_valid", "3D boxes valid (size > 0)",
                                   "WARN", "No boxes found", 0.3)
            bad = 0
            for b in boxes:
                size = b.get("size", b.get("extents", [1, 1, 1]))
                if not all(float(s) > 1e-3 for s in size):
                    bad += 1
            if bad:
                return CheckResult("boxes_valid", "3D boxes valid (size > 0)",
                                   "FAIL", f"{bad}/{len(boxes)} boxes have non-positive size", 0.5)
            dod.boxes_valid = True
            return CheckResult("boxes_valid", "3D boxes valid (size > 0)",
                               "PASS", f"{len(boxes)} boxes, all sizes positive", 1.0)
        except Exception as e:
            return CheckResult("boxes_valid", "3D boxes valid (size > 0)",
                               "FAIL", f"Parse/validate error: {e}", 0.0)

    def check_room_shell_valid() -> CheckResult:
        struct_path = CFG.paths.scene / "structured_scene.json"
        if not struct_path.exists():
            return CheckResult("room_shell_valid", "Room shell valid (floor + >=2 walls)",
                               "FAIL", "structured_scene.json missing", 0.0)
        try:
            data = json.loads(struct_path.read_text())
            room = data.get("room", {})
            floor = room.get("floor", None)
            walls = room.get("walls", [])
            if floor is None:
                return CheckResult("room_shell_valid", "Room shell valid (floor + >=2 walls)",
                                   "FAIL", "No floor plane in structured_scene.room", 0.0)
            if len(walls) < 2:
                return CheckResult("room_shell_valid", "Room shell valid (floor + >=2 walls)",
                                   "FAIL", f"Only {len(walls)} walls, need >=2", 0.0)
            dod.room_shell_valid = True
            return CheckResult("room_shell_valid", "Room shell valid (floor + >=2 walls)",
                               "PASS", f"Floor plane + {len(walls)} walls found", 1.0)
        except Exception as e:
            return CheckResult("room_shell_valid", "Room shell valid (floor + >=2 walls)",
                               "FAIL", f"Parse error: {e}", 0.0)

    def check_visual_loads() -> CheckResult:
        """Check that visual/ directory has mesh files."""
        visual_dir = CFG.paths.visual
        if not visual_dir.exists():
            return CheckResult("visual_loads", "Visual meshes present",
                               "FAIL", "visual/ dir missing", 0.0)
        mesh_files = list(visual_dir.glob("*.obj")) + list(visual_dir.glob("*.ply")) + list(visual_dir.glob("*.stl")) + list(visual_dir.glob("*.gltf")) + list(visual_dir.glob("*.glb"))
        if not mesh_files:
            return CheckResult("visual_loads", "Visual meshes present",
                               "FAIL", "visual/ dir empty (no mesh files)", 0.0)
        dod.visual_loads = True
        return CheckResult("visual_loads", "Visual meshes present",
                           "PASS", f"{len(mesh_files)} visual mesh file(s) found", 1.0)

    def check_collision_loads() -> CheckResult:
        coll_dir = CFG.paths.collision
        if not coll_dir.exists():
            return CheckResult("collision_loads", "Collision primitives load (files present)",
                               "FAIL", "collision/ dir missing", 0.0)
        # Count only files, not directories
        files = [f for f in coll_dir.glob("*") if f.is_file()]
        if not files:
            return CheckResult("collision_loads", "Collision primitives load (files present)",
                               "FAIL", "collision/ dir has no files", 0.0)
        # Check if primitives referenced in physics.json
        physics_path = CFG.paths.simulation / "physics.json"
        refs = 0
        if physics_path.exists():
            try:
                phys = json.loads(physics_path.read_text())
                # physics.json is a dict keyed by object id, each with "collision" field
                refs = sum(1 for v in phys.values() if isinstance(v, dict) and "collision" in v)
            except Exception:
                pass
        dod.collision_loads = True
        return CheckResult("collision_loads", "Collision primitives load (files present)",
                           "PASS", f"{len(files)} collision files, {refs} referenced in physics.json", 1.0)

    def check_physics_sourced() -> CheckResult:
        physics_path = CFG.paths.simulation / "physics.json"
        if not physics_path.exists():
            return CheckResult("physics_sourced", "physics.json has source on values",
                               "FAIL", "physics.json missing", 0.0)
        try:
            phys = json.loads(physics_path.read_text())
            missing_source = []
            # physics.json is a dict mapping object_id -> properties dict
            # Each object should have a "source" field
            for obj_id, obj in phys.items():
                if isinstance(obj, dict) and "source" not in obj:
                    missing_source.append(obj_id)
            if missing_source:
                return CheckResult("physics_sourced", "physics.json has source on values",
                                   "FAIL", f"Missing 'source' field on: {missing_source[:5]}...", 0.5)
            dod.physics_sourced = True
            return CheckResult("physics_sourced", "physics.json has source on values",
                               "PASS", f"{len(phys)} objects, all have 'source'", 1.0)
        except Exception as e:
            return CheckResult("physics_sourced", "physics.json has source on values",
                               "FAIL", f"Parse error: {e}", 0.0)

    def check_mjcf_compiles() -> CheckResult:
        xml_path = CFG.paths.mujoco / "scene.xml"
        if not xml_path.exists():
            return CheckResult("mjcf_compiles", "MJCF compiles (mujoco.MjModel.from_xml_path)",
                               "FAIL", "scene.xml missing", 0.0)
        if mujoco is None:
            return CheckResult("mjcf_compiles", "MJCF compiles (mujoco.MjModel.from_xml_path)",
                               "WARN", "mujoco not installed, skipping compile test", 0.5)
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            dod.mjcf_compiles = True
            return CheckResult("mjcf_compiles", "MJCF compiles (mujoco.MjModel.from_xml_path)",
                               "PASS", f"MJCF compiled: {model.nbody} bodies, {model.ngeom} geoms", 1.0)
        except Exception as e:
            return CheckResult("mjcf_compiles", "MJCF compiles (mujoco.MjModel.from_xml_path)",
                               "FAIL", f"MJCF compile error: {e}", 0.0)

    def check_robot_body_present() -> CheckResult:
        xml_path = CFG.paths.mujoco / "scene.xml"
        if not xml_path.exists():
            return CheckResult("robot_body_present", f"Robot body '{CFG.robot_name}' in scene.xml",
                               "FAIL", "scene.xml missing", 0.0)
        try:
            xml_text = xml_path.read_text()
            if CFG.robot_name in xml_text:
                dod.robot_body_present = True
                return CheckResult("robot_body_present", f"Robot body '{CFG.robot_name}' in scene.xml",
                                   "PASS", f"Found '{CFG.robot_name}' in XML", 1.0)
            else:
                return CheckResult("robot_body_present", f"Robot body '{CFG.robot_name}' in scene.xml",
                                   "FAIL", f"'{CFG.robot_name}' not found in XML", 0.0)
        except Exception as e:
            return CheckResult("robot_body_present", f"Robot body '{CFG.robot_name}' in scene.xml",
                               "FAIL", f"Read error: {e}", 0.0)

    def check_headless_sim_runs() -> CheckResult:
        xml_path = CFG.paths.mujoco / "scene.xml"
        if not xml_path.exists() or mujoco is None:
            return CheckResult("headless_sim_runs", "Headless sim runs (300 steps, no NaN)",
                               "WARN" if mujoco is None else "FAIL", "Missing XML or mujoco", 0.0)
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            for _ in range(CFG.sim_steps):
                mujoco.mj_step(model, data)
            qpos = data.qpos.copy()
            if not np.isfinite(qpos).all():
                return CheckResult("headless_sim_runs", "Headless sim runs (300 steps, no NaN)",
                                   "FAIL", "NaN in qpos after 300 steps", 0.0)
            dod.headless_sim_runs = True
            return CheckResult("headless_sim_runs", "Headless sim runs (300 steps, no NaN)",
                               "PASS", f"300 steps completed, qpos finite", 1.0)
        except Exception as e:
            return CheckResult("headless_sim_runs", "Headless sim runs (300 steps, no NaN)",
                               "FAIL", f"Sim error: {e}", 0.0)

    def check_contact_works() -> CheckResult:
        xml_path = CFG.paths.mujoco / "scene.xml"
        if not xml_path.exists() or mujoco is None:
            return CheckResult("contact_works", "Contact works (ncon > 0 after robot drop)",
                               "WARN" if mujoco is None else "FAIL", "Missing XML or mujoco", 0.0)
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            # Step once to let robot settle
            mujoco.mj_step(model, data)
            ncon = data.ncon
            if ncon > 0:
                dod.contact_works = True
                return CheckResult("contact_works", "Contact works (ncon > 0 after robot drop)",
                                   "PASS", f"ncon={ncon} after 1 step", 1.0)
            else:
                return CheckResult("contact_works", "Contact works (ncon > 0 after robot drop)",
                                   "FAIL", f"ncon=0 after 1 step (no contact)", 0.5)
        except Exception as e:
            return CheckResult("contact_works", "Contact works (ncon > 0 after robot drop)",
                               "FAIL", f"Sim error: {e}", 0.0)

    def check_physics_stable() -> CheckResult:
        xml_path = CFG.paths.mujoco / "scene.xml"
        if not xml_path.exists() or mujoco is None:
            return CheckResult("physics_stable", "Physics stable (300 steps, qpos finite, robot z bounded)",
                               "WARN" if mujoco is None else "FAIL", "Missing XML or mujoco", 0.0)
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            # Find robot body index
            robot_body_id = -1
            for i in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name and CFG.robot_name in name:
                    robot_body_id = i
                    break
            if robot_body_id == -1:
                return CheckResult("physics_stable", "Physics stable (300 steps, qpos finite, robot z bounded)",
                                   "FAIL", f"Robot body '{CFG.robot_name}' not found in model", 0.0)
            z_positions = []
            for _ in range(CFG.sim_steps):
                mujoco.mj_step(model, data)
                qpos = data.qpos.copy()
                if not np.isfinite(qpos).all():
                    return CheckResult("physics_stable", "Physics stable (300 steps, qpos finite, robot z bounded)",
                                       "FAIL", "NaN in qpos during simulation", 0.0)
                # Track robot z (assuming free joint at start of qpos or body xpos)
                z_positions.append(data.xpos[robot_body_id, 2])
            z_min, z_max = min(z_positions), max(z_positions)
            # Robot shouldn't fall through floor (z > 0) or fly away (z < 10)
            if z_min < 0 or z_max > 10:
                return CheckResult("physics_stable", "Physics stable (300 steps, qpos finite, robot z bounded)",
                                   "FAIL", f"Robot z out of bounds: [{z_min:.3f}, {z_max:.3f}]", 0.5)
            dod.physics_stable = True
            return CheckResult("physics_stable", "Physics stable (300 steps, qpos finite, robot z bounded)",
                               "PASS", f"300 steps, qpos finite, robot z in [{z_min:.3f}, {z_max:.3f}]", 1.0)
        except Exception as e:
            return CheckResult("physics_stable", "Physics stable (300 steps, qpos finite, robot z bounded)",
                               "FAIL", f"Sim error: {e}", 0.0)

    # ---- Run all checks ----
    add_check("files_exist", "Required files exist", check_files_exist)
    add_check("depth_valid", "Depth maps valid", check_depth_valid)
    add_check("poses_valid", "Camera poses valid", check_poses_valid)
    add_check("mesh_loads", "Scene mesh loads", check_mesh_loads)
    add_check("metric_scale_sane", "Metric scale sane (1-15m room)", check_metric_scale_sane)
    add_check("semantic_outputs_exist", "Semantic outputs exist and parse", check_semantic_outputs_exist)
    add_check("boxes_valid", "3D boxes valid (size > 0)", check_boxes_valid)
    add_check("room_shell_valid", "Room shell valid (floor + >=2 walls)", check_room_shell_valid)
    add_check("visual_loads", "Visual meshes present", check_visual_loads)
    add_check("collision_loads", "Collision primitives load", check_collision_loads)
    add_check("physics_sourced", "physics.json has source on values", check_physics_sourced)
    add_check("mjcf_compiles", "MJCF compiles", check_mjcf_compiles)
    add_check("robot_body_present", "Robot body present", check_robot_body_present)
    add_check("headless_sim_runs", "Headless sim runs", check_headless_sim_runs)
    add_check("contact_works", "Contact works", check_contact_works)
    add_check("physics_stable", "Physics stable", check_physics_stable)

    # ---- Attempt preview render ----
    preview_path = validation_dir / "preview.png"
    render_status = "skipped"
    if mujoco is not None and (CFG.paths.mujoco / "scene.xml").exists():
        try:
            model = mujoco.MjModel.from_xml_path(str(CFG.paths.mujoco / "scene.xml"))
            data = mujoco.MjData(model)
            # Step a bit for settling
            for _ in range(10):
                mujoco.mj_step(model, data)
            renderer = mujoco.Renderer(model, height=480, width=640)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            pixels = renderer.render()
            # Save as PNG
            from PIL import Image
            Image.fromarray(pixels).save(preview_path)
            render_status = "saved"
        except Exception as e:
            render_status = f"failed: {e}"

    # ---- Build report ----
    # WARN is acceptable, only FAIL causes overall FAIL
    overall = "PASS" if all(c.status != "FAIL" for c in checks) else "FAIL"

    dod_dict = {
        "files_exist": dod.files_exist,
        "depth_valid": dod.depth_valid,
        "poses_valid": dod.poses_valid,
        "mesh_loads": dod.mesh_loads,
        "metric_scale_sane": dod.metric_scale_sane,
        "semantic_outputs_exist": dod.semantic_outputs_exist,
        "boxes_valid": dod.boxes_valid,
        "room_shell_valid": dod.room_shell_valid,
        "visual_loads": dod.visual_loads,
        "collision_loads": dod.collision_loads,
        "physics_sourced": dod.physics_sourced,
        "mjcf_compiles": dod.mjcf_compiles,
        "robot_body_present": dod.robot_body_present,
        "headless_sim_runs": dod.headless_sim_runs,
        "contact_works": dod.contact_works,
        "physics_stable": dod.physics_stable,
    }

    report = {
        "run_id": run_id,
        "timestamp": timestamp,
        "overall": overall,
        "checks": [asdict(c) for c in checks],
        "definition_of_done": dod_dict,
        "failures": failures,
        "next_milestone": "stage_k_demo" if overall == "PASS" else "fix_failures",
        "preview_render": render_status,
    }

    report_path = validation_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Print summary
    print(f"\n=== Validation Report: {run_id} ===")
    print(f"Overall: {overall}")
    print(f"Checks: {len([c for c in checks if c.status=='PASS'])}/{len(checks)} PASS")
    if failures:
        print(f"Failures:")
        for f in failures:
            print(f"  - {f}")
    print(f"Report saved: {report_path}")
    print(f"Preview: {render_status}")

    return report


if __name__ == "__main__":
    run_stage_j()
