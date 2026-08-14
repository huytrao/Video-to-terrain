"""Stage E/F: StructuredScene + Room Shell Reconstruction."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from src.v2t.config import CFG


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    # From rotation matrix to quaternion (SciPy convention: scalar-first)
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
        z = (R[0, 2] + R[2, 0]) / s
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
    return np.array([w, x, y, z], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=np.float32)


def rotation_to_align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix that aligns vector a to vector b.
    a and b should be unit vectors."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            return np.eye(3, dtype=np.float32)  # a == b
        else:
            # 180 degree rotation - pick any perpendicular axis
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float32)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + vx + vx @ vx * ((1 - c) / (s * s))
    return R


def gravity_align(points: np.ndarray, floor_plane: np.ndarray, floor_inliers: np.ndarray,
                  objects: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, dict]:
    """
    Align scene to gravity (Z-up) by rotating floor plane to horizontal.

    Returns: (aligned_points, aligned_floor_plane, aligned_floor_inliers, aligned_objects, transform_info)
    """
    floor_normal = floor_plane[:3]
    floor_normal = floor_normal / (np.linalg.norm(floor_normal) + 1e-8)

    # Target: Z-up
    target_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Compute rotation to align floor normal to Z-up
    R = rotation_to_align_vectors(floor_normal, target_normal)

    # Compute floor centroid for translation
    floor_pts = points[floor_inliers]
    centroid = floor_pts.mean(axis=0)

    # Transform: p' = R @ (p - centroid)
    # This puts floor at z=0 with normal [0,0,1]
    aligned_points = (R @ (points - centroid).T).T

    # New floor plane: after transform, floor points have z=0, so plane is [0,0,1,0]
    aligned_floor_plane = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # Floor inliers remain the same indices (we just transformed the points)
    aligned_floor_inliers = floor_inliers

    # Transform objects
    q_R = quat_from_matrix(R)
    aligned_objects = []
    for obj in objects:
        obj_copy = obj.copy()
        center = np.array(obj.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
        rotation = obj.get("orientation", [1.0, 0.0, 0.0, 0.0])

        # Transform center: p' = R @ (p - centroid)
        center_aligned = R @ (center - centroid)
        obj_copy["center"] = center_aligned.tolist()

        # Transform orientation: q' = q_R * q_obj
        q_obj = np.array(rotation, dtype=np.float32)
        q_obj = q_obj / (np.linalg.norm(q_obj) + 1e-8)
        q_aligned = quat_multiply(q_R, q_obj)
        obj_copy["orientation"] = q_aligned.tolist()

        aligned_objects.append(obj_copy)

    transform_info = {
        "R": R.tolist(),
        "centroid": centroid.tolist(),
        "original_floor_normal": floor_normal.tolist(),
        "original_floor_offset": float(floor_plane[3]),
    }

    return aligned_points, aligned_floor_plane, aligned_floor_inliers, aligned_objects, transform_info


def load_objects_obb(obb_path: Path) -> list:
    """Load objects_obb.json."""
    if not obb_path.exists():
        print(f"OBB file not found: {obb_path}, using stub objects")
        return get_stub_objects()

    with open(obb_path, 'r') as f:
        data = json.load(f)

    return data.get("objects", [])


def get_stub_objects() -> list:
    """Get stub objects for when OBB missing."""
    return [
        {"instance_id": 0, "category": "floor", "center": [0.0, 0.0, 0.0], "size": [4.0, 4.0, 0.05],
         "orientation": [1.0, 0.0, 0.0, 0.0], "bbox_type": "OBB", "confidence": 0.9, "stub": True},
        {"instance_id": 1, "category": "chair", "center": [1.0, 0.5, 0.25], "size": [0.4, 0.4, 0.5],
         "orientation": [1.0, 0.0, 0.0, 0.0], "bbox_type": "OBB", "confidence": 0.8, "stub": True},
        {"instance_id": 2, "category": "table", "center": [-1.0, -0.5, 0.4], "size": [1.2, 0.6, 0.8],
         "orientation": [1.0, 0.0, 0.0, 0.0], "bbox_type": "OBB", "confidence": 0.8, "stub": True},
    ]


def load_point_cloud(pc_path: Path) -> np.ndarray | None:
    """Load point cloud from PLY file using trimesh or open3d."""
    if not pc_path.exists():
        return None

    # Try trimesh first
    try:
        import trimesh
        mesh = trimesh.load(str(pc_path), process=False)
        if hasattr(mesh, 'vertices'):
            return np.array(mesh.vertices, dtype=np.float32)
    except Exception as e:
        print(f"trimesh load failed: {e}")

    # Try open3d
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(pc_path))
        return np.asarray(pcd.points, dtype=np.float32)
    except Exception as e:
        print(f"open3d load failed: {e}")

    return None


def ransac_floor_plane(points: np.ndarray, distance_threshold: float = 0.05):
    """Detect floor plane using Open3D RANSAC. Returns (plane_model, inlier_indices, inlier_points)."""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not available, skipping RANSAC")
        return None

    if points.shape[0] < 100:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # RANSAC plane segmentation
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )

    if len(inliers) < 50:
        return None

    plane_model = np.array(plane_model, dtype=np.float32)
    inliers = np.array(inliers, dtype=np.int32)
    inlier_points = points[inliers]

    # Ensure normal points upward (positive z for up_axis='z')
    normal = plane_model[:3]
    if normal[2] < 0:
        plane_model = -plane_model
        normal = -normal

    return plane_model, inliers, inlier_points


def ransac_wall_planes(points: np.ndarray, floor_plane: np.ndarray, floor_inliers: np.ndarray,
                       distance_threshold: float = 0.05, min_height: float = 1.0,
                       min_inliers: int = 100) -> list:
    """Detect vertical wall planes using RANSAC on non-floor points."""
    try:
        import open3d as o3d
    except ImportError:
        return []

    if points.shape[0] < 100:
        return []

    # Create point cloud and remove floor inliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    non_floor_pcd = pcd.select_by_index(floor_inliers, invert=True)

    if len(non_floor_pcd.points) < 50:
        return []

    floor_normal = floor_plane[:3]
    walls = []
    remaining_pcd = non_floor_pcd

    # Find up to 6 vertical planes (allow more iterations to find 4 good walls)
    for i in range(8):
        if len(remaining_pcd.points) < min_inliers:
            break

        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=2000
        )

        if len(inliers) < min_inliers:
            break

        plane_model = np.array(plane_model, dtype=np.float32)
        normal = plane_model[:3]

        # Check if plane is vertical: normal perpendicular to floor normal (dot ~ 0)
        dot_with_floor = abs(np.dot(normal, floor_normal))
        if dot_with_floor > 0.3:  # Not vertical enough (allow 17 deg tilt)
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            continue

        # Get bounds of this wall
        wall_points = np.asarray(remaining_pcd.points)[inliers]
        z_min, z_max = wall_points[:, 2].min(), wall_points[:, 2].max()

        if z_max - z_min < min_height:
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            continue

        # Project to 2D for bounds
        xy_min = wall_points[:, :2].min(axis=0)
        xy_max = wall_points[:, :2].max(axis=0)

        walls.append({
            "plane": plane_model.tolist(),
            "bounds": [[float(xy_min[0]), float(xy_min[1])], [float(xy_max[0]), float(xy_max[1])]],
            "height": [float(z_min), float(z_max)],
            "inlier_count": int(len(inliers)),
            "normal": normal.tolist()
        })

        # Remove this wall's points
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    return walls


def cluster_walls(walls: list, angle_threshold_deg: float = 15.0) -> list:
    """Cluster walls by normal direction to find 4 dominant wall orientations."""
    if not walls:
        return []

    # Sort by inlier count (largest planes first)
    walls_sorted = sorted(walls, key=lambda w: w.get("inlier_count", 0), reverse=True)

    clustered = []
    used = [False] * len(walls_sorted)

    for i, w in enumerate(walls_sorted):
        if used[i]:
            continue

        normal_i = np.array(w["normal"])
        cluster = [w]
        used[i] = True

        for j, w2 in enumerate(walls_sorted):
            if used[j] or i == j:
                continue
            normal_j = np.array(w2["normal"])
            # Check if normals are aligned or opposite (parallel walls)
            dot = abs(np.dot(normal_i, normal_j))
            angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi
            if angle < angle_threshold_deg or angle > 180 - angle_threshold_deg:
                cluster.append(w2)
                used[j] = True

        # Merge cluster: pick the one with most inliers as representative
        best = max(cluster, key=lambda w: w.get("inlier_count", 0))

        # Combine bounds of cluster
        all_bounds = np.array([c["bounds"] for c in cluster])
        xy_min = all_bounds[:, 0, :].min(axis=0)
        xy_max = all_bounds[:, 1, :].max(axis=0)
        z_min = min(c["height"][0] for c in cluster)
        z_max = max(c["height"][1] for c in cluster)

        clustered.append({
            "plane": best["plane"],
            "bounds": [[float(xy_min[0]), float(xy_min[1])], [float(xy_max[0]), float(xy_max[1])]],
            "height": [float(z_min), float(z_max)],
            "normal": best["normal"],
            "inlier_count": sum(c.get("inlier_count", 0) for c in cluster)
        })

    return clustered


def ensure_four_walls(walls: list, floor_bounds: list) -> list:
    """Ensure we have at least 4 walls forming a rectangle. Add synthetic if needed."""
    if len(walls) >= 4:
        # Keep top 4 by inlier count
        return sorted(walls, key=lambda w: w.get("inlier_count", 0), reverse=True)[:4]

    # If we have 2-3 walls, try to infer the missing ones
    if len(walls) >= 2:
        # Find two dominant orthogonal directions
        normals = [np.array(w["normal"]) for w in walls]
        # Cluster into two groups (parallel pairs)
        # For simplicity, use floor bounds to create missing walls
        pass

    # Fallback: create 4 walls from floor bounds
    x_min, y_min = floor_bounds[0]
    x_max, y_max = floor_bounds[1]
    z_min = floor_bounds[0][0]  # dummy, will use height from existing walls
    z_max = 2.5

    if walls:
        z_min = min(w["height"][0] for w in walls)
        z_max = max(w["height"][1] for w in walls)

    synthetic_walls = [
        {"plane": [0.0, 1.0, 0.0, -y_max], "bounds": [[x_min, y_max], [x_max, y_max]], "height": [z_min, z_max], "normal": [0.0, 1.0, 0.0]},
        {"plane": [0.0, -1.0, 0.0, y_min], "bounds": [[x_min, y_min], [x_max, y_min]], "height": [z_min, z_max], "normal": [0.0, -1.0, 0.0]},
        {"plane": [1.0, 0.0, 0.0, -x_max], "bounds": [[x_max, y_min], [x_max, y_max]], "height": [z_min, z_max], "normal": [1.0, 0.0, 0.0]},
        {"plane": [-1.0, 0.0, 0.0, x_min], "bounds": [[x_min, y_min], [x_min, y_max]], "height": [z_min, z_max], "normal": [-1.0, 0.0, 0.0]},
    ]

    # Replace missing walls with synthetic ones
    existing_normals = [np.array(w["normal"]) for w in walls]
    result = list(walls)

    for syn in synthetic_walls:
        syn_normal = np.array(syn["normal"])
        # Check if we already have a wall with similar normal
        found = False
        for ex_normal in existing_normals:
            dot = abs(np.dot(syn_normal, ex_normal))
            if dot > 0.9:  # Same direction
                found = True
                break
        if not found:
            result.append(syn)

    return result[:4]


def detect_floor_plane_fallback(points: np.ndarray) -> dict:
    """Fallback floor detection using simple statistics."""
    z_vals = points[:, 2]
    z_low = np.percentile(z_vals, 5)
    floor_mask = z_vals < (z_low + 0.1)
    floor_z = float(np.median(z_vals[floor_mask])) if np.any(floor_mask) else float(np.median(z_vals))

    x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
    y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())

    # Plane: z - floor_z = 0 => [0, 0, 1, -floor_z]
    return {
        "plane": [0.0, 0.0, 1.0, -floor_z],
        "bounds": [[x_min, y_min], [x_max, y_max]]
    }


def detect_walls_fallback(points: np.ndarray, floor: dict) -> list:
    """Fallback wall detection using point cloud bounds."""
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
    y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())

    walls = [
        {"plane": [0.0, 1.0, 0.0, -y_max], "bounds": [[x_min, y_max], [x_max, y_max]], "height": [z_min, z_max], "normal": [0.0, 1.0, 0.0]},
        {"plane": [0.0, -1.0, 0.0, y_min], "bounds": [[x_min, y_min], [x_max, y_min]], "height": [z_min, z_max], "normal": [0.0, -1.0, 0.0]},
        {"plane": [1.0, 0.0, 0.0, -x_max], "bounds": [[x_max, y_min], [x_max, y_max]], "height": [z_min, z_max], "normal": [1.0, 0.0, 0.0]},
        {"plane": [-1.0, 0.0, 0.0, x_min], "bounds": [[x_min, y_min], [x_min, y_max]], "height": [z_min, z_max], "normal": [-1.0, 0.0, 0.0]},
    ]
    return walls


def classify_objects(objects: list, floor_plane: np.ndarray) -> list:
    """Classify objects by type based on position relative to floor and size."""
    classified = []
    floor_normal = floor_plane[:3]
    floor_offset = floor_plane[3]

    for obj in objects:
        category = obj.get("category", "unknown").lower()
        center = np.array(obj.get("center", [0, 0, 0]), dtype=np.float32)
        size = np.array(obj.get("size", [0, 0, 0]), dtype=np.float32)

        # Distance from floor plane
        dist_to_floor = abs(np.dot(floor_normal, center) + floor_offset)

        # Classify
        if category in ["floor", "ground"]:
            obj_type = "floor"
        elif category in ["wall"]:
            obj_type = "wall"
        elif category in ["table", "desk"]:
            obj_type = "table"
        elif category in ["chair", "couch", "sofa", "seat"]:
            obj_type = "chair"
        elif category in ["stairs", "stair", "step"]:
            obj_type = "stairs"
        elif dist_to_floor < 0.1 and size[2] < 0.1:
            obj_type = "floor"
        elif size[2] > 1.5 and size[0] * size[1] > 2.0:
            obj_type = "wall"
        elif size[2] > 0.5 and size[0] > 0.5 and size[1] > 0.5:
            obj_type = "table"
        elif size[2] > 0.3:
            obj_type = "chair"
        else:
            obj_type = "unknown"

        classified.append({
            **obj,
            "classified_type": obj_type,
            "dist_to_floor": float(dist_to_floor)
        })

    return classified


def build_free_space(floor_bounds: list, objects: list, inflation: float = 0.4) -> dict:
    """Compute free space polygon (floor minus object footprints)."""
    # Simplified: return floor bounds (object inflation skipped in MVP)
    return {
        "type": "polygon",
        "data": floor_bounds,
        "note": "MVP: free space = floor bounds. Object footprint subtraction deferred.",
        "inflation_margin": inflation
    }


def build_navigation_regions(floor_bounds: list) -> list:
    """Split free space into navigation regions."""
    return [{"id": "nav_0", "polygon": floor_bounds}]


def run_stage_e():
    """Build structured scene and room shell."""
    paths = CFG.paths
    paths.scene.mkdir(parents=True, exist_ok=True)

    # Load objects
    obb_path = paths.holi_spatial / "objects_obb.json"
    objects = load_objects_obb(obb_path)
    print(f"Loaded {len(objects)} objects from {obb_path}")

    # Load point cloud
    pc_path = paths.geometry / "point_cloud.ply"
    points = load_point_cloud(pc_path)

    if points is not None:
        print(f"Loaded point cloud: {points.shape}")

    # Floor detection
    floor = None
    walls = []

    gravity_transform = None  # saved to provenance

    if points is not None and points.shape[0] > 100:
        # Try RANSAC for floor
        ransac_result = ransac_floor_plane(points)

        if ransac_result is not None:
            floor_plane, floor_inliers, floor_pts = ransac_result

            # Gravity-alignment: if detected floor is not horizontal, rotate the
            # entire scene (point cloud + objects) so floor becomes Z-up at z=0.
            floor_normal = floor_plane[:3]
            floor_normal = floor_normal / (np.linalg.norm(floor_normal) + 1e-8)
            tilt_from_z = float(np.degrees(np.arccos(np.clip(abs(floor_normal[2]), 0, 1))))

            if abs(floor_normal[2]) < 0.95:  # floor is tilted > ~18 deg
                print(f"Floor tilted {tilt_from_z:.1f}deg from horizontal -> gravity-aligning scene (Z-up, floor at z=0)")
                points, floor_plane, floor_inliers, objects, gravity_transform = gravity_align(
                    points, floor_plane, floor_inliers, objects
                )
                floor_pts = points[floor_inliers]
                print(f"Gravity-aligned floor: normal=[0,0,1] at z=0, objects repositioned: {len(objects)}")
            else:
                print(f"RANSAC floor normal close to Z-up (tilt {tilt_from_z:.1f}deg), no alignment needed")

            # Now floor is horizontal (Z-up). Compute floor bounds from floor inliers.
            # Floor at z=0, so XY bounds define the footprint.
            xy_min = floor_pts[:, :2].min(axis=0)
            xy_max = floor_pts[:, :2].max(axis=0)
            floor = {
                "plane": floor_plane.tolist(),
                "bounds": [[float(xy_min[0]), float(xy_min[1])], [float(xy_max[0]), float(xy_max[1])]]
            }
            print(f"RANSAC floor: plane={floor_plane}, bounds={floor['bounds']}")

            # Detect walls using RANSAC (pass floor inliers) on aligned points
            walls = ransac_wall_planes(points, floor_plane, floor_inliers)
            print(f"RANSAC raw walls found: {len(walls)}")

            # Cluster walls by normal direction
            walls = cluster_walls(walls)
            print(f"After clustering: {len(walls)} walls")

            # Ensure we have 4 walls
            walls = ensure_four_walls(walls, floor["bounds"])
            print(f"Final walls: {len(walls)}")

        # Fallback if RANSAC failed
        if floor is None:
            print("RANSAC floor detection failed, using fallback")
            floor = detect_floor_plane_fallback(points)
            walls = detect_walls_fallback(points, floor)
    else:
        print("Using synthetic room shell (4x4m x 2.5m)")
        floor = {
            "plane": [0.0, 0.0, 1.0, 0.0],
            "bounds": [[-2.0, -2.0], [2.0, 2.0]]
        }
        walls = [
            {"plane": [0.0, 1.0, 0.0, -2.0], "bounds": [[-2.0, 2.0], [2.0, 2.0]], "height": [0.0, 2.5], "normal": [0.0, 1.0, 0.0]},
            {"plane": [0.0, -1.0, 0.0, -2.0], "bounds": [[-2.0, -2.0], [2.0, -2.0]], "height": [0.0, 2.5], "normal": [0.0, -1.0, 0.0]},
            {"plane": [1.0, 0.0, 0.0, -2.0], "bounds": [[2.0, -2.0], [2.0, 2.0]], "height": [0.0, 2.5], "normal": [1.0, 0.0, 0.0]},
            {"plane": [-1.0, 0.0, 0.0, -2.0], "bounds": [[-2.0, -2.0], [-2.0, 2.0]], "height": [0.0, 2.5], "normal": [-1.0, 0.0, 0.0]},
        ]

    # Ceiling (optional)
    ceiling = None
    if points is not None and points.shape[0] > 100:
        z_vals = points[:, 2]
        z_top = float(np.percentile(z_vals, 95))
        ceiling = {"plane": [0.0, 0.0, -1.0, z_top]}

    # Classify objects
    floor_plane_arr = np.array(floor["plane"], dtype=np.float32)
    classified_objects = classify_objects(objects, floor_plane_arr)

    # Build scene objects
    scene_objects = []
    for obj in classified_objects:
        scene_objects.append({
            "id": obj.get("instance_id", 0),
            "category": obj.get("category", "unknown"),
            "classified_type": obj.get("classified_type", "unknown"),
            "position": obj.get("center", [0.0, 0.0, 0.0]),
            "rotation": obj.get("orientation", [1.0, 0.0, 0.0, 0.0]),
            "size": obj.get("size", [0.0, 0.0, 0.0]),
            "bbox": {
                "type": obj.get("bbox_type", "OBB"),
                "corners": []  # skipped in MVP
            },
            "confidence": obj.get("confidence", 0.5),
            "source_mesh": "geometry/scene_mesh.ply",
        })

    free_space = build_free_space(floor["bounds"], scene_objects)
    nav_regions = build_navigation_regions(floor["bounds"])

    # Build structured scene with room wrapper (expected by validation)
    structured_scene = {
        "units": "meters",
        "up_axis": CFG.up_axis,
        "room": {
            "floor": floor,
            "walls": walls,
            "ceiling": ceiling
        },
        "doors": [],
        "windows": [],
        "objects": scene_objects,
        "free_space": free_space,
        "navigation_regions": nav_regions,
        "provenance": {
            "vlm": "holi_spatial/vlm_labels.json",
            "obb": "holi_spatial/objects_obb.json",
            "mesh": "geometry/scene_mesh.ply",
            "gravity_align": gravity_transform,
        },
    }

    scene_path = paths.scene / "structured_scene.json"
    with open(scene_path, 'w') as f:
        json.dump(structured_scene, f, indent=2)
    print(f"Structured scene saved: {scene_path}")

    # Build scene graph
    scene_graph = {
        "nodes": [{"id": "floor", "type": "region"},
                  {"id": "wall_n", "type": "region"},
                  {"id": "wall_s", "type": "region"},
                  {"id": "wall_e", "type": "region"},
                  {"id": "wall_w", "type": "region"}]
                  + [{"id": f"obj_{o['id']}", "type": "object", "category": o["category"], "classified_type": o["classified_type"]} for o in scene_objects],
        "edges": [],
        "note": "MVP scene graph: adjacency not computed in detail."
    }
    for o in scene_objects:
        scene_graph["edges"].append({"source": "floor", "target": f"obj_{o['id']}", "relation": "supports"})

    graph_path = paths.scene / "scene_graph.json"
    with open(graph_path, 'w') as f:
        json.dump(scene_graph, f, indent=2)
    print(f"Scene graph saved: {graph_path}")

    return {"structured_scene": str(scene_path), "scene_graph": str(graph_path)}


if __name__ == "__main__":
    run_stage_e()
