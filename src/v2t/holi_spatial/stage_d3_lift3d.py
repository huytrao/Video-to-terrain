"""Stage D3: Lift 2D masks to 3D OBB via DA3 depth unprojection."""

from __future__ import annotations
import json
import numpy as np
import os
from pathlib import Path
from typing import Optional
import warnings

from src.v2t.config import CFG


def load_json(path: Path) -> Optional[dict]:
    """Load JSON file if exists."""
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_npy(path: Path) -> Optional[np.ndarray]:
    """Load NPY file if exists."""
    if path.exists():
        return np.load(path)
    return None


def resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize mask to target resolution using nearest neighbor.

    Handles both boolean masks and uint8 masks (0/255).
    Returns boolean mask.
    """
    if mask.shape == (target_h, target_w):
        # Ensure boolean output
        if mask.dtype == np.bool_:
            return mask
        return mask > 0

    import cv2
    # Normalize to 0/255 uint8 for cv2 resize
    if mask.dtype == np.bool_:
        mask_uint8 = mask.astype(np.uint8) * 255
    elif mask.dtype == np.uint8:
        # Already 0/255 or 0/1 - check max
        if mask.max() <= 1:
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = mask
    else:
        # Float or other - normalize to 0/255
        mask_uint8 = (mask * 255).astype(np.uint8)

    resized = cv2.resize(mask_uint8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized > 128


def unproject_mask_to_world(
    mask: np.ndarray,
    depth: np.ndarray,
    K_inv: np.ndarray,
    C2W: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0
) -> np.ndarray:
    """
    Unproject mask pixels to world coordinates using depth and camera matrices.

    Args:
        mask: Boolean mask (H, W)
        depth: Depth map (H, W) in meters
        K_inv: Inverse intrinsics (3, 3)
        C2W: Camera-to-world transform (4, 4)
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth

    Returns:
        World coordinates of mask pixels (N, 3)
    """
    H, W = mask.shape
    # Get pixel coordinates of mask
    yy, xx = np.where(mask)
    if len(yy) == 0:
        return np.empty((0, 3))

    # Homogeneous pixel coordinates (3, N)
    pix_h = np.stack([xx, yy, np.ones_like(xx)], axis=0).astype(np.float32)

    # Unproject to camera space: K_inv @ pix * depth
    cam_dirs = K_inv @ pix_h  # (3, N)
    depths = depth[yy, xx]

    # Filter valid depths
    valid = (depths > min_depth) & (depths < max_depth) & np.isfinite(depths)
    if not np.any(valid):
        return np.empty((0, 3))

    cam_dirs = cam_dirs[:, valid]
    depths = depths[valid]

    # Camera XYZ
    cam_xyz = cam_dirs * depths  # (3, N)

    # Transform to world: C2W @ [cam_xyz; 1]
    cam_xyz_h = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])  # (4, N)
    world_xyz_h = C2W @ cam_xyz_h  # (4, N)
    world_xyz = world_xyz_h[:3, :].T  # (N, 3)

    return world_xyz


def fit_obb(points: np.ndarray) -> dict:
    """
    Fit Oriented Bounding Box to point cloud using trimesh.

    Returns:
        dict with center, extents (full size), quaternion [w,x,y,z], transform matrix
    """
    if len(points) < 4:
        # Fallback: axis-aligned bbox
        center = points.mean(axis=0) if len(points) > 0 else np.zeros(3)
        extents = points.max(axis=0) - points.min(axis=0) if len(points) > 0 else np.ones(3)
        return {
            'center': center.tolist(),
            'extents': extents.tolist(),
            'quaternion': [1.0, 0.0, 0.0, 0.0],  # identity quat
            'transform': np.eye(4).tolist()
        }

    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=points, faces=[])
        obb = mesh.bounding_box_oriented

        # Get center and extents (full size)
        center = obb.centroid
        extents = obb.extents  # full size

        # Get orientation as quaternion [w, x, y, z]
        # trimesh returns transform matrix, extract rotation
        T = obb.transform
        R = T[:3, :3]

        # Convert rotation matrix to quaternion [w, x, y, z]
        q = trimesh.transformations.quaternion_from_matrix(T)
        # trimesh returns [x, y, z, w], convert to [w, x, y, z]
        quat = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

        return {
            'center': center.tolist(),
            'extents': extents.tolist(),
            'quaternion': quat,
            'transform': T.tolist()
        }
    except Exception as e:
        warnings.warn(f"trimesh OBB failed: {e}, falling back to AABB")
        center = points.mean(axis=0)
        extents = points.max(axis=0) - points.min(axis=0)
        return {
            'center': center.tolist(),
            'extents': extents.tolist(),
            'quaternion': [1.0, 0.0, 0.0, 0.0],
            'transform': np.eye(4).tolist()
        }


def estimate_floor_normal(points: np.ndarray, up_axis: str = 'z') -> np.ndarray:
    """
    Estimate floor plane normal from point cloud.
    Assumes floor is the largest horizontal plane.
    """
    if len(points) < 10:
        return np.array([0, 0, 1], dtype=np.float32)  # default up

    # Use PCA to find dominant plane
    from scipy.spatial import ConvexHull
    try:
        # Project to XY plane and find largest extent
        xy = points[:, :2]
        hull = ConvexHull(xy)
        # Floor normal is up_axis
        normal = np.zeros(3)
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[up_axis]
        normal[axis_idx] = 1.0
        return normal
    except Exception:
        normal = np.zeros(3)
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[up_axis]
        normal[axis_idx] = 1.0
        return normal


def create_stub_objects(up_axis: str = 'z') -> list[dict]:
    """Create stub objects when no data is available."""
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[up_axis]

    stubs = [
        {
            'instance_id': 'floor_stub',
            'category': 'floor',
            'center': [0.0, 0.0, -0.05],
            'size': [10.0, 10.0, 0.1],
            'orientation': [1.0, 0.0, 0.0, 0.0],
            'bbox_type': 'OBB',
            'confidence': 0.0,
            'source_frames': [],
            'is_stub': True
        },
        {
            'instance_id': 'wall_stub',
            'category': 'wall',
            'center': [0.0, -5.0, 1.5],
            'size': [10.0, 0.2, 3.0],
            'orientation': [1.0, 0.0, 0.0, 0.0],
            'bbox_type': 'OBB',
            'confidence': 0.0,
            'source_frames': [],
            'is_stub': True
        },
        {
            'instance_id': 'table_stub',
            'category': 'table',
            'center': [0.0, 0.0, 0.75],
            'size': [1.2, 0.8, 0.05],
            'orientation': [1.0, 0.0, 0.0, 0.0],
            'bbox_type': 'OBB',
            'confidence': 0.0,
            'source_frames': [],
            'is_stub': True
        },
        {
            'instance_id': 'chair_stub',
            'category': 'chair',
            'center': [-0.6, 0.0, 0.45],
            'size': [0.5, 0.5, 0.9],
            'orientation': [1.0, 0.0, 0.0, 0.0],
            'bbox_type': 'OBB',
            'confidence': 0.0,
            'source_frames': [],
            'is_stub': True
        }
    ]
    # Adjust for up_axis
    for obj in stubs:
        center = obj['center']
        size = obj['size']
        # Swap axes if needed
        if up_axis == 'y':
            # Z-up to Y-up: z <-> y
            center[1], center[2] = center[2], center[1]
            size[1], size[2] = size[2], size[1]
        elif up_axis == 'x':
            # Z-up to X-up: z <-> x
            center[0], center[2] = center[2], center[0]
            size[0], size[2] = size[2], size[0]
        obj['center'] = center
        obj['size'] = size

    return stubs


def run_stage_d3() -> dict:
    """Main entry for Stage D3: Lift 2D masks to 3D OBBs."""
    print("=" * 60)
    print("Stage D3: Lift 2D masks -> 3D OBB")
    print("=" * 60)

    paths = CFG.paths
    holi_dir = paths.holi_spatial
    sam_dir = paths.sam_masks
    depth_dir = paths.depth
    geom_dir = paths.geometry

    # Load camera intrinsics and poses
    intrinsics_path = geom_dir / "camera_intrinsics.json"
    poses_path = geom_dir / "camera_poses.json"

    intrinsics = load_json(intrinsics_path)
    poses = load_json(poses_path)

    if intrinsics is None or poses is None:
        warnings.warn(f"Missing camera data: intrinsics={intrinsics_path.exists()}, poses={poses_path.exists()}")
        print("Generating stub objects...")
        objects = create_stub_objects(CFG.up_axis)
        is_stub = True
    else:
        # Parse intrinsics - DA3 saves as "intrinsics" array with 100 3x3 K matrices (all same)
        # Use first frame's intrinsics
        K_list = intrinsics.get('intrinsics', [])
        if not K_list:
            warnings.warn("No intrinsics found in camera_intrinsics.json")
            print("Generating stub objects...")
            objects = create_stub_objects(CFG.up_axis)
            is_stub = True
        else:
            K = np.array(K_list[0]).reshape(3, 3)
            K_inv = np.linalg.inv(K)

            # Parse poses - DA3 saves as "extrinsics" array with 100 4x4 identity matrices
            # Since all poses are identity, first camera = world space
            extrinsics_list = poses.get('extrinsics', [])
            if not extrinsics_list:
                warnings.warn("No extrinsics found in camera_poses.json")
                print("Generating stub objects...")
                objects = create_stub_objects(CFG.up_axis)
                is_stub = True
            else:
                # Use identity C2W (first camera frame = world)
                C2W = np.eye(4, dtype=np.float32)

                # Load mask index (metadata.json)
                mask_index_path = sam_dir / "metadata.json"
                mask_index = load_json(mask_index_path)

                if mask_index is None or 'items' not in mask_index:
                    warnings.warn(f"No mask index found at {mask_index_path}")
                    print("Generating stub objects...")
                    objects = create_stub_objects(CFG.up_axis)
                    is_stub = True
                else:
                    print(f"Processing {len(mask_index['items'])} masks...")
                    objects = []
                    is_stub = False

                    # Get depth resolution from first depth file
                    depth_files = sorted(depth_dir.glob("depth_*.npy"))
                    if not depth_files:
                        warnings.warn(f"No depth files found in {depth_dir}")
                        print("Generating stub objects...")
                        objects = create_stub_objects(CFG.up_axis)
                        is_stub = True
                    else:
                        # Load first depth to get resolution
                        sample_depth = np.load(depth_files[0])
                        depth_h, depth_w = sample_depth.shape
                        print(f"Depth resolution: {depth_w}x{depth_h}")

                        # Group masks by category for multi-frame aggregation
                        from collections import defaultdict
                        category_points = defaultdict(list)
                        category_frames = defaultdict(set)

                        for mask_info in mask_index['items']:
                            # Extract frame index from mask info
                            frame_str = mask_info.get('frame', '')
                            # Handle formats like "frame_000000" or "000000"
                            if '_' in frame_str:
                                frame_idx = int(frame_str.split('_')[-1].split('.')[0])
                            else:
                                frame_idx = int(frame_str)

                            label = mask_info.get('label', 'unknown')
                            mask_file = mask_info.get('mask_path', '')

                            # Load mask
                            mask_path = sam_dir / mask_file
                            mask = load_npy(mask_path)
                            if mask is None:
                                # Try .png
                                mask_path_png = mask_path.with_suffix('.png')
                                if mask_path_png.exists():
                                    import cv2
                                    mask_img = cv2.imread(str(mask_path_png), cv2.IMREAD_GRAYSCALE)
                                    mask = mask_img > 128
                                else:
                                    continue

                            # Resize mask from frame resolution (640x480) to depth resolution (378x504)
                            if mask.shape != (depth_h, depth_w):
                                mask = resize_mask(mask, depth_h, depth_w)

                            # Load depth - DA3 uses depth_XXXXXX.npy (6 digits)
                            depth_file = f"depth_{frame_idx:06d}.npy"
                            depth_path = depth_dir / depth_file
                            depth = load_npy(depth_path)
                            if depth is None:
                                # Try 5-digit format as fallback
                                depth_file = f"depth_{frame_idx:05d}.npy"
                                depth_path = depth_dir / depth_file
                                depth = load_npy(depth_path)
                            if depth is None:
                                print(f"  Warning: No depth for frame {frame_idx} ({depth_file})")
                                continue

                            # Verify depth resolution matches
                            if depth.shape != (depth_h, depth_w):
                                print(f"  Warning: Depth shape mismatch for frame {frame_idx}: {depth.shape} vs expected ({depth_h}, {depth_w})")
                                import cv2
                                depth = cv2.resize(depth.astype(np.float32), (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)

                            # Unproject using K_inv and identity C2W (first camera = world)
                            world_pts = unproject_mask_to_world(mask, depth, K_inv, C2W)
                            if len(world_pts) == 0:
                                continue

                            category_points[label].append(world_pts)
                            category_frames[label].add(frame_idx)

                        # Fit OBB per category (aggregate across frames)
                        for label, pts_list in category_points.items():
                            if not pts_list:
                                continue
                            all_pts = np.vstack(pts_list)

                            # Downsample if too many points
                            if len(all_pts) > 5000:
                                idx = np.random.choice(len(all_pts), 5000, replace=False)
                                all_pts = all_pts[idx]

                            obb = fit_obb(all_pts)

                            instance_id = f"{label}_{len(objects)}"
                            objects.append({
                                'instance_id': instance_id,
                                'category': label,
                                'center': obb['center'],
                                'size': obb['extents'],
                                'orientation': obb['quaternion'],
                                'bbox_type': 'OBB',
                                'confidence': 0.8,
                                'source_frames': sorted(list(category_frames[label])),
                                'is_stub': False
                            })

                        if not objects:
                            warnings.warn("No valid objects extracted from masks/depth")
                            print("Generating stub objects...")
                            objects = create_stub_objects(CFG.up_axis)
                            is_stub = True

    # Estimate floor alignment
    # Collect all points from non-stub objects
    all_points = []
    for obj in objects:
        if not obj.get('is_stub', False):
            # Generate points from bbox for floor estimation
            center = np.array(obj['center'])
            size = np.array(obj['size'])
            # Simple: just use center
            all_points.append(center)

    if all_points:
        floor_normal = estimate_floor_normal(np.array(all_points), CFG.up_axis)
    else:
        floor_normal = np.zeros(3)
        floor_normal[{'x': 0, 'y': 1, 'z': 2}[CFG.up_axis]] = 1.0

    floor_alignment = {
        'up_axis': CFG.up_axis,
        'floor_normal': floor_normal.tolist(),
        'floor_label': CFG.floor_label
    }

    # Prepare outputs
    obb_output = {
        'objects': objects,
        'units': 'meters',
        'up_axis': CFG.up_axis,
        'floor_alignment': floor_alignment
    }

    # AABB is just axis-aligned version (min/max)
    aabb_objects = []
    for obj in objects:
        center = np.array(obj['center'])
        size = np.array(obj['size'])
        half = size / 2
        aabb_objects.append({
            'instance_id': obj['instance_id'],
            'category': obj['category'],
            'min': (center - half).tolist(),
            'max': (center + half).tolist(),
            'center': obj['center'],
            'size': obj['size'],
            'confidence': obj['confidence'],
            'source_frames': obj['source_frames'],
            'is_stub': obj.get('is_stub', False)
        })

    aabb_output = {
        'objects': aabb_objects,
        'units': 'meters',
        'up_axis': CFG.up_axis
    }

    # Save
    holi_dir.mkdir(parents=True, exist_ok=True)

    obb_path = holi_dir / "objects_obb.json"
    with open(obb_path, 'w') as f:
        json.dump(obb_output, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"Saved OBB: {obb_path}")

    aabb_path = holi_dir / "objects_aabb.json"
    with open(aabb_path, 'w') as f:
        json.dump(aabb_output, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"Saved AABB: {aabb_path}")

    print(f"Total objects: {len(objects)} (stub={is_stub})")
    return {'obb_path': str(obb_path), 'aabb_path': str(aabb_path), 'num_objects': len(objects)}


if __name__ == "__main__":
    result = run_stage_d3()
    print(f"Result: {result}")