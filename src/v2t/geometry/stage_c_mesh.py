"""Stage C: Mesh generation from point cloud.

Reads point_cloud.ply from Stage B, builds mesh via Open3D TSDF or Poisson,
cleans/decimates, saves scene_mesh.ply + scene_visual.obj to CFG.paths.geometry.
If point_cloud.ply missing, creates a synthetic box room mesh (4m x 4m x 2.5m).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import warnings

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from src.v2t.config import CFG


@dataclass
class MeshMetadata:
    vertex_count: int
    face_count: int
    bounds_min: list[float]
    bounds_max: list[float]
    is_synthetic: bool
    method: str


def create_synthetic_box_room() -> tuple["o3d.geometry.TriangleMesh", MeshMetadata]:
    """Create a synthetic 4m x 4m x 2.5m box room mesh."""
    # Create box: floor (0,0,0) to (4,4,2.5)
    # Using open3d TriangleMesh box
    mesh = o3d.geometry.TriangleMesh.create_box(width=4.0, height=4.0, depth=2.5)
    # Translate so floor is at z=0 (box creates from -2 to 2 by default)
    mesh.translate([0.0, 0.0, 0.0])

    # Compute vertex/face counts
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.triangles)
    bounds_min = [0.0, 0.0, 0.0]
    bounds_max = [4.0, 4.0, 2.5]

    metadata = MeshMetadata(
        vertex_count=vertex_count,
        face_count=face_count,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        is_synthetic=True,
        method="synthetic_box"
    )
    return mesh, metadata


def load_point_cloud(ply_path: Path) -> "o3d.geometry.PointCloud | None":
    """Load point cloud from PLY file."""
    if not ply_path.exists():
        return None
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        return None
    return pcd


def build_mesh_tsdf(pcd: "o3d.geometry.PointCloud", voxel_size: float = 0.02) -> "o3d.geometry.TriangleMesh":
    """Build mesh using TSDF volume integration."""
    # Estimate normals for TSDF
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=10)

    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4.0 * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )

    # We need RGB-D images for TSDF - create dummy depth from point cloud
    # For MVP, use Poisson instead which works directly on point cloud
    return build_mesh_poisson(pcd)


def build_mesh_poisson(pcd: "o3d.geometry.PointCloud", depth: int = 9) -> "o3d.geometry.TriangleMesh":
    """Build mesh using Poisson surface reconstruction."""
    # Ensure normals are estimated
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=10)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Remove low-density vertices (outliers)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.1)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def clean_and_decimate(mesh: "o3d.geometry.TriangleMesh", target_faces: int = 50000) -> "o3d.geometry.TriangleMesh":
    """Clean mesh (remove duplicates, degenerate) and decimate to target face count."""
    # Remove duplicate vertices
    mesh.remove_duplicated_vertices()
    # Remove degenerate triangles
    mesh.remove_degenerate_triangles()
    # Remove non-manifold edges
    mesh.remove_non_manifold_edges()

    # Decimate if too many faces
    current_faces = len(mesh.triangles)
    if current_faces > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)

    return mesh


def save_mesh_files(mesh: "o3d.geometry.TriangleMesh", metadata: MeshMetadata) -> None:
    """Save mesh as PLY and OBJ, plus metadata JSON."""
    geometry_dir = CFG.paths.geometry
    geometry_dir.mkdir(parents=True, exist_ok=True)

    # Save PLY (with normals)
    ply_path = geometry_dir / "scene_mesh.ply"
    o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_normals=True)
    print(f"Saved mesh PLY: {ply_path}")

    # Save OBJ (visual format)
    obj_path = geometry_dir / "scene_visual.obj"
    o3d.io.write_triangle_mesh(str(obj_path), mesh, write_vertex_normals=True)
    print(f"Saved mesh OBJ: {obj_path}")

    # Save metadata JSON
    meta_path = geometry_dir / "mesh_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)
    print(f"Saved metadata: {meta_path}")


def run_stage_c() -> MeshMetadata:
    """Run Stage C: Mesh generation."""
    print("=== Stage C: Mesh Generation ===")

    if o3d is None:
        raise ImportError("open3d not available. Install with: pip install open3d")
    if np is None:
        raise ImportError("numpy not available")

    geometry_dir = CFG.paths.geometry
    ply_path = geometry_dir / "point_cloud.ply"

    # Try to load point cloud from Stage B
    pcd = load_point_cloud(ply_path)

    if pcd is None:
        print(f"WARNING: {ply_path} not found or empty. Creating synthetic box room mesh.")
        mesh, metadata = create_synthetic_box_room()
    else:
        print(f"Loaded point cloud: {len(pcd.points)} points")

        # Voxel downsample to reduce noise
        voxel_size = 0.02
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"After voxel downsample: {len(pcd.points)} points")

        # Build mesh via Poisson (works directly on point cloud)
        print("Running Poisson surface reconstruction...")
        mesh = build_mesh_poisson(pcd, depth=9)

        # Clean and decimate
        print("Cleaning and decimating mesh...")
        mesh = clean_and_decimate(mesh, target_faces=50000)

        # Compute metadata
        vertex_count = len(mesh.vertices)
        face_count = len(mesh.triangles)
        bounds = mesh.get_axis_aligned_bounding_box()
        bounds_min = bounds.min_bound.tolist()
        bounds_max = bounds.max_bound.tolist()

        metadata = MeshMetadata(
            vertex_count=vertex_count,
            face_count=face_count,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            is_synthetic=False,
            method="poisson"
        )

    # Save all outputs
    save_mesh_files(mesh, metadata)

    print(f"Mesh: {metadata.vertex_count} vertices, {metadata.face_count} faces")
    print(f"Bounds: {metadata.bounds_min} -> {metadata.bounds_max}")
    print(f"Synthetic: {metadata.is_synthetic}")
    print("=== Stage C Complete ===\n")

    return metadata


if __name__ == "__main__":
    run_stage_c()