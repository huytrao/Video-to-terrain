#!/usr/bin/env python3
"""
Client Report Generator for Video-to-Terrain Pipeline.

Scans outputs/run_mvp/ for all stage outputs and generates a comprehensive
HTML report with pipeline overview, stage details, statistics, and preview images.

Usage:
    python -m src.v2t.generate_client_report
"""

from __future__ import annotations
import json
import base64
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Optional
from datetime import datetime
import subprocess

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.v2t.config import CFG


@dataclass
class StageReport:
    """Report data for a single pipeline stage."""
    stage: str
    name: str
    status: str  # "complete", "partial", "missing", "error"
    files_found: list[str]
    stats: dict[str, Any]
    errors: list[str]
    preview_images: list[str]  # base64 encoded


class ClientReportGenerator:
    """Generates comprehensive HTML client report from pipeline outputs."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.paths = CFG.paths
        self.stages: list[StageReport] = []
        self.overall_timing: dict[str, float] = {}
        self.preview_images: dict[str, str] = {}  # stage -> base64 image
        
    def run(self) -> Path:
        """Run full report generation and return output path."""
        print(f"Generating client report for: {self.run_dir}")
        
        # Collect all stage data
        self._collect_stage_a()
        self._collect_stage_b()
        self._collect_stage_c()
        self._collect_stage_d1()
        self._collect_stage_d2()
        self._collect_stage_d3()
        self._collect_stage_e()
        self._collect_stage_f()
        self._collect_stage_g()
        self._collect_stage_i()
        self._collect_stage_j()
        
        # Generate HTML
        output_path = self.run_dir / "client_report.html"
        self._write_html(output_path)
        
        print(f"Client report saved to: {output_path}")
        return output_path
    
    def _read_json(self, path: Path) -> Optional[dict]:
        """Safely read JSON file."""
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"_error": str(e)}
    
    def _read_npy_info(self, path: Path) -> Optional[dict]:
        """Read basic info from .npy file without loading full array."""
        if not path.exists():
            return None
        try:
            import numpy as np
            arr = np.load(path, mmap_mode='r')
            return {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(arr.min()) if arr.size > 0 else None,
                "max": float(arr.max()) if arr.size > 0 else None,
                "mean": float(arr.mean()) if arr.size > 0 else None,
            }
        except Exception as e:
            return {"_error": str(e)}
    
    def _encode_image_base64(self, path: Path) -> Optional[str]:
        """Encode image file as base64 data URL."""
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            ext = path.suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            return f"data:{mime};base64,{data}"
        except Exception:
            return None
    
    def _collect_preview_images(self):
        """Collect preview images from various stages."""
        # Point cloud preview (from validation)
        preview_path = self.paths.validation / "preview.png"
        if preview_path.exists():
            self.preview_images["point_cloud"] = self._encode_image_base64(preview_path)
        
        # Try to find mesh preview
        mesh_preview = self.paths.geometry / "scene_preview.png"
        if mesh_preview.exists():
            self.preview_images["mesh"] = self._encode_image_base64(mesh_preview)
        
        # Try to find depth preview
        depth_preview = self.paths.depth / "depth_000000.png"
        if depth_preview.exists():
            self.preview_images["depth"] = self._encode_image_base64(depth_preview)
    
    # ===== Stage Collection Methods =====
    
    def _collect_stage_a(self):
        """Stage A: Frame Extraction."""
        stats = {}
        files_found = []
        errors = []
        
        # Capture metadata
        meta_path = self.paths.frames.parent / "capture_metadata.json"
        if meta_path.exists():
            files_found.append("capture_metadata.json")
            meta = self._read_json(meta_path)
            if meta and "_error" not in meta:
                stats = {
                    "video_path": meta.get("video_path"),
                    "total_frames": meta.get("total_frames"),
                    "fps": meta.get("fps"),
                    "duration_sec": meta.get("duration"),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "sample_rate": meta.get("sample_rate"),
                    "sampled_frames": len(meta.get("sampled_frames", [])),
                    "blurry_frames": sum(1 for f in meta.get("sampled_frames", []) if f.get("is_blurry")),
                }
            else:
                errors.append(f"Failed to parse capture_metadata.json: {meta.get('_error')}")
        else:
            errors.append("capture_metadata.json not found")
        
        # Frame files
        frame_files = list(self.paths.frames.glob("frame_*.jpg")) + list(self.paths.frames.glob("frame_*.png"))
        if frame_files:
            files_found.append(f"{len(frame_files)} frame images")
            stats["frames_on_disk"] = len(frame_files)
        else:
            errors.append("No frame images found in frames/")
        
        status = "complete" if not errors else ("partial" if files_found else "missing")
        
        self.stages.append(StageReport(
            stage="A",
            name="Frame Extraction",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=[]
        ))
    
    def _collect_stage_b(self):
        """Stage B: DA3 Depth Estimation."""
        stats = {}
        files_found = []
        errors = []
        previews = []
        
        # Camera intrinsics
        intr_path = self.paths.geometry / "camera_intrinsics.json"
        if intr_path.exists():
            files_found.append("camera_intrinsics.json")
            intr = self._read_json(intr_path)
            if intr and "_error" not in intr:
                stats["intrinsics"] = intr.get("intrinsics", [])[:1]  # First frame only
                stats["image_width"] = intr.get("width")
                stats["image_height"] = intr.get("height")
                stats["note"] = intr.get("note")
        
        # Camera poses
        poses_path = self.paths.geometry / "camera_poses.json"
        if poses_path.exists():
            files_found.append("camera_poses.json")
            poses = self._read_json(poses_path)
            if poses and "_error" not in poses:
                extr = poses.get("extrinsics", [])
                stats["num_poses"] = len(extr)
                stats["poses_shape"] = f"({len(extr)}, 4, 4)" if extr else "empty"
                stats["poses_note"] = poses.get("note")
        
        # Point cloud
        pc_path = self.paths.geometry / "point_cloud.ply"
        if pc_path.exists():
            files_found.append("point_cloud.ply")
            try:
                import trimesh
                mesh = trimesh.load(str(pc_path))
                if hasattr(mesh, 'vertices'):
                    stats["point_cloud_vertices"] = len(mesh.vertices)
                    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                        stats["point_cloud_has_colors"] = True
            except Exception as e:
                stats["point_cloud_error"] = str(e)
        
        # Depth files
        depth_files = list(self.paths.depth.glob("depth_*.npy"))
        if depth_files:
            files_found.append(f"{len(depth_files)} depth maps (.npy)")
            # Sample first depth map
            sample = self._read_npy_info(depth_files[0])
            if sample and "_error" not in sample:
                stats["depth_sample"] = sample
        
        conf_files = list(self.paths.depth.glob("conf_*.npy"))
        if conf_files:
            files_found.append(f"{len(conf_files)} confidence maps (.npy)")
        
        status = "complete" if not errors and files_found else ("partial" if files_found else "missing")
        
        # Preview: try to render point cloud preview
        if pc_path.exists():
            preview_b64 = self._try_render_point_cloud_preview(pc_path)
            if preview_b64:
                previews.append(preview_b64)
        
        self.stages.append(StageReport(
            stage="B",
            name="DA3 Depth Estimation",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=previews
        ))
    
    def _try_render_point_cloud_preview(self, ply_path: Path) -> Optional[str]:
        """Try to render a point cloud preview image."""
        try:
            import open3d as o3d
            import numpy as np
            pcd = o3d.io.read_point_cloud(str(ply_path))
            if len(pcd.points) == 0:
                return None
            
            # Downsample for rendering
            if len(pcd.points) > 100000:
                pcd = pcd.random_down_sample(100000 / len(pcd.points))
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=800, height=600)
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            img = vis.capture_screen_float_buffer(do_render=True)
            vis.destroy_window()
            
            # Convert to PIL and encode
            img_array = (np.asarray(img) * 255).astype(np.uint8)
            from PIL import Image
            pil_img = Image.fromarray(img_array)
            import io
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception:
            return None
    
    def _collect_stage_c(self):
        """Stage C: Mesh Generation."""
        stats = {}
        files_found = []
        errors = []
        previews = []
        
        # Mesh metadata
        meta_path = self.paths.geometry / "mesh_metadata.json"
        if meta_path.exists():
            files_found.append("mesh_metadata.json")
            meta = self._read_json(meta_path)
            if meta and "_error" not in meta:
                stats = meta
        
        # Mesh files
        for pattern in ["scene_mesh.ply", "scene_visual.obj"]:
            mesh_path = self.paths.geometry / pattern
            if mesh_path.exists():
                files_found.append(pattern)
                try:
                    import trimesh
                    mesh = trimesh.load(str(mesh_path))
                    if hasattr(mesh, 'vertices'):
                        stats["vertices_on_disk"] = len(mesh.vertices)
                    if hasattr(mesh, 'faces'):
                        stats["faces_on_disk"] = len(mesh.faces)
                    if hasattr(mesh, 'bounds'):
                        bounds = mesh.bounds
                        stats["bounds_on_disk"] = bounds.tolist()
                except Exception as e:
                    stats["mesh_load_error"] = str(e)
        
        status = "complete" if not errors and files_found else ("partial" if files_found else "missing")
        
        # Preview: try to render mesh preview
        for pattern in ["scene_mesh.ply", "scene_visual.obj"]:
            mesh_path = self.paths.geometry / pattern
            if mesh_path.exists():
                preview_b64 = self._try_render_mesh_preview(mesh_path)
                if preview_b64:
                    previews.append(preview_b64)
                break
        
        self.stages.append(StageReport(
            stage="C",
            name="Mesh Generation",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=previews
        ))
    
    def _try_render_mesh_preview(self, mesh_path: Path) -> Optional[str]:
        """Try to render a mesh preview image."""
        try:
            import trimesh
            import numpy as np
            mesh = trimesh.load(str(mesh_path))
            
            # Create a scene for rendering
            scene = mesh.scene() if isinstance(mesh, trimesh.Scene) else trimesh.Scene([mesh])
            
            # Set camera
            scene.set_camera()
            
            # Render
            img = scene.save_image(resolution=[800, 600], visible=True)
            if img:
                return f"data:image/png;base64,{base64.b64encode(img).decode('utf-8')}"
        except Exception:
            pass
        return None
    
    def _collect_stage_d1(self):
        """Stage D1: VLM Labeling."""
        stats = {}
        files_found = []
        errors = []
        
        vlm_path = self.paths.holi_spatial / "vlm_labels.json"
        if vlm_path.exists():
            files_found.append("vlm_labels.json")
            data = self._read_json(vlm_path)
            if data and "_error" not in data:
                stats = {
                    "scene": data.get("scene"),
                    "categories": data.get("categories", []),
                    "num_categories": len(data.get("categories", [])),
                    "frames_processed": len(data.get("per_image", {})),
                    "stub": data.get("stub", False),
                    "stub_reason": data.get("stub_reason"),
                }
                
                # Per-frame details
                per_image = data.get("per_image", {})
                if per_image:
                    total_objects = sum(len(v) for v in per_image.values())
                    stats["total_objects_detected"] = total_objects
                    stats["avg_objects_per_frame"] = total_objects / len(per_image)
        else:
            errors.append("vlm_labels.json not found")
        
        status = "complete" if not errors else ("partial" if files_found else "missing")
        
        self.stages.append(StageReport(
            stage="D1",
            name="VLM Labeling",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=[]
        ))
    
    def _collect_stage_d2(self):
        """Stage D2: SAM3 Mask Generation."""
        stats = {}
        files_found = []
        errors = []
        
        sam_dir = self.paths.sam_masks
        meta_path = sam_dir / "metadata.json"
        if meta_path.exists():
            files_found.append("metadata.json")
            data = self._read_json(meta_path)
            if data and "_error" not in data:
                items = data.get("items", [])
                stats = {
                    "num_masks": len(items),
                    "stub": data.get("stub", False),
                    "image_root": data.get("image_root"),
                    "mask_root": data.get("mask_root"),
                }
                
                # Count by label
                label_counts = {}
                frame_counts = {}
                for item in items:
                    label = item.get("label", "unknown")
                    frame = item.get("frame", "unknown")
                    label_counts[label] = label_counts.get(label, 0) + 1
                    frame_counts[frame] = frame_counts.get(frame, 0) + 1
                
                stats["masks_per_label"] = label_counts
                stats["masks_per_frame"] = frame_counts
                stats["unique_labels"] = len(label_counts)
                stats["unique_frames"] = len(frame_counts)
        else:
            errors.append("metadata.json not found in sam_masks/")
        
        # Mask files
        mask_files = list(sam_dir.glob("mask_*.npy"))
        if mask_files:
            files_found.append(f"{len(mask_files)} mask files (.npy)")
            stats["mask_files_on_disk"] = len(mask_files)
            # Sample one mask
            sample = self._read_npy_info(mask_files[0])
            if sample and "_error" not in sample:
                stats["mask_sample_shape"] = sample.get("shape")
        else:
            errors.append("No mask .npy files found")
        
        status = "complete" if not errors else ("partial" if files_found else "missing")
        
        self.stages.append(StageReport(
            stage="D2",
            name="SAM3 Mask Generation",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=[]
        ))
    
    def _collect_stage_d3(self):
        """Stage D3: 3D OBB Lifting."""
        stats = {}
        files_found = []
        errors = []
        
        # OBB
        obb_path = self.paths.holi_spatial / "objects_obb.json"
        if obb_path.exists():
            files_found.append("objects_obb.json")
            data = self._read_json(obb_path)
            if data and "_error" not in data:
                objects = data.get("objects", [])
                stats["obb_count"] = len(objects)
                stats["units"] = data.get("units")
                stats["up_axis"] = data.get("up_axis")
                
                # Categories
                categories = {}
                for obj in objects:
                    cat = obj.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
                stats["categories"] = categories
                
                # Sizes
                sizes = []
                for obj in objects:
                    size = obj.get("size") or obj.get("extents")
                    if size:
                        sizes.append(size)
                if sizes:
                    import numpy as np
                    sizes_arr = np.array(sizes)
                    stats["avg_size"] = sizes_arr.mean(axis=0).tolist()
                    stats["max_size"] = sizes_arr.max(axis=0).tolist()
                    stats["min_size"] = sizes_arr.min(axis=0).tolist()
                
                # Stub check
                stubs = sum(1 for o in objects if o.get("is_stub", False))
                stats["stub_count"] = stubs
        
        # AABB
        aabb_path = self.paths.holi_spatial / "objects_aabb.json"
        if aabb_path.exists():
            files_found.append("objects_aabb.json")
            data = self._read_json(aabb_path)
            if data and "_error" not in data:
                objects = data.get("objects", [])
                stats["aabb_count"] = len(objects)
        
        status = "complete" if not errors and files_found else ("partial" if files_found else "missing")
        
        self.stages.append(StageReport(
            stage="D3",
            name="3D OBB Lifting",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=[]
        ))
    
    def _collect_stage_e(self):
        """Stage E: Scene Structure."""
        stats = {}
        files_found = []
        errors = []
        
        struct_path = self.paths.scene / "structured_scene.json"
        if struct_path.exists():
            files_found.append("structured_scene.json")
            data = self._read_json(struct_path)
            if data and "_error" not in data:
                stats = {
                    "units": data.get("units"),
                    "up_axis": data.get("up_axis"),
                    "has_room": "room" in data,
                    "num_objects": len(data.get("objects", [])),
                }
                
                room = data.get("room", {})
                if room:
                    stats["has_floor"] = "floor" in room
                    stats["num_walls"] = len(room.get("walls", []))
                    stats["has_ceiling"] = "ceiling" in room
                    if "floor" in room:
                        floor = room["floor"]
                        stats["floor_bounds"] = floor.get("bounds")
                        stats["floor_plane"] = floor.get("plane")
                
                free_space = data.get("free_space", {})
                if free_space:
                    stats["free_space_type"] = free_space.get("type")
                    stats["free_space_note"] = free_space.get("note")
                
                nav = data.get("navigation_regions", [])
                stats["nav_regions"] = len(nav)
        
        graph_path = self.paths.scene / "scene_graph.json"
        if graph_path.exists():
            files_found.append("scene_graph.json")
            data = self._read_json(graph_path)
            if data and "_error" not in data:
                stats["graph_nodes"] = len(data.get("nodes", []))
                stats["graph_edges"] = len(data.get("edges", []))
        
        status = "complete" if not errors and files_found else ("partial" if files_found else "missing")
        
        self.stages.append(StageReport(
            stage="E",
            name="Scene Structure",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=[]
        ))
    
    def _collect_stage_f(self):
        """Stage F: Visual Assets."""
        stats = {}
        files_found = []
        errors = []
        
        # Visual mesh
        visual_mesh = self.paths.visual / "scene_visual.obj"
        if visual_mesh.exists():
            files_found.append("scene_visual.obj")
            try:
                import trimesh
                mesh = trimesh.load(str(visual_mesh))
                if hasattr(mesh, 'vertices'):
                    stats["visual_vertices"] = len(mesh.vertices)
                if hasattr(mesh, 'faces'):
                    stats["visual_faces"] = len(mesh.faces)
            except Exception as e:
                stats["visual_mesh_error"] = str(e)
        
        # Materials
        mat_path = self.paths.visual / "materials.json"
        if mat_path.exists():
            files_found.append("materials.json")
            data = self._read_json(mat_path)
            if data and "_error" not in data:
                stats["materials_count"] = len(data)
                stats["material_names"] = list(data.keys())[:10]
        
        # Visual assets
        assets_path = self.paths.visual / "visual_assets.json"
        if assets_path.exists():
            files_found.append("visual_assets.json")
            data = self._read_json(assets_path)
            if data and "_error" not in data:
                stats["visual_scene_mesh"] = data.get("scene_mesh", {}).get("path")
                stats["visual_objects"] = len(data.get("objects", []))
        
        status = "complete" if not errors and files_found else ("partial" if files_found else "missing")
        
        self.stages.append(StageReport(
            stage="F",
            name="Visual Assets",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=[]
        ))
    
    def _collect_stage_g(self):
        """Stage G: Collision + Physics."""
        stats = {}
        files_found = []
        errors = []
        
        # Collision primitives
        coll_dir = self.paths.collision
        floor_plane = coll_dir / "floor_plane.json"
        if floor_plane.exists():
            files_found.append("floor_plane.json")
            data = self._read_json(floor_plane)
            if data and "_error" not in data:
                stats["floor_plane"] = data
        
        walls_dir = coll_dir / "walls"
        if walls_dir.exists():
            wall_files = list(walls_dir.glob("*.json"))
            if wall_files:
                files_found.append(f"{len(wall_files)} wall collision files")
                stats["wall_collision_count"] = len(wall_files)
        
        objects_dir = coll_dir / "objects"
        if objects_dir.exists():
            obj_files = list(objects_dir.glob("*.json"))
            if obj_files:
                files_found.append(f"{len(obj_files)} object collision files")
                stats["object_collision_count"] = len(obj_files)
        
        # Physics
        physics_path = self.paths.simulation / "physics.json"
        if physics_path.exists():
            files_found.append("physics.json")
            data = self._read_json(physics_path)
            if data and "_error" not in data:
                stats["physics_profiles"] = len(data.get("profiles", {}))
                stats["physics_objects"] = len(data.get("objects", []))
                stats["has_sources"] = all("source" in o for o in data.get("objects", []))
        
        status = "complete" if not errors and files_found else ("partial" if files_found else "missing")
        
        self.stages.append(StageReport(
            stage="G",
            name="Collision + Physics",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=[]
        ))
    
    def _collect_stage_i(self):
        """Stage I: MJCF + MJLab Compilation."""
        stats = {}
        files_found = []
        errors = []
        
        # MJCF
        mjcf_path = self.paths.mujoco / "scene.xml"
        if mjcf_path.exists():
            files_found.append("scene.xml")
            try:
                import mujoco
                model = mujoco.MjModel.from_xml_path(str(mjcf_path))
                stats["mjcf_bodies"] = model.nbody
                stats["mjcf_geoms"] = model.ngeom
                stats["mjcf_dofs"] = model.nq
                stats["mjcf_compiles"] = True
            except Exception as e:
                stats["mjcf_compile_error"] = str(e)
                errors.append(f"MJCF compile failed: {e}")
        else:
            errors.append("scene.xml not found")
        
        # MJLab config
        mjlab_path = self.paths.mjlab / "scene_cfg.py"
        if mjlab_path.exists():
            files_found.append("scene_cfg.py")
            stats["mjlab_config_exists"] = True
        
        status = "complete" if not errors and files_found else ("partial" if files_found else "missing")
        
        self.stages.append(StageReport(
            stage="I",
            name="MJCF + MJLab Compilation",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=[]
        ))
    
    def _collect_stage_j(self):
        """Stage J: Validation Report."""
        stats = {}
        files_found = []
        errors = []
        previews = []
        
        report_path = self.paths.validation / "report.json"
        if report_path.exists():
            files_found.append("report.json")
            data = self._read_json(report_path)
            if data and "_error" not in data:
                stats = {
                    "run_id": data.get("run_id"),
                    "timestamp": data.get("timestamp"),
                    "overall": data.get("overall"),
                    "checks_total": len(data.get("checks", [])),
                    "checks_pass": sum(1 for c in data.get("checks", []) if c.get("status") == "PASS"),
                    "checks_fail": sum(1 for c in data.get("checks", []) if c.get("status") == "FAIL"),
                    "checks_warn": sum(1 for c in data.get("checks", []) if c.get("status") == "WARN"),
                    "failures": data.get("failures", []),
                    "next_milestone": data.get("next_milestone"),
                    "preview_render": data.get("preview_render"),
                }
                
                # Definition of done
                dod = data.get("definition_of_done", {})
                stats["definition_of_done"] = dod
                
                # Checks detail
                checks = []
                for c in data.get("checks", []):
                    checks.append({
                        "id": c.get("id"),
                        "name": c.get("name"),
                        "status": c.get("status"),
                        "evidence": c.get("evidence")[:200] if c.get("evidence") else "",
                    })
                stats["checks_detail"] = checks
        
        # Preview image
        preview_path = self.paths.validation / "preview.png"
        if preview_path.exists():
            files_found.append("preview.png")
            preview_b64 = self._encode_image_base64(preview_path)
            if preview_b64:
                previews.append(preview_b64)
        
        status = "complete" if not errors and files_found else ("partial" if files_found else "missing")
        
        self.stages.append(StageReport(
            stage="J",
            name="Validation",
            status=status,
            files_found=files_found,
            stats=stats,
            errors=errors,
            preview_images=previews
        ))
    
    # ===== HTML Generation =====
    
    def _write_html(self, output_path: Path):
        """Write the complete HTML report."""
        html = self._generate_html()
        output_path.write_text(html, encoding='utf-8')
    
    def _generate_html(self) -> str:
        """Generate the full HTML report."""
        # Collect preview images
        self._collect_preview_images()
        
        # Build stages HTML
        stages_html = ""
        for stage in self.stages:
            stages_html += self._render_stage_card(stage)
        
        # Build summary
        total_stages = len(self.stages)
        complete_stages = sum(1 for s in self.stages if s.status == "complete")
        partial_stages = sum(1 for s in self.stages if s.status == "partial")
        missing_stages = sum(1 for s in self.stages if s.status == "missing")
        
        # Pipeline timing from validation if available
        timing_html = ""
        if self.overall_timing:
            timing_html = "<h3>Pipeline Timing</h3><ul>"
            for stage, elapsed in self.overall_timing.items():
                timing_html += f"<li>{stage}: {elapsed:.2f}s</li>"
            timing_html += "</ul>"
        
        # Preview gallery
        preview_html = self._render_preview_gallery()
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video-to-Terrain Pipeline Report - {CFG.run_name}</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Video-to-Terrain Pipeline Client Report</h1>
            <div class="meta">
                <span><strong>Run:</strong> {CFG.run_name}</span>
                <span><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
                <span><strong>Input Video:</strong> {CFG.input_video.name}</span>
                <span><strong>Output Dir:</strong> {self.run_dir}</span>
            </div>
        </header>
        
        <section id="summary">
            <h2>Pipeline Summary</h2>
            <div class="summary-grid">
                <div class="summary-card complete">
                    <div class="summary-value">{complete_stages}</div>
                    <div class="summary-label">Complete</div>
                </div>
                <div class="summary-card partial">
                    <div class="summary-value">{partial_stages}</div>
                    <div class="summary-label">Partial</div>
                </div>
                <div class="summary-card missing">
                    <div class="summary-value">{missing_stages}</div>
                    <div class="summary-label">Missing</div>
                </div>
                <div class="summary-card total">
                    <div class="summary-value">{total_stages}</div>
                    <div class="summary-label">Total Stages</div>
                </div>
            </div>
            {timing_html}
        </section>
        
        <section id="previews">
            <h2>Preview Images</h2>
            {preview_html}
        </section>
        
        <section id="stages">
            <h2>Stage Details</h2>
            {stages_html}
        </section>
        
        <section id="artifacts">
            <h2>Output Artifacts Index</h2>
            {self._render_artifacts_index()}
        </section>
        
        <footer>
            <p>Generated by Video-to-Terrain Pipeline Client Report Generator</p>
            <p>Pipeline Version: MVP | Config: {CFG.run_name}</p>
        </footer>
    </div>
    <script>
        {self._get_js()}
    </script>
</body>
</html>"""
        return html
    
    def _render_stage_card(self, stage: StageReport) -> str:
        """Render a single stage as an HTML card."""
        status_class = stage.status
        status_icon = {
            "complete": "✓",
            "partial": "◐",
            "missing": "✗",
            "error": "✗"
        }.get(stage.status, "?")
        
        # Files
        files_html = ""
        if stage.files_found:
            files_html = "<ul class='file-list'>" + "".join(f"<li>{f}</li>" for f in stage.files_found) + "</ul>"
        
        # Stats
        stats_html = ""
        if stage.stats:
            stats_html = "<table class='stats-table'>"
            for key, value in stage.stats.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                stats_html += f"<tr><td class='stat-key'>{key}</td><td class='stat-value'><pre>{value}</pre></td></tr>"
            stats_html += "</table>"
        
        # Errors
        errors_html = ""
        if stage.errors:
            errors_html = "<div class='errors'><strong>Issues:</strong><ul>" + "".join(f"<li>{e}</li>" for e in stage.errors) + "</ul></div>"
        
        # Previews
        previews_html = ""
        if stage.preview_images:
            previews_html = "<div class='stage-previews'>"
            for img in stage.preview_images:
                previews_html += f'<img src="{img}" alt="Stage {stage.stage} preview" class="preview-img">'
            previews_html += "</div>"
        
        return f"""
        <article class="stage-card {status_class}" data-stage="{stage.stage}">
            <header class="stage-header">
                <span class="stage-badge">{stage.stage}</span>
                <h3>{stage.name} <span class="status-icon">{status_icon}</span></h3>
                <span class="status-label {status_class}">{stage.status.upper()}</span>
            </header>
            <div class="stage-content">
                <div class="stage-section">
                    <h4>Files Found</h4>
                    {files_html or '<p class="empty">No files found</p>'}
                </div>
                <div class="stage-section">
                    <h4>Statistics</h4>
                    {stats_html or '<p class="empty">No statistics available</p>'}
                </div>
                {previews_html}
                {errors_html}
            </div>
        </article>
        """
    
    def _render_preview_gallery(self) -> str:
        """Render the preview images gallery."""
        if not self.preview_images:
            return "<p class='empty'>No preview images available. Run validation stage (J) to generate previews.</p>"
        
        html = "<div class='preview-gallery'>"
        for name, b64 in self.preview_images.items():
            html += f"""
            <div class="preview-item">
                <img src="{b64}" alt="{name} preview" class="gallery-img">
                <div class="preview-caption">{name.replace('_', ' ').title()}</div>
            </div>
            """
        html += "</div>"
        return html
    
    def _render_artifacts_index(self) -> str:
        """Render index of all output files."""
        all_files = []
        for root, dirs, files in os.walk(self.run_dir):
            for f in files:
                if f.endswith(('.json', '.ply', '.obj', '.npy', '.xml', '.py', '.png', '.jpg', '.mp4')):
                    full_path = Path(root) / f
                    rel_path = full_path.relative_to(self.run_dir)
                    size = full_path.stat().st_size
                    all_files.append((str(rel_path), size))
        
        if not all_files:
            return "<p class='empty'>No output files found.</p>"
        
        html = "<table class='artifacts-table'><thead><tr><th>File</th><th>Size</th></tr></thead><tbody>"
        for rel_path, size in sorted(all_files):
            size_str = self._format_size(size)
            html += f"<tr><td><code>{rel_path}</code></td><td>{size_str}</td></tr>"
        html += "</tbody></table>"
        return html
    
    def _format_size(self, size: int) -> str:
        """Format file size human-readable."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _get_css(self) -> str:
        """Return CSS styles."""
        return """
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; }
        header { border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; }
        header h1 { margin: 0 0 10px; color: #2c3e50; }
        .meta { display: flex; flex-wrap: wrap; gap: 20px; font-size: 0.9em; color: #666; }
        .meta span { background: #f8f9fa; padding: 5px 10px; border-radius: 4px; }
        
        h2 { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        h3 { color: #34495e; }
        h4 { color: #2c3e50; margin-top: 20px; }
        
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }
        .summary-card { background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; border-left: 4px solid #3498db; }
        .summary-card.complete { border-left-color: #27ae60; }
        .summary-card.partial { border-left-color: #f39c12; }
        .summary-card.missing { border-left-color: #e74c3c; }
        .summary-card.total { border-left-color: #9b59b6; }
        .summary-value { font-size: 2.5em; font-weight: bold; color: #2c3e50; }
        .summary-label { font-size: 0.9em; color: #666; text-transform: uppercase; }
        
        .preview-gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .preview-item { background: #f8f9fa; border-radius: 8px; overflow: hidden; }
        .gallery-img { width: 100%; height: auto; display: block; }
        .preview-caption { padding: 10px; text-align: center; font-weight: 500; color: #555; background: white; }
        
        .stage-card { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }
        .stage-card.complete { border-left: 4px solid #27ae60; }
        .stage-card.partial { border-left: 4px solid #f39c12; }
        .stage-card.missing { border-left: 4px solid #e74c3c; }
        .stage-card.error { border-left: 4px solid #e74c3c; }
        
        .stage-header { background: #f8f9fa; padding: 15px 20px; border-bottom: 1px solid #e0e0e0; display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
        .stage-badge { background: #3498db; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9em; }
        .stage-header h3 { margin: 0; font-size: 1.2em; }
        .status-icon { font-size: 1.2em; }
        .status-label { padding: 4px 10px; border-radius: 4px; font-size: 0.8em; font-weight: bold; text-transform: uppercase; }
        .status-label.complete { background: #d5f5e3; color: #1e8a49; }
        .status-label.partial { background: #fef9e7; color: #b7950b; }
        .status-label.missing { background: #fadbd8; color: #c0392b; }
        .status-label.error { background: #fadbd8; color: #c0392b; }
        
        .stage-content { padding: 20px; }
        .stage-section { margin-bottom: 20px; }
        .stage-section h4 { margin-bottom: 10px; font-size: 1em; color: #555; }
        
        .file-list { list-style: none; padding: 0; margin: 0; }
        .file-list li { background: #f8f9fa; padding: 8px 12px; margin: 5px 0; border-radius: 4px; font-family: monospace; font-size: 0.9em; }
        
        .stats-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        .stats-table td { padding: 8px 12px; border-bottom: 1px solid #eee; vertical-align: top; }
        .stat-key { font-weight: 600; color: #555; width: 30%; background: #f8f9fa; }
        .stat-value pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-family: inherit; font-size: 0.9em; }
        
        .stage-previews { margin: 20px 0; }
        .preview-img { max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #ddd; margin-right: 10px; }
        
        .errors { background: #fdf2f2; border: 1px solid #fadbd8; border-radius: 4px; padding: 15px; color: #c0392b; }
        .errors ul { margin: 10px 0 0; padding-left: 20px; }
        
        .empty { color: #999; font-style: italic; margin: 0; }
        
        .artifacts-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
        .artifacts-table th, .artifacts-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
        .artifacts-table th { background: #f8f9fa; font-weight: 600; }
        .artifacts-table code { background: #f8f9fa; padding: 2px 6px; border-radius: 3px; }
        
        footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #999; text-align: center; font-size: 0.85em; }
        
        /* Collapsible stages */
        .stage-header { cursor: pointer; }
        .stage-header:hover { background: #f0f0f0; }
        .stage-content { display: block; }
        .stage-card.collapsed .stage-content { display: none; }
        .stage-header::after { content: '▼'; float: right; transition: transform 0.2s; }
        .stage-card.collapsed .stage-header::after { transform: rotate(-90deg); }
        
        @media (max-width: 768px) {
            .container { padding: 15px; }
            .meta { flex-direction: column; gap: 10px; }
            .summary-grid { grid-template-columns: repeat(2, 1fr); }
        }
        """
    
    def _get_js(self) -> str:
        """Return JavaScript for interactivity."""
        return """
        document.addEventListener('DOMContentLoaded', function() {
            // Make stage cards collapsible
            document.querySelectorAll('.stage-header').forEach(function(header) {
                header.addEventListener('click', function() {
                    this.closest('.stage-card').classList.toggle('collapsed');
                });
            });
            
            // Initially collapse all except first
            document.querySelectorAll('.stage-card:not(:first-child)').forEach(function(card) {
                card.classList.add('collapsed');
            });
            
            // Keyboard navigation
            document.addEventListener('keydown', function(e) {
                if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                    e.preventDefault();
                    var cards = document.querySelectorAll('.stage-card');
                    var current = document.activeElement.closest('.stage-card');
                    var index = Array.from(cards).indexOf(current);
                    if (index >= 0) {
                        var next = index + (e.key === 'ArrowDown' ? 1 : -1);
                        if (next >= 0 && next < cards.length) {
                            cards[next].querySelector('.stage-header').focus();
                        }
                    }
                }
            });
        });
        """


def main():
    """Main entry point."""
    run_dir = CFG.paths.run_dir
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1
    
    generator = ClientReportGenerator(run_dir)
    output_path = generator.run()
    
    print(f"\nReport generated: {output_path}")
    print(f"Open in browser: file://{output_path.absolute()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
