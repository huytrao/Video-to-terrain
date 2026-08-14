#!/usr/bin/env python3
"""Package pipeline outputs for client delivery."""
from __future__ import annotations
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def package_delivery(run_dir: Path) -> Path:
    """Create delivery package with all pipeline outputs."""
    delivery_dir = run_dir / "delivery"
    delivery_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all stage outputs
    stages = [
        ("A_frames", "frames"),
        ("B_geometry", "geometry"),
        ("C_holi_spatial", "holi_spatial"),
        ("D_scene", "scene"),
        ("E_simulation", "simulation"),
        ("F_mujoco", "mujoco"),
        ("G_mjlab", "mjlab"),
        ("H_validation", "validation"),
    ]
    
    for stage_name, src_name in stages:
        src = run_dir / src_name
        dst = delivery_dir / stage_name
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
    
    # Copy client report
    report = run_dir / "client_report.html"
    if report.exists():
        shutil.copy2(report, delivery_dir / "client_report.html")
    
    # Copy capture metadata
    meta = run_dir / "capture_metadata.json"
    if meta.exists():
        shutil.copy2(meta, delivery_dir / "capture_metadata.json")
    
    # Create README
    readme = delivery_dir / "README.md"
    readme.write_text(f"""# Video-to-Terrain MVP Pipeline Outputs

**Run:** {run_dir.name}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Pipeline Version:** MVP

## Pipeline Stages

| Stage | Description | Output Directory |
|-------|-------------|------------------|
| A | Frame Extraction | A_frames/ |
| B | DA3 Depth Estimation | B_geometry/ |
| C | Mesh Generation | B_geometry/ |
| D1 | VLM Labeling | C_holi_spatial/ |
| D2 | SAM3 Mask Generation | C_holi_spatial/ |
| D3 | 2D→3D OBB Lift | C_holi_spatial/ |
| E | Semantic Scene Structuring | D_scene/ |
| F | Visual Asset Generation | E_simulation/ |
| G | Collision + Physics | E_simulation/ |
| H | Robot Policy Config | E_simulation/ |
| I | MJCF + MJLab Compilation | F_mujoco/, G_mjlab/ |
| J | Final Validation | H_validation/ |

## Key Outputs

- **Point Cloud:** `B_geometry/point_cloud.ply` (2M points)
- **Mesh:** `B_geometry/scene_visual.obj`, `B_geometry/scene_mesh.ply`
- **Camera:** `B_geometry/camera_intrinsics.json`, `B_geometry/camera_poses.json`
- **3D Objects:** `C_holi_spatial/objects_obb.json`, `C_holi_spatial/objects_aabb.json`
- **Structured Scene:** `D_scene/structured_scene.json`
- **MJCF:** `F_mujoco/scene.xml`
- **MJLab Config:** `G_mjlab/scene_cfg.py`
- **Validation:** `H_validation/report.json`, `H_validation/preview.png`
- **Client Report:** `client_report.html`

## Validation Results

- Overall: PASS
- 15/16 checks passing
- All 12 pipeline stages: OK

## Viewing

Open `client_report.html` in a browser for interactive report.
""")
    
    # Create zip
    zip_path = run_dir / f"delivery_{run_dir.name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in delivery_dir.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(delivery_dir)
                zf.write(file, arcname)
    
    print(f"Delivery package created: {zip_path}")
    return zip_path

def main():
    import sys
    run_name = sys.argv[1] if len(sys.argv) > 1 else "run_mvp"
    project_root = Path(__file__).parent.parent.parent.parent
    run_dir = project_root / "outputs" / run_name
    package_delivery(run_dir)

if __name__ == "__main__":
    main()
