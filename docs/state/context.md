# Master Project Context & Technical Reference (`docs/state/context.md`)

> **Repository:** `Videos_to_3D` (`Full-pipeline-scan-terrain`)  
> **Target Run:** `small_factory` (`input/Small-factory/small_factory_final.mp4`)  
> **Core Objective:** End-to-end reconstruction from 4K industrial/indoor video scans to high-fidelity, robot-ready 3D digital twins (Holi-Spatial 3DGS/TSDF + Hunyuan3D-2/2mv completion + MuJoCo / Unitree simulation + GIS/CAD export).

---

## 1. Executive Summary & Pipeline Architecture

The pipeline processes unconstrained video scans into structured 3D environments ready for robotics, gaming, and spatial computing:

```text
                      4K Video Scan (MP4 / Image Sequence)
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   STAGE 1: Ingest & Camera Poses  │
                     │  (Parallax-aware Keyframing, DA3) │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │    STAGE 2: Metric Depth & PCD    │
                     │  (Depth-Anything-3 @ 1008 res)    │
                     │  Output: 8M Point Cloud + Depth   │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │    STAGE 3: 3D Gaussian Splat     │
                     │ (PGSR Plane-Regularized Surface)  │
                     │ 69k steps, depth-loss supervision │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │    STAGE 4: Dense TSDF Mesh       │
                     │ (Adaptive Voxel Derivation,       │
                     │  Raycasted Depth, Marching Cubes) │
                     └─────────────────┬─────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
      ┌───────────────────────────┐         ┌───────────────────────────┐
      │ STAGE 5A: Semantics & VLM │         │ STAGE 5B: Geometric Split │
      │  (nvdiffrast Mesh2Mask,   │         │ (RANSAC Shell Extraction, │
      │   VLM Voting, SAM 3,      │         │  Half-space Removal,      │
      │   Lift3D Bounding Boxes)  │         │  Connected Components)    │
      └─────────────┬─────────────┘         └─────────────┬─────────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ STAGE 6: Object Completion Bridge │
                     │  (HoliSpatial → Hunyuan3D-2mv)    │
                     │  - Quality Gate: 8-Azimuth Cover  │
                     │  - Adapter A: A ∪ B Union Mask    │
                     │  - Hunyuan3D-DiT Multi-View Shape │
                     │  - Adapter B: Multi-view IoU/Depth│
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ STAGE 7: Multi-Modal Deliverables │
                     │ - 1.5M/39M Decimated GLB/OBJ/STL  │
                     │ - 512×510 Heightfield (ASC/PNG)   │
                     │ - URDF Collision Tree + OBB Boxes │
                     │ - MuJoCo Scene + Unitree Robot    │
                     │ - 10 Rendered Audit Videos (MP4)  │
                     └───────────────────────────────────┘
```

---

## 2. Mathematical Invariants & Coordinate Conventions

The system bridges three distinct coordinate systems. Conversions must follow strict matrix transforms:

| Coordinate System | Axes Definition | Usage Domains |
| :--- | :--- | :--- |
| **OpenCV** | $+X$ Right, $+Y$ Down, $+Z$ Forward | Video frames, raw image plane, COLMAP, DA3 depth |
| **OpenGL / NeRF** | $+X$ Right, $+Y$ Up, $-Z$ Forward | 3DGS, NeRF, WebGL/Three.js, GLB viewers |
| **MuJoCo / ROS** | $+X$ Forward, $+Y$ Left, $+Z$ Up | URDF, MJCF simulation, Heightmaps, RL World Frame |

### Transformation Math
To transform OpenCV Camera-to-World ($c2w_{cv}$) to OpenGL Camera-to-World ($c2w_{gl}$):
$$c2w_{gl} = c2w_{cv} \cdot \operatorname{diag}(1, -1, -1, 1)$$

- **Determinant Invariant**: Rotation matrices must satisfy $\det(R) = +1.0 \pm 10^{-5}$. A negative determinant signals a mirrored coordinate frame and must fail immediately.
- **Intrinsic Matrix Scaling**: When resizing images $(W_{orig}, H_{orig}) \to (W_{proc}, H_{proc})$:
  $$f_x' = f_x \cdot \frac{W_{proc}}{W_{orig}}, \quad f_y' = f_y \cdot \frac{H_{proc}}{H_{orig}}, \quad c_x' = c_x \cdot \frac{W_{proc}}{W_{orig}}, \quad c_y' = c_y \cdot \frac{H_{proc}}{H_{orig}}$$
- **Scale Mode Invariant**: Monocular video cannot recover absolute physical scale without external metric references. Scale is labeled `unit="da3_estimate"` with `v3d_scale_mode="assumed"`. One physical tape measurement (`--metres-per-unit`) rescales all distances, masses, and forces.

---

## 3. Quality Rails & Statistically Derived Knobs

All mesh and fusion parameters in `src/v3d/geometry/quality.py` are derived dynamically from scene depth statistics, preventing bedroom-default hardcodes on large industrial halls:

1. **Voxel Size ($v$)**:
   Derived from median scene depth $d_{med}$, focal length $f$, and pixel footprint:
   $$v = \operatorname{clamp}\left(1.75 \cdot \frac{d_{med}}{f}, \; \frac{D_{scene}}{6000}, \; \frac{D_{scene}}{2500}\right)$$
   *(Memory guard guarantees $\le 6000$ voxels across scene diagonal $D_{scene}$).*
2. **Fusion Truncation Distance ($d_{fuse}$)**:
   $$d_{fuse} = P_{98}(d) \cdot 1.05$$
3. **Mask Distance ($d_{mask}$)**:
   $$d_{mask} = 1.5 \cdot d_{fuse}$$
4. **PGSR Schedule Scaling**:
   Tied to frame count $N$ to maintain constant gradient steps per image:
   $$\text{max\_steps} = \operatorname{round}\left(30000 \cdot \frac{N}{261}\right), \quad \text{densify\_until} = \frac{\text{max\_steps}}{2}$$
   `position_lr_max_steps` is strictly matched to `max_steps`.

---

## 4. Hardware & Cloud Execution Environment

Workloads execute across three synchronized environments:

| Node | Hardware Spec | Connectivity | Responsibilities |
| :--- | :--- | :--- | :--- |
| **RUN (Kaggle)** | NVIDIA RTX PRO 6000 Blackwell (95.6 GiB VRAM, `sm_120`), 30 GB RAM | **Offline (No Internet)** | All GPU compute: DA3, PGSR, TSDF, nvdiffrast, SAM 3, Hunyuan3D-2mv |
| **BUILD (Kaggle)** | Multi-core CPU, 30 GB RAM | **Online** | `pip download`, Kaggle Dataset uploads, mirror syncing |
| **Laptop / Local** | Local Dev Machine | **Online** | Git control, CLI orchestration (`kx.sh`), MuJoCo playback |

- **Transfer Speeds**: BUILD $\to$ RUN via relay is **~23.8 MB/s**; Laptop $\to$ Kernel is **~4.6 MB/s**.
- **PyTorch/CUDA Pins**: PyTorch `2.10.0+cu128`, NumPy `2.0.2`, CUDA `12.8`.
- **Memory Partitions**: `/kaggle/working` is limited to 20 GB. Heavy model weights and temporary point clouds/tars live in `/tmp` (1.1 TB).

---

## 5. Critical Upstream Bug Fixes & Architectural Lessons

| Bug / Defect Identified | Root Cause | Implemented Solution |
| :--- | :--- | :--- |
| **Silent PGSR Depth Supervision Loss** | `cameras.py` searched for depth in an arbitrary machine directory (`DptV3/data/...`) and silently fell back to `None` without depth L1 loss. | Patched path resolution; `kernel_pgsr.py` now asserts depth existence before launch. Gaussians increased from 616k to 2.9M. |
| **Chunked DA3 Device Crash** | `chunk_inference` returned GPU tensors concatenated via `np.concatenate`, throwing a device conversion error. | Added explicit `.cpu()` transfer in `CHUNK_TO_HOST` patch, freeing VRAM incrementally. |
| **Native 1920 Res Scale Inconsistency** | Chunking 600 views into 20 chunks @ 1920 res produced 6–27% scale drift due to per-chunk Umeyama fits. | Adopted **1008 res with 200 views/chunk** (3 chunks), delivering 4.1× paper resolution at 0.11% scale consistency. |
| **Open3D GLB Corrupt Export** | Open3D writes invalid glTF buffer lengths that fail re-reading. | Use Open3D strictly for Quadric Decimation; use `trimesh` for GLB exports and verify round-trips. |
| **Heightmap Undersampling Holes** | Heightfield sampled points proportional to triangle count, yielding 58% empty pit cells. | Fixed to grid-relative sampling ($16\times$ samples per cell), slashing holes to 0.18%. |
| **SAM 3 Checkpoint Architecture** | SAM 3.1 multiplex lacked `convs.3` weights; official repo vendoring missed `sam3/train/`. | Authenticated gated `sam3.pt` checkpoint (1465 tensors) and rebuilt `.gz` tokenizer vocab. |
| **nvdiffrast sm_120 Compilation** | Blocked across PyTorch 2.10 on Blackwell architecture. | Built native CUDA rasterizer context on sm_120 in 27s. |

---

## 6. Geometric Shell Extraction & Object Decomposition

Implemented in `src/v3d/scene/decompose.py` for model-free object separation:

1. **RANSAC Shell Plane Fitting**:
   - Identifies Dominant Floor (verticality $> 0.99$, span $> 95\%$), Ceiling, and Walls.
   - Guarded by Boundary Threshold (`max_shell_edge_frac = 0.10`) to prevent deleting interior machine guard panels.
2. **Half-Space Removal**:
   - Clears wall thickness artifacts. Guarded so single planes remove $\le 45\%$ and total removal $\le 90\%$.
3. **Connected Component Clustering**:
   - Extracts isolated 3D meshes for distinct machines/crates without requiring prior VLM detection.

---

## 7. HoliSpatial $\leftrightarrow$ Hunyuan3D-2mv Completion Bridge

When video coverage misses the backside of equipment, the pipeline invokes Hunyuan3D-2mv:

```text
Holi 3DGS Scene ──► Quality Gate (8-Azimuth Bins) ──► Coverage < 50%
                                                            │
                                                            ▼
                                               Adapter A: A ∪ B Union Mask
                                               (SAM3 Mask ∪ Rembg Alpha Matte)
                                                            │
                                                            ▼
                                               Hunyuan3D-2mv DiT Multi-View
                                               (Front / Left / Back / Right)
                                                            │
                                                            ▼
                                               Adapter B: 4-Candidate Alignment
                                               (Yaw Search, Silhouette IoU, Depth)
                                                            │
                                                            ▼
                                               Hybrid Reconstructed 3D Scene
```

- **Exclusion Filter**: Structural architectural surfaces (`floor`, `wall`, `ceiling`, `pipe`, `wire`, `label`, `sticker`) are skipped.
- **Smart View Selection**: Picks non-boundary, highest-sharpness frames centered on 518×518 pure white canvases.
- **Multi-View Verification**: Renders candidate meshes back to Holi camera views; scores via:
  $$\text{Score} = \text{IoU}_{\text{silhouette}} - \lambda \cdot \text{Error}_{\text{depth}}$$

---

## 8. Benchmark Deliverables: Small Factory Scene

- **Input:** 600 keyframes from 4K industrial scan (`small_factory_final.mp4`).
- **Scene Dimensions:** $6.01 \times 3.73 \times 6.35$ DA3 units (Diagonal: 9.51 units, Height: 2.307 units).
- **Point Cloud & Splats:** 8,000,000 points; 2,902,811 Gaussians at 69,000 iterations (PSNR 24.80 dB).
- **Mesh:** 39.8M TSDF triangles (decimated to 1.5M GLB/OBJ).
- **Extracted Assets:** 43 decomposed object meshes + 4 doors + 9 shell planes.
- **Terrain:** $512 \times 510$ grid (0.01183 units/cell), $0.18\%$ hole rate, 16-bit PNG heightfield + top-down photo texture.
- **Simulation Validation:** Unitree `vr_h3_2` humanoid spawned on optimal flat interior floor (clearance 0.848m, peak $|qvel| = 8.87$, heightfield RMS 5.16mm).
- **Dataset Archive:** Kaggle dataset `traoanhuy/v3d-small-factory-scene` (198 files, 4.62 GB).

---

## 9. Rendered Visualizations & Verification Suite (10 MP4s)

1. `01_depth.mp4` — Per-frame DA3 metric depth streams.
2. `02_point_cloud_turntable.mp4` — 360° orbit of 8M-point cloud.
3. `03_mesh_turntable.mp4` — Textured TSDF surface orbit.
4. `04_height_map.mp4` — 2D top-down elevation heatmap.
5. `05_slope_deg.mp4` — Ground terrain slope analysis.
6. `06_step_height.mp4` — Obstacle traversability step height map.
7. `07_mesh_guided_masks.mp4` — `nvdiffrast` rendered silhouette overlays.
8. `08_sam3_masks.mp4` — SAM 3 zero-shot segmentation tracking.
9. `09_semantic_boxes.mp4` — 3D oriented bounding boxes (OBBs) in space.
10. `10_mujoco_rollout.mp4` — Physics settling and Unitree robot standing simulation.

---

## 10. Operational Rules & Pipeline Runbook

- **Never Pipe Commands Through Grep**: Remote execution wrappers might mask fatal stack traces while printing standard status codes.
- **Kaggle 24MB API Upload Limit**: Always upload large archives via chunked scripts (`scripts/kaggle/kupload.py`) with remote SHA256 validation.
- **Default Timeouts**: Set `kx.sh exec --timeout 300+` to prevent default 600s process kills during heavy training iterations.
- **File System Cleanliness**: Strip macOS `._*` AppleDouble resource forks before unpacking tars inside Linux containers.
- **Regressions Testing**: All geometry changes must pass:
  ```bash
  python tests/test_quality_knobs.py
  python tests/test_pose_conventions.py
  ```