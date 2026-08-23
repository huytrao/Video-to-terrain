# 📊 Videos_to_3D Architecture & Dataflow Graphs

This document provides visual diagrams (Mermaid graphs) illustrating the architecture, mathematical coordinate bridges, dataflow pipelines, and infrastructure mechanics of the `Videos_to_3D` project.

---

## 1. 🔄 End-to-End Pipeline Dataflow (Stage A $\to$ Stage J)

```mermaid
flowchart TD
    classDef input fill:#2b3a4a,stroke:#4a90e2,stroke-width:2px,color:#fff;
    classDef stage fill:#1e293b,stroke:#38bdf8,stroke-width:1.5px,color:#fff;
    classDef model fill:#312e81,stroke:#818cf8,stroke-width:1.5px,color:#fff;
    classDef export fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef sim fill:#701a75,stroke:#f472b6,stroke-width:2px,color:#fff;

    Vid["📹 Raw Input Video (MP4 / ROS2 DB3)"]:::input
    
    subgraph S_A ["Stage A: Ingestion & Poses"]
        Frames["Frame Extraction (1080p RGB)"]:::stage
        PosePool["Focal Length Pooling & Pose Consensus"]:::stage
    end

    subgraph S_B_C ["Stage B & C: Geometry Reconstruction"]
        DA3["Depth-Anything-3 (Native Metric Depth)"]:::model
        PGSR["Planar 3DGS (PGSR Optimization)"]:::model
        DepthStats["Depth Statistics & Quality Derivation (voxel, d_fuse)"]:::stage
    end

    subgraph S_D ["Stage D: Volumetric Fusion"]
        TSDF["Adaptive TSDF Fusion (Voxel Grid)"]:::stage
        MeshExt["Marching Cubes & Clean Mesh Extraction"]:::stage
    end

    subgraph S_E_F ["Stage E & F: Semantic Perception & 3D Lifting"]
        VLM["VLM Multi-View Object Voting (Nemotron / Groq)"]:::model
        SAM3["SAM3 2D Instance Segmentation"]:::model
        Lift3D["Lift3D: 2D Masks + Depth -> 3D Bounding Boxes"]:::stage
    end

    subgraph S_G ["Stage G: Scene Decomposition"]
        RANSAC["Floor, Ceiling & Wall Plane RANSAC"]:::stage
        Decomp["Object Bounding Box & Hull Decomposition"]:::stage
    end

    subgraph S_H ["Stage H: Multimodal Exporters"]
        ExpPLY["Dense Point Cloud (XYZRGB .ply)"]:::export
        ExpMesh["Watertight Collision Mesh (.ply/.obj)"]:::export
        ExpHmap["Terrain Heightfield (.png/.npy)"]:::export
        ExpGLB["Interactive 3D Web Scene (.glb)"]:::export
        ExpURDF["Simulation URDF & Layout JSON"]:::export
    end

    subgraph S_I_J ["Stage I & J: Simulation & Validation"]
        MJCFGen["make_unitree_scene.py (MuJoCo XML Assembler)"]:::sim
        RobotModel["Unitree Robot Model (vr_h3_2 / vr_m3_1)"]:::input
        PhysicsAudit["validate_unitree_scene.py (Settling & Penetration Audit)"]:::sim
        RaytraceRender["render_unitree_scene.py (Headless 360° Orbit MP4)"]:::sim
    end

    Vid --> Frames
    Frames --> PosePool
    
    Frames --> DA3
    PosePool --> DA3
    Frames --> PGSR
    PosePool --> PGSR
    
    DA3 --> DepthStats
    DepthStats --> TSDF
    PGSR --> TSDF
    TSDF --> MeshExt
    
    Frames --> VLM
    VLM --> SAM3
    DA3 --> Lift3D
    SAM3 --> Lift3D
    
    MeshExt --> RANSAC
    Lift3D --> Decomp
    RANSAC --> Decomp
    
    Decomp --> ExpPLY
    Decomp --> ExpMesh
    Decomp --> ExpHmap
    Decomp --> ExpGLB
    Decomp --> ExpURDF
    
    ExpHmap & ExpMesh & ExpURDF & RobotModel --> MJCFGen
    MJCFGen --> PhysicsAudit
    PhysicsAudit --> RaytraceRender
```

---

## 2. ☁️ Remote Kaggle GPU Hybrid Bridge Infrastructure

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Engineer / Local AI Agent
    participant Local as Local Host (macOS / Linux)
    participant Proxy as Jupyter Proxy Bridge (HTTPS / WSS)
    participant Kaggle as Kaggle GPU Runtime (RTX PRO 6000 / A100)
    participant PreStaged as Pre-Mounted Kaggle Datasets (Offline)

    Dev->>Local: Run ./scripts/kaggle/kx.sh ping
    Local->>Proxy: Authenticate with JWT Token (.env)
    Proxy-->>Local: HTTP 200 OK (Bridge Active)

    Dev->>Local: Run ./scripts/kaggle/kstage.sh --upload-code
    Local->>Local: Zip src/v3d & scripts/
    Local->>Proxy: Parallel Chunk Upload via Contents API
    Proxy->>Kaggle: Write code.zip to /kaggle/working/

    Local->>Proxy: WebSocket: Execute kernel_unpack_code.py
    Kaggle->>Kaggle: Extract codebase into runtime

    Note over Kaggle,PreStaged: Kernel has NO internet access
    Kaggle->>PreStaged: Link Pre-staged DA3 / PGSR weights & wheels
    Kaggle->>Kaggle: pip install --no-index from wheelhouse

    Dev->>Local: Run ./scripts/kaggle/kx.sh run python scripts/kernel/kernel_stage.py --stage all
    Local->>Proxy: Dispatch Stage Execution Command (WSS Terminal)
    Proxy->>Kaggle: Run Ingest -> DA3 -> PGSR -> TSDF -> Decompose -> Export
    Kaggle->>Kaggle: Generate outputs/ and delivery.tar.gz

    Dev->>Local: Run python scripts/kaggle/deliver.py
    Local->>Proxy: Download delivery.tar.gz
    Proxy-->>Local: Stream delivery archive
    Local->>Local: Unpack to outputs-smallroom/ (Ready for MuJoCo)
```

---

## 3. 📐 Coordinate Frame Transformations & Mathematical Invariants

```mermaid
graph LR
    subgraph OpenCV ["📸 OpenCV Camera Frame"]
        direction TB
        CV_X["+X : Right"]
        CV_Y["+Y : Down"]
        CV_Z["+Z : Forward"]
        CV_Desc["Used in: Raw Video, Frames, COLMAP"]
    end

    subgraph OpenGL ["🌐 OpenGL / NeRF Frame"]
        direction TB
        GL_X["+X : Right"]
        GL_Y["+Y : Up"]
        GL_Z["-Z : Forward"]
        GL_Desc["Used in: DA3, 3DGS, PGSR, WebGL"]
    end

    subgraph MuJoCo ["🤖 MuJoCo / ROS World Frame"]
        direction TB
        MJ_X["+X : Forward"]
        MJ_Y["+Y : Left"]
        MJ_Z["+Z : Up"]
        MJ_Desc["Used in: Robot URDF, RL Policy, MJCF"]
    end

    OpenCV -- "c2w_gl = c2w_cv * diag(1, -1, -1, 1)" --> OpenGL
    OpenGL -- "Gravity RANSAC + Axis Mapping" --> MuJoCo
```

---

## 4. 🧩 Package & Module Architecture (`src/v3d/`)

```mermaid
classDiagram
    class Ingest {
        +extract_frames(video_path, fps)
        +estimate_poses(frames)
        +consensus_intrinsic(intrinsics)
    }

    class Geometry {
        +KnobPolicy
        +derive(depth, w2c, K)
        +tsdf_fusion(depths, poses, voxel_size)
        +marching_cubes(tsdf_grid)
    }

    class Semantic {
        +vlm_multiview_vote(frames, prompts)
        +reconcile_labels(candidate_votes)
    }

    class Scene {
        +ransac_shell_planes(mesh_vertices)
        +decompose_objects(mesh, boxes)
        +room_from_mesh(objects_json)
    }

    class Export {
        +export_glb(mesh, boxes, output_path)
        +export_heightfield(mesh, cell_size)
        +export_collision_hulls(objects)
    }

    class Validate {
        +audit_physics_settling(mjcf_path)
        +check_stage_j_geometry(layout_json)
        +verify_pose_determinant(rot_matrix)
    }

    Ingest --> Geometry : Camera Poses & Frames
    Geometry --> Scene : Fused Mesh & Point Cloud
    Semantic --> Scene : 3D Oriented Bounding Boxes
    Scene --> Export : Decomposed Scene Graph
    Export --> Validate : MuJoCo XML & Heightmap
```
