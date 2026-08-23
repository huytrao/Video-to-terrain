# scripts/ — index

The subdirectory says **where a script runs**, which is the distinction that
matters most here: a script in `kernel/` executes on a Kaggle GPU with no
internet and no local filesystem, while one in `kaggle/` runs on this laptop and
talks to that kernel over an HTTP proxy.

Scripts in `scripts/<group>/` sit **two** levels below the repo root, so they
resolve it as `Path(__file__).resolve().parents[2]` (or `dirname .../../..` in
shell). Getting that depth wrong is the failure mode this layout invites.

---

## kernel/ — runs inside the Kaggle GPU kernel

Uploaded by `kx.sh run`, which sends a **local** path to `_run/<basename>` and
executes it there. The grouping is local only; remote paths are unaffected.

| Script | Purpose |
| :--- | :--- |
| `kernel_stage.py` | Master stage dispatcher (`--stage da3\|pgsr\|mesh\|...`) |
| `kernel_ingest.py` | Extracts frames, pools focal lengths, builds camera poses |
| `kernel_da3_native.py` | DA3 metric depth at native sensor resolution |
| `kernel_da3_maxres.py` | Max-resolution DA3 with adaptive memory scaling |
| `kernel_da3_depth.py` | Standalone DA3 depth with confidence-mask filtering |
| `kernel_calibrate_da3_res.py` | Measures the DA3 VRAM/resolution boundary |
| `kernel_verify_native.py` | Validates native-resolution depth and cloud density |
| `kernel_pgsr.py` | Planar 3D Gaussian Splatting training |
| `kernel_mesh.py` | TSDF fusion and mesh extraction |
| `kernel_chunk_scale.py` | Chunk scaling and boundary handling for large scans |
| `kernel_cutaway.py` | Mesh sectioning, floor/wall boundary extraction |
| `kernel_semantics.py` | VLM object tagging and multi-view label voting |
| `kernel_outputs.py` | Master exporter: clouds, meshes, heightmaps, URDF, JSON |
| `kernel_export_formats.py` | Converts geometry to PLY/OBJ/GLB/heightfield |
| `kernel_glb.py` | Combined GLB scene with bounding boxes |
| `kernel_mujoco_video.py` | Headless MuJoCo render and video on GPU |
| `kernel_assemble_delivery.py` | Packages the delivery archive and manifest |
| `kernel_prepare_run.py` | Creates workspace dirs, checks stage prerequisites |
| `kernel_datasets.py` | Mounts pre-staged offline Kaggle datasets |
| `kernel_install_deps.py` | Installs offline wheels inside the kernel |
| `kernel_unpack_code.py` | Unpacks the uploaded code archives |
| `kernel_fetch_scene.py` | Fetches a ScanNet++ scene and its ground-truth boxes |
| `kernel_inventory.py` | GPU / CUDA / VRAM capability audit |
| `kernel_probe.py`, `kernel_probe2.py` | Low-level diagnostics and network probes |

## kaggle/ — runs locally, talks to Kaggle

| Script | Purpose |
| :--- | :--- |
| `kx.sh` | Retrying front-end to the executor. Start here: `kx.sh ping` |
| `kenv.sh` | Greps proxy URLs out of `.env` without sourcing it |
| `kstage.sh` | Long-running stage control: `start`, `follow`, `status`, `stop` |
| `deploy_code.sh` | Tars `src/v3d` + the paper checkout and uploads both |
| `deploy_kaggle.sh` | Full deployment wrapper with retry, resume, JWT expiry checks |
| `kupload.py` | Chunked upload with SHA-256 verified on the kernel |
| `deliver.py` | Downloads, verifies and extracts the delivery package |
| `pull_robot_sim.py` | Pulls rendered MuJoCo videos and scene layouts |
| `relay_push.py` / `relay_pull.py` | Move individual files across the proxy |
| `relay_hf_repo.py` | Syncs HuggingFace weights through the Kaggle bridge |
| `kaggle_dataset_push.py` | Pushes datasets via the official Kaggle API |
| `kaggle_upload_example.py` | Example upload script (gitignored: holds a key) |

## deploy/ — runs locally, deploys to an SSH GPU server

| Script | Purpose |
| :--- | :--- |
| `deploy_gpu.sh` | rsync + conda provisioning + DA3 run on a remote GPU box |
| `deploy_gpu.py` | Same flow via paramiko/fabric, with progress and retry |

> Both carry the server password as a literal default in the source. Move it to
> `.env` before this repo is shared any wider.

## sim/ — MuJoCo scene build, validation, render

| Script | Purpose |
| :--- | :--- |
| `make_room_scene.py` | **One command.** Room mesh + robot -> textured MJCF, gravity fixed, collider built, physics-checked, rendered |
| `play_policy.py` | Runs an exported RSL-RL ONNX policy in the room scene and reports whether the robot stands |
| `make_unitree_scene.py` | `scene_layout.json` + heightmap -> MJCF with a robot |
| `validate_unitree_scene.py` | Settling, penetration and contact-force audit |
| `render_unitree_scene.py` | Inspection stills and a 360-degree orbit mp4 |
| `stage_j_controls.py` | Control/RL policy compatibility against the ground |
| `stage_j_selftest.py` | Stage J scene geometry and bounds self-test |
| `selftest_decompose.py` | Object decomposition and floor-plane RANSAC self-test |

Reachable as `make room-scene`, `make room-play`, `make room-view`, `make sim-build`, `make sim-validate`, `make sim-render`.

Room scene method and traps: [../docs/guides/room-to-robot-scene.md](../docs/guides/room-to-robot-scene.md)

## adapter/ — HoliSpatial to Hunyuan3D, stages 01..10

Numbered by execution order. Gaps are stages not yet written.

| Script | Purpose |
| :--- | :--- |
| `01_normalize_holi_output.py` | Normalises HoliSpatial output into the adapter schema |
| `02_quality_gate_and_adapter_a.py` | Quality gate, then adapter A (reads `OUTPUT_C3` masks, writes RGBA 518 crops) |
| `03_resegment_obb_sam3.py` | **GPU.** Re-segments every view by prompting SAM 3 with the projected metric OBB |
| `04_run_hunyuan_shape.py` | Hunyuan3D-2 / 2mv DiT shape generation |
| `05_align_and_verify_adapter_b.py` | Aligns generated meshes back, then adapter B |
| `06_assemble_final_scene.py` | Assembles the final composed scene |
| `07_texture_bake_objects.py` | Bakes per-object textures |
| `09_run_images_to_3d_batch.py` | Batch images-to-3D on remote GPU |
| `10_robotics_manipulation_gate.py` | Robot-readiness gate, emits MJCF |

| `tools/make_contact_sheet.py` | Before/after review page for the adapter A crops |

Walkthrough: [../docs/guides/holispatial-hunyuan3d-adapter.md](../docs/guides/holispatial-hunyuan3d-adapter.md)
Segmentation method and runbook: [../docs/guides/obb-anchored-segmentation.md](../docs/guides/obb-anchored-segmentation.md)

## tools/ — standalone utilities

| Script | Purpose |
| :--- | :--- |
| `verify_geometry_setup.py` | CUDA / torch / PyG / v3d import health (`make verify`) |
| `bake_texture.py` | Projects RGB frames onto mesh UVs barycentrically |
| `export_blender.py` | Gravity-aligned GLB/OBJ export for Blender |
| `extract_video_from_db3.py` | Pulls video streams out of ROS2 `.db3` bags |
| `download_dl3dv.py` | Downloads the DL3DV-10K dataset at a chosen resolution |
| `setup_remote_hunyuan_gpu.py` | Provisions Hunyuan3D on a remote GPU |
| `nv_audit.sh` | Second-opinion audit through the NVIDIA NIM endpoint |

## patches/

Patches held against upstream dependencies until they land there. Currently:
`kaggle_executor-download-must-fail-loudly.patch`, which makes a failed download
exit non-zero instead of reporting success on a transfer that moved no bytes.
