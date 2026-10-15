# Documentation index

Four folders, split by what you want from a document rather than who wrote it.

| Folder | Answers | Changes |
| :--- | :--- | :--- |
| [architecture/](architecture/) | How is the system shaped? | Rarely |
| [guides/](guides/) | How do I run this? | When a procedure changes |
| [reference/](reference/) | What is the value / why is it that value? | When something is learned |
| [state/](state/) | Where is the current run? | Every session |

## architecture/ — how the system is shaped

| Document | Contents |
| :--- | :--- |
| [architecture-graph.md](architecture/architecture-graph.md) | **System Architecture & Technical Dataflow**: End-to-end pipeline stages (A–J), collision explosion resolution via RANSAC plane separation & footprint carving, 2.5D heightfield doorway wall-slab synthesis, real-to-sim validation with humanoid robot & RSL-RL policies, Kaggle offline GPU bridge sequence, coordinate frame chirality ($\det(R)=+1$), and package architecture (`src/v3d/`). |
| [architecture-critique.md](architecture/architecture-critique.md) | **Architecture Critique & Industrial Rework**: In-depth analysis of structural bottlenecks (scale provenance, missing manifest validation, feedback-free semantic chains, uncalibrated meshes) and blueprint for industrial factory digital twins. |

## guides/ — how to run a given thing

| Document | Contents |
| :--- | :--- |
| [pipeline-runbook.md](guides/pipeline-runbook.md) | Benchmark runbook for the Small Factory scene |
| [kaggle-deployment.md](guides/kaggle-deployment.md) | Robust Kaggle deployment: proxy expiry, retries, resume |
| [gpu-deployment.md](guides/gpu-deployment.md) | SSH GPU server provisioning and conda setup |
| [holispatial-hunyuan3d-adapter.md](guides/holispatial-hunyuan3d-adapter.md) | Step-by-step adapter pipeline (`scripts/adapter/`) |
| [room-to-robot-scene.md](guides/room-to-robot-scene.md) | `make room-scene`: room mesh + Unitree robot -> a textured MuJoCo scene with correct gravity, a real collider, and two physics checks |
| [obb-anchored-segmentation.md](guides/obb-anchored-segmentation.md) | Cutting objects out accurately for Hunyuan3D: prompting SAM 3 with the metric OBB instead of a text label, the six traps, and the runbook |
| [production-checklist.md](guides/production-checklist.md) | Production readiness gates |
| [text-to-terrain-integration.md](guides/text-to-terrain-integration.md) | Text-to-Terrain integration guide: Generative synthesis, 2.5D heightfield `<hfield>` physics, and co-spawning with reconstructed factories for robot stress-tests |

## reference/ — facts, and how they were learned

| Document | Contents |
| :--- | :--- |
| [output-quality.md](reference/output-quality.md) | What limits output quality, and the parameter behind each limit |
| [video-to-boxes.md](reference/video-to-boxes.md) | Mechanics of 2D-to-3D bounding box lifting |
| [lift3d-postprocess-history.md](reference/lift3d-postprocess-history.md) | How box post-processing evolved, and why |
| [bugs-found.md](reference/bugs-found.md) | Bug catalogue and regression-prevention record |
| [deb-technical.md](reference/deb-technical.md) | Transport-layer findings (chunking, truncation limits) |
| [audit-prompt-da3-resolution.md](reference/audit-prompt-da3-resolution.md) | Audit prompt for the DA3 resolution calibration |

## state/ — live run state

These change every session. Treat them as the current reading, not as a spec.

| Document | Contents |
| :--- | :--- |
| [SESSION_STATE.txt](state/SESSION_STATE.txt) | Stage completion and open gates, updated as stages land |
| [context.md](state/context.md) | Active execution context for the current run |
| [strategy-context.md](state/strategy-context.md) | Strategy for high-quality runs on the offline RTX 6000 |
| [input_prompt.txt](state/input_prompt.txt) | The prompt text the current run was framed with |

## Elsewhere

- [../README.md](../README.md) — project overview and quickstart
- [../AGENTS.md](../AGENTS.md) — rules and invariants for AI assistants
- [../scripts/README.md](../scripts/README.md) — what each executable does
