# Pipeline runbook — everything done, in order

A record of the actual run on pilot scene `00777c41d4`, with measured timings
and the state each step left behind. Written so the run can be resumed from any
point, or repeated on a new scene.

Companion documents: `bugs-found.md` (defects and their fixes),
`deb.technical` (outstanding work).

---

## 0. The three kernels

Proxy URLs expire when a session restarts; take fresh ones from each notebook's
⋮ → "Kaggle Jupyter Server" panel.

| Role | Hardware | Internet | Used for |
|---|---|---|---|
| **RUN** | RTX PRO 6000 Blackwell, 102 GB, sm_120 | no | every pipeline stage |
| **BUILD** | Tesla P100, sm_60 | yes | `pip download`, CUDA builds, HF downloads, Kaggle pushes |
| **FETCH** | CPU only | yes | relaying URLs |

BUILD cannot *run* torch on its own GPU (its torch supports sm_70–sm_120, not
sm_60) but has `nvcc` and the same Python 3.12 / torch 2.10.0+cu128 as RUN, so
it cross-compiles for sm_120 correctly. That combination is what makes the
whole offline setup possible.

Measured transfer rates, which decide where work should happen:

| Route | Rate |
|---|---|
| Kaggle kernel → Kaggle dataset | **87–100 MB/s** |
| Kernel → kernel (`relay_push.py`) | **35 MB/s** |
| HuggingFace → Kaggle kernel | 3.5 GB in **10 s** |
| Kernel → this laptop | **4.6 MB/s** |

The laptop is twenty times slower than any kernel-to-kernel path. Never route
bulk data through it.

---

## 1. Bootstrap

### 1.1 PGSR CUDA extensions (built on BUILD, run on RUN)

```bash
# BUILD: patch + build for sm_120
python3 -m v3d.geometry.pgsr_build <PGSR/submodules> --wheels wheels --arch 12.0
```

One source fix is applied first: `simple_knn.cu` uses `FLT_MAX` without
`#include <float.h>`, which CUDA 12.8 no longer supplies transitively.

Result: `simple_knn` and `diff_plane_rasterization` wheels, 76 s.
Verified on RUN — `distCUDA2` over 50k points returns finite distances, and
`diff_plane_rasterization` exports `rasterize_gaussians` plus its backward.

Both wheels plus the 120-wheel dependency set are on Kaggle as
**`traoanhuy/v3d-build-cache-sm120`** (3.58 GB). Attaching that dataset
replaces this entire step on a fresh session.

### 1.2 Python dependencies

`pip download` on BUILD, `relay_push.py` to RUN, `pip install --no-index`.

Two packages must be installed with `--no-deps`, and the reason matters:

- **`evo`** (DA3 needs it for `pose_align`) — a full resolve pulls numpy 2.5.2
  and pandas 3.0.5, which break the torch 2.10 build PGSR depends on.
- **`ftfy`** (official SAM3 needs it) — SAM3's `pyproject.toml` pins
  `numpy>=1.26,<2`, so a normal install *downgrades* numpy below 2 and takes
  torch with it.

After every install, check `numpy.__version__` is still 2.0.2 and
`torch.cuda.is_available()` is still True. Both were verified after each.

### 1.3 nvdiffrast (needed by mesh2mask)

Not on PyPI — clone `NVlabs/nvdiffrast` (22 MB) on BUILD, relay, then:

```bash
pip install --no-index --no-deps --no-build-isolation -e /tmp/nvd/nvd/nvdiffrast
```

`--no-build-isolation` is required; plain pip fetches build dependencies from
the network and fails offline. Its CUDA is JIT-compiled by torch at first use,
which RUN can do because it has nvcc.

No source patch is needed. Current nvdiffrast deprecates `RasterizeGLContext`
and routes it to `RasterizeCudaContext` internally, so the paper's code runs
unmodified and no EGL headers are involved. Both contexts rasterise an
identical test triangle on sm_120 — 1352 covered pixels each.

### 1.4 Model weights on RUN

| Model | Size | Location | How it got there |
|---|---|---|---|
| DA3NESTED-GIANT-LARGE | 6.76 GB | `/tmp/DA3NESTED-GIANT-LARGE` | `relay_hf_repo.py`, byte-verified |
| Qwen3-VL-8B (Holi-Spatial fine-tune) | 17.55 GB | `/tmp/qwen3vl8b` | `relay_hf_repo.py`, 17 files byte-verified |
| SAM 3.1 multiplex + official source | 3.31 GB | `/tmp/sam3new/sam3bundle` | relay, `RELAY_PUSH_OK` |
| SAM 3 (`sam3.pt`) | 3.45 GB | in progress | gated, needs an approved HF token |

`facebook/sam3` and `facebook/sam3.1` are both **manually gated**. The token in
`.env` gets 403 on both; the working token belongs to HF account `huytrao123`.

### 1.5 Scene data

`v3d.data.scannetpp.fetch_scene` pulls one ScanNet++ scene out of the
Holi-Spatial release by byte range — the shards are 14.4 GB and must never be
downloaded whole. Shards are ordered by scene id, so the right one is found by
binary search over first-member names, about ten range requests.

Pilot scene `00777c41d4` is 165.8 MB: 261 images at 1752×1168,
`transforms_undistorted.json` (244 posed frames, full intrinsics), COLMAP
`cameras/images/points3D.txt`. Ground truth is a separate 6.65 GB tar; its
per-scene JSON is 13 MB with **223 instances**.

---

## 2. Stages run, with results

All through `v3d.paper_runner`, which supplies local paths and stage ordering
and otherwise runs the paper's code unchanged.

### Stage A — DA3 depth and point cloud ✅

```python
from v3d.paper_runner import stage_da3
stage_da3('00777c41d4')          # rc 0, ~60 s
```

- 261 depth maps, 1168×1752 float32, **100% finite**, range **0.444–4.942 m**
- 504,026,496 points, downsampled to 4,000,000 → `pointcloud_da3.ply`, 60 MB

Required patch: `inference_da3_scannetppv2.py` hardcodes the HuggingFace repo
id. `apply_offline_patches()` rewrites it to the local directory.

### Stage B — PGSR / 3DGS training ✅

```python
stage_pgsr('00777c41d4')         # rc 0, 54 min 06 s
```

- 30,000 iterations, **PSNR 26.77** at 30k (25.03 at 7k), L1 0.0296
- **1,247,783 gaussians**, checkpoint 309.5 MB

Required patch: `PGSR/scene/dataset_readers.py` falls back to
`output_scannet/scannetppv2/0b031f3119/input.ply` — an absolute path from the
authors' machine pointing at a different scene — whenever `--ply_path` is
absent, and `3dgs_train.sh` never passes it. Both `train.py` and `render.py`
hit this; `render.py` has no flag to route around it, so the source is patched
to derive the cloud from the scene, matching what `readScannetV2SceneInfo`
already does correctly.

A dropped websocket during this stage does **not** kill it — the job is a
kernel subprocess. Check `ps -ef | grep train.py` before restarting anything.

### Stage C — mesh extraction and mesh-guided masks ✅

```python
stage_mesh('00777c41d4')         # rc 0, 139 s
```

- TSDF fusion over 261 frames, `extract_triangle_mesh`, 140,334 clusters
- **10,706,756 → 9,517,867 vertices**, 18,543,482 triangles, 498 MB
- mesh2mask: **261/261 masks** at ~30 it/s

For comparison, the abandoned v2t pipeline produced 352,452 vertices by running
Poisson directly on a raw point cloud — 27× fewer, and without pose-consistent
fusion.

### Stage D — class discovery (Qwen3-VL, in-process) ✅

```python
stage_vlm('00777c41d4')          # rc 0, ~610 s, all 261 frames
```

Runs the paper's prompt verbatim through `transformers` instead of a vLLM
server, because vLLM is not installed on RUN and 102 GB of VRAM is ample for
an 8B model.

**Do not subsample frames.** The prompt carries the accumulated vocabulary into
every later frame and instructs the model to reuse existing labels; skipping
frames breaks that continuity and the same chair becomes "chair" in one frame
and "office chair" in another.

That same mechanism amplifies any junk that gets in. Four runs were needed:

| Attempt | Rule | Classes | Outcome |
|---|---|---|---|
| 1 | none | 58 | 26 were elaborations of "wood" |
| 2 | fold on **first** word | 33 | *best-looking number, worst result* — nine real classes absorbed into the colour "green" |
| 3 | fold on **last** word | 62 | objects safe, wood cascade back |
| 4 | **head noun first, then leading word** | **35** | correct |

Final vocabulary: barrel, bench, bin, board, book, bookshelf, bottle, box,
bucket, cabinet, ceiling, container, door, drawer, drum, floor, hose, ladder,
lamp, light, machine, pipe, plastic, radiator, shelf, sign, stand, stool,
table, tool, trash, wall, wheel, window, wood.

Every run reported `rc 0` and 261/261 frames. The only thing that distinguished
them was reading the label list.

### Stage E — SAM3 masks ⏳ blocked, being resolved

`sam3.py` needs three things that were all missing:

1. The Holi-Spatial repo vendors an **incomplete** `sam3/` — 70 `.py` files but
   no `sam3/train/`, which five core modules import from, so it cannot load a
   model at all. Replaced with `facebookresearch/sam3` at HEAD; the broken copy
   is kept as `sam3.vendored.broken` for comparison.
2. `ftfy`, installed `--no-deps` as above.
3. The right checkpoint — see below.

**SAM 3.1 is not a drop-in for SAM 3 here.** `download_ckpt_from_hf` differs
only by filename between versions, which is misleading: `version="sam3.1"` is
only accepted by `build_sam3_predictor` → `build_sam3_multiplex_video_predictor`.
`build_sam3_image_model` — what `sam3.py` calls — has no version parameter and
builds the SAM 3 image architecture only.

Loading `sam3.1_multiplex.pt` into it "succeeds" and reports four missing keys:

```
backbone.vision_backbone.convs.3.conv_1x1.{weight,bias}
backbone.vision_backbone.convs.3.conv_3x3.{weight,bias}
```

Those tensors stay randomly initialised. The checkpoint has `convs.0–2` built
from `dconv_2x2_*` modules; the image model expects four blocks of
`conv_1x1`/`conv_3x3`. Different architecture, not a version bump. Use
`sam3.pt` for this pipeline and keep `sam3.1_multiplex.pt` for the video
predictor.

### Stages F–H — lifting, post-processing, scene.json ⛔ not yet run

- `3d_bounding_instance_gs_rerun_da3.py` (1,893 lines) lifts masks into 3D
  using the trained gaussians
- `postprocess_3d_bbox_aabb.py` floor-aligns the boxes
- `v3d.scene.from_holi.build_scene` converts the result to `scene.json`

`rerun` is absent and `rerun-sdk` 0.36 no longer exposes a `rerun` module (the
paper used 0.24.1). The lifting script guards that import and prints
"未安装 rerun，跳过可视化", so only visualisation is lost.

### Stage J — the gate ⛔ waiting on a scene.json

```bash
python3 -m v3d.validate.stage_j /tmp/run/scene.json \
    --gt /tmp/data/gt/00777c41d4.json --out /tmp/run/stage_j.json
```

Read the size-band table, not just the verdict. The ScanNet++ annotations go
down to a 6 cm cord, so a flat 5 cm tolerance is 2% of a door and 80% of a
cord; a PASS carried by fragments while the `large` band fails is a failure.

Reference points already measured on this scene:

- ground-truth room: floor slab **6.42 × 6.22 m**, height **3.304 m**
- PGSR mesh height **3.432 m** — about **13 cm** over, the first real number to
  explain

---

## 3. Visualisations

`v3d.export.visualize` renders offscreen through Open3D's EGL backend, which
works on RUN (EGL 1.5 / OpenGL 4.5).

| Function | Output |
|---|---|
| `orbit_views` | 8 views circling the room, 62 s on the full 9.5M-vertex mesh |
| `topdown_view` | floor plan with the ceiling sliced off |
| `boxes_over_mesh` | mesh with ground-truth boxes, shell in blue, objects in orange |
| `depth_sheet` | contact sheet of DA3 depth maps on a shared metre scale |
| `render_vs_photo` | trained 3DGS render beside the real photograph |

`render_vs_photo` is the most convincing single artefact: the renders are
almost indistinguishable from the photographs, which is what PSNR 26.77 looks
like.

Three bugs in these were found only by opening the PNGs — all had passed every
automated check. See `bugs-found.md` B2–B4.

---

## 4. Where everything is

**On RUN**

```
/tmp/paper/holi-spatial/          the paper's code, patched
/tmp/data/scannetppv2/data/00777c41d4/
    dslr/resized_undistorted_images/   261 frames
    dslr/nerfstudio/transforms_undistorted.json
    depth_da3/                          261 .npy
    pointcloud_da3.ply                  60 MB
    mask/                               261 mesh-guided masks
/tmp/data/gt/00777c41d4.json      223 ground-truth instances
/tmp/run/gs/00777c41d4/           gaussians, mesh, renders
/tmp/run/scene_objects/00777c41d4.json   35 classes
/tmp/DA3NESTED-GIANT-LARGE/       6.76 GB
/tmp/qwen3vl8b/                   17.55 GB
/tmp/sam3new/sam3bundle/          SAM 3.1 + official source
/tmp/viz/                         16 rendered images
```

`/kaggle/working` is a 20 GB loop mount; `/tmp` has ~1.1 TB. Everything large
belongs in `/tmp` — but note a session restart wipes it, which is roughly
25 minutes of re-fetching at the rates above.

**On Kaggle**

- `traoanhuy/v3d-build-cache-sm120` — 3.58 GB, verified present
- `traoanhuy/v3d-sam31-multiplex` — private, push reported success, **not yet
  verified**
- a DA3 mirror was attempted and the push reported success, but no dataset
  appeared under any search; unresolved, and low priority since DA3 re-fetches
  from HF in ten seconds

**On this laptop** — `outputs/delivery/`

```
INPUT/    159 MB, the complete scene in pipeline layout
OUTPUT/visualizations/   6 renders including render_vs_photo.png
OUTPUT/mesh/room_300k.glb  12.7 MB, opens in Blender
```

---

## 5. Resuming

```bash
cd kaggle-runtime-toolkit-agent-skills
export KAGGLE_PROXY_URL='<RUN proxy>'
python3 kaggle_executor.py exec "
import sys; sys.path.insert(0,'/kaggle/working')
from v3d.paper_runner import stage_report
print(stage_report('00777c41d4'))
"
```

`stage_report` names what already exists — frames, depth maps, point cloud,
mesh, masks, VLM labels — so a stage is resumed rather than repeated.

Sync local code to a kernel before running anything:

```bash
COPYFILE_DISABLE=1 tar --exclude='__pycache__' -czf /tmp/v3d.tar.gz -C src v3d
python3 kaggle_executor.py upload /tmp/v3d.tar.gz v3d.tar.gz
# then extract, delete ._* AppleDouble files, and clear v3d from sys.modules
```

Two rules that cost hours when broken: never pipe a deploy command through
`grep`, and `kaggle_executor.py exec` defaults to a 600 s timeout.
