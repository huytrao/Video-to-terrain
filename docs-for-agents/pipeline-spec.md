# Specification — one command, any video, every output

The target interface:

```bash
v3d run capture.mp4                      # works with nothing else supplied
v3d run capture.mp4 --config factory.yaml
v3d run a.mp4 b.mp4 --prompt "factory floor, CNC machines, pallets"
```

Everything else — which kernels to use, whether they have internet, model
paths, every threshold — is an input with a default. Nothing is hardcoded in a
stage, which is not true today: `RENDER_MAX_DEPTH`, `VOXEL_SIZE` and the rest
are module constants in `paper_runner.py`.

---

## 1. The configuration object

Three layers, each overriding the one before: built-in defaults → config file →
command line. A field absent everywhere takes its default, so a bare video path
is a complete invocation.

```python
@dataclass
class RunConfig:
    # ---- input -------------------------------------------------------
    videos: list[Path]
    scene_id: str | None = None          # defaults to the video stem
    prompt: str | None = None            # steers the class vocabulary
    domain: str = "generic"              # generic | factory | warehouse | office

    # ---- scale: the one input that cannot be guessed ------------------
    scale: ScaleSpec = ScaleSpec(mode="assumed")

    # ---- compute: where stages run ------------------------------------
    compute: ComputeSpec = ComputeSpec()

    # ---- stage parameters, all currently hardcoded ---------------------
    frames: FrameSpec = FrameSpec()
    poses: PoseSpec = PoseSpec()
    geometry: GeometrySpec = GeometrySpec()
    semantics: SemanticSpec = SemanticSpec()

    # ---- output --------------------------------------------------------
    outputs: tuple[str, ...] = ("scene_json", "mesh", "boxes", "blend", "report")
    out_dir: Path = Path("runs")
```

### Compute — the offline/online split as a first-class input

This is the part the current code cannot express at all: the three-kernel
arrangement lives in shell commands and environment variables.

```python
@dataclass
class ComputeSpec:
    mode: str = "local"                  # local | kaggle | hybrid

    # kaggle mode: proxy URLs, which expire when a session restarts
    run_proxy: str | None = None         # GPU, typically no internet
    build_proxy: str | None = None       # nvcc + internet: pip, CUDA builds
    fetch_proxy: str | None = None       # internet only: relaying URLs

    run_has_internet: bool = False       # drives whether relaying is needed
    dataset_cache: str | None = "traoanhuy/v3d-build-cache-sm120"
    hf_token: str | None = None          # gated checkpoints
```

Behaviour follows from `run_has_internet` rather than from separate code paths:

- **True** — stages fetch their own weights; no relay, no build host needed.
- **False** — every weight and wheel is staged first, from `dataset_cache` if
  it is attached, otherwise built on `build_proxy` and relayed. This is the
  path that works today, and it is why `build_proxy` must have `nvcc` *and*
  internet: CUDA extensions cross-compile there for the GPU it cannot run.

A missing `build_proxy` with `run_has_internet=False` is a configuration error
and should be reported as one, not discovered thirty minutes in.

### Scale — required to be explicit

```python
@dataclass
class ScaleSpec:
    mode: str = "assumed"     # assumed | rgbd | marker | measured | vio
    marker_size_m: float | None = None      # marker mode
    measured_distance_m: float | None = None
    measured_between: tuple[str, str] | None = None   # what was measured
    depth_dir: Path | None = None           # rgbd mode
```

`mode="assumed"` is allowed and is the default, because a bare video has no
other option. What it must not do is stay invisible: it propagates into
`scene.json`, and Stage J then reports **relative** proportion error and
refuses to state an absolute figure. The previous pipeline's 8.6 m ceiling is
what happens when this field does not exist.

### Stage parameters — the constants that are hardcoded today

```python
@dataclass
class FrameSpec:
    max_frames: int | None = None       # None = all
    min_sharpness: float = 0.0          # variance of Laplacian; 0 disables
    min_baseline_m: float = 0.0         # reject near-duplicate viewpoints

@dataclass
class PoseSpec:
    method: str = "colmap"              # colmap | da3_streaming | provided
    transforms: Path | None = None      # method="provided"

@dataclass
class GeometrySpec:
    da3_model: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    iterations: int = 30_000            # PSNR was still climbing here
    resolution: int = 1
    render_max_depth: float = 4.0       # too small for an 8 x 9 m room
    mask_max_depth: float = 6.0
    voxel_size: float = 0.006           # the detail ceiling
    num_cluster: int = 1                # drops disconnected real geometry
    use_depth_filter: bool = False      # off is why there are 140k fragments

@dataclass
class SemanticSpec:
    vlm_model: str = "Holi-Spatial/HoliSpatial-2M-QA-Qwen3-VL-8B"
    sam3_checkpoint: str = "facebook/sam3"   # gated; 3.1 is video-only
    confidence: float = 0.6
    consolidate_vocabulary: bool = True
    min_points: int = 30                # lift3d instance filter
    erode_pixels: int = 15
```

The comments are not decoration — those five values in `GeometrySpec` are the
subject of `output-quality.md`, and they are exactly the ones a user needs to
reach in order to improve a result.

---

## 2. Outputs — everything the paper produces

| Key | Artefact | Produced by |
|---|---|---|
| `depth` | per-frame metric depth | DA3 |
| `point_cloud` | fused cloud, `pointcloud_da3.ply` | DA3 |
| `gaussians` | trained 3DGS checkpoint | PGSR |
| `mesh` | `tsdf_fusion_post.ply` | render.py |
| `masks` | mesh-guided + SAM3, RLE with scores | mesh2mask, sam3.py |
| `classes` | discovered vocabulary, per frame | VLM |
| `boxes` | oriented 3D boxes, floor-aligned | lift3d, postprocess |
| `scene_json` | the single source of truth | from_holi |
| `blend` / `glb` / `mjcf` / `usd` | exports, all reading scene.json | export/ |
| `report` | Stage J verdict and size bands | validate/stage_j |
| `videos` | turntable, renders, comparisons | export/visualize |

`outputs` in the config selects which are produced; the dependency graph pulls
in whatever they need. Asking for `boxes` implies geometry and masks; asking
for `report` implies everything.

---

## 3. Run contract

**Resumable.** Every stage writes a manifest entry: input hashes, parameters,
code version, output paths. A stage re-runs when any of those differ, and is
skipped when they match. This is what `stage_report` gestures at today by
checking file existence — which cannot tell a stale artefact from a current
one, and would happily pair masks from one vocabulary with classes from
another.

**Fails early.** Configuration is validated before any GPU work: proxies
reachable, checkpoints present or fetchable, `build_proxy` supplied when
needed, scale spec internally consistent. PGSR alone is 54 minutes; discovering
a missing token after it is unacceptable.

**Fails loudly.** Each stage asserts a postcondition — depth finite and in
range, gaussian count and PSNR within band, mesh extent consistent with the
camera trajectory, non-trivial mask coverage per class. Every one of those
numbers was computed by hand during the pilot run; they belong in code.

**Records provenance.** `scene.json` carries how scale was established, which
models ran, which parameters, which code version. An output whose origin cannot
be reconstructed cannot be trusted, and this project has already shipped one
that reported sixteen passes on a room 8.6 m tall.

---

## 4. What running it looks like

```bash
# a video and nothing else: defaults throughout, scale flagged assumed
v3d run capture.mp4

# offline GPU kernel, build host for wheels, gated checkpoints
v3d run capture.mp4 \
    --run-proxy   https://…/proxy \
    --build-proxy https://…/proxy \
    --hf-token    hf_… \
    --dataset-cache traoanhuy/v3d-build-cache-sm120

# a factory, with scale anchored to a measured doorway
v3d run floor.mp4 \
    --domain factory \
    --prompt "CNC machines, pallets, forklifts, control panels, safety cages" \
    --scale measured --measured-distance 0.92 --measured-between door_left door_right \
    --render-max-depth 8.0 --use-depth-filter --voxel-size 0.004 \
    --outputs scene_json mesh boxes mjcf report
```

The third command contains every fix `output-quality.md` recommends, expressed
as flags rather than edits to a module — which is the point of the whole
specification.

---

## 5. Order of implementation

1. `RunConfig` and the specs, with the current hardcoded constants moved into
   `GeometrySpec` and `SemanticSpec` unchanged. Pure refactor, no behaviour
   change, immediately makes §4's third command expressible.
2. `ComputeSpec` wrapping the relay scripts, so the three-kernel arrangement
   stops living in shell history.
3. Manifest and postconditions — small, and they make every later result
   trustworthy.
4. `ingest` producing the flat layout, at which point a bare video actually
   runs the chain.
5. `ScaleSpec` modes beyond `assumed`.

Steps 1–3 change no algorithm and make the system usable and checkable. Step 4
is the one the user is asking for, and it depends on 1–3 being in place, since
otherwise a video run has neither reproducibility nor a way to say whether its
output is any good.
