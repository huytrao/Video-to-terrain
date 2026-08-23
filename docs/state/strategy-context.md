# Strategy context — small_factory, maximum quality

Live working document for the run of
`input/Small-factory/small_factory_final.mp4` through the Holi-Spatial
video→3D pipeline on the offline RTX PRO 6000, at the highest quality the
hardware allows. Written so the run can be picked up cold.

Companions: `docs/guides/pipeline-runbook.md` (the benchmark run),
`docs/reference/output-quality.md` (what limits quality),
`docs/reference/deb-technical.md` (outstanding work).

---

## 1. The ask, and what each part of it means concretely

| Asked for | Concretely |
|---|---|
| Highest quality | raise DA3's `process_res` above the paper's 504, derive the mesh knobs from the scene instead of inheriting bedroom constants, scale PGSR's schedule to the frame count |
| **Double the point cloud** | `target_num_points` 4 M → 8 M, and verify the extra points are *measured* rather than interpolated |
| General method | no scene id, image size or frame count baked in; every constant becomes an environment read or a derived ratio |
| Robot-ready output | collision mesh, height map, MJCF, semantic boxes, clearances — not just a visual mesh |
| Cut objects out | per-object assets extracted from the scene, see §7 |
| Outputs stay on Kaggle | push to a Kaggle dataset from BUILD; never route artefacts through the laptop |
| An agent that only catches errors | `.claude/agents/main-chat-auditor.md`, already written |

---

## 2. Hardware and the three kernels

| Role | Hardware | Internet | Use |
|---|---|---|---|
| **RUN** | RTX PRO 6000 Blackwell, 95.6 GiB, sm_120 | no | every pipeline stage |
| **BUILD** | CPU only | yes | `pip download`, Kaggle dataset pushes |
| laptop | — | yes | orchestration and small text only |

Measured transfer rates: BUILD→RUN via `relay_push.py` **23.8 MB/s**;
laptop→kernel **~4.6 MB/s**. Everything bulk moves kernel-to-kernel.

RUN ships **torch 2.10.0+cu128, numpy 2.0.2, transformers 5.0.0, nvcc 12.8**.
That combination matters: the two PGSR CUDA extensions in the build cache were
compiled against exactly it. The same cache also contains `torch-2.13.0`,
`numpy-2.5.2`, `pandas-3.0.5` and a CUDA 13 runtime, so installs are an
**allowlist**, never `deps/*.whl`. `scripts/kernel/kernel_stage.py` enforces this and
re-checks both pins after every install phase.

---

## 3. Staging — done, and proved rather than assumed

`scripts/kernel/kernel_stage.py` reports `STAGE_OK`. What it establishes, by running the
thing rather than reading a version string:

| Component | Evidence |
|---|---|
| open3d 0.19.0 | builds a `ScalableTSDFVolume`, clusters a mesh |
| plyfile | `PlyElement.describe` round trip |
| `simple_knn.distCUDA2` | finite distances over 50 000 real points on sm_120 |
| `diff_plane_rasterization` | exports `rasterize_gaussians` |
| **nvdiffrast** | `RasterizeCudaContext` covers **1352 px** of a test triangle |
| evo | `PosePath3D` imports — DA3's `pose_align` needs it |
| pycocotools, skimage | RLE encode; `morphology` — sam3.py:12, lift3d:19/23 |
| DA3 | `depth_anything_3.api` imports |

nvdiffrast is `docs/reference/deb-technical.md` **P1-1**, the item that blocked masks → SAM3 →
boxes across previous sessions. It is resolved: built for sm_120 in 27 s. 1352 px
is the same figure the runbook recorded on the benchmark, so the backend agrees
with the earlier verification.

Three packages were absent from the cache and hard-required — `evo`,
`pycocotools`, `scikit-image` — fetched on BUILD under a `numpy==2.0.2`
constraint and relayed. `pandas` and `seaborn` came down as evo's transitive
dependencies and are deliberately **not** installed: pandas 3.0.5 needs
numpy ≥ 2.1 and would break the pin. evo only reaches pandas through
`evo.tools.pandas_bridge`, which DA3 never imports.

Two cache entries are sdists nested one directory deeper than their name
(`deps/iopath-0.1.10/iopath-0.1.10/setup.py`) on a read-only mount; they are
copied to `/tmp` and built there. open3d's import chain is settled by asking the
interpreter what it wants rather than enumerating it — it wanted `janus`.

DA3 is **not** pip-installed: it declares `hatchling` as its build backend, which
is not in the cache and cannot be fetched, so `--no-build-isolation` fails too
(pip still imports the backend). It is pure Python, so a `.pth` in site-packages
is the whole install — and unlike a `PYTHONPATH` export it reaches every stage
subprocess. Caveat learned: a `.pth` does **not** affect the already-running
IPython kernel, whose `sys.path` was fixed before the file existed.

---

## 4. Ingest — done

`scripts/kernel/kernel_ingest.py` reports `INGEST_OK`.

**Frames.** 3458 decoded, 57.69 s, 1920×1080 @ 59.94 fps. Selection is spaced by
*accumulated parallax*, not by time, because even temporal sampling spends frames
where the camera hovered and starves the seconds where it moved.

| | Value |
|---|---|
| sharpness | median 23.4, p10 10.0 |
| pan / parallax | 3.47 px / 1.37 px, ratio **0.396** |
| kept | **600** at 1.79 px parallax spacing |
| rejected | 153 blurred, 7 mis-exposed, 393 too close |

The ratio is the number worth reading: 0.396 means a real **walk**, not a pan.
A tripod pan produces parallax near zero and triangulates nothing however many
frames it contributes.

**Poses.** DA3 with no pose input, 600 views at `process_res=504`.

| | Value |
|---|---|
| processed at | 504×280 |
| intrinsics | fx 1760.1, fy 1780.6, cx 960.0, cy 540.0 — hfov 57.2° |
| focal spread | 0.037 relative, no time trend → pooling to one intrinsic is justified |
| pose round trip | rotation err 3.3e-16, translation 1.8e-15, det ∈ [1.0000, 1.0000] |
| trajectory | path 20.40 over extent 6.07, **ratio 3.36** |
| split | 525 train + 75 test |
| **peak VRAM** | **57.2 GiB** of 95.6, in 62 s |

504×280 is exactly what reading `input_processor.py` predicted, which validates
the sizing model in §5.

### Scale is `assumed`, and stays that way

Nothing in a video fixes metres. Every length in this run is in DA3's own units;
`transforms_undistorted.json` carries `v3d_scale_mode: "assumed"`. This project
has already shipped a room measured **8.6 m tall** with a validator passing it
16/16, and the mechanism was exactly this field not existing. Stage J must report
*relative proportion* error, not an absolute figure.

To get metres, one of these is needed and none is free: an RGB-D stream, a marker
or standard pallet (1200 × 800 mm) in shot, or **one tape measurement** of
something identifiable in the video. The last is cheapest — one number turns
every output metric.

---

## 5. Resolution — the main quality decision

`api.py:145` defaults `process_res=504` with `upper_bound_resize`, which scales
the **longest** side. `model_inference.py:128` maps
`{"high_res": "lower_bound_resize", "low_res": "upper_bound_resize"}` — so the
paper's default is DA3's own *low-resolution* preset.

For 1920×1080 at 504, `_resize_longest_side` gives 504×283.5 → 284, then
`_make_divisible_by_resize` rounds to the nearest multiple of `PATCH_SIZE=14`
→ **504×280 = 141,120 samples**, bilinearly upsampled to 2,073,600 by
`inference_da3_scannetppv2.py:598-603`. **14.7× areal interpolation.**

That ceiling reaches further than the point cloud: `PGSR/scene/cameras.py:89`
reads `depth_da3/<frame>.npy` as training supervision, so DA3's real resolution
feeds the final surface too.

Raising it is safe from the assertion two lines below the model call.
`inference_da3_scannetppv2.py:588` asserts predicted intrinsics match the input,
but `api.py:352` overwrites `prediction.intrinsics` with the input matrix before
anything compares them — the difference is exactly zero at any resolution. The
authors' own note on the commented-out line says "this can lead to OOM issue",
which is a memory reason, not an accuracy one.

| `process_res` | predicted | samples/frame | tok/view | % of native |
|---|---|---|---|---|
| 504 (paper) | 504×280 | 141,120 | 720 | 6.8 |
| 1008 | 1008×574 | 578,592 | 2,952 | 27.9 |
| 1920 | 1918×1078 | 2,067,604 | 10,549 | 99.7 |

`scripts/kernel/kernel_da3_maxres.py` decides this empirically. One measurement cannot
distinguish a token-linear cost from a view-quadratic one, so it takes three
probes — 600v@504, 300v@504, 150v@1008, the last chosen to match the first's
token count at 4.1× the resolution — fits `peak = fixed + slope · tokens`, and
reports the residual so the model says when it should not be trusted.

**The OOM policy is the point.** It escalates resolution downward from native and
gives way on **chunk size**, not resolution, because with poses supplied and
`align_to_input_ext_scale=True` every chunk is registered to the same global pose
set, whereas resolution lost is unrecoverable. `MAX_CHUNK_SIZE` is already the
paper's own mechanism for >800 views.

**Risk this creates, and the test for it.** Each inference call fits its own
Umeyama scale and does `prediction.depth /= scale` (`api.py:353-362`), so
chunking could in principle scale parts of the room differently. The check:
`v3d.ingest.poses.reprojection_consistency` between frames from *different*
chunks. Within-chunk error low and cross-chunk error high is exactly what a scale
disagreement looks like, and nothing else produces that signature.

---

## 6. Doubling the point cloud — what it does and does not buy

Two call sites pass `target_num_points=4000000`
(`inference_da3_scannetppv2.py:753, 762`) while the function's own default and
docstring say 20 million (line 469, 476).

The honest arithmetic: 600 frames × 141,120 real samples ≈ **85 M measured
points** before confidence filtering. Keeping 4 M is under 5 % of them, so the
cut discards *measurements*, not interpolation — and doubling to 8 M recovers
real surface rather than inflating a count. Raising `process_res` raises that
85 M budget too.

`create_from_pcd` (`PGSR/scene/gaussian_model.py:170-189`) turns **every** point
in the PLY into an initial gaussian with no downsampling, so 8 M points doubles
initialisation memory. The pilot started from 4 M and finished at 1.25 M after
opacity pruning, so the initial count is not what bounds the final model — but it
does bound peak VRAM at iteration zero, which is why it is measured, not assumed.

`apply_offline_patches` now injects these as environment reads —
`V3D_TARGET_POINTS`, `V3D_PROCESS_RES`, `V3D_PROCESS_RES_METHOD`,
`V3D_CONF_THRESHOLD`, `V3D_MAX_CHUNK_SIZE` — with the paper's literals as
defaults, so an unset environment reproduces the paper exactly.

### The patch mechanism itself was a defect

`apply_offline_patches` previously recorded `applied=False` both when a patch was
already in place *and* when its anchor text could not be found at all, and
printed the same word for both. An upstream edit of one character would silently
stop a patch applying. Every patch is now verified against **its own output**:
absent both ways raises `PatchError`, before the weights load. For the
point-cloud and resolution patches, a silent no-op would mean running the
paper's settings and reporting them as this run's.

---

## 7. Cutting objects out — the plan

The user's `nghiên cứu cách cắt vật thể ra` is object *extraction*: each object as
its own asset, not a box in a list. Note `scripts/kernel/kernel_cutaway.py` is a
different thing — it slices the room open for rendering.

The pipeline already produces what is needed: SAM3 gives per-class masks, lift3d
gives oriented boxes and the point membership behind them. The extraction is
then, per instance:

1. select cloud points inside the OBB, with a small margin;
2. **split instances** — text-prompted masks give one mask per *class* per frame,
   so five pallets in a row become one `pallet` mask and one box around all five.
   Connected components in 3D after lifting is the fix, and it is required rather
   than optional in a factory where repeated objects are the normal case;
3. crop the visual mesh to the same region;
4. fit a convex or primitive collision shape;
5. write `objects/<class>_<n>/{visual.ply, collision.obj, meta.json}`.

Judged without ground truth: every box inside the room shell, nothing larger than
a sensible fraction of the room, objects that should rest on a surface doing so,
and no two same-class boxes overlapping beyond a threshold — that last one
catches the duplicate-instance failure directly.

---

## 8. Order of work, and where it stands

| Step | State |
|---|---|
| Stage the kernel, prove every component | **done** — `STAGE_OK` |
| Frames + poses from the video | **done** — `INGEST_OK`, 600 views |
| Measure max `process_res`, set the OOM fallback | **running** |
| DA3 depth + 8 M point cloud | next |
| Derive mesh knobs from measured depth | code written, unwired |
| PGSR at the scaled schedule | ~2 h at 600 frames |
| Mesh + mesh-guided masks (nvdiffrast) | unblocked |
| VLM vocabulary | **needs a decision — see §9** |
| SAM3 masks | **needs a decision — see §9** |
| lift3d → boxes → `scene.json` → Stage J | wired, never yet run end to end |
| Collision, height map, MJCF, object cutting | to build |
| Push everything to a Kaggle dataset | to do, from BUILD |

### Knobs derived rather than inherited

`src/v3d/geometry/quality.py` replaces four constants copied from the paper's
`mesh.sh`, each of which is a length tuned on ScanNet++ bedrooms:

- `max_depth=4.0` — `render.py:122` does `ref_depth[ref_depth > max_depth] = 0`.
  Not downweighted, **erased**. A factory bay is routinely wider than 4 units.
- `voxel_size=0.006` — the mesh's resolution ceiling. Derived instead from the
  pixel footprint at typical depth, so it sharpens with better input and never
  mentions a metre.
- `num_cluster=1` — `render.py:166` keeps only the largest component, deleting any
  freestanding machine or crate the fusion did not weld to the floor. That is
  exactly the geometry the semantic-box and collision deliverables are about.
- `use_depth_filter=0` — off means multi-view-inconsistent noise is fused; the
  pilot measured **140,334** components. The interaction is the real finding:
  `num_cluster=1` is only survivable *because* the filter is off, so the cluster
  cull is doing the filter's job bluntly and paying for it in real objects.

It also scales PGSR's schedule with the frame count — 30,000 iterations is 115
views per image at 261 images but only 50 at 600 — and ties
`position_lr_max_steps` to it, without which a longer run trains on at a position
learning rate that can no longer move a gaussian.

---

## 9. Open decisions — all but one now closed

1. **The VLM.** ~~No weights on this kernel.~~ **Resolved.** The paper's exact
   fine-tune `Holi-Spatial/HoliSpatial-2M-QA-Qwen3-VL-8B` (17.55 GB, 19 files)
   downloaded on BUILD in 79 s and relayed to RUN. Running the paper's own model
   rather than a NIM substitute means the vocabulary is the one the method was
   tuned for.

   One trap when tarring it: `/kaggle/working` is a 20 GB loop mount and already
   held the SAM3 staging, so a 17.5 GB tar died at 13.8 GB with
   `Wrote only 8192 of 10240 bytes`. Stage through `/tmp` (1.1 TB).

2. **SAM 3 vs SAM 3.1.** **Resolved, and the real SAM 3 is the right checkpoint.**
   `HF_TOKEN` authenticates as `huytrao123`, the account with the approved gate.
   Inspecting `sam3.pt` (1465 tensors, prefixes `detector`/`tracker`) shows all
   four tensors the runbook recorded as *missing* from SAM 3.1 multiplex:

   ```
   detector.backbone.vision_backbone.convs.3.conv_1x1.weight  (256, 1024, 1, 1)
   detector.backbone.vision_backbone.convs.3.conv_3x3.weight  (256, 256, 3, 3)
   ```

   Two offline blockers had to be cleared as well, neither obvious:

   - The paper vendors an **incomplete** `sam3/` — no `sam3/train/`, which five core
     modules import from — so it cannot load a model at all. Replaced by symlinking
     the official package; the broken copy is kept as `sam3.vendored.broken`.
   - Kaggle **decompressed** `assets/bpe_simple_vocab_16e6.txt.gz` on dataset
     upload, and `model_builder.py` resolves exactly the `.gz` name through
     `pkg_resources`. The mount is read-only, so the package is copied to
     `/tmp/sam3pkg` and the archive restored. Verified by constructing
     `SimpleTokenizer` and encoding a sentence.
   - `ftfy` must be installed `--no-deps`: SAM3's `pyproject.toml` pins
     `numpy>=1.26,<2`, so a normal resolve downgrades numpy below 2 and takes
     torch 2.10 with it. Pins re-checked after: numpy 2.0.2, torch 2.10.0+cu128.

   `build_sam3_image_model(..., checkpoint_path=..., load_from_HF=True)` only
   reaches HuggingFace `if load_from_HF and checkpoint_path is None`, so passing the
   checkpoint keeps it offline.

3. **The factory prompt.** Set, and it is the highest-leverage string in the
   semantic half because nothing outside the vocabulary can become a box:
   `pallet, forklift, conveyor, control panel, safety cage, machine guard,
   hose reel, barrel, drum, crate, toolbox, workbench, shelving rack, cable tray,
   fire extinguisher, pipe, ladder`. Override with `V3D_PROMPT`.

4. **One measured distance — still open, and still the cheapest large win.**
   Everything is in DA3 units. One tape measurement of anything identifiable in the
   video converts every output from proportion to metres. Nothing else in this
   pipeline can supply it.

---

## 10. What each stage now checks, and what it caught

The pattern throughout is that a stage asserts something that can fail, before the
expensive step rather than after. What that actually caught:

| Check | Caught |
|---|---|
| Patch verified against its own output | nothing yet, but a silent no-op on the point-cloud patch would have reported the paper's 4 M as this run's 8 M |
| Depth reachable before training | PGSR reading depth from a path that never existed — 600 warnings, loss term silently skipped |
| Cross-chunk depth scale | native-resolution chunking disagreeing by 2.0%, which would have been baked into the surface |
| Grid-relative sampling | 58% holes in a height map of a *closed* room |
| Synthetic room with a known answer | shell removal leaving 27 objects where there were 3, then deleting all of them, then eating a thin panel |
| Quaternion control | nothing — but the swapped-order control differs by 1.9971, which is what makes the 4.4e-16 agreement meaningful |
| MJCF compiled and stepped | nothing yet; asserts the scene loads and does not explode |


---

## 10. Rules that cost hours when broken

- Never pipe a deploy command through `grep` — `kaggle_relay.py` prints `DONE`
  even when its assemble step raised, and a filter hid the traceback.
- `kaggle_executor.py exec` defaults to a **600 s** timeout; pass `--timeout`.
- `/kaggle/working` is **20 GB**; `/tmp` has **1.1 TB**. Big files go in `/tmp`,
  and a session restart wipes it.
- Kaggle's Contents API **silently truncates** PUTs near 24 MB, and
  `kaggle_executor.upload_file` sends one unchunked PUT. `scripts/kaggle/kupload.py`
  chunks and verifies a SHA256 computed *on the kernel*, because the local side
  cannot detect a short write.
- macOS tars carry `._*` AppleDouble members; delete them after extracting or
  they shadow real modules.
- A dropped websocket does not kill a stage — the job is a kernel subprocess.
  Check `ps -ef` before restarting anything.
