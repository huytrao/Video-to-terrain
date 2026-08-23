Independent correctness audit. You are NOT the model that wrote this code — your value is catching what its author could not see. Read-only: do not edit anything.

Repository: `/Users/traoanhuy/source_code/Full-pipeline-scan-terrain/Videos_to_3D ` — the directory name ends with a SPACE, so quote every path.

Do NOT read `.env`, `server-connect.txt`, or anything under `~/.codex/` — they hold live credentials. Do not open binaries (`.ply`, `.npy`, `.JPG`, `.safetensors`, `.tar`) and never print base64.

## Background

This reimplements the Holi-Spatial paper's video→3D pipeline. Stages shell out to a checkout of the paper's code; the upstream sources are vendored read-only under `source-reference/` and are the ground truth for what upstream actually does. Two standing requirements: it must work for ANY input video (so baked-in scene ids / image sizes / frame counts are defects), and quality is the top priority (so silent downsampling or silently dropped geometry are defects even when nothing crashes). The accuracy gate is ±5 cm against ScanNet++ ground truth.

Target GPU: one NVIDIA RTX PRO 6000 Blackwell, 96 GiB VRAM, compute capability 12.0.

## Claim to check — this is the main thing I want challenged

I concluded that the largest quality limiter in the pipeline is that DA3 runs its depth inference at low resolution and the result is then bilinearly upsampled, so the point cloud carries far less real geometric detail than its pixel count suggests. My evidence:

- `source-reference/Depth-Anything-3/src/depth_anything_3/api.py:145-146` — `process_res: int = 504`, `process_res_method: str = "upper_bound_resize"`.
- `source-reference/Depth-Anything-3/src/depth_anything_3/app/modules/model_inference.py:128` — `method_mapping = {"high_res": "lower_bound_resize", "low_res": "upper_bound_resize"}`, which I read as: `upper_bound_resize` IS the low-resolution preset.
- `source-reference/holi-spatial/inference_da3_scannetppv2.py:557` — the call passes no `process_res`, and the line that would raise it is commented out with the note that it can cause OOM. So the default 504 applies.
- Same file, lines 591-609 — depth, conf and images are `F.interpolate`d up to the original image size before unprojection.
- Same file, line 762 — `downsample_pointcloud(..., target_num_points=4000000)`, while the function's own default and docstring say 20 million.

Verify or refute each of these five, by reading the files. In particular:

1. Is my reading of `upper_bound_resize` vs `lower_bound_resize` correct? Read the actual implementation in `source-reference/Depth-Anything-3/src/depth_anything_3/utils/io/input_processor.py` (see `_resize_image`, `_resize_longest_side`, `_resize_shortest_side`, `_make_divisible_by_resize`) and state precisely what each method does to a 1752×1168 input at `process_res=504`, including the rounding to a multiple of the patch size. Give the resulting pixel dimensions.

2. Does raising `process_res` actually increase the *measured* depth resolution, or does something downstream re-quantise it anyway? Follow the data path from `model.inference(...)` through to `depth_to_pointcloud`.

3. Is there any correctness reason — not just memory — that the authors left `process_res` at the default? For example, does the model have a training resolution beyond which accuracy degrades? Look for any resolution assumption in the model code or configs.

4. `inference_da3_scannetppv2.py:588-589` asserts the predicted intrinsics and extrinsics match the inputs to within 1e-6. If `process_res` changes, the images are resized — would that assertion still hold, or would raising resolution trip it? Read `_resize_ixt` in the input processor and `_preprocess_inputs` in `api.py` before answering. This one matters: if it breaks, raising resolution fails immediately.

## Second thing to check: my calibration script

`scripts/kernel/kernel_calibrate_da3_res.py` (I just wrote it; never reviewed) is meant to measure peak VRAM at several resolutions on 16 views and extrapolate to the full 261-view run. Audit its correctness:

- The extrapolation is `est = weights + (probe_peak - weights) * 261/16`, i.e. it assumes activation memory is **linear** in the number of views. Is that right for this model? If DA3 uses attention *across* views, cost could be quadratic and this would badly under-predict, causing an OOM an hour into a real run. Look at how the model attends across views (`source-reference/Depth-Anything-3/src/depth_anything_3/`) and tell me whether linear is defensible, and if not, what scaling to use.
- `torch.cuda.max_memory_allocated()` ignores memory held by the caching allocator but not "allocated", and ignores anything allocated outside PyTorch. Does that make the measurement optimistic in a way that matters at 96 GiB?
- Does resetting stats with `reset_peak_memory_stats()` and then reading the peak actually isolate one candidate from the previous one, given the loop only calls `empty_cache()` and `gc.collect()`?
- `grid()` recomputes what it believes DA3's token grid will be. Compare it against the real implementation and report any disagreement — if my grid math is wrong, the whole table is mislabelled.
- Any plain bug: wrong variable, values that could be unbound, a failure path that reports success.

## Third: the mesh knobs

`src/v3d/paper_runner.py:81-85` sets `RENDER_MAX_DEPTH="4.0"`, `MASK_MAX_DEPTH="6.0"`, `VOXEL_SIZE="0.006"`, `NUM_CLUSTER="1"`, passed to `PGSR/render.py` and `PGSR/mesh2mask.py`. I believe `--max_depth 4.0` deletes every surface farther than 4 m from a camera (via `ref_depth[ref_depth>max_depth] = 0`) and `--num_cluster 1` keeps only the largest connected component, discarding detached objects. Confirm or refute both against `source-reference/holi-spatial/PGSR/render.py`, and say whether passing 4.0 to render but 6.0 to mesh2mask is coherent or means the masks describe geometry the mesh does not contain.

## Output

A numbered list, most severe first. For each finding: `file:line`, what is wrong, and the concrete scenario where it produces a wrong result or a late failure. Mark each as CONFIRMED (you read the code that proves it) or SUSPECTED (needs a run). Where I am wrong, say so directly and show the line that refutes me. Where I am right, one line is enough — do not pad.
