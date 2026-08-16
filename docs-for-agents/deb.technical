# Technical debt — v3d (video → metric 3D)

Branch: `v3d-rewrite`. Every item below is written so it can be picked up cold:
what is wrong, why it matters, the exact commands, and how to know it is fixed.
Work items top to bottom — P1 blocks the pipeline, P2 is correctness, P3 is
cleanup.

## How to run anything at all

Three Kaggle kernels are involved. Their proxy URLs expire when a session
restarts; get fresh ones from the notebook's ⋮ → "Kaggle Jupyter Server" panel.

| Role | GPU | Internet | Use it for |
|---|---|---|---|
| **RUN** | RTX PRO 6000 Blackwell, 102 GB, sm_120 | no | every pipeline stage |
| **BUILD** | Tesla P100, sm_60 (unusable by its own torch) | yes | `pip download`, compiling CUDA extensions |
| **FETCH** | none (CPU) | yes | relaying files |

BUILD has `nvcc` and the *same* Python 3.12 + torch 2.10.0+cu128 as RUN, so it
cross-compiles for sm_120 correctly even though it cannot run the result.

```bash
cd kaggle-runtime-toolkit-agent-skills
export KAGGLE_PROXY_URL='<RUN proxy url>'
python3 kaggle_executor.py exec --timeout 3000 "print('hello')"
python3 kaggle_executor.py upload <local> <remote-relative-to-/kaggle/working>
```

Rules learned the hard way — violating these wastes hours:

- **Never pipe a deploy command through `grep`.** `kaggle_relay.py` prints
  `DONE` even when its assemble step raised, and a grep filter hid the
  traceback; 5 GB of orphaned parts filled the disk while the log looked clean.
  Read full output, then verify on the kernel.
- `kaggle_executor.py exec` defaults to a **600 s** timeout. Pass `--timeout`.
- `/kaggle/working` is only **20 GB**. `/tmp` has **~1.1 TB**. Big files go in
  `/tmp`.
- A subprocess inherits none of the `sys.path` edits made in a notebook cell.
  `PipelineConfig.subprocess_env()` handles this; use it.
- `relay_push.py` must stay at `--subchunk 8` or lower. Kaggle's Contents API
  silently truncates PUTs somewhere near 24 MB.

Sync local code to a kernel:

```bash
COPYFILE_DISABLE=1 tar --exclude='__pycache__' -czf /tmp/v3d.tar.gz -C src v3d
python3 kaggle_executor.py upload /tmp/v3d.tar.gz v3d.tar.gz
python3 kaggle_executor.py exec "
import subprocess,sys
subprocess.run(['tar','xzf','/kaggle/working/v3d.tar.gz','-C','/kaggle/working'])
subprocess.run('find /kaggle/working/v3d -name \"._*\" -delete', shell=True)
sys.path.insert(0,'/kaggle/working')
for m in list(sys.modules):
    if m.startswith('v3d'): del sys.modules[m]
"
```

macOS tars carry `._*` AppleDouble files; always delete them after extracting.

---

## P1-1 — `nvdiffrast` is missing, which blocks everything downstream

**State.** DA3, PGSR training and mesh extraction all completed on the pilot
scene `00777c41d4`. `PGSR/mesh2mask.py` then failed:

```
File "PGSR/Mesh2DepthHelper/Mesh2DepthHelper.py", line 4
    import nvdiffrast.torch as dr
ModuleNotFoundError: No module named 'nvdiffrast'
```

No masks means no `sam3.py` input, which means no 3D lifting, which means no
object boxes and no `scene.json`. This is the single item blocking a first
end-to-end result.

**Fix.** Build it on BUILD exactly the way PGSR's extensions were built, then
relay the wheel. `nvdiffrast` is a normal setuptools CUDA extension.

```bash
# on BUILD
export KAGGLE_PROXY_URL='<BUILD proxy url>'
python3 kaggle_executor.py exec --timeout 3000 "
import subprocess, sys, os, glob
subprocess.run(['git','clone','--depth','1',
                'https://github.com/NVlabs/nvdiffrast','/kaggle/working/nvdiffrast'])
env = dict(os.environ, TORCH_CUDA_ARCH_LIST='12.0', MAX_JOBS='4')
r = subprocess.run([sys.executable,'-m','pip','wheel','/kaggle/working/nvdiffrast',
                    '-w','/kaggle/working/wheels_nv','--no-build-isolation','--no-deps'],
                   capture_output=True, text=True, env=env)
print('RC', r.returncode); print((r.stdout+r.stderr)[-3000:])
print(glob.glob('/kaggle/working/wheels_nv/*.whl'))
subprocess.run(['tar','czf','/kaggle/working/nvdiff.tar.gz','-C','/kaggle/working','wheels_nv'])
"
```

Then push to RUN and install:

```bash
export KAGGLE_INTERNET='<BUILD proxy url>'
export KAGGLE_WITHOUT_INTERNET='<RUN proxy url>'
python3 scripts/relay_push.py /kaggle/working/nvdiff.tar.gz \
    --dest /tmp/nvdiff.tar.gz --extract-to /tmp/nvdiff
# then on RUN: pip install --no-index --find-links /tmp/nvdiff/wheels_nv nvdiffrast
```

**Likely complications.**

- `nvdiffrast` builds some of its CUDA at *first use*, not at install, and it
  needs a writable cache: set `TORCH_EXTENSIONS_DIR=/tmp/torchext` on RUN.
- It also needs OpenGL/EGL headers for the GL backend. Use the **CUDA backend**
  (`nvdiffrast.torch.RasterizeCudaContext`) to avoid EGL entirely. Check what
  `PGSR/Mesh2DepthHelper/Mesh2DepthHelper.py` instantiates; if it asks for
  `RasterizeGLContext`, switch it to `RasterizeCudaContext` and record that as
  a patch in `v3d/paper_runner.py` next to the two existing ones.
- If the build fails outright, the fallback is to rasterise the mesh with
  Open3D's offscreen renderer instead — but try nvdiffrast first, because
  mesh2mask's depth comparison is tuned to it.

**Done when.** On RUN:

```python
import nvdiffrast.torch as dr; dr.RasterizeCudaContext()
```
succeeds, and:
```python
from v3d.paper_runner import stage_mesh; stage_mesh('00777c41d4')
```
returns 0 with `/tmp/data/scannetppv2/data/00777c41d4/mask` non-empty
(expect roughly one mask file per frame — 261).

---

## P1-2 — SAM3, lifting and post-processing stages are not wired up

**State.** `v3d/paper_runner.py` implements `stage_da3`, `stage_pgsr`,
`stage_mesh`. The remaining three paper stages exist in
`source-reference/holi-spatial/` but have no wrapper, so there is no path from
masks to `scene.json`.

**Fix.** Add three functions to `v3d/paper_runner.py`, following the shape of
`stage_mesh` exactly (take `scene: str | SceneLayout` and `cfg:
PipelineConfig`, call `_run`, return the exit code). Do **not** invent
arguments — read each script's `parse_args` and pass what it already accepts.

1. `stage_vlm` — replaces `classic_vllm.py`. Do not call that script; it posts
   to a vLLM server that is not installed. Use `v3d.semantic.vlm.VLMLabeler`,
   which runs the same prompt in-process, and write its output in whatever
   shape `sam3.py --scene-json` expects (check `DEFAULT_SCENE_JSON` in
   `sam3.py` and match the format of the file it reads).
2. `stage_sam3` — `python sam3.py --data-root <data_root> --scene-json <labels>
   --output-dir <masks> --sam3-checkpoint <cfg.sam3_checkpoint>`.
   The checkpoint is already on RUN at `/kaggle/working/sam3` and
   `transformers` 5.0.0 has `Sam3Model` natively, so nothing needs downloading.
3. `stage_lift3d` — `python 3d_bounding_instance_gs_rerun_da3.py --scene <id>
   --data-root … --mask-root … --model-path <cfg.gs_root/scene> --output-dir …`
   then `python postprocess_3d_bbox_aabb.py --input_dir … --output_dir …`.

`rerun` is not installed on RUN and `rerun-sdk` 0.36 no longer exposes a
`rerun` module (the paper used 0.24.1). The lifting script wraps its rerun
import in try/except and prints "未安装 rerun，跳过可视化", so this only costs
visualisation. Leave it.

**Done when.** `postprocess_3d_bbox_aabb.py` has written a per-scene JSON, and

```python
from v3d.scene.from_holi import build_scene
doc = build_scene('00777c41d4', '<that json>')
doc.write('/tmp/run/scene.json')
```
produces a `scene.json` whose object count is in the same order as the ground
truth's 223 instances.

---

## P1-3 — grade the first end-to-end result

Once P1-2 lands, this is the whole point of the project:

```bash
python3 -m v3d.validate.stage_j /tmp/run/scene.json \
    --gt /tmp/data/gt/00777c41d4.json --out /tmp/run/stage_j.json
```

Report the size-band table, not just the verdict. Read the numbers honestly:
a PASS driven by hundreds of `sticker` and `cord` fragments while the `large`
band fails is a failure. The `large` band — walls, doors, cabinets — is what
sim-to-sim depends on.

Known reference points already measured on this scene:

- ground truth room: floor slab **6.42 × 6.22 m**, height **3.304 m**
- PGSR mesh: 9,517,867 vertices, 18,543,482 triangles, 498 MB, height
  **3.432 m** (≈ 13 cm over) — that gap is the first real thing to explain

---

## P2-1 — six live NVIDIA API keys sit in a tracked file

`.env.example` contains six real `nvapi-…` keys in the working tree. Git
history is clean and `HEAD`'s copy has only placeholders, so nothing has leaked
yet — but the file **is tracked**, so any `git add -A` publishes them.

Five of the six exist *only* in `.env.example`; they are not in `.env`. So do
not simply scrub the file, that destroys the key pool.

**Fix.** Move them, then scrub:

1. Append the five missing keys to `.env` (gitignored) under
   `NVIDIA_API_KEY_ACCOUNT2..6`.
2. Replace every real key in `.env.example` with
   `nvapi-REPLACE_WITH_REAL_KEY_N`, which is what the file's own line 81 says
   should be there.
3. Verify: `git grep -cE 'nvapi-[A-Za-z0-9_-]{20,}' -- .env.example` reports 0
   real keys, and `grep -c nvapi- .env` reports 6.
4. Rotate the keys anyway if the repo is ever pushed public with them staged.

---

## P2-2 — the ±5 cm gate is not size-aware by default

ScanNet++ annotates to the level of `sticker`, `label`, `wood`, `panel` and a
6 cm `cord`. A flat 5 cm tolerance is 2% of a 2.46 m door and 80% of that cord,
so a single pass/fail number is dominated by fragments.

`stage_j.run(..., min_size_m=…)` and `--min-size` already exist and default to
0.0, which gates everything — this is deliberate, it is what was asked for.
The size-band table makes the asymmetry visible.

**Decision needed from the user, not from an agent:** whether the headline gate
should move to `--min-size 0.3`. Do not change the default unilaterally.

---

## P2-3 — no Chamfer distance, so surface quality is unmeasured

Stage J compares dimensions only. Two meshes can agree on every extent while
one has a wall bowed by 15 cm. The released `GS_model_w_mesh_pp/*.tar` shards
contain the authors' own mesh per scene, which is the natural reference.

**Fix.** Add `v3d/validate/chamfer.py`: load both meshes with Open3D, sample
~200k points each, compute symmetric point-to-point distance, report mean and
95th percentile. Wire it into `StageJReport` as a reported number, **not** a
gate — the gate stays dimensional as agreed.

Fetching the reference mesh: `v3d/data/hf_tar.RemoteTar` reads a member out of
a remote tar by byte offset; `find_scene_shard` locates which shard holds a
scene. The GS shards are 14.4 GB each, so never download one whole.

---

## P2-4 — `.blend` export does not exist

The user asked for a Blender file and none of the pipeline produces one. The
old `scripts/export_blender.py` only ever wrote `.glb`; there is no `bpy` call
anywhere in the repo.

Agreed approach: relay a headless Blender into RUN and export there.

**Fix.** Write `v3d/export/blender.py` with a `bpy` script that reads
`scene.json` and nothing else — room shell as meshes, one named object per
entry with its OBB transform, correct metric scale, then `bpy.ops.wm.save_as_
mainfile`. Run it as `blender --background --python …`. `src/v3d/export/` is
currently an empty package and is where this belongs.

Everything must be derived from `scene.json`; that file is the single source of
truth by design, and an exporter that reads meshes directly will drift from it.

---

## P3-1 — repo hygiene

- `outputs_trash/` holds ~1.3 GB of the old v2t run. It is gitignored. Keep
  `apartment_room1_full/` as the "before" evidence; the rest can go.
- `src/v2t/` (31 files) is untracked on `v3d-rewrite` but still present on
  `main`. Nothing should be restored from it; it exists only for reference.
- Commit `47538c5` swept in pre-existing uncommitted edits to
  `scripts/bake_texture.py` and `scripts/export_blender.py` that were not part
  of this work. Review them and split them out if they matter.
- There is no test suite. `scripts/stage_j_selftest.py` is the only automated
  check; it builds an honest scene and a deliberately broken one from real
  ground truth and asserts the gate separates them. Run it after touching
  anything in `v3d/scene/` or `v3d/validate/`:
  `python3 scripts/stage_j_selftest.py` must print
  `SELFTEST: PASS`.

## P3-2 — session assumptions

No checkpoint/resume machinery exists, on the stated assumption that Kaggle
sessions do not die. Note that PGSR training takes **54 minutes** for 30k
iterations, and a websocket drop during it is survivable (the job is a kernel
subprocess and keeps running — verify with `ps -ef | grep train.py` rather than
assuming a silent log means death). A session *restart* would lose
`/tmp` entirely: ~17.5 GB of VLM weights, 6.8 GB of DA3, the scene, and every
artefact. Re-fetching all of it is roughly 25 minutes at the measured
30–35 MB/s.

---

## Verified facts worth not rediscovering

- PGSR's two CUDA extensions build for sm_120 after one fix: `simple_knn.cu`
  uses `FLT_MAX` without `#include <float.h>`, which CUDA 12.8 no longer
  supplies transitively. `v3d/geometry/pgsr_build.py` does this.
- `pytorch3d` is *not* needed. The whole paper repo imports one symbol from it,
  `quaternion_to_matrix`, which is pure tensor arithmetic.
  `v3d/geometry/pytorch3d_shim.py` provides it and `verify()` proves the
  convention against scipy — real-part-first, agreeing to 4e-16, with the
  swapped ordering deliberately reported as a 1.997 control.
- PGSR's `readNerfSyntheticInfo` falls back to
  `output_scannet/scannetppv2/0b031f3119/input.ply`, an absolute path from the
  authors' machine, whenever `--ply_path` is absent — and `3dgs_train.sh` never
  passes it. Both `train.py` and `render.py` hit this. Patched in
  `apply_offline_patches`.
- `evo` must be installed with `--no-deps`. A full resolve pulls numpy 2.5.2
  and pandas 3.0.5, which break the torch 2.10 install that PGSR needs.
- Ground-truth boxes are **oriented and never axis-aligned**. Comparing a
  world-axis mesh extent against them inflated the pilot room from 6.42 m to
  11.49 m. Footprints are compared as rotated rectangles; see
  `Room.footprint_extent_m` and `room_dims_from_gt`.
