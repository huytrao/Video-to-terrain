# Bugs found — v3d

Every defect discovered while getting the Holi-Spatial pipeline running on the
offline Blackwell kernel, with the evidence that revealed it and the fix that
was applied. Grouped by where the fault lives, because that determines who can
fix it and how much to trust the surrounding code.

A theme runs through almost all of these: **the failure was silent**. Each one
reported success, or reported nothing, while producing wrong output. Where a
fix was applied, it was paired with a check that fails loudly the next time.

---

## A. Bugs in the published Holi-Spatial / PGSR code

These are in `source-reference/holi-spatial/`, i.e. the released ICML code, not
in anything written here. They are patched at runtime by
`v3d.paper_runner.apply_offline_patches()` rather than by editing the vendored
tree, so the patches are visible, idempotent, and survive a re-clone.

### A1 — hardcoded absolute path from the authors' machine (blocking)

`PGSR/scene/dataset_readers.py:294`

```python
if ply_path == None:
    print("ply_path is None, make sure this is eval")
    ply_path = "output_scannet/scannetppv2/0b031f3119/input.ply"
```

`0b031f3119` is a *different scene*, and the path is relative to a directory
that exists only on the authors' machine. `3dgs_train.sh` — their own entry
point — never passes `--ply_path`, so every ScanNet++ scene reaches this
fallback. Execution then continues into `storePly(ply_path, ...)`, which tries
to **write** to that non-existent directory:

```
FileNotFoundError: [Errno 2] No such file or directory:
  'output_scannet/scannetppv2/0b031f3119/input.ply'
```

Hit by both `train.py` and `render.py`. Passing `--ply_path` fixes training,
but `render.py` has no such flag, so the source must be patched.

**Fix.** The same repository already does this correctly a few hundred lines
down, in `readScannetV2SceneInfo`:

```python
if ply_path is None:
    ply_path = os.path.join(path, "pointcloud_da3.ply")
```

The patch makes the ScanNet++ reader agree. Its `path` is the transforms
directory, so the DA3 cloud is two levels up. This also matches what the paper
describes — 3DGS is initialised from the DA3 point cloud.

**Watch out.** If a future layout puts the transforms file elsewhere, the
`"..", ".."` in the patch becomes wrong. Prefer passing `--ply_path`
explicitly wherever the script accepts it; the patch is only for `render.py`.

### A2 — `simple_knn.cu` misses an include on modern CUDA (blocking)

```
simple_knn.cu(90): error: identifier "FLT_MAX" is undefined
simple_knn.cu(154): error: identifier "FLT_MAX" is undefined
```

CUDA 12.8 no longer pulls in `<float.h>` transitively. Two lines of the 2023
source depend on it having done so.

**Fix.** `v3d.geometry.pgsr_build.patch_float_header()` prepends
`#include <float.h>` to any `.cu` that uses `FLT_MAX` without it. After that
both extensions compile clean for `sm_120` in 76 s.

### A3 — misleading log line (cosmetic, but it wastes debugging time)

`PGSR/scene/__init__.py` prints

```
Found transforms_train.json file, assuming Blender data set!
```

when it detects `transforms_undistorted.json`. The branch is correct — that
*is* the ScanNet++ path in this fork — but the message names the wrong file
and the wrong dataset, and sent this investigation down a false trail.
Not patched; documented here instead.

### A4 — `rerun` import is version-locked

`3d_bounding_instance_gs_rerun_da3.py` expects `rerun` 0.24.1. Current
`rerun-sdk` (0.36) installs `rerun_bindings` and a CLI but no top-level
`rerun` module, so the import fails. The script wraps it in try/except and
prints `未安装 rerun，跳过可视化`, so **only visualisation is lost**. Left alone
deliberately; pin `rerun-sdk==0.24.1` if the 3D viewer is ever wanted.

---

## B. Bugs in this repository's own code

### B1 — `room_dims_from_gt` inflated every rotated room (silent, severe)

The first real comparison of the finished mesh against ground truth reported
the footprint **3.5 m short**. The mesh was fine; the measurement was wrong.

`room_dims_from_gt` took all eight corners of every shell box and wrapped them
in a world-axis bounding box. None of the ScanNet++ shell boxes are axis
aligned, so a **6.42 × 6.22 m floor wrapped to 11.49 × 11.44 m**. Any correct
reconstruction of a room that happens to sit at an angle would have been failed
by 3.5 m — and every real room sits at an angle.

**Fix.** A footprint is a polygon and its dimensions are edge lengths.
`scene.json` now stores the floor slab's own rotated rectangle,
`Room.footprint_extent_m()` returns its edge lengths, and `room_dims_from_gt`
reads the floor box's own extents. Both sides are rotation-invariant.
`GTBox.size` is sorted descending along the box's own axes, which is what makes
the comparison frame-independent.

**How it was caught.** `scripts/stage_j_selftest.py` failed the moment the two
halves of the comparison disagreed. Honest scene afterwards: footprint within
**0.2 cm**, height exact.

### B2 — the up axis points DOWN in this scene, and the floor-plan render assumed otherwise

Measured from the data:

```
GT floor   centre z = -0.036
GT ceiling centre z = -3.327     → +z points DOWN
camera z ≈ -1.44                 → ~1.4 m above the floor, as expected
```

`topdown_view` cut `z < cut_z` and placed the eye at `+z`, which in this frame
**kept the ceiling, discarded the floor, and looked up from beneath the floor**.
The render looked plausible — a striped surface that reads as a floor — and was
actually the aluminium ceiling slats.

**Status: OPEN.** `v3d/export/visualize.py::topdown_view` still assumes `+z` is
up. Fix by determining the up direction from data rather than assuming it, and
never hardcode a sign:

- best signal: the floor is the dominant horizontal plane, and its normal
  points toward the camera positions;
- cheap signal that works here: cameras sit between floor and ceiling, nearer
  the floor;
- always expose an `up_sign` parameter so a caller can override.

This is the same class of defect as the frame-convention bugs recorded earlier
in this project. Treat any code that writes a literal `2` for "the z axis" or
assumes `+` is up as suspect.

### B3 — `render_vs_photo` silently produced blank panels

PGSR writes renders as `.jpg`; ScanNet++ ships photographs as `.JPG`. The
comparison sheet looked for an exact filename match, found none, and drew an
empty axis with a caption — a figure that appeared to show a comparison and
showed nothing.

**Fix.** Try `.JPG/.jpg/.png/.PNG/.jpeg`, and **raise** naming the missing
stems if no photograph is found. A missing input is now a failure, not a blank.

Also relevant: `PGSR/render.py --skip_test` leaves `train/ours_30000/gt/`
empty, so the photographs must come from the input image directory, not from
the model output tree.

### B4 — box renders framed on the mesh, cropping the boxes off

`boxes_over_mesh` computed the camera distance from the mesh bounds. Object
boxes extend past the mesh, so the outer ones were cut off. Fixed by including
box corners in the framing computation.

### B5 — `relay_push.py` lost 4.4 MB of a 148 MB file and reported success

The transfer completed, printed no error, and the destination was short. Only
gzip's own integrity check caught it.

Two separate faults:

1. **No verification.** The assemble step printed the assembled size next to
   the expected size and carried on regardless.
2. **Chunk size.** Kaggle's Contents API silently truncates PUTs somewhere
   near 24 MB. `kaggle_relay.py` never hit this because `iter_content` yields
   *short* chunks; `relay_push.py` read exact-size blocks and hit it every time.

**Fix.** Default `--subchunk` lowered to 8 MB; the assemble step now checks
every part exists, compares the total byte count, refuses to extract on a
mismatch, and emits `RELAY_PUSH_OK` which the caller requires. Same 148 MB file
afterwards: **148,090,824 bytes, exact**.

### B6 — an in-place edit left a syntax error, hidden by a shell pipe

A multi-line argparse `help=` string was written without quoting, making
`relay_push.py` unrunnable. The failure was invisible because the calling
command piped output through `grep`, so the traceback was filtered out and the
transfer appeared to do nothing quietly.

**Fix.** `python3 -c "import ast; ast.parse(open(f).read())"` over every source
file after any scripted edit. And see the standing rule below.

### B7 — `.gitignore` inline comment did nothing

```
kaggle_upload_example.py   # contains a plaintext Kaggle API key
```

`.gitignore` has no inline comments; the whole line became a literal pattern
and matched nothing, so the file stayed unignored. Verified with
`git check-ignore -v` — which is the only way to know a rule actually applies.
Comments must be on their own line.

---

## C. Environment traps (not bugs, but they behave like them)

### C1 — `kaggle_relay.py` reports DONE on failure

It opens the destination without creating the parent directory, and prints
`DONE in Ns` as soon as the assemble step returns, regardless of whether that
step raised. Relaying a 17 GB checkpoint into a fresh folder failed on **every
file** while the log looked healthy, and left 5 GB of orphaned parts on a
20 GB disk. Use `scripts/relay_hf_repo.py`, which creates the destination and
stats every file on the receiving kernel against the size HuggingFace
advertises.

### C2 — subprocesses inherit no `sys.path` from the notebook

`depth_anything_3` imported fine when checked interactively and then failed
inside a stage two seconds in, because the check ran in a kernel after a
`sys.path.insert` and the stage runs as a subprocess.
`PipelineConfig.subprocess_env()` passes `PYTHONPATH` explicitly.

### C3 — `pip install evo` breaks torch

A full dependency resolve pulls numpy 2.5.2 and pandas 3.0.5, which are
incompatible with the torch 2.10 build PGSR needs. Install with `--no-deps`
and add only the transitive packages that are genuinely absent. Verify after:
`numpy.__version__` unchanged and `torch.cuda.is_available()` still True.

### C4 — disk and timeout limits

`/kaggle/working` is a 20 GB loop mount; `/tmp` has ~1.1 TB. Big artefacts
belong in `/tmp`. `kaggle_executor.py exec` defaults to a **600 s** timeout,
which silently truncates any long job — pass `--timeout`.

### C5 — a dropped websocket does not kill the job

PGSR training survived a client disconnect and ran to completion, because it is
a subprocess of the kernel. Silence in the log is not evidence of death; check
`ps -ef | grep train.py` and `nvidia-smi` before restarting anything expensive.

---

## D. Standing rules these bugs produced

1. **Never pipe a deploy or transfer command through `grep`, `head`, or
   `tail -n`.** Three separate failures in this project were hidden that way:
   a relay whose every file failed, a script with a syntax error, and a
   transfer that lost 4.4 MB. Read the full output, then verify the effect
   independently.
2. **A transfer or stage that cannot verify its own result has failed.** Check
   byte counts, not exit codes.
3. **Every measurement needs a control.** The quaternion shim is trusted
   because the deliberately swapped ordering reports 1.997 while the correct
   one reports 4e-16; without the control, the small number proves nothing.
   Same reason `stage_j_selftest.py` builds a deliberately broken scene.
4. **Look at the picture before shipping it.** B2, B3 and B4 were all found by
   opening the rendered PNG, and all three had passed every automated check.
5. **Assume no frame convention.** Derive up, scale and handedness from the
   data and assert them; this project has now been bitten by quaternion order,
   gravity direction and axis-aligned-vs-oriented boxes.
