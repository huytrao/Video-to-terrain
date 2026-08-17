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

**Status: FIXED, after the first fix was itself wrong.** `detect_up_sign` was
added to derive the direction from the camera heights instead of assuming it —
and mapped the comparison to the opposite sign:

```python
return -1 if abs(cam - lo) < abs(cam - hi) else 1     # wrong
return  1 if abs(cam - lo) < abs(cam - hi) else -1     # right
```

The comparison finds which end of the vertical range is the floor, correctly.
But if that end is the *low* end, up is `+z`, so the answer is `+1`. The original
returned `-1`. On the real numbers — floor −0.036, ceiling −3.327, cameras
−1.44 — it returned `+1` for a scene that is Z-down, which is precisely the value
the bug needed to survive. So `topdown_view` kept rendering the ceiling from
below, behind a helper whose docstring described the correct behaviour.

Two lessons, both cheap:

- **A fix that is not exercised is not a fix.** This one was written, reviewed,
  read back, and never run on the numbers in its own docstring.
- **Sign conventions need a two-sided test.** One case cannot distinguish
  "correct" from "inverted"; a Z-up case *and* a Z-down case can. Both are now
  pinned in `visualize.selftest()`, which also asserts the section cut removes
  the half of the room nearest the camera at eight azimuths:

```
detect_up_sign: z-up -> +1, z-down -> -1
kept fraction per azimuth: 0.497 … 0.504   (must be ~0.5)
max (kept - centre) · n <= 0               (nothing left on the camera side)
```

Guidance for the original fix still applies:

- best signal: the floor is the dominant horizontal plane, and its normal
  points toward the camera positions;
- cheap signal that works here: cameras sit between floor and ceiling, nearer
  the floor;
- always expose an `up_sign` parameter so a caller can override.

This is the same class of defect as the frame-convention bugs recorded earlier
in this project. Treat any code that writes a literal `2` for "the z axis" or
assumes `+` is up as suspect.

### B2b — the frame of a delivered `.glb` cannot be assumed either way

**This entry previously stated that `deliver_out/mesh/room_300k.glb` is Y-up with
the rotation baked into the vertices. That was wrong, and it was never measured.**
What the file actually contains:

```
nodes: [{"mesh": 0}]              → no node transform at all
vertex bounds z: -3.465 … -0.033  → extent [8.02, 9.056, 3.431]
GT floor z = -0.036, ceiling z = -3.327
```

The smallest extent is on z and equals the room height, and the z range matches
the annotations to within a few centimetres. The mesh is in the **reconstruction
frame**, unrotated — `load_mesh_in_box_frame` selects `as_loaded` for it and
prints no `applied gltf_undo` line. Open3D's writer did not bake the Y-up
rotation, whatever the glTF specification says about the axis convention.

The wrong claim was harmless only because the code around it never trusted it.
That is the part worth keeping:

**Fix.** `load_mesh_in_box_frame` tries both frames and keeps the one whose bounds
agree with the annotation bounds on all three axes, raising when neither does.
Deciding from the data is what makes it safe to point at an arbitrary mesh, and
what made the incorrect documentation cost nothing. The raise is what stops a
misaligned scene from being delivered quietly.

`v3d.export.blender.import_mesh` *does* need the correction, so both conventions
genuinely occur in this project's outputs. Which one a given file uses is a
question to measure per file, not to remember.

**Also done.** The mesh is now exported as `.ply` as well
(`deliver_out/mesh/room_300k_boxframe.ply`, 12.3 MB, 163,808 vertices). PLY has
no up-axis convention and no chunked container, and this one is written after the
frame has been resolved — so a downstream stage needs no correction at all. It is
read back and compared vertex-for-vertex before the write is reported, for the
reason in B9.

### B9 — Open3D 0.19 writes a `.glb` it cannot read back, and returns an empty mesh

The cutaway render failed with `ValueError: no geometry in room_300k.glb`, one
line after this:

```
[Open3D WARNING] Unable to load file .../room_300k.glb with ASSIMP:
GLTF: Buffer view with offset/length (0/3600000) is out of range.
```

The obvious hypothesis was truncation — the B5 class of fault, and this project
has been bitten by it. **It was wrong.** The file is byte-complete and identical
on both machines: 12,664,000 bytes, matching the total declared in its own
12-byte header.

The real cause is a defect in Open3D 0.19's own round trip. Its exporter puts the
entire geometry in a **base64 `data:` URI inside the JSON chunk** and emits no
BIN chunk. Its ASSIMP-based importer expects the BIN chunk, finds nothing there,
and returns a **mesh with zero vertices** having printed only a warning. Blender
and every spec-compliant loader read the same file correctly; the data was never
damaged.

Two distinct faults, as usual:

1. **A warning where an error belongs.** `read_triangle_mesh` returns an empty
   mesh rather than raising, so any caller that does not check vertex count
   proceeds with nothing. `orbit_views` and `turntable_video` would both have
   rendered an empty room and encoded a clean video of it.
2. **A write that was never verified by reading.** The `.glb` had been in the
   delivery for days. Nothing had ever loaded it back.

**Fix.** `visualize.read_triangle_mesh` tries Open3D, and on an empty result
falls back to `_mesh_from_gltf`, which decodes the glTF buffers directly —
`data:` URIs, external `.bin` files and BIN chunks alike, with interleaved
`byteStride`, normalized integer attributes and node TRS transforms all handled.
It raises for a non-glTF file it cannot read rather than returning nothing.
Validated three ways: against the real 12.6 MB file (163,808 vertices, 300,000
triangles), against a synthetic control exercising every path the real file does
not take, and against three negative controls — a truncated file, a missing BIN
chunk, and a buffer whose declared `byteLength` is shorter than an accessor
reads. The third of those initially **passed when it should have failed**,
because the reader used the whole blob instead of trimming to `byteLength`; it
now reports `accessor 1 reads to byte 96 of a 20-byte buffer`.

**Watch out.** Anything that writes a mesh with Open3D and hands the path to
another stage has this defect. Prefer `.ply` for machine consumption and keep the
`.glb` for viewers.

### B10 — every cutaway still framed its subject at a third of the frame

The first sections that showed the interior at all were correct in every
respect except the one a person notices first. Measured from the delivered PNGs:

```
cutaway_nobox_00..03: subject fills 0.286–0.289 of frame height, 12–14% of pixels
cutaway_00..03:       width_fraction 1.000, touches_edge True  → boxes cropped
```

Two independent causes.

**The camera distance was solved for the wrong thing.** `_prepare_cutaway` set
`radius` from the *whole room's* footprint diagonal (`‖xy_hi − xy_lo‖ / 2 × zoom`
≈ 7.5 m) and `_cutaway_eye` placed the eye that far from the room centre. But a
cutaway has removed half the room, so the content sits on the far side —
11.5 m away, where a 60° vertical field of view spans about 19 m. An 8 m wide,
3.2 m tall section lands at roughly a third of the frame. The number was not
arbitrary; it framed a room that was no longer there.

**Nothing framed on the boxes.** `boxes_over_mesh` had this fixed in B4 — box
corners are included in its framing — and the cutaway path never got the same
treatment, so the shell boxes ran off all four edges.

**Fix.** `_fit_eye_distance` solves for the distance instead of estimating it.
With the eye at `target − d·forward`, a point at camera-space `(x, y, z)` is
inside the frustum when `|x| ≤ (d+z)·tan(fov_h/2)` and `|y| ≤ (d+z)·tan(fov_v/2)`,
so every point imposes a lower bound on `d` and the answer is the largest one.
The points fed in are exactly what the frame draws: the kept mesh vertices plus
the corners of the boxes that survived the cut. `margin` shrinks the usable
frustum, so the content spans `1/margin` of the binding axis and nothing sits
flush against an edge. The projection is now pinned with `set_projection` rather
than inherited from Open3D's default, because the solver has to know the field of
view exactly.

Three supporting fixes fell out of it:

- **Shell boxes are off by default in the cutaway.** In this dataset they are
  oriented slabs whose corner wrap spans 11.44 × 11.49 m across a 6.42 × 6.22 m
  floor — B1's number, still present in the annotations — one "wall" is
  9.48 × 4.50 × 3.66 m, and one "ceiling" is a 24 cm sliver. Framing on them
  dragged the camera back, and the section has already removed the wall and
  ceiling they describe. `show_shell=True` restores them.
- **Box visibility now respects the ceiling cut**, not only the vertical section
  plane. Boxes near the ceiling were being drawn over empty space above the cut
  wall tops.
- **Framing ignores per-axis outliers** (`content_percentile`, default 0.5).
  TSDF output has isolated vertices metres outside the walls, and one of them
  would undo the whole fit. `_prepare_cutaway` already used percentiles for the
  same reason. Box corners are never clipped — that would reintroduce B4.

**How it is checked.** Two ways, because the geometry side cannot see the
outcome:

1. `selftest()` fits an analytic case — a unit cube viewed along +x, where
   `d = 1/tan(fov/2) + 1` exactly — and then a wide low room seen obliquely,
   where it *measures* the result by projecting the points back
   (`_screen_bounds` / `_screen_span`) rather than re-deriving the same algebra.
   That check is two-sided, for the reason B2 taught: a fit that merely zoomed far
   enough out would satisfy "everything is inside the frame" just as well, so
   pulling 5% closer must overflow.
2. `image_content_extent` measures the **rendered pixels** — whether Open3D
   honoured the projection it was handed is not observable anywhere else. The
   kernel driver requires the subject to fill ≥ 88% of one axis and fails the
   run otherwise.

**A trap inside the check itself.** The first version of `image_content_extent`
compared each pixel against the `BG_LIGHT` constant, and reported
`width=1.000 height=1.000 coverage=1.000` — perfect framing — for the very
images that motivated it. Open3D tone-maps before writing, so a background set to
`0.94` arrives as **228, not 240**, and dithers by a level or two. Every pixel
therefore counted as subject. The background is now sampled from the four corners
and the function raises if they disagree. *A measurement that cannot fail is not
a measurement* — and this one was three lines from being trusted.

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

### B8 — clearing `sys.modules` to retry an import breaks C extensions

`open3d/__init__.py` imports `open3d.visualization` and `open3d.ml`
unconditionally, and on an offline kernel those reach absent third-party
packages at module level: first `dash` (the interactive plotly viewer), then
`addict` (the ml config loader). Each one costs a round trip to the kernel to
discover, so `ensure_importable` was written to resolve them in a loop: try the
import, read the name off the `ModuleNotFoundError`, write a stub, retry.

Retrying is the part that was wrong. A failed import leaves partially
initialised modules behind, so the loop cleared `open3d*` out of `sys.modules`
first — correct for pure Python, **fatal for a C extension**. open3d's pybind11
layer registers `open3d.cuda.pybind.visualization` and its siblings in
`sys.modules` during its one-time module init; deleting those entries makes them
unrecoverable, because re-importing the `.so` returns the cached extension
object without re-running the init that registered them. The next round then
failed with

```
ModuleNotFoundError: No module named 'open3d.cuda.pybind.visualization';
'open3d.cuda.pybind' is not a package
```

which reads as a broken open3d install and is nothing of the kind. Two rounds of
this would have been spent reinstalling a package that was never damaged.

**Fix.** Probe in a throwaway subprocess — `python -c "import open3d"`, read the
missing name off stdout, write the stub, probe again — and let the real process
import it exactly once, after every dependency is already on disk. No module is
ever re-imported, so there is nothing to corrupt.

**What caught it.** A guard written for a different reason: the loop refuses to
stub a name that starts with the target's own prefix, on the grounds that a
missing *submodule* means a broken install rather than an absent dependency.
That guard turned a silent nonsense stub (`open3d/cuda/pybind/` full of
permissive fakes shadowing a working extension) into an immediate, legible
failure. Guards that state an invariant catch faults their author never had in
mind.

**Also.** `open3d.ml` is inert under stubs — anything under `open3d.ml` will
return empty containers rather than working. Nothing in this pipeline calls it;
the import is the only reason it is present. The stub log line names every
package that got faked, so this stays visible instead of becoming folklore.

### B11 — Stage J's containment check tested a box the room does not occupy

`objects_inside_room` asked whether each object's centre satisfied
`min(xs) - 0.25 ≤ cx ≤ max(xs) + 0.25` over the floor polygon's vertices — i.e.
against the polygon's **axis-aligned bounding box**, not the polygon. This
scene's footprint is a 6.42 × 6.22 m rectangle rotated about 21°, whose
axis-aligned span is 8.15 × 8.26 m. The check therefore admitted objects up to
1.27 m outside the room along the diagonals while flagging others merely for
sitting near a corner of a box the room never occupied.

This is B1 again, in a different function. `Room.footprint_extent_m` already
documents having fixed exactly this rotated-rectangle mistake for the room's own
dimensions; the containment test never got the same treatment.

**The invariant was also wrong.** Even corrected, "every object centre is inside
the floor polygon" is false of correct data: a window's box lies *in* the wall,
and the polygon is fitted to a shell that disagrees with the other four (B1).
Measured on the released annotation for `00777c41d4`, **75 of 218 ground-truth
centres fall outside the polygon, the worst by 0.82 m** — all `glass` and `panel`
labels, i.e. windows and wall panels. By nearest wall face rather than centre it
is 44 of 218, worst 0.55 m. So the check as specified could never pass, on any
reconstruction, however good.

**Fix.** `footprint_overhang_m` measures signed distance to the **polygon**, via
outward edge normals with the winding resolved from the shoelace signed area, so
a clockwise polygon does not invert every normal and report the room inside out.
The check became `no_stray_objects`, with a limit that is a fraction of the
room's short edge (`STRAY_FRACTION_OF_ROOM = 0.20`) rather than an absolute pad —
it scales to any room, which a hardcoded 0.25 m does not. 0.20 puts the
annotation's own worst case (0.82 m) inside a 1.24 m limit with about a third to
spare. **That calibration is against real correct data, not a number chosen to
make this scene pass** — and the comment in the source says so, with the numbers.

**Confirmed on a second scene.** `stage_j_selftest.py` runs against
`d986399f4c` — 116 objects, a 48.2 m³ room — where the same fraction yields a
0.61 m limit and 4 of 116 objects sit outside by at most 0.14 m: PASS. The buggy
variant's `obj_ceiling_0` lands 2.83 m out and fires. A limit that is right at
both 1.24 m and 0.61 m without being touched is a limit that scales; the 0.25 m
absolute pad it replaced was right in neither room.

Box size was deliberately left out of the measure. Folding in the box's own
half-extent makes a wardrobe against a wall score worse than a mug in the garden,
which is backwards for a check asking "is this object in this room"; size is
already gated to 5 cm by `object_size_within_tol`. That was caught by an analytic
test returning −2.5 m where −1.5 m was correct, not by reading the code.

### B12 — Stage J passed a scene with half its objects missing

The metric verdict grades **matched** objects. Recall was computed, stored on the
report, and printed — but never fed into a check, so it could not fail anything.
Measured with the negative-control battery:

```
half the objects gone   structural=PASS metric=PASS OVERALL=PASS   recall 50.0%
```

A pipeline that found one wardrobe and missed everything else would score a
perfect ±5 cm. The tolerance was a claim about the easy half of the scene, and
the number the whole benchmark exists to produce would have been reported in good
faith on a reconstruction that had dropped half the furniture — which for a robot
sim is furniture it walks straight through.

**Fix.** `recall_check` gates recall, `metric_ids` includes it. The floor is
**per size band**, because a flat floor would be B11's mistake again: 99 of this
annotation's 218 objects are under 30 cm — stickers, labels, cords, switches —
and no video reconstruction segments a 6 cm sticker, so a flat 80% could never
pass honestly. `MIN_RECALL_BY_BAND = {"large": 0.80, "small": 0.50}`; the tiny
band is reported and ungated (absent from the dict). Large objects are both the
ones any reconstruction can see and the ones a robot collides with, so they are
held highest. Empty floors report **SKIP**, not PASS — "we did not check" and "we
checked and it was fine" must not print the same word.

These floors are **declared expectations, not measured calibration**, and the
source says so: the only end-to-end run so far feeds the released annotation to
itself, where recall is 100% by construction and proves nothing. Revisit them
against the first real DA3+PGSR run.

### B13 — a gate made to pass on ground truth had proved nothing

Having fixed B11 and B12, Stage J returned OVERALL PASS on the released
annotation. That is not evidence: the previous validator answered PASS sixteen
times on a scene with an 8.6 m ceiling and would have kept doing so however
carefully it was read. `scripts/stage_j_controls.py` perturbs a known-good scene
eight ways and asserts both the verdict **and which check fires** — a control
that fires on the wrong check is a gate that is right for the wrong reason, which
will not survive the next edit.

```
[ok] honest                           PASS recall 100.0%  fired: -
[ok] sizes +8cm                       FAIL recall 100.0%  fired: object_size_within_tol
[ok] sizes +4cm                       PASS recall 100.0%  fired: -
[ok] object 0.3m inside stray limit   PASS recall  99.5%  fired: -
[ok] object 0.3m past stray limit     FAIL recall  99.5%  fired: no_stray_objects
[ok] half the objects gone            FAIL recall  50.0%  fired: object_recall
[ok] all large objects gone           FAIL recall  69.3%  fired: object_recall
[ok] ceiling 8.6m                     FAIL recall 100.0%  fired: ceiling_plausible, room_dims_within_tol
8/8 controls behaved as claimed
```

The ±4 cm / ±8 cm and the ±0.3 m stray pair are two-sided: they locate the
boundary rather than poking one side of it, which is the only way to know a
threshold is where it is documented to be.

**A control that lied.** The first mislocalisation control shifted an object
5.7 m along **+x** and was recorded as catching a stray object. In a room rotated
21° that shift slides *along* the long axis: the object ends up 0.22 m **inside**
the footprint, and the corrected gate passes it — correctly. The control had been
passing for a reason unrelated to the distance in its name, and it only surfaced
because the recall fix changed the verdict and forced the number to be explained.
Perturbations in that file are now derived from the scene's own outward normals,
never from a constant measured in one room. *Controls need controls.*

### B14 — the delivery check could not run where the delivery is reviewed

`image_content_extent` — the function that measures whether a shipped frame is
framed correctly, and the only check that can see B10's actual outcome — called
`open3d.io.read_image`. Open3D is installed on the kernel and not on the laptop,
so verifying a delivered video off-kernel raised `ModuleNotFoundError`. The one
check that would have caught loose framing before shipping could not be pointed
at the shipped file. That is the same gap that let B10 ship in the first place.

**Fix.** `_read_image_any` tries Pillow → imageio → OpenCV → Open3D and reports
what it tried when all four are absent. Nothing in the function needs a GPU or a
render context, so nothing about it justified requiring the render library.

**B10 now closes with off-kernel evidence.** Frames 0/18/36/54 extracted from the
delivered `outputs/delivery/OUTPUT/videos/07_cutaway_interior.mp4` with ffmpeg
and measured on the laptop: `fill = 0.947–0.950, touches_edge = False,
background = [229,229,229]` (was 0.286). The viz stills `cutaway_0*.png` measure
0.944–0.948, `edge = False`.

**Still open, opposite direction:** `cutaway_nobox_0{0,1,3}.png` measure
0.973–0.978 with `touches_edge = True`. Framing without the boxes is now slightly
*too* tight — inside the ≥ 0.88 driver threshold, so nothing fails, but content
is flush against an edge. The margin should leave headroom on both paths.

### B15 — four pipeline stages existed and could not be run

`lift3d`, `postprocess`, `scene` and `gate` were implemented but absent from
`STAGES`, so the CLI could not reach them: everything downstream of the mesh,
including the ±5 cm number the benchmark exists to produce, was unreachable
without importing the module by hand. Wired up with `ALL_ORDER`, plus `--gt` and
`--tol`, and a `tail` alias for `lift3d` onward.

Two faults surfaced while wiring it, both of the same kind — code that had never
executed:

- `stage_scene` referenced `doc.room.kind`, which does not exist, and hand-rolled
  the json instead of calling `doc.write()`.
- `main()` patched the vendored paper checkout for **every** stage but `report`,
  so `--stage scene` — which needs no paper code at all — died on a missing
  `inference_da3_scannetppv2.py`. Now gated on `PAPER_STAGES` intersecting the
  computed order.

### B16 — seven live API keys were sitting in the tracked `.env.example`

Found while previewing what a commit would stage. `.env.example` — a file whose
entire purpose is to be committed — held **six live `nvapi-` keys and one live
`gsk_` Groq key** as its default values. `.gitignore` correctly ignores `.env`
and `.env.*` while excepting `.env.example`, so the one file the ignore rules
deliberately let through was the one carrying the secrets. A single `git add -A`
would have published them.

**Not leaked.** All 38 commits were scanned, blob by blob, for
`nvapi-|gsk_|sk-ant-|hf_` patterns: the only commit that ever touched
`.env.example` is clean, and no real-looking key appears anywhere in history. The
exposure was working-tree only.

**Fix, in an order that could not lose a key.** Five of the six NVIDIA keys
existed *nowhere else* — `.env` had only `NVIDIA_API_KEY_1`, which duplicated the
example's `ACCOUNT4` — so redacting first would have destroyed them. Instead:
both files were backed up (`.env.*` is already ignored), the example's real values
were appended to `.env` as `NVIDIA_API_KEY_2..6`, their presence there was
**verified before anything was overwritten**, and only then were the example's
values replaced with `<REPLACE_WITH_YOUR_…>`. Verified after: 0 live secrets in
`.env.example`, 0 values present before and missing after, 6 distinct keys in
`.env`. No key value was printed at any point — matching the standing instruction
never to display credentials or binary content.

`server-connect.txt` was also about to be committed. The path segment of a Kaggle
Jupyter proxy URL *is* a ~400-character JWT, so that file is a credential with an
expiry rather than configuration; it is now ignored, with live URLs in `.env`.

**Still worth the user's attention:** `.env` defines `GROQ_API_KEY` twice, with
different values. Most dotenv loaders take the last, silently — so which key is
in use depends on line order, which is not a property anyone intends.

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

### C6 — the Jupyter proxy fails often enough that single-shot calls are unsafe

Two distinct failures wear the same face, and the difference matters:

- **Transient.** The `/api` handshake that every `kaggle_executor.py` call opens
  with read-times-out at 30 s, or returns HTTP 500, for no reason and then works.
  An earlier session lost a 121 MB upload to this and recorded it as "proxy
  instability".
- **Terminal.** `HTTP 404 / not found` means the **proxy token has expired** —
  the URL is dead and no amount of retrying revives it. A fresh URL must come
  from the notebook's *Kaggle Jupyter Server* panel. Kernel URLs died three times
  in one session, so treat the URL as a perishable credential, not configuration.

**Fix.** `scripts/kx.sh` wraps the executor, retries only on transport patterns
(`ReadTimeout|ConnectionError|HTTP 5..|Cannot reach Jupyter API|RemoteDisconnected|Connection aborted`)
with linear backoff, and refuses to retry a real error from the remote code so a
failing script says so on attempt 1 rather than attempt 6. `KX_TRIES` (default 6).
Live URLs live in `.env` as `KAGGLE_WITHOUT_INTERNET` / `KAGGLE_HAS_INTERNET`;
`server-connect.txt` is the fallback.

**A trap in the wrapper's own first draft.** It was invoked as
`if python3 … | tail -60; then` — which reports **`tail`'s** exit status. A dead
proxy read as "OK on attempt 1". This is standing rule 1 again, in the tool
written to enforce it. `kx.sh` now redirects to a file and tests the executor's
own status. (Also: macOS has no `timeout`, and zsh has no `${!var}` indirect
expansion — both bit this script.)

**And a second trap in the wrapper.** The transport pattern above contains
`Cannot reach Jupyter API`, but the executor reports an expired token as
`Cannot reach Jupyter API: HTTP 404` — so the *terminal* failure matched the
*transient* pattern and every dead URL cost six attempts and 126 s of backoff
before admitting it. The 404 case is now tested **before** the transport pattern
and exits immediately, naming the two `.env` keys to update. When a classifier
has a catch-all arm, check that the specific case is tested first.

### C7 — `. ./.env` executes the file, and printed a live key into a terminal log

Loading the environment with `set -a && . ./.env` runs every line as a shell
command. `.env` contained `CLAUDEFARE-API-KEY=cfut_…`; a dash is not legal in a
shell identifier, so zsh could not parse it as an assignment, tried to run it as
a **command**, and put the whole line — key value included — into the
`command not found` error. The credential was then in a terminal log and a
session transcript, which is the one place a credential must never be.

The blast radius is not limited to malformed lines: any `.env` line beginning
with a backtick or `$(` would be *executed* by the same mechanism.

**Fix.** `scripts/kenv.sh` greps out only the keys the caller needs
(`KAGGLE_WITHOUT_INTERNET`, `KAGGLE_HAS_INTERNET`, `HF_TOKEN`) and prints them as
`export` lines for `eval`. It cannot execute anything else in the file no matter
what the file contains. Never `source` a `.env` that a human edits by hand.
The exposed key must be **rotated** — redacting the file afterwards does not
un-log it. (See also B16: this is the second credential incident from treating
dotenv files as ordinary config.)

### C8 — two probe results that read as failure and were not

Both cost real time by sending a session looking for a problem that did not exist.

- **Attached datasets are at `/kaggle/input/datasets/<owner>/<name>`**, not
  `/kaggle/input/<name>`. Probing the short path reported all three model
  datasets ABSENT while every one was attached and readable, and produced a
  written conclusion that a 6.76 GB dataset push had "silently failed". It had
  not. `scripts/kernel_probe.py` now uses the long form.
- **`huggingface.co/api/whoami-v2` answers HTTP 401 without a token**, and
  `urllib` raises `HTTPError` for that. A bare `except` around it reported
  "NO internet" on a kernel whose network was fine. Any HTTP status proves the
  host answered; only `URLError`/socket errors mean unreachable. Both the probe
  and any future reachability check must distinguish the two.

### C9 — `kx.sh` shows nothing until the job ends, so long stages look hung

The retry wrapper redirects the executor's output to a temp file and `cat`s it on
success — necessary to test the executor's own exit status (see C6), but it means
a stage that runs for 54 minutes prints **nothing** for 54 minutes, which is
indistinguishable from a hang. It also cannot retry a job that already burned an
hour of GPU.

**Use `kx.sh` for short, retryable calls** (uploads, probes, installs) and
`kaggle_executor.py run` **directly** for the long GPU stages, where the
websocket streams stdout live and a dropped connection does not kill the remote
job (C5). Log to a file on the kernel as well, so a dropped websocket can be
recovered from by tailing rather than re-running.

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
6. **An invariant must be checked against known-correct data before it is
   trusted as a gate.** B11's containment check could never have passed: 75 of
   218 correct ground-truth centres violate it. Run every new check against the
   annotation first — if it fails on data known to be right, the check is wrong,
   not the data.
7. **A threshold measured in one room must be a fraction of something.**
   `STRAY_FRACTION_OF_ROOM = 0.20` yields 1.24 m on the 6.42 × 6.22 m pilot and
   0.61 m on `d986399f4c` (48.2 m³), and both are right; the 0.25 m absolute pad
   it replaced was right nowhere. The user's standing instruction — the pipeline
   must work for *any* input video — makes this a correctness requirement, not a
   style preference. Verify on a second scene, which is what caught this.
8. **A number that is computed and printed but not wired into a verdict gates
   nothing.** B12's recall was visible in every report for as long as the gate
   existed and stopped exactly zero bad scenes.
9. **Controls need controls.** A negative control must assert *which* check
   fires, and should be two-sided where a threshold is involved (±4 cm passes,
   ±8 cm fails). B13's mislocalisation control had been passing for a reason
   unrelated to the distance in its name.
10. **A check must be runnable where the artefact is reviewed.** B14's framing
    check imported Open3D, so it could not be pointed at a delivered video on the
    laptop — which is precisely where the framing bug it was written for shipped
    from.
11. **Code that has never executed is not code.** Four stages in `paper_runner`
    were unreachable from the CLI; wiring them up surfaced a reference to a
    non-existent attribute and an unconditional patch step in the first minute.
    If there is no command that runs it, assume it is broken.
12. **Never `source` a file containing secrets.** `. ./.env` executes it; one
    key whose *name* contained a dash was therefore run as a command and its
    value printed into a terminal log (C7). Parse out the keys you need with
    `grep` — `scripts/kenv.sh` — so the file's contents can never be executed.
    Secrets have now leaked twice in this project from treating dotenv files as
    ordinary config (B16, C7).
13. **A probe that reports absence must be tested against known-present data.**
    Two probes in one session cried failure at healthy systems: a dataset path
    missing its `/datasets/<owner>/` segment declared three attached datasets
    ABSENT, and an HTTP 401 was read as "no internet" (C8). This is rule 6
    pointed at diagnostics: a check that cannot pass on a working system tells
    you about itself, not the system.
