# Roadmap — from a benchmark scene to any video

What exists, what is missing, and the order in which the gaps should be closed.
The benchmark half is nearly done; the video half has not started, and its hard
problem is named in §2.

Companions: `pipeline-runbook.md` (what was run), `bugs-found.md` (defects),
`deb.technical` (task list).

---

## 1. Where the pipeline stands

```
video ──?──▶ posed frames ──▶ DA3 ──▶ PGSR ──▶ mesh ──▶ masks ──▶ SAM3 ──▶ lift3d ──▶ scene.json ──▶ exporters
   ▲                             ✅       ✅       ✅        ✅       ✅        ⛔          ⛔            ◐
   └── §2, not started                                                                      └── Stage J grades this
```

Done and measured on ScanNet++ scene `00777c41d4`:

| Stage | Evidence |
|---|---|
| DA3 | 261 depth maps, 100% finite, 0.44–4.94 m |
| PGSR | 30k iterations, PSNR 26.77, 1,247,783 gaussians |
| Mesh | 9,517,867 vertices / 18,543,482 triangles |
| mesh2mask | 261/261 |
| VLM classes | 35 clean classes from 261 frames |
| SAM3 | 9,326 masks, 261/261 images, 31 classes matched |

Not yet run: `3d_bounding_instance_gs_rerun_da3.py` (lifting),
`postprocess_3d_bbox_aabb.py`, and therefore no `scene.json` and no ±5 cm
number. Those are P1-2 and P1-3 in `deb.technical`.

Partially done: exporters. `.blend` and `.glb` work today from mesh plus boxes;
they should be re-pointed at `scene.json` once it exists, since that file is
the single source of truth by design.

---

## 2. The gap that decides everything: video → posed frames

Every stage above consumes **frames plus a camera transforms file**. ScanNet++
supplies both. A video supplies neither, and the two are not equally hard.

Poses can be estimated. **Metric scale cannot be recovered from pixels at all** —
a room and a photograph of a room twice its size produce identical images.
Nothing downstream can fix a wrong scale, and every ±5 cm claim rests on it.

This is not a theoretical worry: the abandoned v2t pipeline reported a room
**8.6 m tall** because DA3 guessed the scale and guessed wrong, and its
validator passed it.

### Options for scale, best first

| Route | Gets metric? | Cost |
|---|---|---|
| **RGB-D capture** (RealSense, iPhone LiDAR via Record3D) | yes, from hardware | needs the right camera; the depth stream *is* the answer |
| **Known reference object** in frame (A4 sheet, printed marker) | yes, one solve | changes how you film; needs a detection step |
| **Calibrated intrinsics + a measured distance** | yes | one checkerboard session per camera, one tape measurement per room |
| **VIO / ARKit pose stream** | yes, from the IMU | requires a capture app that exports poses |
| **DA3 metric mode alone** | **no** | this is what produced 8.6 m |

The user's earlier clips came from an Intel RealSense — the mp4s were extracted
from a ROS bag topic `/device_0/sensor_1/Color_0/image/data` — but only the
colour stream was kept. **The depth and the factory intrinsics were discarded.**
If those bags still exist anywhere, recovering them is worth more than any
amount of pipeline work.

### Implementation sketch

```
v3d/ingest/video.py
    extract_frames()     sharpness-filtered, evenly spaced
    solve_poses()        COLMAP, or DA3 streaming with loop closure
    resolve_scale()      one of the routes above, recorded in scene.json
    write_layout()       images/ + transforms_undistorted.json
```

Emit the **flat layout** already defined in `v3d/config.py`, and everything
downstream runs unchanged — that is what `SceneLayout` exists for.

Two checks belong here rather than later, because they are cheap and they fail
loudly: reject a video whose frames are too blurred or too sparsely overlapping
to reconstruct, and refuse to emit a layout with no scale provenance.

---

## 3. Closing the benchmark half

**3.1 Lifting.** `3d_bounding_instance_gs_rerun_da3.py` takes `--scene`,
`--data-root`, `--mask-root`, `--model-path`. `--mask-root` is the directory
*containing* `<scene>/mask_index.json` — note sam3.py nests the scene name
again, so masks land at `sam_masks/<scene>/<scene>/`. Expect the same class of
problems already seen twice: a hardcoded path, or an assumption about dtype.

**3.2 scene.json.** `v3d.scene.from_holi.build_scene` already converts the
eight-corner instance format into the schema. It derives the room shell from
the floor/wall/ceiling instances and marks anything it cannot measure as
`assumed`, which Stage J rejects.

**3.3 Grade it.** `python3 -m v3d.validate.stage_j scene.json --gt … --out …`.
Read the **size-band table**, not the verdict: ScanNet++ annotates down to a
6 cm cord, so a flat 5 cm tolerance is 2% of a door and 80% of that cord. A
PASS carried by fragments while the `large` band fails is a failure.

First number to explain: the mesh is **3.432 m** tall against a ground-truth
**3.304 m** — about 13 cm over.

**3.4 Then five scenes.** One scene proves the pipeline runs; five prove it was
not luck. `find_image_shard` locates any scene by binary search, so adding
scenes is cheap.

---

## 4. What a complete pipeline still needs

**Chamfer distance.** Stage J compares dimensions only, and two meshes can
agree on every extent while one has a wall bowed 15 cm. The released
`GS_model_w_mesh_pp/*.tar` contain the authors' own mesh per scene — the
natural reference. Report it; do not gate on it.

**Physics.** `scene.json` carries geometry but nothing about mass, friction or
articulation. Sim-to-sim needs collision primitives that are convex and cheap,
not the 18.5 M-triangle visual mesh, and every physical value should carry the
same `source` tag the geometry does so an assumed friction coefficient is
visible.

**Exporters from scene.json.** GLB, MJCF, USD and `.blend` should each read the
schema and nothing else. Anything reading meshes directly will drift from the
document that Stage J grades.

**Scale-invariant reporting.** For scenes with no ground truth — which is every
one of the user's own videos — report relative-proportion error and the
plausibility checks, and say plainly that absolute scale is an input, not a
measurement.

---

## 5. Sequence

1. Lifting → `scene.json` → **the first ±5 cm number** (§3.1–3.3)
2. Five benchmark scenes (§3.4)
3. Video ingest with an explicit scale route (§2)
4. Chamfer, physics, exporters (§4)

Steps 1 and 2 are finishing something already built. Step 3 starts the half
that has not begun, and it is where the project's original goal actually lives.
