# Carving movable objects out of the factory scan, and giving them physics

**Session date:** 2026-08-20, 14:04 local
**Branch:** `v3d-rewrite`
**Working scene:** `outputs-full-factory/delivery/`

---

## What was asked for, in the order it was asked

1. Take named objects out of `OUTPUT_B1/mesh/room_300k.glb` and fill the holes they leave.
2. Be able to click and drag those objects in a web page.
3. Give the room gravity; each object 10 kg; dragging one up and releasing it makes it fall.
4. Objects standing on the workbench must fall when the workbench is taken away.
5. Paint the parts of the mesh with no texture grey.
6. Fill the missing back of each object using the generated mesh at
   `OUTPUT_J/<name>/mesh_textured.glb`.
7. Add friction and other physical properties, and put the objects into a factory scene.

---

## The one fact everything else depends on

**Up in this scan is `-Z`, not `+Z`.**

- The two dominant horizontal planes are at `Z = -3.18` (35.3 m²) and `Z = -0.21` (30.6 m²);
  the room is 2.97 m tall.
- Four of the five original objects have their *maximum* z within 4–15 cm of `Z = -0.21`.
  `obj_0202_blade` is 1.12 m off it because it sits on a bench, not the floor.
- `ground_truth_boxes.json` labels `ceiling` at z ≈ -3.32.

So the **floor is `Z ≈ -0.155`** (the value the tools compute), the ceiling is at -3.18, and
**gravity acts along `+Z`**. `carve_manifest.json` records this as `up_axis: "-Z"`.

Anything that assumes `+Z` up drops objects onto the ceiling. `scripts/sim/make_room_scene.py`
already knew this and fixes it with `FLIP_X180` — a 180° rotation about X. Do **not** "simplify"
that to negating Z: negating one axis mirrors the scene and silently flips every rotation in it.

---

## Pipeline that now exists

```
OUTPUT_B1/mesh/room_300k.glb          the fused scan: 163,808 verts, 300k faces, VERTEX COLOURS
   |
   |  scripts/tools/carve_objects.py
   v
OUTPUT_K/room_carved.glb              room with objects removed and holes patched
OUTPUT_K/objects_raw/<name>.glb       the carve's own output, never written to again
OUTPUT_K/carve_manifest.json          up axis, floor z, gravity, per-object record
   |
   |  scripts/tools/backfill_objects.py   (reads objects_raw/, writes objects/)
   v
OUTPUT_K/objects/<name>.glb           scan front + generated back, where registration passed
   |
   |  scripts/sim/make_factory_physics.py
   v
OUTPUT_K/sim/factory_physics.xml      MuJoCo scene: static room + free bodies
OUTPUT_K/sim/assets/*.obj             visual meshes and convex colliders
```

Viewer: `OUTPUT_K/viewer/index.html`, served locally.

### Reproduce

```bash
cd "<REPO_ROOT>"  # e.g. cd /path/to/Videos_to_3D

python3 scripts/tools/carve_objects.py --no-close-objects \
  --objects obj_0054_machine obj_0180_workbench obj_0179_workbench \
            obj_0202_blade obj_0203_trash_can obj_0185_box obj_0186_box \
  --extra obj_0300_crate:5.55,5.70,-1.53,6.67,6.33,-0.99

python3 scripts/tools/backfill_objects.py \
  --objects obj_0203_trash_can obj_0185_box obj_0186_box obj_0202_blade \
            obj_0054_machine obj_0179_workbench obj_0180_workbench

.venv/bin/python scripts/sim/make_factory_physics.py          # slow: CoACD on the room
.venv/bin/python scripts/sim/validate_factory_physics.py

cd outputs-full-factory/delivery/OUTPUT_K && python3 -m http.server 8765
# then open http://127.0.0.1:8765/viewer/index.html
```

**Use `.venv/bin/python` for anything needing `coacd` or `fast_simplification`.** The system
python3 has neither, and a lot of this session was spent working around their absence before
the venv was noticed. The venv lacks `rtree`, `shapely` and `mapbox_earcut`, so `trimesh.ray`,
`trimesh.proximity.closest_point` and `triangulate_polygon(engine="earcut")` are all unavailable
in both interpreters — the tools use `scipy.spatial.cKDTree` instead throughout.

---

## Objects in the scene

| object | scan faces | back-fill | notes |
| :-- | --: | :-- | :-- |
| `obj_0054_machine` | 14,580 | rejected, 5.7 cm | back still open |
| `obj_0179_workbench` | 6,403 | rejected, 8.6 cm | generated mesh is a slatted panel |
| `obj_0180_workbench` | 5,696 | rejected, 7.0 cm | back still open |
| `obj_0203_trash_can` | 915 | +4,468 gen faces, 4.5 cm | closed |
| `obj_0185_box` | 420 | +5,489 gen faces, 4.2 cm, yaw **180°** | closed |
| `obj_0186_box` | 394 | +309 gen faces, 2.9 cm | closed |
| `obj_0202_blade` | 202 | skipped — scan already covers it | tiny fragment |
| `obj_0300_crate` | 463 | no generated mesh exists | hand-specified region |

### Support chain (verified by replaying the viewer's raycast in Python)

```
obj_0185_box     rests on  obj_0179_workbench
obj_0186_box     rests on  obj_0179_workbench
obj_0300_crate   rests on  obj_0179_workbench
obj_0202_blade   rests on  the room
the rest         rest on   the floor plane
```

Nothing falls forever.

### Findings about the source data

- **`obj_0179_workbench` and `obj_0219_table` are the same physical bench, detected twice.**
  Only 0179 is carved. Carving both would double-cut.
- **The third crate in the photo has no usable detection.** It came back as `obj_0181_box` with
  a 5 × 3 × 18 cm OBB — a sliver. Found geometrically instead (a connected component on the
  bench top: 572 faces, 0.598 m², 51 cm tall) and carved via `--extra` as `obj_0300_crate`.
- **Stage G's yaw is close to a coin flip on slab-shaped objects.** `obj_0179` scores
  `[0.360, 0.213, 0.343, 0.246]` across the four quarter-turns; 0° and 180° are nearly tied.
  That is why registration has to search yaw rather than trust the stored transform.
- **`OUTPUT_J/obj_0179_workbench/mesh_textured.glb` is itself a ribbed panel**, not a workbench.
  The "ugly ladder" seen in the viewer came from Hunyuan's own generation, not from the fusion.
  Rendered unmodified at `OUTPUT_K/review_raw_gen.png`.

---

## Bugs found and fixed, with the evidence

Each of these produced a wrong result that looked plausible, which is why the numbers matter.

### 1. Hole patches were detached discs, not patches

`patch_holes` allocated **fresh** rim vertices and fanned onto those, so the cap floated beside
the hole and the hole stayed open. Evidence: the room had 2,121 open boundary loops before the
carve and **2,440 after "patching 405 loops"** — patching *increased* the count.

Fix: the rim reuses the loop's original vertex indices.

### 2. Vertex colours were destroyed on export

The colour array was built with `np.vstack` + `.round()`, making it `float64`. `ColorVisuals`
reads float colours as 0–1, so every value ≥ 1 clamped to white. The whole room exported at
mean colour `[255, 255, 255]`.

Fix: cast to `uint8`. `carve_objects.py` now raises if the carved room's colour std is under 5.

### 3. Delaunay left concave rim edges uncovered

Delaunay triangulates the convex hull; filtering by polygon containment then drops triangles and
leaves rim edges belonging to no triangle. On the jagged boundary of a torn scan this reopens the
hole. Fix: detect uncovered rim edges and stitch each to its nearest interior grid vertex.

Connecting them all to one shared hub instead — which was tried — covers every edge but paints
the patch with long radial spokes, clearly visible in the render.

Loop count through the fixes: **+319 → +68 → −93** (2,121 → 2,028; the carved room is now more
closed than the original).

### 4. Patches rebuilt the object they replaced — the "ghost table"

Interior patch points took their depth by IDW from the nearest room vertices. Next to a big hole
those are the **rim** vertices, which lie on the removed object's own silhouette. So the patch
reconstructed a table-shaped lid at table height.

Measured: 3.13 m² still inside the bench's OBB after carving, of which **1.94 m² was the patch**.

Fix: vertices on an open edge are excluded from the background KD-tree entirely, and the interior
estimate blends in with distance from the rim. The patch now drapes to the floor — 67% of the
patch inside the bench's box sits within 10 cm of the floor plane.

### 5. Floor protection preserved the object's own base

Faces within 3.5 cm of the floor plane with vertical normals were protected as "floor". A bench's
bottom shelf lies in that same band, so a table-shaped footprint was left behind — 1.05 m² of it.

Fix: a face counts as floor only if the connected surface it belongs to continues at least 30 cm
past the object's box. Real floor does; an object's base does not.

### 6. ICP cheated by shrinking

Registration was run with scale free. Trimmed ICP has a degenerate escape: shrinking the source
collapses it into the covered patch and drives the trusted residual down while wrecking the fit.
It pulled `obj_0179_workbench` to **0.558×** and `obj_0180_workbench` to **0.613×**, reporting
*better* residuals while being far more wrong.

Fix: rigid-only by default (`--allow-scale` to re-enable). Stage G already fixed size from the OBB.

### 7. The backfill overwrote the file it read

`backfill_objects.py` wrote to `objects/<name>.glb` and read from the same path, so a second run
fused on top of the first. `obj_0185_box` went from 420 scan faces to 6,278.

Fix: `carve_objects.py` writes `objects_raw/` (never touched again); the backfill reads from there
and writes to `objects/`. Verified idempotent — two consecutive runs produce identical numbers.

### 8. The viewer dropped objects into the void

```js
if (!isFinite(clearance)) { body.mesh.position.y -= wanted; }   // "let it leave"
```

Dragged through a wall there is no triangle below to hit, so the object fell forever.

Fix: three layers — an unbounded floor plane at the floor height (support exists at every xz),
a bounding box 6 m larger than the room, and recall to the home position past that.

### 9. Drag-clamp rays could not see what they had sunk into

`supportBelow` casts down **from** the feet, so once a body is under the bench the bench is behind
the ray. A separate `liftOutOfFloor` starts its rays 30 cm above each foot, which is what makes an
already-penetrated surface visible at all.

### 10. Convex-hull backing covered the front

Tried and reverted. A convex hull lies outside every concavity, so on a one-sided scan shell it
occludes the real surface — the workbench rendered as a plain grey wedge. `--fallback none` is now
the default; `hull` is kept only for experiments and documented as unsafe.

### 11. The material table missed every class with a space in it

`MATERIALS` was keyed `trash_can`; the VLM writes `trash can`. The dict lookup fell straight
through to the default, and the bin silently got a workbench's friction (0.60 instead of 0.35).

Fix: `material_for()` normalises the key and **reports a miss** rather than defaulting quietly.

### 12. Convex room colliders fill the voids the carve created

The validator's first run: **77.8 cm of penetration before a single step**, and bodies drifting
73–402 cm during settling — `obj_0185_box` was thrown 4 m.

| pair | worst penetration |
| :-- | --: |
| `obj_0054_machine` ↔ room | 77.8 cm |
| `obj_0179_workbench` ↔ room | 41.4 cm |
| `obj_0180_workbench` ↔ room | 32.5 cm |
| `obj_0203_trash_can` ↔ room | 20.0 cm |
| `obj_0186_box` ↔ `obj_0300_crate` | 7.8 cm |

Carving an object leaves a **concave void**, and a convex hull cannot represent a void. CoACD
spans it, so the room collider is solid exactly where the object stands.

### 13. Two wrong fixes for 12, and what the numbers said

**Vertex test.** `carve_hulls` first checked whether a hull's **vertices** fell inside an object's
box. A large collider can span the void with every vertex far outside it — precisely the machine's
case. Penetration stayed at 77.8 cm and drift got *worse* (380 cm).

**Delete instead of clip.** Dropping every intruding hull took penetration to 12.0 cm and then all
eight bodies **fell through the world at 29 m/s, travelling 44 m** — the dropped hulls were the
floor they stood on.

**What works:** test whether any point *inside the box* is also inside the hull (convex, so:
behind every face plane), then clip the hull by the box's six outward half-spaces. `hull minus box`
is not convex, but the six pieces `hull ∩ outside-of-face-k` cover it and MuJoCo tolerates the
overlap. The count now *rises* (94 → 174) as clipping should.

`trimesh.intersections.slice_mesh_plane` needs `shapely`, which is not installed, and a bare
`except Exception: continue` around it hid that completely — every slice failed and the carve
silently degraded to delete-only. `clip_convex()` does it in numpy instead, and the exception path
raises.

### 13b. Earlier note kept for the record

`carve_hulls` checked whether a hull's **vertices** fell inside an object's box. A large collider
can span the void with every vertex far outside it — which is precisely the machine's case.
Penetration stayed at 77.8 cm and drift got worse (380 cm).

Fix: test whether any point **inside the box** is also inside the hull (convex, so: behind every
face plane), then slice the hull by the box's six outward half-spaces. `hull minus box` is not
convex, but the six pieces `hull ∩ outside-of-face-k` cover it, and MuJoCo tolerates their overlap.

### 14. The hand-drawn crate region swallowed the boxes under it

`--extra obj_0300_crate:5.55,5.70,-1.53,6.67,6.33,-0.99` spanned z from the bench top all the way
up, so it **contained** `obj_0185_box` (x 5.55–6.09, y 5.93–6.39, z -1.24…-0.97) outright. Face
claiming split the triangles, but the convex hulls of the crate's pieces still enclosed the boxes:
12.0 cm of body-vs-body penetration.

Corrected to `5.57,5.72,-1.53,6.67,6.31,-1.26` — only the part sitting *above* the boxes.

### 15. Reusing carved colliders as input compounded the carve

`--reuse-assets` reloaded `room_collider_*.obj` — already-carved output — and carved it again.
Each rebuild shrank the set instead of growing it (93 → 77, when slicing should *increase* the
count). Fix: CoACD's output is cached as `room_raw_*.obj` and never modified; colliders are
derived from it fresh on every run.

---

## Tools written this session

### `scripts/tools/carve_objects.py`

Carves objects and patches the room. Notable behaviour:

- Faces are claimed **once**, smallest OBB first, so a crate on a bench keeps its own geometry
  instead of being swallowed by the bench. 0179 yielded 371 faces to the crates; 0180 yielded 915
  to the trash can, which sits inside its box.
- `--extra NAME:x0,y0,z0,x1,y1,z1` carves a hand-specified axis-aligned region, for objects stage G
  got wrong.
- `--grey-below` / `--grey-feather` / `--grey-colour` repaint unobserved surface — 8,947 vertices
  fully repainted, 13,259 touched by the blend, at a luminance threshold of 26 with a 14-wide band.
  1.51% of faces (3.84 m² of 225 m²) were dark enough to read as holes rather than geometry.
- `--close-objects` caps each object's own boundary loops (a pure centroid fan, provably closed).
  **Currently run with `--no-close-objects`**, because the backfill is the better answer and the
  fan caps read as umbrella spokes.

### `scripts/tools/backfill_objects.py`

Registers `OUTPUT_J/<name>/mesh_textured.glb` to the carved scan piece and keeps only what the
scan does not cover within `--cover` (5 cm).

- Vertex-cluster decimation, because no decimation library is available under system python and
  the generated meshes run 367k–1M faces.
- **Trimmed ICP**: fit on the best-matching `--keep` fraction (0.55) only. Plain ICP fails here —
  the target is a one-sided surface, so every generated back-face would drag toward the front.
- Four yaw starts, best residual wins. `obj_0185_box` turned out to be at 180°.
- `--max-residual` (5.5 cm) rejects a generated mesh that will not register. Three of seven are
  rejected, and they keep their open scan shells.

### `scripts/tools/render_scene_png.py`, `review_carve.py`

Offscreen renderers written from scratch: samples the surface in proportion to area, projects
through a pinhole camera, resolves visibility with a z-buffer. No GPU, no OpenGL. Needed because
`pyrender`, `pyglet` and `open3d` are all absent, and `trimesh.ray` needs `rtree`.
Camera clearance is probed with a KD-tree walk instead of raycasting.

Review images in `OUTPUT_K/`: `review_pairs.png`, `review_backfill.png`, `review_ghost.png`,
`review_raw_gen.png`.

### `outputs-full-factory/delivery/OUTPUT_K/viewer/index.html`

three.js from CDN, `three-mesh-bvh` for raycasting. Geometry is rotated +90° about X at load so
the scan's `-Z` up becomes three.js `+Y`, and everything downstream can assume Y-up.

- Gravity 9.81 m/s², 10 kg per object, quadratic air drag (the only place mass enters the motion).
- Support comes from the room, every **other** body, and the unbounded floor plane.
- `wakeAll()` on any movement: a body that moves can pull the ground out from under whatever is
  standing on it, so nothing stays asleep on a stale contact.
- Dragging passes through walls sideways but is clamped at whatever is directly below.
- No rotation — carved objects are open shells and rotating one shows it is hollow.

### `scripts/sim/make_factory_physics.py` + `validate_factory_physics.py`

MuJoCo 3.11 scene. Room as static terrain, all eight objects as free bodies.

| class | friction (slide/torsion/roll) | solref | mass |
| :-- | :-- | :-- | --: |
| machine | 0.70 / 0.030 / 0.0010 | 0.008 1.00 | 10 kg |
| workbench, table | 0.60 / 0.020 / 0.0008 | 0.010 1.00 | 10 kg |
| box, crate | 0.55 / 0.010 / 0.0005 | 0.012 0.95 | 10 kg |
| trash_can | 0.35 / 0.005 / 0.0002 | 0.018 0.70 | 10 kg |
| blade | 0.45 / 0.010 / 0.0004 | 0.006 1.00 | 10 kg |
| room (static) | 1.00 / 0.020 / 0.0010 | — | — |

- Everything collidable uses **`condim="6"`**. At the default condim 3 the torsional and rolling
  friction numbers do nothing and a crate spins on the spot forever.
- Both the room and every object go through CoACD. MuJoCo collides a mesh geom as its convex hull,
  so an undecomposed workbench is a solid wedge with no space underneath it.
- Body mass is split across hulls by **volume**, not evenly, so the inertia tensor is right.
- Visual meshes carry `mass="0"`, `contype="0"`, `conaffinity="0"`, group 2.

---

## Physics scene: measured

`OUTPUT_K/sim/factory_physics.xml` — 8 free bodies, 275 geoms, 174 room colliders.

How penetration and drift moved across five builds:

| build | room colliders | penetration at t=0 | heavy-body drift |
| :-- | --: | --: | :-- |
| 1 — no object volume removed | 94 | 77.8 cm | 143–380 cm |
| 2 — vertex test (wrong) | 93 | 77.8 cm | 143–380 cm |
| 3 — delete intruding hulls | 75 | 12.0 cm | **free fall, 44 m at 29 m/s** |
| 4 — proper half-space clipping | 174 | 12.0 cm | 4.3–12.8 cm |
| 5 — crate region corrected | 174 | **6.3 cm** | **3.1–12.8 cm** |

Final settle, 3 s:

```
obj_0054_machine       3.1 cm    at rest
obj_0179_workbench     6.2 cm    at rest
obj_0180_workbench    12.8 cm    at rest
obj_0185_box          26.0 cm    at rest
obj_0300_crate        55.6 cm    at rest
obj_0186_box          76.3 cm    at rest
obj_0203_trash_can    99.8 cm    still moving at 0.164 m/s
obj_0202_blade       127.6 cm    at rest
```

Friction, 40 N for 0.25 s then 2.5 s to coast: everything slides 0.0–1.7 cm and stops, except the
trash can. That range is right for 10 kg at mu ~= 0.5 — static friction is about 49 N, so a 40 N
shove barely breaks it loose.

**Still wrong:**

- **The trash can never stops.** Friction 0.35 with rolling friction 0.0002 on a round bin: it
  rolls. Physically plausible, but it fails the settle test and needs either more rolling friction
  or a flat-bottomed collider.
- **The small objects walk off the bench** — 26 to 128 cm of drift. They start about 10 cm above
  the bench top because their undersides were never scanned, and they slide off as they land.
  Baking a settled keyframe would freeze them wherever they ended up, destroying the
  boxes-on-a-bench arrangement, so it has not been done.
- **6.3 cm of residual penetration** remains, now object-vs-room on the blade and the machine.

## Outstanding

1. Stop the trash can rolling, and stop the small objects sliding off the bench. Both need the
   colliders to be better, not the friction numbers to be tuned.
2. `obj_0054_machine`, `obj_0179_workbench`, `obj_0180_workbench` still have **open backs**. Their
   generated meshes cannot be registered under 5.5 cm and 0179's is the wrong shape entirely.
   The only real fix is re-running stage D→F on a GPU with better inputs
   (`OUTPUT_D_v2/adapter_a/<name>/mv_inputs/`). The user has an RTX 6000 Pro on another machine
   and Kaggle infrastructure; nothing has been deployed.
3. `obj_0179_workbench` and `obj_0054_machine` sit 12.2 cm and 8.1 cm **below** the floor plane
   after backfill — ICP placed the generated bottoms low. They rest fine, but grabbing one in the
   viewer makes it jump up to the floor.
4. The three crates drop ~10 cm on page load because their undersides were never scanned. The user
   chose to keep this as visible proof gravity is live.
5. 33 other objects in `OUTPUT_G` have not been carved. Append names to `--objects` to add them.
6. **The viewer has never been opened.** The user verifies in their own browser and declined
   browser automation. Every number in this document is geometry measured outside a browser.

## Working preferences recorded

- Do not use the Claude-in-Chrome tools; hand over a URL instead.
- Prefer the pipeline's existing generated meshes over inventing geometry with hole-capping,
  convex hulls, or plane fills.

Both are saved as memories under
`~/.claude/projects/-Users-traoanhuy-source-code-Full-pipeline-scan-terrain-Videos-to-3D/memory/`.
