# Raising output quality

The pilot scene runs end to end, and the result is not good enough. This is the
list of what limits it, ordered by expected gain per unit of effort, with the
parameter that controls each one and its value in the run that produced the
current output.

Every item names how to tell whether the change helped, because "looks better"
has already misled this project once — see `bugs-found.md`.

---

## What the current output actually looks like

Measured on `00777c41d4`:

| | Value |
|---|---|
| PGSR | 30,000 iterations, PSNR 26.77, L1 0.0296, 1,247,783 gaussians |
| Mesh | 9,517,867 vertices, 18,543,482 triangles |
| Room height | mesh **3.432 m** vs ground truth **3.304 m** — **12.8 cm over** |
| TSDF voxel | 0.006 m |
| Fusion depth limit | 4.0 m |
| Input | 261 frames, 1752×1168, resolution 1 |

Visible in the renders: surfaces are soft and waxy rather than flat; boundaries
of the observed region are ragged; thin structures — railings, cables, table
legs — are either bloated or missing; isolated spikes stick out of otherwise
clean walls.

---

## 1. Fusion depth limit is truncating the room (highest impact)

`mesh.sh` defaults `RENDER_MAX_DEPTH=4.0`, and `v3d.paper_runner` keeps it to
stay faithful. The room is **8.0 × 9.1 m**. Any surface more than 4 m from a
camera is discarded during TSDF fusion, so far walls are reconstructed only
from the few frames that happened to stand close to them — which is exactly
where ragged edges and holes appear.

DA3's own depth range on this scene is **0.44–4.94 m**, so there is real signal
past 4 m being thrown away.

**Try** `RENDER_MAX_DEPTH` 6.0–8.0. Raising it also admits more noise at range,
so pair it with §2.

**Measure** hole area on the floor and walls, and the height error. If far
surfaces were being lost, coverage rises and the 12.8 cm height error should
move.

## 2. Depth filtering is switched off

`mesh.sh` defaults `USE_DEPTH_FILTER=0`, so `render.py` fuses raw rendered depth
with no multi-view consistency check. That is the standard cause of the spikes
and the waxy inflation around thin structures: a depth value wrong in one view
gets fused as if it were real.

**Try** `--use_depth_filter`. It is a flag `render.py` already accepts.

**Measure** count of connected components before cluster filtering — currently
**140,334**, which is a lot of floating debris — and Chamfer distance against
the released reference mesh.

## 3. TSDF voxel size sets the ceiling on detail

`VOXEL_SIZE=0.006` (6 mm). No amount of downstream work recovers detail finer
than this, and a 6 mm voxel cannot represent a 3 mm cable or a sharp table edge.

**Try** 0.004 or 0.003. Cost is roughly cubic in memory and time; at 0.003 the
mesh will be several times its current 9.5 M vertices, so decimate for delivery
rather than for computation.

**Measure** Chamfer distance, and whether thin objects survive as separate
components rather than merging into their support.

## 4. Gaussian count is low for a room this size

1.25 M gaussians across roughly 70 m² of floor and 8 m of ceiling height is
thin. PGSR densifies until a gradient threshold stops firing; the default is
tuned for object-scale captures, not rooms.

**Try** a longer schedule (50–60 k iterations) and a lower densification
gradient threshold. PSNR 26.77 at 30 k after 25.03 at 7 k was still climbing —
the curve had not flattened, which is direct evidence training stopped early.

**Measure** PSNR at the new iteration count, and Chamfer on the extracted mesh.
PSNR alone is not enough: it rewards texture fidelity and is nearly blind to a
wall bowed by 10 cm.

## 5. Frames are not screened

All 261 frames go in regardless of blur, exposure or redundancy. A motion-blurred
frame contributes wrong gradients to every gaussian it sees.

**Try** a variance-of-Laplacian sharpness score, drop the worst decile, and
check pose coverage so dropping frames does not open a gap.

**Measure** PSNR on a held-out set — not on training frames, where dropping the
hard ones flatters the number by construction.

## 6. The single-cluster filter may be discarding real geometry

`NUM_CLUSTER=1` keeps only the largest connected component after fusion. That
removes debris, and it also removes anything genuinely disconnected — a shelf
seen only through a doorway, an object on a high rack.

**Try** 3–5 clusters and inspect what the extra ones contain.

**Measure** recall against the ground-truth boxes: an object whose geometry was
dropped cannot be lifted into the scene graph, so this shows up as missing
instances rather than as visual error.

## 7. Appearance modelling is on but unexamined

PGSR saved an `app_model`, meaning per-image appearance compensation is active.
That helps PSNR under changing exposure and can bake lighting into geometry if
it absorbs error that belongs to the surface.

**Try** one run with it disabled, and compare mesh accuracy rather than PSNR.

---

## How to judge any of this

Three numbers, and none of them is sufficient alone:

1. **Chamfer distance** to the released reference mesh in
   `GS_model_w_mesh_pp/*.tar` — the authors' own reconstruction of the same
   scene. This is the only direct measure of surface accuracy and it does not
   exist yet (`docs/reference/deb-technical.md` P2-3).
2. **Dimensional error** through Stage J, read by size band. The `large` band —
   walls, doors, cabinets — is what matters for simulation; a PASS carried by
   6 cm cords is not a pass.
3. **Held-out PSNR**, which measures whether the radiance field explains images
   it never trained on.

Change one thing at a time. Six parameters interact: raising the depth limit
without depth filtering adds noise, and shrinking the voxel without more
gaussians produces finer sampling of the same blur.

---

## Where the real ceiling is

Items 1–7 are tuning, and tuning will not fix the two structural limits.

**A mesh is the wrong representation for thin structures.** Railings, cables and
cage panels are thinner than any practical voxel. They need primitive fitting —
cylinders and planes — driven by the semantic labels the pipeline already
produces. That is a new stage, not a parameter.

**Sharp edges need planes, not samples.** Room corners are the intersection of
two planes, and TSDF rounds them because it samples a distance field. PGSR's
planar regularisation helps during training but the extraction throws the
structure away. Fitting planes to the wall and floor instances and snapping
geometry to their intersections would produce corners that are actually sharp —
and would improve the ±5 cm dimensional numbers directly, because a rounded
corner biases every wall-to-wall measurement.

Of everything on this page, that last one is most likely to move the 12.8 cm
height error, since floor and ceiling are exactly the surfaces whose plane fits
would be best constrained.
