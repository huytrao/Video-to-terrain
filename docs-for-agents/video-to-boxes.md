# Video in, accurate 3D with object boxes out

The goal in one line: hand the system a video, get back metric geometry and a
labelled oriented box per object. This is what has to be true for that to work,
what already is, and what is not.

Read `output-quality.md` for why the geometry is not yet accurate enough, and
`architecture-critique-and-factory-design.md` for the wider design. This file is
only about the shortest path to the stated goal.

---

## The chain, and where it breaks

```
video ──▶ frames ──▶ poses ──▶ scale ──▶ geometry ──▶ masks ──▶ boxes ──▶ scene.json
          ⛔         ⛔        ⛔         ✅           ✅        ◐         ◐
```

Everything from geometry rightwards runs today, on a dataset scene. Everything
to the left of it does not exist. That is the honest position: the hard half of
"just give it a video" has not been started, and the three missing pieces are
not equally hard.

**Frames** is easy. **Poses** is a solved problem with known tools. **Scale is
not recoverable from video at all**, and no amount of work downstream changes
that.

---

## 1. Frames

Sample evenly, then drop what cannot contribute:

- **blur** — variance of the Laplacian per frame, drop the worst decile
- **exposure** — discard frames that are largely clipped
- **redundancy** — a stationary camera contributes nothing; require a minimum
  baseline between kept frames
- **coverage** — after dropping, check no viewing direction lost all its frames

The last one matters more than it sounds. Dropping blurred frames is usually
safe, except when the blurred ones are the *only* views of a corner — then the
reconstruction loses that corner entirely and the mesh gains a hole nobody
attributes to the frame filter.

## 2. Poses

Two routes, and they fail differently.

**COLMAP** is accurate and slow, and fails on textureless surfaces — which is
most of a factory wall. When it fails it usually fails loudly, by not
registering images.

**DA3 streaming** (already vendored, `depth-anything-3/da3_streaming/`) has loop
closure and is fast, and drifts on long sequences. It fails quietly, which is
worse.

Either way the output must be the same nerfstudio-style `transforms.json` the
benchmark path already consumes, so that everything downstream is unchanged.
`v3d/config.py` already defines that shape as `flat_layout`.

**Check the poses before trusting them**: reproject a handful of triangulated
points and look at pixel error; verify the trajectory is continuous rather than
teleporting; confirm every kept frame actually registered.

## 3. Scale — the part that cannot be solved by trying harder

A room, and a photograph of a room built twice the size, produce identical
images. Scale is not in the pixels. Every ±5 cm claim in this project rests
entirely on how metres were established, and the previous pipeline's 8.6 m
ceiling is what happens when that question is skipped.

Four honest answers, best first:

| Route | How it works | What it costs |
|---|---|---|
| **RGB-D capture** | the depth stream is metric by construction | needs the right camera — and note the earlier RealSense bags had exactly this before the depth was discarded |
| **Known object in frame** | detect it, solve one scale factor | a printed marker or a standard pallet (1200 × 800 mm) in shot |
| **One measured distance** | user measures a doorway, pipeline anchors to it | one tape measurement per space |
| **VIO / ARKit poses** | the IMU supplies metric baselines | a capture app that exports poses |

And one dishonest answer: trusting a monocular depth model's metric mode. That
is what produced the 8.6 m room.

Whichever is used, `scene.json` must record **which one**, so a downstream
reader can tell a measurement from a guess. Stage J should refuse to report an
absolute error when scale is `assumed` — reporting relative proportion error
instead.

---

## 4. Making the boxes good

Masks already work: 9,326 of them over 261 frames, 31 classes. The box quality
depends on three things, none of which is the segmentation model.

**Which classes get looked for at all.** The vocabulary comes from a VLM prompt.
A generic prompt on a workshop returned `wood`, `board`, `pipe`. A factory needs
`pallet`, `forklift`, `conveyor`, `control panel`, `safety cage`. Nothing
outside the vocabulary can ever become a box, so this is the single highest
-leverage input — and it is a string.

**Instance separation.** Text-prompted masks give one mask per class per frame,
not per object. Five pallets in a row become one `pallet` mask, and lift3d then
produces one box around all five. Repeated objects are the normal case in a
factory, so instance splitting — connected components in 3D after lifting, or
per-instance prompting — is required, not optional.

**Multi-view consistency.** A box built from one frame's mask inherits that
frame's errors. The lifting stage already aggregates across views; the knobs are
`--min-points` (currently 30) and `--erode-pixels` (currently 15, which trims
mask boundaries that bleed onto neighbouring surfaces).

### Judging the boxes

Not by eye, and not by count. Against ground truth: matched fraction (recall),
dimensional error by size band, and centre error. Stage J does this already.
Without ground truth — every real video — what remains is:

- every box inside the room shell
- no box larger than a sensible fraction of the room
- boxes that should rest on a surface actually resting on it
- no two boxes of the same class overlapping by more than a threshold, which
  catches the duplicate-instance failure directly

---

## 5. What to build, in order

1. **Finish the benchmark chain.** Lifting and post-processing are wired but
   unrun, so the project has never produced a box from its own pipeline. Until
   it does, everything about box quality is speculation.
2. **Grade it.** Stage J against the 223 ground-truth instances. This is the
   first honest number the project will have.
3. **Instance splitting.** Without it, factories reconstruct as one giant
   `pallet`.
4. **Video ingest**, with scale provenance as a required field rather than an
   afterthought.
5. **Domain prompts**, so the vocabulary matches the space being scanned.

Steps 1 and 2 are finishing what exists. Step 3 is the first genuinely new
capability. Steps 4 and 5 are what turn it into "just give it a video".

---

## 6. What "accurate" should mean here

Worth deciding explicitly, because it drives everything above:

- **Geometry**: dimensional error on walls, doors and large fixtures. The
  current mesh is 12.8 cm over on room height, which is the number to beat.
- **Boxes**: recall of real objects, and size error on the ones that matter for
  a robot — not on the 6 cm cords the benchmark is full of.
- **Scale**: stated, not assumed. An output that says "proportions correct,
  absolute scale unverified" is more useful than one that quietly asserts
  metres it cannot support.

The last point is the one this project has already learned the expensive way.
A validator that reported sixteen passes on a room 8.6 m tall was worse than no
validator, because it converted an unknown into a false certainty.
