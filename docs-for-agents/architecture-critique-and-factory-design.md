# Where this architecture breaks, and what a factory pipeline needs instead

Two halves. §1 is an honest account of what is weak in what exists today, each
point grounded in something that actually went wrong during the run rather than
in general principle. §2 designs the thing the project is actually for: drop a
video in, get a simulable factory, with everything tunable and sane defaults
when nothing is given.

---

## 1. Weaknesses of the current architecture

### 1.1 Scale has no provenance, and nothing downstream can detect a wrong one

The pipeline consumes frames plus a camera transforms file and trusts whatever
scale those imply. On ScanNet++ that scale is a dataset fact. On a video it is
a guess, and **no later stage can catch a bad guess** — the mesh, the boxes and
the physics will all be self-consistent and all wrong together.

This already happened: the abandoned v2t pipeline reported a room **8.6 m tall**
and its validator passed it sixteen times out of sixteen.

`scene.json` should carry a `scale` block naming *how* metres were established
(depth sensor / reference object / measured distance / assumed), and Stage J
should refuse to report an absolute error when the answer is "assumed".

### 1.2 Stages talk through the filesystem with no manifest

Each stage writes files where the next one looks. `stage_report` checks that
those files **exist**, not that they match the current inputs. Re-run the VLM
with a different prompt and the old SAM3 masks stay on disk, still valid-looking,
and lift3d will happily consume them.

During this session the VLM ran four times with different consolidation rules.
Nothing in the pipeline would have noticed if masks from run 1 had been paired
with classes from run 4.

Fix: a per-scene manifest recording, for each stage, the input hashes, the code
version, the parameters and the output paths. A stage is "done" only when the
manifest entry matches what would be produced now.

### 1.3 Validation exists only at the end

Stage J grades the final scene. Everything before it is checked with "did the
process exit 0". PGSR runs for **54 minutes**; a bad mesh is discovered after
mesh extraction, mask rendering, ten minutes of VLM and SAM3 — roughly ninety
minutes of work on a foundation already known to be wrong at minute fifty-five.

Each stage should assert its own postcondition against a cheap, falsifiable
check: depth maps finite and in a plausible metric range; gaussian count and
PSNR within band; mesh extent within a factor of the camera trajectory's extent;
mask coverage per class non-trivial. Every one of those numbers was computed
manually during this run — they should be code.

### 1.4 The semantic chain has no feedback

The VLM invents a vocabulary, SAM3 finds what it can (31 of 35 classes), lift3d
takes whatever survives. No stage tells the previous one it produced something
unusable. Four classes yielded zero masks and nobody was informed.

Worse, the failure mode is silent *and* self-amplifying: the prompt carries the
accumulated vocabulary forward, so one bad label is repeated back to the model
for every remaining frame. That is how 26 of 58 classes became elaborations of
"wood".

### 1.5 Coordinate conventions are assumed, never asserted

`+z` points **down** in this scene — ground truth puts the floor at z = −0.04
and the ceiling at z = −3.33. The floor-plan renderer assumed otherwise and
produced a picture that looked entirely reasonable and was upside down.

The project has now been bitten three times by conventions: quaternion order,
gravity direction, and axis-aligned-versus-oriented boxes. Frames should be
derived from data and asserted once at the boundary, not assumed per module.

### 1.6 The mesh is a rendering artefact, not a simulation asset

18,543,482 triangles, not watertight, no convex decomposition, no material
semantics. MuJoCo or Isaac cannot use it directly; a robot cannot collide
against it at any useful rate. Producing a *visual* mesh and a *collision*
representation are different jobs and only the first is done.

### 1.7 Objects are boxes, not mechanisms

Drawers, doors and machine guards come out as oriented boxes. For a robotics
environment they are **articulated bodies with joints, limits and handles** —
which is most of what makes an environment interesting to a manipulation
policy. The current output cannot express "this drawer slides 40 cm along its
local X".

### 1.8 Runtime patching of vendored source

Four patches are applied to the paper's code at run time by string replacement.
They are idempotent and recorded, which is better than editing in place, but
they will silently stop applying if upstream text shifts by a character —
`apply_offline_patches` reports "already applied" for both *applied* and *not
found*. That ambiguity should be an error.

### 1.9 Everything assumes one scene at a time

No batching, no parallelism, no queue. A factory has many rooms and many
capture sessions; the current design runs them one after another by hand.

### 1.10 The environment is not reproducible

25 GB of weights and wheels live in `/tmp`, which a session restart erases. The
build cache is on Kaggle now, which helps, but there is no single command that
takes a fresh kernel to a working one.

---

## 2. The factory pipeline

### 2.1 What the user actually asked for

```
video(s) in  ─────────────────────────────────▶  simulable factory out
     + optional prompt / config                  + Blender, MJCF, USD
     + nothing at all works too
```

Three input tiers, and the design must be usable at each:

| Tier | Input | Behaviour |
|---|---|---|
| **Zero-config** | just a video | everything defaults; scale is flagged `assumed`, absolute claims suppressed |
| **Guided** | video + prompt, e.g. *"factory floor with CNC machines, pallets, forklifts"* | prompt seeds the class vocabulary and the physics priors |
| **Full** | video + prompt + config + one measured distance | metric output, tuned thresholds, named articulation |

### 2.2 The prompt is not decoration — it is the class prior

The current VLM prompt is domain-agnostic and produced `wood`, `board`, `pipe`
for a woodworking shop. A factory needs `pallet`, `forklift`, `conveyor`,
`control panel`, `safety cage`, `machine guard`, `hose reel`, `barrel`. That
distinction decides what SAM3 looks for and therefore what ends up in the scene
graph at all.

```python
@dataclass
class SceneRequest:
    video: Path | list[Path]
    prompt: str | None = None            # "factory floor with CNC machines…"
    domain: str = "generic"              # generic | factory | warehouse | office
    scale_hint: ScaleHint | None = None  # measured distance, marker size, RGB-D
    target: tuple[str, ...] = ("blend", "mjcf")
    quality: str = "balanced"            # draft | balanced | max
```

`domain` selects a class prior and physics defaults; `prompt` refines it;
absent both, the generic vocabulary runs and the output says so. **The prompt
must never be able to silently narrow the scene** — anything the model sees but
the prompt did not mention still gets labelled, or a forklift outside the
prompt vanishes from the factory.

### 2.3 Stage graph

```
ingest      video → frames (sharpness-filtered, overlap-checked)
poses       COLMAP or DA3-streaming → transforms + trajectory
scale       depth sensor | marker | measured distance | assumed   ← recorded
geometry    DA3 → PGSR → TSDF mesh
semantics   VLM(prompt, domain) → SAM3 → lift3d → boxes
structure   floor/wall/ceiling planes, openings, support relations
articulate  doors, drawers, panels → joints with limits
physics     mass, friction, convex collision decomposition
scene.json  the single source of truth
export      Blender · MJCF · USD · GLB
validate    Stage J + per-stage postconditions
```

Only the last four are new work of substance; the middle is what already runs.

### 2.4 Factory-specific concerns the current design would get wrong

- **Scale matters more, not less.** A pallet is 1200 × 800 mm by standard. That
  is a free, verifiable ruler present in almost every factory frame — a stronger
  scale anchor than any generic reference object, and worth detecting explicitly.
- **Repeated identical objects.** Racking, pallets and drums repeat. Instance
  segmentation will fragment them; the scene graph should collapse repeats into
  one asset with many transforms, which also makes the sim vastly cheaper.
- **Large volumes, thin structures.** Mesh railings, cable trays and cage panels
  reconstruct badly and matter for collision. They likely need primitive fitting
  rather than mesh.
- **Reflective and moving machinery.** 3DGS assumes a static scene; a running
  conveyor or a specular machine housing violates that. Frames should be
  screened for motion, and moving regions excluded rather than fused into the
  static mesh.
- **Safety-critical clearances.** For a robot, the number that matters is the
  gap between the rack and the wall, not the rack's size. Stage J should gate on
  **free-space clearances** as well as object extents.

### 2.5 Sim-to-sim for robots

The target is a policy trained in one simulator transferring to another, and
ultimately to the real cell. What that demands of this pipeline:

**Geometry that survives the trip.** One canonical `scene.json` with metric
geometry, exported to MJCF and USD by two exporters reading the same file. If
either exporter reads meshes directly, the two simulators diverge and the
transfer gap becomes an artefact of the toolchain rather than of the physics.

**Physics with provenance.** Mass, friction and restitution should carry the
same `measured | derived | assumed` tag the geometry does. A policy that only
works because a friction coefficient was guessed at 0.8 is worth knowing about
before it reaches hardware.

**Collision separate from visuals.** Convex decomposition per object, plus a
simplified static shell. Target something on the order of hundreds of collision
primitives, against the 18.5 M-triangle visual mesh.

**Articulation as a first-class output.** Joints with axes, limits and handle
poses. A manipulation policy needs the drawer to open.

**Domain randomisation hooks.** The scene graph should expose what may vary —
friction ranges, lighting, object placement within support surfaces, texture
swaps — so sim-to-sim can sample rather than replay one fixed world.

**A transfer metric, not a vibe.** Same policy, same seeds, both simulators,
compare success rate and trajectory divergence. Report it the way Stage J
reports geometry: with a control that proves the metric can fail.

### 2.6 What to build first

1. **Per-stage postconditions** (§1.3) — cheapest, and it stops wasting
   ninety-minute runs on foundations already known to be bad.
2. **Manifest + hashes** (§1.2) — makes every later result trustworthy.
3. **Video ingest with explicit scale** (§1.1) — unlocks the actual goal.
4. **Collision + articulation** (§1.6, §1.7) — turns a picture into a simulation.
5. **MJCF and USD from `scene.json`** — makes sim-to-sim measurable at all.

The ordering is deliberate: 1 and 2 are small and make everything after them
verifiable, which is the lesson this project has now learned expensively three
times over — a result nobody can check is not a result.
