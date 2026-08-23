# lift3d and postprocess — the first run, and what it actually produced

`docs/reference/deb-technical.md` carried these as **P1-2** and **P1-3** across several sessions:
lifting existed in the paper's repo, had a wrapper in `v3d/paper_runner.py`, and had
never once been executed. So the project had never produced a 3D box from its own
pipeline, and everything said about box quality was speculation.

They ran on 2026-08-18, on `small_factory` — 600 frames from one handheld video.
This records what it took, what came out, and the finding that matters more than the
milestone.

---

## 1. What was actually blocking it

Four things, none of which was the lifting code:

**`nvdiffrast` (P1-1).** `PGSR/mesh2mask.py` imports it, and without masks there is no
SAM3 input, no lifting, no boxes. It is not on PyPI as a wheel for this platform.
Built on the internet-enabled kernel for `sm_120` in 27 s, relayed, and verified by
rasterising a test triangle: **1352 covered pixels**, the same figure the earlier
benchmark run recorded, so the CUDA backend agrees with the previously-verified one.

**The SAM 3 gate.** `facebook/sam3` is `"gated": "manual"`; both checkpoint URLs return
HTTP 401 unauthenticated. Resolved with an approved token. The runbook's warning that
`sam3.1_multiplex.pt` is *not* a substitute was confirmed and then made moot: the real
`sam3.pt` state dict (1465 tensors, prefixes `detector`/`tracker`) contains all four
tensors the 3.1 checkpoint lacks —
`detector.backbone.vision_backbone.convs.3.conv_1x1.{weight,bias}` (256×1024×1×1) and
`…convs.3.conv_3x3.{weight,bias}` (256×256×3×3).

**Two offline traps in the SAM3 package.** The paper vendors an *incomplete* `sam3/`
with no `sam3/train/`, which five core modules import from, so it cannot build a model
at all — replaced with the official package. And Kaggle **decompressed**
`assets/bpe_simple_vocab_16e6.txt.gz` during dataset upload, while `model_builder.py`
resolves exactly the `.gz` name through `pkg_resources`; since the mount is read-only
the package was copied to `/tmp` and the archive restored. `ftfy` also had to go in
`--no-deps`, because SAM3's `pyproject.toml` pins `numpy>=1.26,<2` and a normal
resolve drags torch down with numpy.

**`manifold3d`.** lift3d performs mesh booleans through `trimesh`, which raises
`ImportError: No boolean backend` without it. This is the one that actually stopped
lift3d on its first attempt. Fetched, relayed, and verified with a real boolean
(difference of two unit cubes, volume 0.5).

Checked pre-emptively before the run, because `docs/reference/deb-technical.md` says to expect a
hardcoded path or a dtype assumption: every argument `stage_lift3d` passes exists in
the script's `parse_args`; `postprocess`'s `--floor_label` default `"floor"` matches a
class the VLM actually discovered; and `sam3.agent.helpers.visualizer`, which lift3d
imports and which is Holi-Spatial's own addition, does exist in the official package
that replaced the vendored one. `--help` on both scripts exercises every top-level
import.

---

## 2. What ran

| Stage | Output | Result |
|---|---|---|
| SAM3 | `sam_masks/…/mask_index.json` | **14,905 masks**, 600/600 images, 22 of 23 classes, median confidence 0.754 |
| lift3d | `boxes_raw/small_factory.json` | 21.22 MB, **95 instances**; also a GLB with 5 M points + 95 boxes |
| postprocess | `boxes/small_factory.json` | 21.22 MB, **95 instances** floor-aligned |
| `build_scene` | `scene.json` | **82 objects** (13 shell instances dropped) |
| Stage J | `stage_j.json` | **FAIL** — see §4 |

lift3d aggregated **1,123,373,795 points** before downsampling. `rerun` is absent and
its import is guarded, so only visualisation was lost, exactly as documented.

---

## 3. The finding that matters more than the milestone

The 82 objects break down like this:

| category | count | | category | count |
|---|---:|---|---|---:|
| **label** | **26** | | light | 3 |
| **sticker** | **23** | | machine | 3 |
| cable | 7 | | wood | 3 |
| pipe | 5 | | sign | 2 |
| window | 5 | | handle | 1 |
| door | 4 | | | |

**Forty-nine of 82 objects are `label` and `sticker`** — flat printed markings, not
things a robot manipulates. Meanwhile nine classes that SAM3 *did* find masks for were
filtered to **zero** 3D instances by lifting:

```
box  button  gauge  knob  lever  panel  switch  table  tool
```

That list is most of the machinery in a control-panel factory. `machine` survived with
3 instances; `panel`, `switch`, `lever`, `knob`, `button` and `table` produced none.

This is precisely the asymmetry `docs/reference/deb-technical.md` **P2-2** predicts — "a PASS driven by
hundreds of `sticker` and `cord` fragments while the `large` band fails is a failure" —
observed for the first time on real output rather than argued from the benchmark's
annotation statistics. The lifting filters (`--min-points 30`, `--erode-pixels 15`,
and the iterative re-merge) survive flat high-contrast markings and discard the small
3D structures that matter, because a switch subtends few pixels and few points while a
printed label is large, flat and unambiguous.

**So "lift3d runs" is true and is not the same as "the semantic boxes are useful."**
The next work on this stage is not making it run; it is `--min-points` and
`--erode-pixels`, and probably per-instance prompting, so that a lever becomes an
instance and a sticker does not dominate the count.

---

## 4. Stage J fails, and the gate is right to fail

```
[FAIL] no_assumed_room_geometry    room.ceiling marked 'assumed'
[FAIL] no_object_larger_than_room  room volume 20.9; oversized obj_machine_43, 44
[FAIL] no_stray_objects            36/82 reach past the footprint, worst 0.92, limit 0.28
```

The last two share one cause. `scene.json`'s floor polygon is a thin quadrilateral
spanning **5.90 × 1.43**, and three independent witnesses disagree with it:

| witness | footprint |
|---|---|
| fused mesh extent | 6.013 × 6.348 |
| camera path extent | 4.922 × 3.454 |
| **mesh floor plane** | **3,798,304 points spanning 96.3% of the footprint** |

The mesh floor covers essentially the whole footprint, so the lifted floor is
**under-segmented** — it captured a walkable strip. Objects standing beside that strip
then read as "stray", and the room volume computed from it makes two machines look
oversized.

**Do not loosen the gate to make this pass.** A validator in this project once
returned 16/16 PASS on a room reconstructed 8.6 m tall; this one refused to certify a
scene whose shell it could not reconcile, which is the behaviour that was wanted.

The fix is to derive the room shell from the **mesh**, which `v3d.scene.decompose`
already does correctly — its RANSAC shell detection finds one floor (span 0.963), five
walls and a ceiling, and its audit passes 5/5 — and to have `v3d.scene.from_holi`
consume that instead of the lifted floor instances.

---

## 5. Two routes to objects, and they are complementary

The same mesh yields two object sets, and neither is simply better:

| | semantic (SAM3 → lift3d) | geometric (`decompose.cut`) |
|---|---|---|
| objects | 82 | **41** |
| labels | yes, 11 categories | none |
| dominated by flat markings | **yes**, 49 of 82 | no |
| room shell | 5.90 × 1.43 (wrong) | floor spans 96.3% |
| self-audit | Stage J **FAIL** | **PASS** 5/5 |
| needs a gated checkpoint | yes | no |

The geometric cut defines objects soundly and cannot name them; the semantic route
names them and mis-defines the shell. The composition that follows from this — geometry
decides *what is an object*, semantics decides *what it is called* — is what
`optitional_output_proposal.md` argues for, and this run is the evidence for it rather
than the motivation.
