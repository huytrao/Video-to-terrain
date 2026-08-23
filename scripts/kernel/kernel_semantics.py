"""The semantic half: vocabulary, masks, 3D boxes, scene.json, and the gate.

Runs after the mesh. Each stage here is the paper's own code with only paths
substituted, and each is checked against a postcondition that can fail, because
this half of the pipeline fails *quietly* more than any other part:

  vlm         a bad label is repeated back to the model for every remaining frame,
              so one junk class becomes twenty-six. Four runs were needed on the
              benchmark scene before the vocabulary was usable, and every one of
              them reported rc 0 on all frames. The only thing that distinguished
              them was reading the list.
  sam3        four classes yielded zero masks on the benchmark and nothing said so.

So the vocabulary is printed in full, per-class mask counts are printed, and any
class with no masks is named. None of that is optional output — it is the only
place those failures are visible.

Do NOT subsample frames for the VLM. The prompt carries the accumulated vocabulary
into every later frame and instructs the model to reuse existing labels; skipping
frames breaks that continuity and the same chair becomes "chair" in one frame and
"office chair" in another.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")
DA3_SRC = "/tmp/da3src/Depth-Anything-3/src"
if DA3_SRC not in sys.path:
    sys.path.insert(0, DA3_SRC)
os.environ.setdefault("V3D_IMPORT_PATHS", f"/kaggle/working/src{os.pathsep}{DA3_SRC}")
# Where the gated checkpoint actually landed. PipelineConfig defaults to
# /kaggle/working/sam3, which is not it.
os.environ.setdefault("V3D_SAM3", "/tmp/sam3/sam3.pt")
os.environ.setdefault("V3D_VLM_MODEL", "/tmp/vlm")

SCENE = os.environ.get("V3D_SCENE", "small_factory")
WORK = Path(os.environ.get("V3D_WORK_ROOT", "/tmp/run"))
# The class prior. Nothing outside the vocabulary can ever become a box, so this
# string is the single highest-leverage input in the semantic half. A generic prompt
# on a workshop returned `wood`, `board`, `pipe`.
DOMAIN_HINT = os.environ.get(
    "V3D_PROMPT",
    "industrial factory floor: pallet, forklift, conveyor, control panel, "
    "safety cage, machine guard, hose reel, barrel, drum, crate, toolbox, "
    "workbench, shelving rack, cable tray, fire extinguisher, pipe, ladder",
)

STEPS: dict[str, object] = {}
FAILED: dict[str, str] = {}


def main() -> None:
    from v3d.config import PipelineConfig, detect_layout
    from v3d import paper_runner as pr

    cfg = PipelineConfig()
    layout = detect_layout(SCENE, cfg.data_root)
    model_dir = cfg.gs_root / layout.scene_id

    print("=" * 78)
    print("semantics: vocabulary -> masks -> boxes -> scene.json -> gate")
    print("=" * 78)
    print(f"  scene      {layout.scene_id}")
    print(f"  gaussians  {model_dir}")
    print(f"  sam3       {cfg.sam3_checkpoint}")
    print(f"  vlm        {cfg.vlm_model}")
    for p in pr.apply_offline_patches(cfg):
        print(f"  {p.describe()}")

    n_frames = len(list(layout.images_dir.glob("*.JPG")))
    n_masks = len(list(layout.mask_dir.glob("*"))) if layout.mask_dir.is_dir() else 0
    print(f"  frames {n_frames}, mesh-guided masks {n_masks}")
    if n_masks == 0:
        raise SystemExit("no mesh-guided masks; run the mesh stage first")

    # --- 1. vocabulary -------------------------------------------------------
    vlm_out = pr.vlm_output(layout.scene_id, cfg)
    if vlm_out.exists():
        doc = json.loads(vlm_out.read_text())
        print(f"\n--- 1. vocabulary: reusing {vlm_out} ---")
    else:
        print(f"\n--- 1. vocabulary ({n_frames} frames, no subsampling) ---")
        if not Path(cfg.vlm_model).is_dir():
            raise SystemExit(
                f"no VLM weights at {cfg.vlm_model}. This kernel has no internet, so "
                f"they must be relayed in. Without them the vocabulary cannot be "
                f"discovered and nothing downstream can produce a box."
            )
        t = time.time()
        rc = pr.stage_vlm(layout, cfg)
        print(f"    exit {rc} after {(time.time() - t) / 60:.1f} min")
        if rc != 0:
            raise SystemExit("stage_vlm failed")
        doc = json.loads(vlm_out.read_text())

    vocab = doc.get("vocabulary", [])
    per_image = doc.get("per_image", {})
    print(f"    {len(vocab)} classes over {len(per_image)} labelled frames")
    print(f"    vocabulary: {', '.join(sorted(vocab))}")
    merges = doc.get("merged_specialisations", {})
    if merges:
        print(f"    folded {len(merges)} specialisations into their base class")
    # Frequency matters: a class seen in one frame of 600 is far more likely to be a
    # hallucination than a real object, and it is the kind of thing only a count shows.
    freq = Counter(l for labels in per_image.values() for l in labels)
    rare = [c for c in vocab if freq[c] <= 2]
    print(f"    seen in <=2 frames (likely spurious): {rare or 'none'}")
    STEPS["vocabulary"] = {"classes": len(vocab), "vocabulary": sorted(vocab),
                           "rare": rare, "frames_labelled": len(per_image)}

    # --- 2. SAM3 masks -------------------------------------------------------
    mask_root = cfg.work_root / "sam_masks" / layout.scene_id
    index = mask_root / layout.scene_id / "mask_index.json"
    if index.exists():
        print(f"\n--- 2. SAM3: reusing {index} ---")
    else:
        print("\n--- 2. SAM3 text-prompted masks ---")
        t = time.time()
        rc = pr.stage_sam3(layout, cfg)
        print(f"    exit {rc} after {(time.time() - t) / 60:.1f} min")
        log = cfg.log_root / f"{layout.scene_id}_sam3.log"
        if rc != 0:
            tail = log.read_text().splitlines()[-40:] if log.exists() else ["<no log>"]
            raise SystemExit("stage_sam3 failed:\n" + "\n".join(tail))
    if not index.exists():
        raise SystemExit(f"sam3 reported success but {index} is absent")

    idx = json.loads(index.read_text())
    # The real schema, read off the file sam3.py writes rather than guessed:
    #   {scene, image_root, mask_root, items: [{image, label, mask_path,
    #    mask_rle: {size, counts}, bbox, score}], missing_images}
    # An earlier version of this looked for "masks"/"entries" and therefore counted
    # zero, which read as "SAM3 produced nothing" when it had produced 14,905 masks.
    # Guessing a schema is how a working stage gets reported as a failed one.
    entries = idx["items"] if isinstance(idx, dict) and "items" in idx else (
        idx if isinstance(idx, list) else [])
    if not entries:
        raise SystemExit(
            f"{index} holds no 'items'. Top-level keys: "
            f"{sorted(idx) if isinstance(idx, dict) else type(idx).__name__}. "
            f"Do not proceed on a guess about this file's shape."
        )
    by_class: Counter = Counter()
    images = set()
    scores: list[float] = []
    for e in entries:
        if isinstance(e, dict):
            by_class[e.get("label", "?")] += 1
            if e.get("image"):
                images.add(e["image"])
            if isinstance(e.get("score"), (int, float)):
                scores.append(float(e["score"]))
    import statistics
    print(f"    {sum(by_class.values())} masks over {len(images) or '?'} images, "
          f"{len(by_class)} classes matched")
    if scores:
        print(f"    confidence: median {statistics.median(scores):.3f}, "
              f"min {min(scores):.3f}, max {max(scores):.3f}")
    print(f"    images with no mask at all: {len(idx.get('missing_images', []))}")
    for c, n in by_class.most_common(20):
        print(f"      {c:24} {n:>6}")
    missed = sorted(set(vocab) - set(by_class))
    # Named, not counted: a class the segmenter never found is a class that cannot
    # become an object, and on the benchmark four such classes went unreported.
    print(f"    classes with ZERO masks ({len(missed)}): {missed or 'none'}")
    STEPS["sam3"] = {"masks": sum(by_class.values()), "classes_matched": len(by_class),
                     "per_class": dict(by_class.most_common()), "zero_mask_classes": missed,
                     "images_covered": len(images),
                     "median_score": (statistics.median(scores) if scores else None)}

    # --- 3. postprocess (floor-align the boxes) -------------------------------
    aligned = pr.boxes_output(layout.scene_id, cfg)
    if aligned.is_dir() and any(aligned.glob("*.json")):
        print(f"\n--- 3. postprocess: reusing {aligned} ---")
    else:
        print("\n--- 3. postprocess (floor-align the boxes) ---")
        rc = pr.stage_postprocess(layout, cfg)
        print(f"    exit {rc}")
        if rc != 0:
            log = cfg.log_root / f"{layout.scene_id}_postprocess.log"
            tail = log.read_text().splitlines()[-30:] if log.exists() else ["<no log>"]
            raise SystemExit("stage_postprocess failed:\n" + "\n".join(tail))

    # --- 4. scene.json -------------------------------------------------------
    print("\n--- 4. scene.json ---")
    rc = pr.stage_scene(layout, cfg)
    scene_path = pr.scene_output(layout.scene_id, cfg)
    print(f"    exit {rc} -> {scene_path}")
    if not scene_path.exists():
        raise SystemExit("stage_scene produced no scene.json")
    scene = json.loads(scene_path.read_text())
    objs = scene.get("objects", [])
    print(f"    {len(objs)} objects")
    kinds = Counter(o.get("label", o.get("class", "?")) for o in objs)
    for c, n in kinds.most_common(15):
        print(f"      {c:24} {n:>4}")
    STEPS["scene"] = {"objects": len(objs), "by_class": dict(kinds.most_common()),
                      "path": str(scene_path)}

    # --- 5. the gate ---------------------------------------------------------
    # No ground truth exists for a video, so Stage J runs its structural half only
    # and must say so. It is a real check even without annotation: every box inside
    # the room shell, nothing absurdly large, objects resting where they should.
    print("\n--- 5. Stage J (structural only; scale is 'assumed') ---")
    try:
        rc = pr.stage_gate(layout, cfg, gt=None)
        print(f"    gate exit {rc}")
        gate_path = pr.gate_output(layout.scene_id, cfg)
        if gate_path.exists():
            STEPS["gate"] = json.loads(gate_path.read_text())
    except Exception as exc:
        FAILED["gate"] = f"{type(exc).__name__}: {exc}"
        print(f"    gate raised: {exc}")

    out = WORK / f"semantics_summary_{SCENE}.json"
    out.write_text(json.dumps({"scene": SCENE, "prompt_hint": DOMAIN_HINT,
                               "steps": STEPS, "failed": FAILED}, indent=2, default=str))
    print(f"\n  -> {out}")
    print("\nSEMANTICS_OK" if not FAILED else "\nSEMANTICS_PARTIAL")


main()
