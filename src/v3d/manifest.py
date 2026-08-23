"""What Pipeline 1 produced, with the evidence to check it.

A run that ends with files on disk has not finished. The question a manifest
answers is the one that matters a week later: *are these the artefacts that run
produced, on what scale, and which parts of them are weaker than they look.*

Three properties drive the shape of this file.

**Absent artefacts are recorded, not omitted.** A manifest that simply leaves
out what failed reads as complete, and the reader has no way to tell a stage
that was skipped from a stage that never existed. Every expected artefact gets
an entry; a missing one says so, and says which stage was responsible.

**Checksums are over content.** Sizes and counts agree with each other far too
easily -- a directory of 852 files that are all the previous run's still counts
852. A directory digest is taken over ``(name, sha256)`` pairs in sorted order,
so it is independent of filesystem ordering and changes if any single file
does.

**The scale is stated, and stated as unverified.** Depth here is DA3's own,
fitted to the supplied camera poses. It is metric only to the extent those
poses are, and nothing in P1 has checked that against a measured distance. So
the units field says ``da3_prior`` and the flag says ``UNVERIFIED`` rather than
quietly implying metres. Settling it is calibration, and it belongs to P2 --
using a reference an arbitrary video can also supply, never the capture's
on-device sensor depth, which this pipeline does not read.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from v3d.config import PipelineConfig, SceneLayout, detect_layout

SCHEMA = "v3d/p1-manifest/1"

# The one scale gate P1 enforces on itself. DA3 fits a scale per inference call
# and the run is chunked, so this is the residual disagreement between chunks
# after the common scalar is removed -- the part no rescaling can recover, and
# the part PGSR bakes into the surface as depth supervision.
CHUNK_RESIDUAL_TOLERANCE_PCT = 1.0

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def digest_dir(path: Path, pattern: str = "*") -> tuple[str, int, int]:
    """``(digest, file_count, bytes)`` over a directory's contents.

    The digest covers names as well as contents, so a file that is renamed or
    dropped changes it. Sorting makes it independent of the order the
    filesystem happens to return.
    """
    files = sorted(p for p in path.glob(pattern) if p.is_file())
    h = hashlib.sha256()
    total = 0
    for p in files:
        h.update(p.name.encode())
        h.update(sha256_file(p).encode())
        total += p.stat().st_size
    return h.hexdigest(), len(files), total


@dataclass(frozen=True)
class Expected:
    """One artefact P1 is supposed to produce, and who produces it."""

    key: str
    path: Path
    stage: str
    kind: str = "file"          # "file" or "dir"
    pattern: str = "*"
    note: str = ""


def expected_artefacts(layout: SceneLayout, cfg: PipelineConfig) -> list[Expected]:
    """Everything P1 promises, whether or not this run got that far."""
    from v3d.paper_runner import (
        MESH_PATH_REL, boxes_output, gate_output, scene_output, vlm_output,
    )

    gs = cfg.gs_root / layout.scene_id
    return [
        Expected("frames", layout.images_dir, "ingest", "dir", "*.[jJpP][pPnN]*[gG]",
                 "the posed images every later stage reads"),
        Expected("transforms", layout.transforms, "ingest",
                 note="camera poses and intrinsics, nerfstudio convention"),
        Expected("depth", layout.depth_dir, "da3", "dir", "*.npy",
                 "DA3 depth; PGSR reads this as training supervision"),
        Expected("point_cloud", layout.point_cloud, "da3",
                 note="unprojected depth; PGSR initialises from it"),
        Expected("chunk_scales", layout.scene_dir / "da3_chunk_scales.json", "da3",
                 note="per-chunk scale join and its residual"),
        Expected("gaussians", gs / "point_cloud", "pgsr", "dir", "**/*.ply",
                 "trained 3D Gaussians"),
        Expected("mesh", gs / MESH_PATH_REL, "mesh",
                 note="TSDF fusion of the rendered depth"),
        Expected("masks", layout.mask_dir, "sam3", "dir", "*",
                 "2D instance masks"),
        Expected("vlm_labels", vlm_output(layout.scene_id, cfg), "vlm",
                 note="class vocabulary discovered for this scene"),
        Expected("boxes", boxes_output(layout.scene_id, cfg), "postprocess", "dir", "*.json",
                 "floor-aligned 3D oriented boxes"),
        Expected("scene_json", scene_output(layout.scene_id, cfg), "scene",
                 note="the assembled structured scene"),
        Expected("gate", gate_output(layout.scene_id, cfg), "gate",
                 note="quality gate verdict"),
    ]


def _entry(exp: Expected, checksums: bool) -> dict:
    p = exp.path
    out: dict[str, object] = {"path": str(p), "stage": exp.stage, "kind": exp.kind}
    if exp.note:
        out["note"] = exp.note
    if exp.kind == "dir":
        if not p.is_dir():
            out["present"] = False
            return out
        if checksums:
            digest, n, total = digest_dir(p, exp.pattern)
            out.update(present=True, files=n, bytes=total, sha256=digest)
        else:
            files = [q for q in p.glob(exp.pattern) if q.is_file()]
            out.update(present=True, files=len(files),
                       bytes=sum(q.stat().st_size for q in files))
        out["present"] = out["files"] > 0
        return out
    if not p.exists():
        out["present"] = False
        return out
    out.update(present=True, bytes=p.stat().st_size)
    if checksums:
        out["sha256"] = sha256_file(p)
    return out


def _chunk_scale_summary(layout: SceneLayout) -> dict:
    """The measured scale consistency, and whether it passed its own gate.

    Absent is not the same as fine: a single-chunk run has nothing to join, but
    a chunked run with no record means the join was never measured, and that is
    the state this whole mechanism exists to make impossible.
    """
    rec = layout.scene_dir / "da3_chunk_scales.json"
    if not rec.exists():
        return {"measured": False,
                "meaning": "one inference call, or a run that predates the "
                           "chunk-scale join -- no inter-chunk scale evidence"}
    hist = json.loads(rec.read_text())
    resid = [h["residual_pct"] for h in hist]
    offs = [abs(h["offset_pct"]) for h in hist]
    worst = max(resid) if resid else 0.0
    return {
        "measured": True,
        "joins": len(hist),
        "worst_residual_pct": round(worst, 3),
        "worst_offset_removed_pct": round(max(offs), 3) if offs else 0.0,
        "tolerance_pct": CHUNK_RESIDUAL_TOLERANCE_PCT,
        "passed": worst <= CHUNK_RESIDUAL_TOLERANCE_PCT,
        "per_chunk": hist,
    }


def _run_parameters() -> dict:
    """The knobs this process was told to use, read back from the environment.

    Recorded because "which resolution was that run at" is otherwise
    unanswerable once the shell that launched it is gone.
    """
    names = ("V3D_PROCESS_RES", "V3D_PROCESS_RES_METHOD", "V3D_MAX_CHUNK_SIZE",
             "V3D_ANCHOR_FRAMES", "V3D_TARGET_POINTS", "V3D_CONF_THRESHOLD",
             "PYTORCH_ALLOC_CONF")
    return {n: os.environ[n] for n in names if os.environ.get(n)}


def build(
    scene: str | SceneLayout,
    cfg: PipelineConfig | None = None,
    state_path: Path | None = None,
    checksums: bool = True,
) -> dict:
    """Assemble the manifest. `checksums=False` for a quick look, not for delivery."""
    cfg = cfg or PipelineConfig()
    layout = scene if isinstance(scene, SceneLayout) else detect_layout(scene, cfg.data_root)

    state = {}
    sp = state_path or Path(f"/kaggle/working/outputs/{layout.scene_id}/STATE.json")
    if sp.exists():
        state = json.loads(sp.read_text())

    artefacts = {}
    for exp in expected_artefacts(layout, cfg):
        artefacts[exp.key] = _entry(exp, checksums)

    stages = state.get("stages", {})
    # Degradation is reported from what the stages recorded plus what is on
    # disk, never inferred from one alone: a stage can exit 0 and produce
    # nothing (the DA3 skip-guard did exactly that), and an artefact can exist
    # from an earlier run under a stage that has since failed.
    degraded = []
    for key, ent in artefacts.items():
        stage = ent["stage"]
        st = stages.get(stage, {})
        if not ent["present"]:
            degraded.append({"artefact": key, "stage": stage,
                             "status": st.get("status", "not run"),
                             "why": st.get("error") or
                                    f"stage {stage} did not produce {ent['path']}"})
        elif st.get("status") == "failed":
            degraded.append({"artefact": key, "stage": stage, "status": "failed",
                             "why": "artefact exists but its stage failed in this "
                                    "run; it is from an earlier one"})

    chunk = _chunk_scale_summary(layout)
    return {
        "schema": SCHEMA,
        "scene": layout.scene_id,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checksums": checksums,
        "layout": {"scene_dir": str(layout.scene_dir),
                   "source_dir": str(layout.source_dir),
                   "images_dir": str(layout.images_dir)},
        "units": {
            "depth": "da3_prior",
            "scale_status": "UNVERIFIED",
            "meaning": "depth is DA3's own, fitted to the supplied poses; metric "
                       "only insofar as those poses are. Nothing in P1 compared it "
                       "to a measured distance.",
            "how_to_verify": "metric calibration is P2 stage [2], and it must use a "
                             "reference the general video path can also supply -- a "
                             "known object dimension, a measured baseline, or the pose "
                             "scale itself. On-device sensor depth is deliberately not "
                             "used: it exists only for this ScanNet++ capture, so "
                             "calibrating against it would not survive an arbitrary video.",
        },
        "depth_scale_consistency": chunk,
        "run_parameters": _run_parameters(),
        "stages": stages,
        "artefacts": artefacts,
        "degraded": degraded,
        "complete": not degraded,
    }


def write(scene: str | SceneLayout, cfg: PipelineConfig | None = None,
          out: Path | None = None, **kw) -> Path:
    cfg = cfg or PipelineConfig()
    layout = scene if isinstance(scene, SceneLayout) else detect_layout(scene, cfg.data_root)
    doc = build(layout, cfg, **kw)
    out = out or Path(f"/kaggle/working/outputs/{layout.scene_id}/manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(out)
    return out
