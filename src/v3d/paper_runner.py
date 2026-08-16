"""Run the Holi-Spatial pipeline unmodified, on a kernel with no network.

The paper's code is 4,300 lines that went through review; reimplementing it
would trade a known-good method for fresh bugs. So it runs as published, and
this module supplies only what an offline machine cannot: local paths in place
of HuggingFace repo ids, and the stage ordering.

Almost nothing needs patching. `sam3.py` and the lifting script already take
their checkpoints and roots as command-line flags. Exactly one file hardcodes
a repo id, and `apply_offline_patches` rewrites that single line.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Where things live on the offline Blackwell kernel.
PAPER_ROOT = Path("/tmp/paper/holi-spatial")
DATA_ROOT = Path("/tmp/data/scannetppv2/data")
DA3_MODEL = Path("/tmp/DA3NESTED-GIANT-LARGE")
SAM3_CHECKPOINT = Path("/kaggle/working/sam3")
WORK_ROOT = Path("/tmp/run")

# The literal in inference_da3_scannetppv2.py that reaches for the network.
DA3_REPO_ID = "depth-anything/DA3NESTED-GIANT-LARGE"


@dataclass
class Patch:
    path: Path
    before: str
    after: str
    applied: bool


def apply_offline_patches(paper_root: Path = PAPER_ROOT, da3_model: Path = DA3_MODEL) -> list[Patch]:
    """Point the one hardcoded repo id at a local directory. Idempotent."""
    target = paper_root / "inference_da3_scannetppv2.py"
    text = target.read_text()
    patches: list[Patch] = []

    if DA3_REPO_ID in text:
        patched = text.replace(f'"{DA3_REPO_ID}"', f'"{da3_model}"')
        target.write_text(patched)
        patches.append(Patch(target, DA3_REPO_ID, str(da3_model), applied=True))
    else:
        patches.append(Patch(target, DA3_REPO_ID, str(da3_model), applied=False))

    return patches


def _run(cmd: list[str], cwd: Path, log: Path | None = None, env: dict | None = None) -> int:
    """Run a stage, streaming to a log so a long GPU job can be watched."""
    log = log or (WORK_ROOT / "logs" / (cmd[1].replace("/", "_") + ".log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def stage_da3(scene_id: str, paper_root: Path = PAPER_ROOT, data_root: Path = DATA_ROOT) -> int:
    """DA3 depth and point cloud. Writes depth_da3/*.npy and pointcloud_da3.ply."""
    return _run(
        [sys.executable, "inference_da3_scannetppv2.py",
         "--scannetppv2_dir", str(data_root / scene_id)],
        cwd=paper_root,
        log=WORK_ROOT / "logs" / f"{scene_id}_da3.log",
    )


def da3_outputs(scene_id: str, data_root: Path = DATA_ROOT) -> dict[str, Path]:
    scene = data_root / scene_id
    return {
        "depth_dir": scene / "depth_da3",
        "point_cloud": scene / "pointcloud_da3.ply",
    }


def stage_report(scene_id: str, data_root: Path = DATA_ROOT) -> dict[str, object]:
    """What exists so far, so a stage can be resumed rather than repeated."""
    out = da3_outputs(scene_id, data_root)
    depth_dir = out["depth_dir"]
    return {
        "scene": scene_id,
        "depth_maps": len(list(depth_dir.glob("*.npy"))) if depth_dir.is_dir() else 0,
        "point_cloud_mb": round(out["point_cloud"].stat().st_size / 1e6, 1)
        if out["point_cloud"].exists() else None,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene")
    ap.add_argument("--stage", default="da3", choices=["patch", "da3", "report"])
    ap.add_argument("--paper-root", type=Path, default=PAPER_ROOT)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--da3-model", type=Path, default=DA3_MODEL)
    args = ap.parse_args(argv)

    if args.stage in ("patch", "da3"):
        for p in apply_offline_patches(args.paper_root, args.da3_model):
            state = "patched" if p.applied else "already local"
            print(f"  {p.path.name}: {state} ({p.before} -> {p.after})")

    if args.stage == "da3":
        rc = stage_da3(args.scene, args.paper_root, args.data_root)
        print(f"  DA3 exit code {rc}")
        print(f"  {stage_report(args.scene, args.data_root)}")
        return rc

    if args.stage == "report":
        print(stage_report(args.scene, args.data_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
