"""Pipeline 1 — run the stages that turn a posed frame set into every derived artefact.

This is the piece that makes P1 survivable. The proxy JWT expires in one to two
hours and the stages together take longer than that, so the run cannot live
inside the connection that started it. It lives in a detached subprocess on the
kernel instead, and its progress lives in a file.

`STATE.json` is the whole contract. Each stage writes its outcome there the
moment it finishes, so a reconnecting operator learns what happened from the
filesystem rather than from a log they were not watching. Re-running skips
whatever already succeeded, which means "the websocket dropped" and "the stage
failed" stop looking alike — the first resumes, the second shows a `failed`
entry with the return code that produced it.

Degradation is recorded, not silent. A stage that fails is written down as
failed and the run stops there rather than feeding a missing artefact to the
next stage, because a stage that cannot do its job should say so instead of
producing a plausible-looking wrong answer.

    python3 /kaggle/working/_run/kernel_p1_run.py            # detached, resumes
    python3 /kaggle/working/_run/kernel_p1_run.py --status   # just read STATE.json

Environment:
    V3D_SCENE     scene id under V3D_DATA_ROOT
    V3D_STAGES    comma-separated stage list, in order
    V3D_STATE     where STATE.json is written
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# `/kaggle/working` first, deliberately. An older deploy left a second copy of the
# package at `/kaggle/working/src/v3d`, and inserting that at position 0 shadowed
# the one every upload actually writes to. The symptom is a stage failing on an
# argument that plainly exists in the source you just checked -- here,
# `stage_pgsr() got an unexpected keyword argument 'hold_out_test'` while the file
# on disk had the parameter and hashed identical to local. Keep the stale tree
# reachable as a fallback, but never ahead of the live one.
sys.path.insert(0, "/kaggle/working")
if os.path.isdir("/kaggle/working/src"):
    sys.path.append("/kaggle/working/src")

SCENE = os.environ.get("V3D_SCENE", "scannetpp_00777c41d4")
STATE = Path(os.environ.get("V3D_STATE", f"/kaggle/working/outputs/{SCENE}/STATE.json"))
DEFAULT_STAGES = "da3,pgsr,mesh"


def derive_knobs_for(scene: str):
    """Measure this scene once; return `(Knobs, schedule)` for the stages to split.

    Both come from `v3d.paper_runner.derive_knobs`, which reads the depth DA3
    actually predicted and the camera path it actually recovered, rather than
    inheriting the paper's ScanNet++-bedroom constants. The result is cached
    beside the run, so the numbers reported here are the ones the stage got.
    """
    from v3d.paper_runner import derive_knobs

    return derive_knobs(scene)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"scene": SCENE, "started": now(), "stages": {}}


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)          # atomic: a reader never sees a half-written file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=os.environ.get("V3D_STAGES", DEFAULT_STAGES))
    ap.add_argument("--status", action="store_true", help="print STATE.json and exit")
    ap.add_argument("--force", default="", help="comma-separated stages to redo even if done")
    args = ap.parse_args()

    state = load()
    if args.status:
        print(json.dumps(state, indent=2))
        return 0

    from v3d.paper_runner import STAGES  # imported here so --status needs no deps

    force = {s for s in args.force.split(",") if s}
    order = [s for s in args.stages.split(",") if s]
    unknown = [s for s in order if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(unknown)}; have {sorted(STAGES)}")

    print(f"scene  {SCENE}")
    print(f"stages {' -> '.join(order)}")
    print(f"state  {STATE}", flush=True)

    for name in order:
        prior = state["stages"].get(name)
        if prior and prior.get("status") == "ok" and name not in force:
            print(f"[skip] {name}: already ok at {prior['finished']}", flush=True)
            continue

        print(f"[run ] {name} at {now()}", flush=True)
        state["stages"][name] = {"status": "running", "started": now()}
        save(state)

        # PGSR is the one stage whose defaults are wrong for a frame count other
        # than the paper's. It samples one image per iteration, so a fixed 30,000
        # iterations means fewer views of each image as the count grows -- the
        # opposite of what more input should buy. `pgsr_schedule` derives the
        # iteration count, the learning-rate horizon and the view-pairing
        # distances from this scene's own cameras. Passing none is what silently
        # ran 746 images on the 261-image schedule.
        kwargs = {}
        if name in ("pgsr", "mesh"):
            # Both stages are measured off the same scene, so derive once and
            # hand each the part it takes. `mesh` getting nothing is what let the
            # paper's mesh.sh constants back in: `max_depth 4.0` erases 9.5% of
            # this scene's believable depth, and `num_cluster 1` deletes every
            # freestanding object the fusion did not weld to the floor.
            knobs, schedule = derive_knobs_for(SCENE)
            if name == "pgsr":
                kwargs["schedule"] = schedule
                # V3D_HOLD_OUT_TEST=1 holds out the transforms file's 106
                # `test_frames` so the run can report a generalisation PSNR
                # instead of a train-set ceiling. Costs 12.4% of the views.
                if os.environ.get("V3D_HOLD_OUT_TEST") == "1":
                    kwargs["hold_out_test"] = True
                    print("       holding out 106 test frames (--eval)", flush=True)
                print(f"       schedule {json.dumps(schedule, default=str)}", flush=True)
                state["stages"][name]["schedule"] = schedule
            else:
                kwargs["knobs"] = knobs
                rec = {k: v for k, v in vars(knobs).items() if k != "evidence"}
                print(f"       knobs {json.dumps(rec, default=str)}", flush=True)
                state["stages"][name]["knobs"] = rec
            save(state)

        t = time.time()
        try:
            rc = STAGES[name](SCENE, **kwargs)
        except Exception as exc:                       # noqa: BLE001 - recorded, then re-raised
            state["stages"][name] = {
                "status": "failed", "started": state["stages"][name]["started"],
                "finished": now(), "seconds": round(time.time() - t, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }
            save(state)
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            return 1

        entry = {
            "status": "ok" if rc == 0 else "failed",
            "started": state["stages"][name]["started"],
            "finished": now(),
            "seconds": round(time.time() - t, 1),
            "returncode": rc,
        }
        state["stages"][name] = entry
        save(state)
        print(f"[{entry['status']:4s}] {name} rc={rc} in {entry['seconds']}s", flush=True)
        if rc != 0:
            print("stopping: a later stage would consume a missing artefact", flush=True)
            return rc

    state["finished"] = now()
    save(state)
    print("P1_STAGES_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
