"""Prove the factory physics scene actually behaves, rather than merely compiling.

Three questions, because a scene can pass the compiler and still be useless:

1. **Does everything settle?** A body that never comes to rest is sitting in a
   bad contact -- usually interpenetrating terrain the solver keeps ejecting it
   from. Reported as how far each body drifted from where it was placed.
2. **Is anything born inside the walls?** Penetration at t=0 is a placement bug,
   not a physics one, and it is invisible once the solver has pushed things
   apart. Measured before the first step.
3. **Does friction hold?** A shove is applied to each body and the distance it
   travels afterwards is reported. Zero would mean the contact is frozen; metres
   would mean the surface is ice. Both are wrong and neither shows up in a
   settle test.
"""
from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def body_positions(model, data, names):
    return np.array([data.body(n).xpos.copy() for n in names])


def worst_penetration(model, data):
    if data.ncon == 0:
        return 0.0, None
    depths = np.array([data.contact[i].dist for i in range(data.ncon)])
    worst = int(np.argmin(depths))
    pair = data.contact[worst]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, pair.geom1)
    return float(-depths.min()), name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default="outputs-full-factory/delivery/OUTPUT_K/sim/factory_physics.xml")
    ap.add_argument("--settle", type=float, default=3.0, help="seconds to settle")
    ap.add_argument("--push", type=float, default=40.0, help="newtons of shove")
    ap.add_argument("--push-time", type=float, default=0.25)
    ap.add_argument("--coast", type=float, default=2.5)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(ROOT / args.xml))
    data = mujoco.MjData(model)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    ]
    print(f"compiled: {model.nbody-1} bodies, {model.ngeom} geoms, {model.nq} dof-q")

    mujoco.mj_forward(model, data)
    depth, culprit = worst_penetration(model, data)
    print(f"at rest before stepping: {data.ncon} contacts, worst penetration "
          f"{depth*100:.1f} cm" + (f" at {culprit}" if culprit else ""))

    start = body_positions(model, data, names)
    for _ in range(int(args.settle / model.opt.timestep)):
        mujoco.mj_step(model, data)
    settled = body_positions(model, data, names)
    drift = np.linalg.norm(settled - start, axis=1)

    speed = np.array([
        np.linalg.norm(data.body(n).cvel[3:]) for n in names
    ])

    print(f"\nafter {args.settle:.0f} s of settling:")
    print(f"  {'body':24}{'drift cm':>10}{'speed m/s':>11}{'':>4}")
    for name, d, v in zip(names, drift, speed):
        flag = "still moving" if v > 0.02 else ""
        print(f"  {name:24}{d*100:10.1f}{v:11.3f}   {flag}")

    # --- friction: shove each body along +x and see how far it coasts --------
    print(f"\nfriction check: {args.push:.0f} N for {args.push_time:.2f} s, "
          f"then {args.coast:.1f} s to stop")
    print(f"  {'body':24}{'slid cm':>10}{'stopped?':>11}")
    for index, name in enumerate(names):
        probe = mujoco.MjData(model)
        for _ in range(int(args.settle / model.opt.timestep)):
            mujoco.mj_step(model, probe)
        before = probe.body(name).xpos.copy()
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for _ in range(int(args.push_time / model.opt.timestep)):
            probe.xfrc_applied[body_id, 0] = args.push
            mujoco.mj_step(model, probe)
        probe.xfrc_applied[body_id] = 0
        for _ in range(int(args.coast / model.opt.timestep)):
            mujoco.mj_step(model, probe)
        slid = float(np.linalg.norm(probe.body(name).xpos - before))
        rest = float(np.linalg.norm(probe.body(name).cvel[3:])) < 0.02
        print(f"  {name:24}{slid*100:10.1f}{('yes' if rest else 'NO'):>11}")


if __name__ == "__main__":
    main()
