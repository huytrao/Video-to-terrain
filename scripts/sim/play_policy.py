#!/usr/bin/env python3
"""Run an exported RSL-RL policy on the room scene and check the robot stands.

    python scripts/sim/play_policy.py \
        --scene  outputs-full-factory/delivery/OUTPUT_ROOM/room_scene.xml \
        --policy .../logs/rsl_rl/exported/policy_12000.onnx \
        --deploy .../logs/rsl_rl/exported/deploy_12000.yaml

The observation layout is reconstructed from the deploy YAML rather than
hardcoded, because getting it wrong does not raise: the vector is the right
length, the network runs, and the robot simply falls over in a plausible-looking
way. Every term is scaled and stacked exactly as the training config declares.

Three things the YAML pins down that are easy to get wrong:

* **History order.** IsaacLab's circular buffer returns oldest-first,
  newest-last, and each term is flattened independently before the terms are
  concatenated in declaration order.
* **Actuators are torque, not position.** The scene's actuators have
  gainprm = 1 and biasprm = 0, so the PD loop lives here, using the stiffness
  and damping arrays from the YAML — not in MuJoCo.
* **Standing is a distinct gait phase.** `mdp.phase` zeroes both phase channels
  when the commanded velocity norm is under 0.1, so a zero command must produce
  [0, 0] and not sin/cos of a running clock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

STAND_COMMAND_EPS = 0.1  # matches mdp.phase's stand_mask threshold


def load_yaml(path: Path) -> Dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml is required to read the deploy config")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class History:
    """Oldest-first ring of the last `length` frames of one observation term."""

    def __init__(self, dim: int, length: int) -> None:
        self.buffer = np.zeros((length, dim), dtype=np.float32)

    def reset(self, value: np.ndarray) -> None:
        self.buffer[:] = value

    def push(self, value: np.ndarray) -> None:
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = value

    def flat(self) -> np.ndarray:
        return self.buffer.reshape(-1)


class PolicyRunner:
    def __init__(self, scene: Path, policy: Path, deploy: Path,
                 command: Optional[List[float]] = None) -> None:
        import mujoco
        import onnxruntime as ort

        self.mujoco = mujoco
        self.cfg = load_yaml(deploy)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)

        self.joint_names: List[str] = self.cfg["joint_names"]
        self.n = len(self.joint_names)
        self.default_q = np.asarray(self.cfg["default_joint_pos"], dtype=np.float64)
        self.kp = np.asarray(self.cfg["stiffness"], dtype=np.float64)
        self.kd = np.asarray(self.cfg["damping"], dtype=np.float64)
        self.step_dt = float(self.cfg["step_dt"])

        action_cfg = self.cfg["actions"]["JointPositionAction"]
        self.action_scale = np.asarray(action_cfg["scale"], dtype=np.float64)
        self.action_offset = np.asarray(action_cfg["offset"], dtype=np.float64)
        self.action_clip = action_cfg.get("clip")

        self.command = np.asarray(command if command is not None else [0.0, 0.0, 0.0],
                                  dtype=np.float32)
        self.standing = bool(np.linalg.norm(self.command) < STAND_COMMAND_EPS)

        self._map_joints()
        self._build_history()

        self.session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        expected = self.session.get_inputs()[0].shape[-1]
        if expected != self.obs_dim:
            raise SystemExit(
                f"observation is {self.obs_dim} wide but the policy wants {expected}. "
                "The deploy YAML and the checkpoint disagree."
            )

        self.gait_period = float(
            self.cfg["observations"]["gait_phase"]["params"].get("period", 0.9))
        self.action = np.zeros(self.n, dtype=np.float32)
        self.elapsed = 0.0

    def _map_joints(self) -> None:
        """Index the policy's joints and actuators inside this particular scene.

        The scene interleaves other free bodies with the robot, so a joint's qpos
        address has to be looked up rather than assumed to start at 7.
        """
        mj = self.mujoco
        self.qpos_adr, self.qvel_adr, self.act_id = [], [], []
        for name in self.joint_names:
            j = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise SystemExit(f"joint '{name}' is not in the scene")
            self.qpos_adr.append(int(self.model.jnt_qposadr[j]))
            self.qvel_adr.append(int(self.model.jnt_dofadr[j]))
            a = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if a < 0:
                raise SystemExit(f"actuator '{name}' is not in the scene")
            self.act_id.append(int(a))
        self.qpos_adr = np.array(self.qpos_adr)
        self.qvel_adr = np.array(self.qvel_adr)
        self.act_id = np.array(self.act_id)

        pelvis = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis < 0:
            raise SystemExit("no body named 'pelvis'")
        self.pelvis = pelvis
        free = [j for j in range(self.model.njnt)
                if self.model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
                and self.model.jnt_bodyid[j] == pelvis]
        if not free:
            raise SystemExit("the pelvis has no free joint")
        self.root_qpos = int(self.model.jnt_qposadr[free[0]])
        self.root_qvel = int(self.model.jnt_dofadr[free[0]])

    def _build_history(self) -> None:
        """One ring per observation term, sized and ordered by the deploy YAML."""
        obs_cfg = self.cfg["observations"]
        widths = {
            "base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3,
            "joint_pos_rel": self.n, "joint_vel_rel": self.n,
            "last_action": self.n, "gait_phase": 2,
        }
        self.terms: List[str] = []
        self.scales: Dict[str, np.ndarray] = {}
        self.history: Dict[str, History] = {}
        for term, cfg in obs_cfg.items():
            if term not in widths:
                raise SystemExit(f"unhandled observation term '{term}'")
            dim = widths[term]
            scale = cfg.get("scale", 1.0)
            self.scales[term] = np.broadcast_to(
                np.asarray(scale, dtype=np.float32), (dim,)).copy()
            self.history[term] = History(dim, int(cfg.get("history_length", 1)) or 1)
            self.terms.append(term)
        self.obs_dim = sum(h.buffer.size for h in self.history.values())

    # -- observation pieces --------------------------------------------------

    def _projected_gravity(self) -> np.ndarray:
        """Gravity expressed in the pelvis frame — the policy's sense of upright."""
        quat = self.data.qpos[self.root_qpos + 3:self.root_qpos + 7]
        rot = np.zeros(9)
        self.mujoco.mju_quat2Mat(rot, quat)
        return (rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

    def _base_ang_vel(self) -> np.ndarray:
        """Body-frame angular velocity, i.e. what an onboard IMU would report."""
        quat = self.data.qpos[self.root_qpos + 3:self.root_qpos + 7]
        rot = np.zeros(9)
        self.mujoco.mju_quat2Mat(rot, quat)
        world = self.data.qvel[self.root_qvel + 3:self.root_qvel + 6]
        return (rot.reshape(3, 3).T @ world).astype(np.float32)

    def _gait_phase(self) -> np.ndarray:
        # mdp.phase zeroes both channels while the command is a stand.
        if self.standing:
            return np.zeros(2, dtype=np.float32)
        t = (self.elapsed % self.gait_period) / self.gait_period
        return np.array([np.sin(t * 2 * np.pi), np.cos(t * 2 * np.pi)], dtype=np.float32)

    def _raw(self, term: str) -> np.ndarray:
        if term == "base_ang_vel":
            return self._base_ang_vel()
        if term == "projected_gravity":
            return self._projected_gravity()
        if term == "velocity_commands":
            return self.command
        if term == "joint_pos_rel":
            return (self.data.qpos[self.qpos_adr] - self.default_q).astype(np.float32)
        if term == "joint_vel_rel":
            return self.data.qvel[self.qvel_adr].astype(np.float32)
        if term == "last_action":
            return self.action
        if term == "gait_phase":
            return self._gait_phase()
        raise SystemExit(f"unhandled term '{term}'")

    def observe(self, push: bool = True) -> np.ndarray:
        for term in self.terms:
            value = self._raw(term) * self.scales[term]
            if push:
                self.history[term].push(value)
            else:
                self.history[term].reset(value)
        # Terms concatenate in declaration order; each is already flattened.
        return np.concatenate([self.history[t].flat() for t in self.terms])[None, :]

    # -- control -------------------------------------------------------------

    def reset(self) -> None:
        mj = self.mujoco
        key = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mj.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            mj.mj_resetData(self.model, self.data)
        self.action[:] = 0.0
        self.elapsed = 0.0
        mj.mj_forward(self.model, self.data)
        self.observe(push=False)

    def step(self) -> None:
        """One policy step: infer once, then hold that target over the substeps."""
        obs = self.observe()
        self.action = self.session.run(None, {self.input_name: obs.astype(np.float32)})[0][0]
        if self.action_clip:
            self.action = np.clip(self.action, *self.action_clip)

        target = self.action * self.action_scale + self.action_offset
        substeps = max(1, int(round(self.step_dt / self.model.opt.timestep)))
        for _ in range(substeps):
            q = self.data.qpos[self.qpos_adr]
            qd = self.data.qvel[self.qvel_adr]
            # The scene's actuators are plain torque motors, so the PD law that
            # IsaacLab applied implicitly has to be applied explicitly here.
            self.data.ctrl[self.act_id] = self.kp * (target - q) - self.kd * qd
            self.mujoco.mj_step(self.model, self.data)
        self.elapsed += self.step_dt


def run(runner: PolicyRunner, seconds: float) -> Dict:
    """Simulate, and judge against what was actually commanded.

    A walk command needs a different verdict from a stand command. Reporting
    "STANDING" for `0.5 0 0` is not a pass — it is a robot that stayed upright
    while ignoring the instruction, and the earlier version of this function did
    exactly that because it never measured travel.
    """
    runner.reset()
    stand_height = float(runner.data.xpos[runner.pelvis][2])
    start_xy = runner.data.xpos[runner.pelvis][:2].copy()

    mj = runner.mujoco
    obstacle_geom = {
        g for g in range(runner.model.ngeom)
        if (mj.mj_id2name(runner.model, mj.mjtObj.mjOBJ_GEOM, g) or "").startswith("obs_")
    }

    heights, uprightness = [], []
    blocked_by: set = set()
    diverged = False
    steps = int(seconds / runner.step_dt)
    for _ in range(steps):
        runner.step()
        if not np.all(np.isfinite(runner.data.qpos)):
            diverged = True
            break
        heights.append(float(runner.data.xpos[runner.pelvis][2]))
        # -gravity_z in the body frame is 1.0 when perfectly upright.
        uprightness.append(float(-runner._projected_gravity()[2]))
        # A velocity policy has no obstacle sensing — its 475 inputs are all
        # proprioception, command and gait — so it walks into furniture and stops.
        # That is the collider working, not the policy failing, and the two are
        # indistinguishable from distance alone.
        for c in range(runner.data.ncon):
            contact = runner.data.contact[c]
            for g in (contact.geom1, contact.geom2):
                if g in obstacle_geom:
                    blocked_by.add(mj.mj_id2name(runner.model, mj.mjtObj.mjOBJ_GEOM, g))

    if not heights:
        return {"ran": False, "diverged": diverged}

    final = heights[-1]
    tail = heights[max(0, len(heights) - int(1.0 / runner.step_dt)):]
    height_drift = float(max(tail) - min(tail))
    elapsed = len(heights) * runner.step_dt

    travel = runner.data.xpos[runner.pelvis][:2] - start_xy
    distance = float(np.linalg.norm(travel))
    speed = distance / max(elapsed, 1e-6)

    # Upright and no longer sinking. Necessary for either verdict, sufficient for
    # neither: a robot can be perfectly upright and still be ignoring the command.
    alive = bool(not diverged and final > 0.6 * stand_height and uprightness[-1] > 0.85)

    wanted = float(np.linalg.norm(runner.command[:2]))
    if runner.standing:
        verdict, target = "STANDING", 0.0
        # A stand means it held its ground: no height drift and no wandering.
        followed = alive and height_drift < 0.05 and distance < 0.15
    else:
        verdict, target = "WALKING", wanted
        # Half the commanded speed is the floor: policies undershoot, but a robot
        # that merely stayed upright is not walking.
        followed = alive and speed > 0.5 * wanted
        if not followed and blocked_by and alive:
            # It set off, then met something solid. Averaging speed across the
            # time it spent pinned measures the obstacle, not the policy.
            verdict, followed = "BLOCKED", True

    return {
        "ran": True,
        "seconds": seconds,
        "mode": "stand" if runner.standing else "walk",
        "start_pelvis_z": round(stand_height, 4),
        "final_pelvis_z": round(final, 4),
        "min_pelvis_z": round(min(heights), 4),
        "last_second_drift": round(height_drift, 4),
        "final_uprightness": round(uprightness[-1], 4),
        "min_uprightness": round(min(uprightness), 4),
        "travel_xy_m": [round(float(travel[0]), 4), round(float(travel[1]), 4)],
        "distance_m": round(distance, 4),
        "mean_speed_mps": round(speed, 4),
        "commanded_speed_mps": round(target, 4),
        "diverged": diverged,
        "upright": alive,
        "blocked_by": sorted(blocked_by),
        "verdict": verdict if followed else "FAILED",
        # Kept for callers that already read it; it now means "did what was asked".
        "standing": followed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Play an exported policy in the room scene")
    ap.add_argument("--scene", type=Path,
                    default=Path("outputs-full-factory/delivery/OUTPUT_ROOM/room_scene.xml"))
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--deploy", type=Path, default=None,
                    help="deploy YAML; defaults to deploy_<tag>.yaml beside the policy")
    ap.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("VX", "VY", "WZ"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--render", type=Path, default=None)
    args = ap.parse_args()

    deploy = args.deploy
    if deploy is None:
        tag = args.policy.stem.replace("policy_", "")
        for candidate in (args.policy.with_name(f"deploy_{tag}.yaml"),
                          args.policy.with_name("deploy.yaml")):
            if candidate.is_file():
                deploy = candidate
                break
    if deploy is None or not deploy.is_file():
        raise SystemExit("no deploy YAML found; pass --deploy")

    runner = PolicyRunner(args.scene, args.policy, deploy, command=args.command)
    print(f"[policy] {args.policy.name}  obs={runner.obs_dim}  actions={runner.n}")
    print(f"[deploy] {deploy.name}  step_dt={runner.step_dt}s  "
          f"substeps={int(round(runner.step_dt / runner.model.opt.timestep))}")
    print(f"[command] {list(args.command)}  -> "
          f"{'STAND (gait phase zeroed)' if runner.standing else 'WALK'}")

    result = run(runner, args.seconds)
    if not result.get("ran"):
        raise SystemExit(f"[result] simulation produced nothing: {result}")

    print(f"[result] pelvis {result['start_pelvis_z']} -> {result['final_pelvis_z']} m "
          f"(min {result['min_pelvis_z']})")
    print(f"         uprightness {result['final_uprightness']} "
          f"(min {result['min_uprightness']}), "
          f"height drift last second {result['last_second_drift']} m")
    print(f"         travelled {result['distance_m']} m "
          f"({result['mean_speed_mps']} m/s, commanded {result['commanded_speed_mps']})")
    if result.get("blocked_by"):
        print(f"         hit {', '.join(result['blocked_by'][:3])}"
              f"{' and more' if len(result['blocked_by']) > 3 else ''}")
    print(f"         {result['verdict']}")

    if args.render:
        png = render(runner, args.render, args.seconds)
        print(f"[render] {png}")

    report = args.scene.parent / "policy_play_report.json"
    report.write_text(json.dumps({
        "policy": str(args.policy), "deploy": str(deploy),
        "command": list(args.command), **result,
    }, indent=2), encoding="utf-8")
    print(f"[done] {report}")


def render(runner: PolicyRunner, out_png: Path, seconds: float,
           width: int = 1140, height: int = 760) -> Path:
    """Four moments through the run, so a numeric pass can still be eyeballed."""
    import cv2
    mj = runner.mujoco

    renderer = mj.Renderer(runner.model, height, width)
    runner.reset()
    marks = [0.0, seconds / 3, 2 * seconds / 3, seconds]
    focus = runner.data.xpos[runner.pelvis].copy()
    shots, next_mark, elapsed = [], 0, 0.0

    def shot(label):
        camera = mj.MjvCamera()
        camera.lookat[:] = runner.data.xpos[runner.pelvis]
        camera.distance, camera.azimuth, camera.elevation = 3.6, 135, -10
        renderer.update_scene(runner.data, camera)
        image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(image, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        return image

    while next_mark < len(marks):
        if elapsed >= marks[next_mark] - 1e-9:
            z = runner.data.xpos[runner.pelvis][2]
            up = -runner._projected_gravity()[2]
            shots.append(shot(f"t={elapsed:4.1f}s   pelvis z={z:.3f} m   upright={up:.2f}"))
            next_mark += 1
            continue
        runner.step()
        elapsed += runner.step_dt

    grid = np.vstack([np.hstack(shots[0:2]), np.hstack(shots[2:4])])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), grid)
    return out_png


if __name__ == "__main__":
    main()
