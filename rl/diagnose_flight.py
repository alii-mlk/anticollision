#!/usr/bin/env python3
"""Why does the MuJoCo drone fall when asked to move sideways?

check_env.py showed that the drone hovers perfectly and climbs at the speed
its gains predict, but drops to the ground as soon as it is asked to move
horizontally. Hover and climb need only collective thrust; horizontal flight
needs the drone to tilt. So the suspicion is the attitude feedback: the
package's controller may be seeing a stale orientation, or one in the wrong
quaternion order (MuJoCo stores w,x,y,z; the PyBullet code this package was
ported from used x,y,z,w).

Two parts:

  A. bookkeeping  for the first control steps of a sideways command, print
                  the drone's true orientation from MuJoCo next to what the
                  package hands its controller
  B. flights      a 4 s sideways flight flown six ways:
                    package path, waypoint 0.2 m ahead
                    package path, waypoint 1.0 m ahead
                    own call to the package's controller, quaternion w,x,y,z
                    own call to the package's controller, quaternion x,y,z,w
                    own call, but a new controller every control step, which
                      is what the package's PID action mode does
                    own call, quaternion w,x,y,z, waypoint 2.5 m ahead
                  If the package path falls and one of the others flies level,
                  the fault is in how the package feeds its controller, and
                  the fix is to call the controller ourselves.

First run (2026-10-11): the package path falls even with a gentle waypoint,
while the same controller called with one instance kept for the whole flight
flies level at 0.42 m/s. Part A showed the orientation it receives is correct;
the pitch instead oscillates with growing amplitude until the drone flips.

The report also goes to rl/logs/diagnose_flight.txt.

Usage (in the VM, after  source rl/env.sh):
  python rl/diagnose_flight.py
"""

import math
import sys
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
LOG = HERE / "logs" / "diagnose_flight.txt"
START_Z = 3.0
SIM_FREQ, CTRL_FREQ = 240, 48
SETTLE = 48
LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(text)


def make_sim(act_type):
    from multi_drone_mujoco.envs.base_aviary import BaseAviary
    from multi_drone_mujoco.utils.enums import (DroneModel, ObservationType,
                                                Physics)
    out = HERE / "runs" / "aviary"
    out.mkdir(parents=True, exist_ok=True)
    return BaseAviary(drone_model=DroneModel.CF2X, num_drones=1,
                      initial_xyzs=np.array([[0.0, 0.0, START_Z]]),
                      physics=Physics.MJC, sim_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
                      gui=False, record=False, obstacles=False,
                      obs_type=ObservationType.KIN, act_type=act_type,
                      output_folder=str(out))


def free_joint(sim):
    """qpos and qvel addresses of the drone's free joint."""
    m = sim.model
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
    raise RuntimeError("no free joint in the model")


def true_state(sim, qadr, vadr):
    q = sim.data.qpos
    return q[qadr:qadr + 3].copy(), q[qadr + 3:qadr + 7].copy()  # pos, quat w,x,y,z


def tilt_deg(quat_wxyz):
    """Angle between the body z axis and the world z axis."""
    _, x, y, _ = quat_wxyz
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def pitch_deg(quat_wxyz):
    w, x, y, z = quat_wxyz
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))


def part_a():
    from multi_drone_mujoco.utils.enums import ActionType
    sim = make_sim(ActionType.PID)
    qadr, vadr = free_joint(sim)
    sim.reset()
    hold = np.array([0.0, 0.0, START_Z, 0.0])
    for _ in range(SETTLE):
        sim.step(hold)
    say("step  true quat w,x,y,z              package quat                    "
        "package rpy (deg)        true pitch  x")
    for k in range(12):
        pos, quat = true_state(sim, qadr, vadr)
        pq = np.asarray(getattr(sim, "quat", np.full((1, 4), np.nan))[0])
        pr = np.degrees(np.asarray(getattr(sim, "rpy", np.full((1, 3), np.nan))[0]))
        say(f"{k:>4}  {np.array2string(quat, precision=4, suppress_small=True):<30}"
            f"{np.array2string(pq, precision=4, suppress_small=True):<32}"
            f"{np.array2string(pr, precision=2, suppress_small=True):<25}"
            f"{pitch_deg(quat):>8.2f}  {pos[0]:.3f}")
        sim.step(np.array([pos[0] + 1.0, pos[1], pos[2], 0.0]))


def fly(mode, lookahead, seconds=4.0):
    """Fly towards +x and return a summary of what happened."""
    from multi_drone_mujoco.control.pid_control import PIDControl
    from multi_drone_mujoco.utils.enums import ActionType

    own = mode != "package"
    sim = make_sim(ActionType.RPM if own else ActionType.PID)
    qadr, vadr = free_joint(sim)
    sim.reset()
    ctrl = PIDControl(sim) if own else None

    def command(target):
        nonlocal ctrl
        if not own:
            sim.step(np.array([*target, 0.0]))
            return
        if mode == "own_fresh":
            # What the package does: a new controller for every control step.
            ctrl = PIDControl(sim)
        pos, quat = true_state(sim, qadr, vadr)
        cur_quat = quat if mode == "own_wxyz" else np.array([quat[1], quat[2], quat[3], quat[0]])
        rpm, _, _ = ctrl.computeControl(
            control_timestep=1.0 / CTRL_FREQ, cur_pos=pos, cur_quat=cur_quat,
            cur_vel=sim.data.qvel[vadr:vadr + 3].copy(),
            cur_ang_vel=sim.data.qvel[vadr + 3:vadr + 6].copy(),
            target_pos=np.asarray(target, dtype=float), target_rpy=np.zeros(3))
        sim.step(np.asarray(rpm, dtype=float).reshape(4))

    for _ in range(SETTLE):
        command((0.0, 0.0, START_Z))
    start, _ = true_state(sim, qadr, vadr)

    n = int(seconds * CTRL_FREQ)
    xs, zs, tilts = [], [], []
    for _ in range(n):
        pos, quat = true_state(sim, qadr, vadr)
        command((pos[0] + lookahead, pos[1], pos[2]))
        pos, quat = true_state(sim, qadr, vadr)
        xs.append(pos[0])
        zs.append(pos[2])
        tilts.append(tilt_deg(quat))

    half = n // 2
    speed = (xs[-1] - xs[half]) / ((n - 1 - half) / CTRL_FREQ)
    fell = min(zs) < 0.5
    return {
        "x moved": xs[-1] - start[0],
        "lowest z": min(zs),
        "max tilt": max(tilts),
        "speed": speed,
        "verdict": "FALLS" if fell else ("flies" if speed > 0.05 else "stuck"),
    }


def part_b():
    say(f"{'mode':<22}{'ahead':>6}{'x moved':>10}{'lowest z':>10}{'max tilt':>10}"
        f"{'speed':>9}  verdict")
    for mode, ahead in (("package", 0.2), ("package", 1.0),
                        ("own_wxyz", 1.0), ("own_xyzw", 1.0),
                        ("own_fresh", 1.0), ("own_wxyz", 2.5)):
        try:
            r = fly(mode, ahead)
            say(f"{mode:<22}{ahead:>6.1f}{r['x moved']:>9.2f}m{r['lowest z']:>9.2f}m"
                f"{r['max tilt']:>9.1f}d{r['speed']:>7.2f}m/s  {r['verdict']}")
        except Exception:
            say(f"{mode:<22}{ahead:>6.1f}  ERROR")
            for line in traceback.format_exc().strip().splitlines()[-5:]:
                say(f"    {line}")


def main():
    say(f"flight diagnosis {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for title, fn in (("A. orientation bookkeeping", part_a),
                      ("B. sideways flights", part_b)):
        say()
        say(f"=== {title} ===")
        try:
            fn()
        except Exception:
            for line in traceback.format_exc().strip().splitlines()[-6:]:
                say(f"    {line}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(LINES) + "\n")
    print(f"\nreport written to {LOG}")


if __name__ == "__main__":
    main()
