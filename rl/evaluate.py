#!/usr/bin/env python3
"""Score a trained policy scenario by scenario, through its deployment path.

The policy is flown in the training environment, but every decision goes
through rl/policy.py, the same wrapper that flies it in Gazebo. At every step
the observation the wrapper builds from the drone's state is compared with
the one the environment builds for training. If the two are identical, the
Gazebo node gives the network the same input it was trained on.

The 25 single-drone scenarios of the Gazebo sweep are flown both ways, flying
straight and with the policy, and listed next to what PX4 did there flying
straight. This shows which scenarios the policy should save, and gives a
scenario by scenario comparison with Gazebo for the policy as well.

The report also goes to rl/logs/evaluate_<run>.txt.

Usage (in the VM, after  source rl/env.sh):
  python rl/evaluate.py rl/runs/mujoco_1
  python rl/evaluate.py rl/runs/mujoco_1 --dynamics point
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402
from policy import AvoidancePolicy  # noqa: E402

GAZEBO_METRICS = HERE.parent / "results" / "2026-09-06_3d_sweep" / "metrics.csv"
SCENARIOS = [(n, s) for n in (4, 8, 12, 16, 20) for s in (1, 2, 3, 4, 5)]
LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(text)


def gazebo_straight():
    if not GAZEBO_METRICS.exists():
        return {}
    with open(GAZEBO_METRICS) as f:
        return {(int(r["n_obstacles"]), int(r["seed"])): int(r["reached"]) == 1
                for r in csv.DictReader(f) if r["k_drones"] == "1"}


def fly(env, n, seed, policy=None, check=None):
    obs, _ = env.reset(options={"n": n, "seed": seed})
    while True:
        if policy is None:
            to_target = env.target - env.pos
            a = to_target / max(np.linalg.norm(to_target), 1e-9)
        else:
            built = policy.observe(env.pos, env.vel, env.target, env.boxes)
            check.append(float(np.max(np.abs(built - obs))))
            a = policy.action(built)
        obs, _, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="run directory, or a model .zip inside one")
    ap.add_argument("--dynamics", choices=["mujoco", "point"], default=None,
                    help="flight model (default: the one the policy was trained on)")
    args = ap.parse_args()

    policy = AvoidancePolicy(args.run)
    dynamics = args.dynamics or (policy.dynamics if policy.dynamics in ("mujoco", "point")
                                 else "mujoco")
    env = maze.MazeEnv(dynamics=dynamics, lidar=policy.lidar)
    gazebo = gazebo_straight()

    say(f"evaluation {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"policy  {policy.describe()}")
    say(f"flown in the {dynamics} model")
    say()
    say(f"{'scenario':<13}{'gazebo':<10}{'straight':<15}{'policy':<15}"
        f"{'closest':>8}{'ratio':>8}{'highest':>9}")

    diffs = []
    straight_counts, policy_counts = Counter(), Counter()
    saved, lost = [], []
    for n, s in SCENARIOS:
        st = fly(env, n, s)
        po = fly(env, n, s, policy, diffs)
        straight_counts[st["outcome"]] += 1
        policy_counts[po["outcome"]] += 1
        if st["outcome"] != "reached" and po["outcome"] == "reached":
            saved.append((n, s))
        if st["outcome"] == "reached" and po["outcome"] != "reached":
            lost.append((n, s))
        gz = gazebo.get((n, s))
        gz_text = "-" if gz is None else ("reached" if gz else "crashed")
        ratio = po["travelled"] / po["euclidean"] if po["outcome"] == "reached" else float("nan")
        say(f"n{n:<2} seed {s:<4}{gz_text:<10}{st['outcome']:<15}{po['outcome']:<15}"
            f"{max(po['min_clearance'], -1):>7.2f}m{ratio:>8.3f}{po['max_altitude']:>8.1f}m")

    say()
    say(f"straight: {dict(straight_counts)}")
    say(f"policy:   {dict(policy_counts)}")
    say(f"saved by the policy (straight crashes, policy arrives): "
        f"{', '.join(f'n{n} s{s}' for n, s in saved) or 'none'}")
    say(f"lost by the policy (straight arrives, policy does not): "
        f"{', '.join(f'n{n} s{s}' for n, s in lost) or 'none'}")
    say()
    worst = max(diffs) if diffs else float("nan")
    say(f"deployment observation vs training observation, over {len(diffs)} decisions: "
        f"largest difference {worst:.2e}")
    ok = worst < 1e-5
    say("PASS  the wrapper builds the same observation as training" if ok else
        "FAIL  the wrapper's observation differs from training")

    out = HERE / "logs" / f"evaluate_{policy.model_file.parent.name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(LINES) + "\n")
    print(f"\nreport written to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
