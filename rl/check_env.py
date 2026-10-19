#!/usr/bin/env python3
"""Checks for rl/env.py, run for both flight models.

  1. check_env      Stable-Baselines3's own checker for the Gymnasium interface
  2. calibration    how the drone actually flies for a given action: drift
                    while hovering, speed and height loss flying level, climb
                    rate. In MuJoCo these come from the package's controller,
                    so they have to be measured, not assumed.
  3. random policy  should mostly crash; arriving by accident should be rare
  4. straight line  flying straight at the target, the no-avoidance baseline,
                    on the 25 single-drone scenarios of the Gazebo sweep and
                    compared run by run with what PX4 did there. If the
                    environment is faithful, the same runs should collide in
                    both. It is also the number the trained policy has to beat.
  5. speed          decisions per second, which sets the training time

The report also goes to rl/logs/check_env.txt.

Usage (in the VM, after  source rl/env.sh):
  python rl/check_env.py
  python rl/check_env.py --dynamics point        # only the point model
"""

import argparse
import csv
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402

LOG = HERE / "logs" / "check_env.txt"
GAZEBO_METRICS = HERE.parent / "results" / "2026-09-06_3d_sweep" / "metrics.csv"
LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(text)


def section(title):
    say()
    say(f"=== {title} ===")


def straight(env):
    """The baseline: aim at the target."""
    to_target = env.target - env.pos
    d = np.linalg.norm(to_target)
    return to_target / d if d > 1e-9 else np.zeros(3)


def run_episode(env, policy, options=None, seed=None):
    env.reset(seed=seed, options=options)
    total = 0.0
    while True:
        a = policy(env)
        _, r, terminated, truncated, info = env.step(a)
        total += r
        if terminated or truncated:
            info["return"] = total
            return info


def check_interface(env):
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    obs, _ = env.reset(seed=0)
    say(f"    observation {obs.shape} in [{obs.min():.2f}, {obs.max():.2f}], "
        f"action {env.action_space.shape}")
    say("    check_env passed")


def calibrate(env):
    dyn = env.dyn
    dt = 1.0 / maze.CTRL_FREQ

    def fly(direction, seconds):
        pos, _ = dyn.reset()
        start = pos.copy()
        track = [pos.copy()]
        for _ in range(int(seconds / dt)):
            pos, _ = dyn.step(np.asarray(direction, dtype=float))
            track.append(pos.copy())
        return start, np.array(track)

    start, track = fly((0, 0, 0), 3.0)
    drift = float(np.linalg.norm(track[-1] - start))
    say(f"    hover 3 s:          drift {drift:.3f} m")

    start, track = fly((1, 0, 0), 6.0)
    half = len(track) // 2
    speed = float(np.linalg.norm(track[-1] - track[half]) / ((len(track) - 1 - half) * dt))
    sag = float(start[2] - track[half:, 2].min())
    say(f"    level flight +x:    {speed:.2f} m/s steady, height loss {sag:.2f} m")

    start, track = fly((0, 0, 1), 3.0)
    climb = float((track[-1, 2] - track[len(track) // 2, 2]) / (1.5))
    say(f"    climb +z:           {climb:.2f} m/s")
    say(f"    (waypoint {maze.LOOKAHEAD} m ahead; point model set to "
        f"{maze.POINT_SPEED} m/s)")


def outcome_line(counts, n):
    order = ["reached", "collided", "ground", "out_of_bounds", "timeout"]
    return ", ".join(f"{k} {counts.get(k, 0)}/{n}" for k in order if counts.get(k))


def random_policy(env, episodes):
    rng = np.random.default_rng(0)
    infos = [run_episode(env, lambda e: rng.uniform(-1, 1, 3), seed=i)
             for i in range(episodes)]
    counts = Counter(i["outcome"] for i in infos)
    say(f"    {outcome_line(counts, episodes)}")
    say(f"    mean length {np.mean([i['decisions'] for i in infos]):.0f} decisions, "
        f"mean return {np.mean([i['return'] for i in infos]):.1f}")


def gazebo_k1_runs():
    if not GAZEBO_METRICS.exists():
        return []
    with open(GAZEBO_METRICS) as f:
        rows = [r for r in csv.DictReader(f) if r["k_drones"] == "1"]
    return [(int(r["n_obstacles"]), int(r["seed"]), int(r["reached"]) == 1) for r in rows]


def straight_vs_gazebo(env):
    runs = gazebo_k1_runs()
    if not runs:
        say(f"    no Gazebo results at {GAZEBO_METRICS}, skipped")
        return
    agree = 0
    say(f"    {'scenario':<12} {'gazebo':<9} {'here':<14} min clearance")
    for n, s, gz_reached in sorted(runs):
        info = run_episode(env, straight, options={"n": n, "seed": s})
        here_reached = info["outcome"] == "reached"
        agree += here_reached == gz_reached
        mark = "" if here_reached == gz_reached else "   <- differs"
        say(f"    n{n:<2} seed {s:<3} {'reached' if gz_reached else 'crashed':<9} "
            f"{info['outcome']:<14} {info['min_clearance']:6.2f} m{mark}")
    say(f"    agreement with Gazebo: {agree}/{len(runs)}")


def straight_random(env, episodes):
    infos = [run_episode(env, straight, seed=1000 + i) for i in range(episodes)]
    counts = Counter(i["outcome"] for i in infos)
    say(f"    {outcome_line(counts, episodes)}")
    reached = [i for i in infos if i["outcome"] == "reached"]
    if reached:
        ratio = np.mean([i["travelled"] / i["euclidean"] for i in reached])
        say(f"    ratio over the ones that arrived: {ratio:.3f}")


def throughput(env):
    rng = np.random.default_rng(1)
    env.reset(seed=1)
    n, start = 0, time.perf_counter()
    while time.perf_counter() - start < 5.0:
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 3))
        n += 1
        if term or trunc:
            env.reset()
    rate = n / (time.perf_counter() - start)
    say(f"    {rate:,.0f} decisions/s, {rate * maze.DECISION_EVERY:,.0f} control steps/s, "
        f"one environment")
    say(f"    so 1 million decisions take about {1e6 / rate / 60:.0f} min on one environment")


def run_checks(name, episodes):
    section(f"flight model: {name}")
    try:
        env = maze.MazeEnv(dynamics=name)
    except Exception:
        say("FAIL  could not build the environment")
        for line in traceback.format_exc().strip().splitlines()[-6:]:
            say(f"    {line}")
        return [name + ": build"]

    failed = []
    steps = [
        ("1 check_env", lambda: check_interface(env)),
        ("2 calibration", lambda: calibrate(env)),
        (f"3 random policy, {episodes} episodes", lambda: random_policy(env, episodes)),
        ("4a straight line on the Gazebo scenarios", lambda: straight_vs_gazebo(env)),
        (f"4b straight line on {episodes * 4} random training mazes",
         lambda: straight_random(env, episodes * 4)),
        ("5 speed", lambda: throughput(env)),
    ]
    for title, fn in steps:
        say(f"--- {title} ---")
        start = time.time()
        try:
            fn()
            say(f"PASS  ({time.time() - start:.1f} s)")
        except Exception:
            failed.append(f"{name}: {title}")
            for line in traceback.format_exc().strip().splitlines()[-6:]:
                say(f"    {line}")
            say("FAIL")
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dynamics", choices=["mujoco", "point", "both"], default="both")
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    say(f"environment checks {time.strftime('%Y-%m-%d %H:%M:%S')}")
    names = ["point", "mujoco"] if args.dynamics == "both" else [args.dynamics]
    failed = []
    for name in names:
        failed += run_checks(name, args.episodes)

    section("summary")
    say("all checks ran" if not failed else f"{len(failed)} failed: " + "; ".join(failed))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(LINES) + "\n")
    print(f"\nreport written to {LOG}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
