#!/usr/bin/env python3
"""Checks for rl/swarm.py, the maze with several drones.

  1. one drone      a scene with one drone must fly exactly like rl/env.py:
                    same outcome and same travelled distance on the 25
                    single-drone Gazebo scenarios
  2. independence   three drones that ignore each other must each fly exactly
                    as the same drone does alone in the same maze. Each drone
                    has its own simulator instance in the same process, and
                    this shows the instances do not disturb each other.
  3. straight line  flying straight with 2 and 3 drones on the 50 scenarios of
                    the Gazebo sweep, compared run by run with how many drones
                    arrived there
  4. the policy     an already trained policy on the same scenarios with 1, 2
                    and 3 drones, flown twice: together, and each drone alone
                    in the same maze. If it does as badly alone, the mazes are
                    harder; if it does worse together, the other drones are
                    the problem.
  5. training side  the vector environment PPO trains on: shapes, value
                    ranges, what comes back when a mission ends, and speed,
                    without and with the neighbour inputs
  6. neighbours     the inputs that describe the nearest drones
                    (env.neighbour_features), against cases worked out by
                    hand, and that the two drones of a pair get opposite
                    passing vectors

The report also goes to rl/logs/check_swarm.txt.

Usage (in the VM, after  source rl/env.sh):
  python rl/check_swarm.py --dynamics point       # one minute, no MuJoCo
  python rl/check_swarm.py                        # mujoco, with rl/runs/mujoco_1
  python rl/check_swarm.py --policy rl/runs/swarm_1
"""

import argparse
import csv
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402
from swarm import Scene, SwarmVecEnv, fly_scenarios  # noqa: E402
from train import GAZEBO_SCENARIOS, straight_policy  # noqa: E402

LOG = HERE / "logs" / "check_swarm.txt"
SWEEP = HERE.parent / "results" / "2026-10-19_rl_sweep"
LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(text)


def gazebo_reached(arm, k, path=None):
    """{(n, seed): drones that reached} for one arm of the Gazebo sweep."""
    path = path or SWEEP / arm / "metrics.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {(int(r["n_obstacles"]), int(r["seed"])): int(r["reached"])
                for r in csv.DictReader(f) if int(r["k_drones"]) == k}


def by_scene(infos):
    """Group per-drone results: {scene seed and size: {drone: info}}."""
    scenes = {}
    for info in infos:
        scenes.setdefault((info["n_obstacles"], info["scene_seed"]), {})[info["drone"]] = info
    return scenes


def one_drone(dynamics):
    env = maze.MazeEnv(dynamics=dynamics)
    scene = Scene(1, dynamics=dynamics, respawn=False)
    same = 0
    for n, seed in GAZEBO_SCENARIOS:
        obs, _ = env.reset(options={"n": n, "seed": seed})
        while True:
            obs, _, terminated, truncated, single = env.step(straight_policy(env, obs))
            if terminated or truncated:
                break
        multi = fly_scenarios(scene, straight_policy, [(n, seed)])[0]
        ok = (single["outcome"] == multi["outcome"]
              and abs(single["travelled"] - multi["travelled"]) < 1e-6
              and abs(single["min_clearance"] - multi["min_clearance"]) < 1e-6)
        same += ok
        if not ok:
            say(f"    n{n} seed {seed}: env.py {single['outcome']} "
                f"{single['travelled']:.4f} m, scene {multi['outcome']} "
                f"{multi['travelled']:.4f} m")
    say(f"    identical to rl/env.py on {same}/{len(GAZEBO_SCENARIOS)} scenarios")
    if same != len(GAZEBO_SCENARIOS):
        raise AssertionError("a one-drone scene does not fly like rl/env.py")


def independence(dynamics):
    together = Scene(3, dynamics=dynamics, respawn=False, interact=False)
    alone = Scene(1, dynamics=dynamics, respawn=False)
    same = total = 0
    for seed in (1, 2, 3, 4, 5):
        drones, boxes, _ = together.draw(12, seed)

        def fly(scene, missions):
            obs = scene.begin(missions, boxes, seed=seed)
            done_infos = {}
            while not scene.finished:
                flying = scene.present.copy()
                actions = np.zeros((scene.k, 3))
                actions[flying] = straight_policy(scene, obs[flying])
                obs, _, dones, infos = scene.step(actions)
                done_infos.update({i["drone"]: i for i, d in zip(infos, dones) if d})
            return done_infos

        group = fly(together, drones)
        for i, drone in enumerate(drones):
            single = fly(alone, [drone])[0]
            ok = (single["outcome"] == group[i]["outcome"]
                  and abs(single["travelled"] - group[i]["travelled"]) < 1e-6)
            same += ok
            total += 1
            if not ok:
                say(f"    seed {seed} drone {i}: alone {single['outcome']} "
                    f"{single['travelled']:.4f} m, in the group "
                    f"{group[i]['outcome']} {group[i]['travelled']:.4f} m")
    say(f"    {same}/{total} drones fly the same alone and in a group of three")
    if same != total:
        raise AssertionError("the simulator instances disturb each other")


def straight_vs_gazebo(dynamics):
    for k in (2, 3):
        gazebo = gazebo_reached("straight", k)
        scene = Scene(k, dynamics=dynamics, respawn=False)
        scenes = by_scene(fly_scenarios(scene, straight_policy, GAZEBO_SCENARIOS))
        agree = here_total = gazebo_total = touched = 0
        differ = []
        for (n, seed), drones in sorted(scenes.items()):
            here = sum(i["outcome"] == "reached" for i in drones.values())
            touched += sum(i["outcome"] == "hit_drone" for i in drones.values())
            here_total += here
            if (n, seed) in gazebo:
                gazebo_total += gazebo[(n, seed)]
                if gazebo[(n, seed)] == here:
                    agree += 1
                else:
                    differ.append(f"n{n} s{seed} (Gazebo {gazebo[(n, seed)]}, here {here})")
        say(f"    k={k}: same number of arrivals in {agree}/{len(gazebo)} runs; "
            f"drones reached: Gazebo {gazebo_total}, here {here_total} of "
            f"{k * len(scenes)}; drones that hit another drone here: {touched}")
        if differ:
            say(f"         differ: {', '.join(differ)}")
    if not gazebo:
        say(f"    (no Gazebo results found under {SWEEP})")


def policy_alone_and_together(dynamics, run, gazebo_metrics=None):
    from policy import AvoidancePolicy
    policy = AvoidancePolicy(run)
    say(f"    policy {policy.describe()}")

    def act(scene, obs):
        return policy.model.predict(obs, deterministic=True)[0]

    say(f"    {'k':>3} {'drones':>7} {'Gazebo':>8} {'together':>9} {'alone':>7}   "
        f"crashes together (obstacle / drone / other)")
    for k in (1, 2, 3):
        gazebo = gazebo_reached("policy", k, gazebo_metrics)
        results = {}
        for mode, interact in (("together", True), ("alone", False)):
            scene = Scene(k, dynamics=dynamics, respawn=False, interact=interact,
                          lidar=policy.lidar, neighbours=policy.neighbours)
            results[mode] = fly_scenarios(scene, act, GAZEBO_SCENARIOS)
        counts = {m: Counter(i["outcome"] for i in infos) for m, infos in results.items()}
        t = counts["together"]
        other = sum(v for o, v in t.items() if o not in ("reached", "collided", "hit_drone"))
        gz = f"{sum(gazebo.values())}" if gazebo else "-"
        say(f"    {k:>3} {len(results['together']):>7} {gz:>8} {t['reached']:>9} "
            f"{counts['alone']['reached']:>7}   {t['collided']} / {t['hit_drone']} / {other}")
    say("    (columns Gazebo, together and alone: drones that reached their target)")


def neighbour_inputs():
    f = maze.neighbour_features
    zero = np.zeros(3)
    close = dict(atol=1e-4)

    assert not f(zero, zero, [], [], 3).any()
    assert f(zero, zero, [[7.0, 0, 0]], [zero], 3).shape == (30,)
    assert not f(zero, zero, [[7.0, 0, 0]], [zero], 3).any(), "beyond 6 m must read as nobody"

    # Head on along x, 4 m apart, 1 m/s each: they meet in 2 s at the same
    # point. Closeness 1 - 4/6, relative velocity -2/3, passing distance 0 so
    # full length sideways, and 1 - 2/5 for how soon.
    me = f(zero, [1.0, 0, 0], [[4.0, 0, 0]], [[-1.0, 0, 0]], 1)
    other = f([4.0, 0, 0], [-1.0, 0, 0], [zero], [[1.0, 0, 0]], 1)
    assert np.allclose(me[0:3], [1 / 3, 0, 0], **close), me
    assert np.allclose(me[3:6], [-2 / 3, 0, 0], **close), me
    assert np.isclose(np.linalg.norm(me[6:9]), 1.0, **close) and abs(me[6]) < 1e-6, me
    assert np.isclose(me[9], 0.6, **close), me
    assert np.allclose(other[6:9], -me[6:9], **close), "a head-on pair must get opposite sides"

    # Passing 1 m to the side: at (3, 1, 0) flying -x past a hovering drone.
    # Closest in 3 s at (0, 1, 0): length 1 - 1/4 along y, and 1 - 3/5.
    g = f(zero, zero, [[3.0, 1.0, 0]], [[-1.0, 0, 0]], 1)
    d = np.sqrt(10.0)
    assert np.allclose(g[0:3], (1 - d / 6) * np.array([3, 1, 0]) / d, **close), g
    assert np.allclose(g[6:9], [0, 0.75, 0], **close), g
    assert np.isclose(g[9], 0.4, **close), g

    # Moving apart: the passing is where it is now, and nothing is coming.
    g = f(zero, zero, [[2.0, 0, 0]], [[1.0, 0, 0]], 1)
    assert np.allclose(g[6:9], [0.5, 0, 0], **close) and g[9] == 0.0, g

    # Nearest first, and only as many as asked for.
    g = f(zero, zero, [[5.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [2.0, 0, 0]],
          np.zeros((4, 3)), 3).reshape(3, 10)
    assert np.allclose(g[:, 0], [1 - 1 / 6, 1 - 2 / 6, 1 - 3 / 6], **close), g[:, 0]

    # Any pair: what one drone is told about the other is the mirror image
    # of what the other is told about it.
    rng = np.random.default_rng(1)
    for _ in range(500):
        pa, pb = rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3)
        va, vb = rng.uniform(-1.2, 1.2, 3), rng.uniform(-1.2, 1.2, 3)
        a, b = f(pa, va, [pb], [vb], 1), f(pb, vb, [pa], [va], 1)
        assert np.allclose(a[0:9], -b[0:9], atol=1e-5) and np.isclose(a[9], b[9], atol=1e-5)
        assert np.abs(a).max() <= 1.0 + 1e-6
    say("    hand-worked cases and 500 random pairs agree")


def training_side(dynamics, neighbours=0):
    sizes = [1, 2, 3] if not neighbours else [2, 6]
    venv = SwarmVecEnv(sizes, dynamics=dynamics, neighbours=neighbours)
    try:
        rng = np.random.default_rng(0)
        n_rays = maze.Lidar().n_rays
        n_obs = n_rays + 8 + neighbours * maze.NEIGHBOUR_FEATURES
        obs = venv.reset()
        assert obs.shape == (sum(sizes), n_obs), obs.shape
        steps = 3000 if dynamics == "point" else 400
        outcomes, later = Counter(), 0
        start = time.monotonic()
        for _ in range(steps):
            # Roughly towards the target, so some arrive and some crash.
            actions = obs[:, n_rays:n_rays + 3] + 0.4 * rng.standard_normal((len(obs), 3))
            obs, rewards, dones, infos = venv.step(np.clip(actions, -1, 1))
            assert obs.shape == (sum(sizes), n_obs) and obs.dtype == np.float32
            assert rewards.shape == dones.shape == (sum(sizes),)
            assert np.isfinite(obs).all() and np.isfinite(rewards).all()
            assert obs.min() >= -1.0 and obs.max() <= 1.0
            for done, info in zip(dones, infos):
                if done:
                    assert info["outcome"] is not None
                    assert info["terminal_observation"].shape == (n_obs,)
                    outcomes[info["outcome"]] += 1
                    later += info["mission"] > 0
        rate = steps * sum(sizes) / (time.monotonic() - start)
        say(f"    scenes with {sizes} drones, {neighbours} neighbours, {steps} steps: "
            f"{sum(outcomes.values())} missions ended, {later} of them later missions")
        say(f"    {dict(outcomes)}")
        say(f"    {rate:,.0f} decisions/s over {len(sizes)} processes")
    finally:
        venv.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dynamics", choices=["mujoco", "point"], default="mujoco")
    ap.add_argument("--policy", type=Path, default=HERE / "runs" / "mujoco_1",
                    help="trained run to fly in check 4 (skipped if it does not exist)")
    ap.add_argument("--policy-sweep", type=Path, default=None,
                    help="metrics.csv of the Gazebo sweep flown with that policy, "
                         "for the Gazebo column of check 4 (default: the October sweep)")
    args = ap.parse_args()

    say(f"swarm checks {time.strftime('%Y-%m-%d %H:%M:%S')}, {args.dynamics} flight model")
    steps = [
        ("1 one drone, against rl/env.py", lambda: one_drone(args.dynamics)),
        ("2 independence of the simulator instances", lambda: independence(args.dynamics)),
        ("3 straight line with 2 and 3 drones, against Gazebo",
         lambda: straight_vs_gazebo(args.dynamics)),
        ("5 training side", lambda: training_side(args.dynamics)),
        ("5 training side, with neighbour inputs",
         lambda: training_side(args.dynamics, neighbours=3)),
        ("6 neighbour inputs", neighbour_inputs),
    ]
    if (args.policy / "best.zip").exists() or args.policy.suffix == ".zip":
        steps.insert(3, (f"4 the policy {args.policy.name}, together and alone",
                         lambda: policy_alone_and_together(args.dynamics, args.policy,
                                                           args.policy_sweep)))
    else:
        say(f"(check 4 skipped: no trained run at {args.policy})")

    failed = []
    for title, fn in steps:
        say()
        say(f"--- {title} ---")
        start = time.monotonic()
        try:
            fn()
            say(f"PASS  ({time.monotonic() - start:.1f} s)")
        except Exception:
            failed.append(title)
            for line in traceback.format_exc().strip().splitlines()[-8:]:
                say(f"    {line}")
            say("FAIL")

    say()
    say("all checks passed" if not failed else f"FAILED: {'; '.join(failed)}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(LINES) + "\n")
    print(f"\nreport written to {LOG}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
