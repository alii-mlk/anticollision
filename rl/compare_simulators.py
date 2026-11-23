#!/usr/bin/env python3
"""Fly the Gazebo scenarios again in MuJoCo and compare drone by drone.

The professor asked why the results in Gazebo and in MuJoCo are not the same.
This script looks at it in two ways.

1. Outcomes. Every scenario of a Gazebo batch is flown in MuJoCo with the same
   policy, the same drones, starts and targets. For every drone the two
   outcomes are paired: arrived in both, only in Gazebo, only in MuJoCo, or in
   neither. McNemar's exact test says whether the drones that differ lean to
   one side more than chance would explain. The drones that differ are
   listed, so their tracks can be looked at.

2. How the drones follow the commands. At every decision the policy asks for
   a velocity. Between two decisions the drone flies some distance, and the
   distance divided by the time is the velocity it really flew. The
   difference between the two is the tracking error. It is computed the same
   way in both simulators: in MuJoCo from this replay, in Gazebo from the
   decisions.csv files that run_mission.py writes since 2026-11-19 (older
   batches have none). The error after one second of flight, between where
   the commands said the drone would be and where it was, is the number the
   safety layer of stage 5 has to allow for.

--decision-every sets how many control steps (1/48 s) a decision is held in
MuJoCo: 5 is 0.104 s as in training, 7 is 0.146 s, close to the 0.15 s the
Gazebo runs before 2026-11-19 decided at.

The report goes to rl/logs/compare_<name>.txt and every drone to
rl/logs/compare_<name>.csv.

Usage (in the VM, after  source rl/env.sh):
  python rl/compare_simulators.py --name final_0104
  python rl/compare_simulators.py --decision-every 7 --name final_0146
  python rl/compare_simulators.py --gazebo px4/runs3d/<new batch>/drones.csv \\
      --gazebo-batch px4/runs3d/<new batch> --name final_fixed
"""

import argparse
import csv
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402
from swarm import Scene, fly_scenarios  # noqa: E402

RESULTS = HERE.parent / "results" / "2026-11-15_swarm_sweep"
DEFAULT_POLICY = RESULTS / "model" / "final.zip"
DEFAULT_GAZEBO = [RESULTS / "policy" / "drones.csv", RESULTS / "k4_policy" / "drones.csv"]
LOOKAHEAD_S = 1.0
LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(text)


def read_gazebo(paths):
    """{k: {(n, seed, drone): row}} from one or more drones.csv files."""
    drones = defaultdict(dict)
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (int(r["n_obstacles"]), int(r["seed"]), int(r["drone"]))
                drones[int(r["k_drones"])][key] = r
    return drones


def mcnemar(only_a, only_b):
    """Two-sided exact McNemar test on the drones that differ."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    low = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(low + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def tracking(rows):
    """Tracking errors of one drone from its decisions, oldest first.

    rows: (t, position, measured velocity, commanded velocity). Returns two
    lists: the velocity error of every decision interval, and the position
    error after LOOKAHEAD_S seconds from every decision that has one.
    """
    velocity_errors, position_errors = [], []
    for a, b in zip(rows, rows[1:]):
        dt = b[0] - a[0]
        if dt > 0:
            flown = (b[1] - a[1]) / dt
            velocity_errors.append(float(np.linalg.norm(flown - a[3])))
    for i, start in enumerate(rows):
        expected = np.zeros(3)
        for a, b in zip(rows[i:], rows[i + 1:]):
            expected += a[3] * (b[0] - a[0])
            if b[0] - start[0] >= LOOKAHEAD_S - 1e-6:
                position_errors.append(float(np.linalg.norm(b[1] - start[1] - expected)))
                break
    return velocity_errors, position_errors


def describe_errors(name, velocity_errors, position_errors):
    if not velocity_errors:
        say(f"    {name}: no decisions recorded")
        return
    v = np.array(velocity_errors)
    p = np.array(position_errors) if position_errors else np.array([np.nan])
    say(f"    {name:<8} velocity error  median {np.median(v):.3f}  95% {np.percentile(v, 95):.3f}"
        f"  largest {v.max():.3f} m/s   ({len(v):,} intervals)")
    say(f"    {'':<8} after {LOOKAHEAD_S:.0f} s       median {np.nanmedian(p):.3f}  "
        f"95% {np.nanpercentile(p, 95):.3f}  largest {np.nanmax(p):.3f} m")


def read_gazebo_decisions(batch):
    """{(run, drone): rows} from the decisions.csv files of a Gazebo batch."""
    tracks = {}
    for path in sorted(Path(batch).glob("*/run/drone*/decisions.csv")):
        with open(path) as f:
            rows = []
            for r in csv.DictReader(f):
                rows.append((float(r["t"]),
                             np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
                             np.array([float(r["vx"]), float(r["vy"]), float(r["vz"])]),
                             np.array([float(r["cmd_x"]), float(r["cmd_y"]), float(r["cmd_z"])])))
        tracks[(path.parent.parent.parent.name, path.parent.name)] = rows
    return tracks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--gazebo", type=Path, nargs="+", default=DEFAULT_GAZEBO,
                    help="drones.csv of the Gazebo batches to compare with")
    ap.add_argument("--gazebo-batch", type=Path, nargs="*", default=[],
                    help="Gazebo batch folders with decisions.csv files, for the tracking part")
    ap.add_argument("--decision-every", type=int, default=maze.DECISION_EVERY,
                    help="control steps per decision in MuJoCo (5 = 0.104 s, as in training)")
    ap.add_argument("--dynamics", choices=["mujoco", "point"], default="mujoco")
    ap.add_argument("--name", default=time.strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    from policy import AvoidancePolicy
    policy = AvoidancePolicy(args.policy)
    decision_dt = args.decision_every / maze.CTRL_FREQ
    limit = round(maze.MAX_DECISIONS * maze.DECISION_EVERY / args.decision_every)
    maze.DECISION_EVERY = args.decision_every
    gazebo = read_gazebo(args.gazebo)

    say(f"Gazebo against MuJoCo, {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"policy {policy.describe()}")
    say(f"flown in the {args.dynamics} model with a decision every {decision_dt:.3f} s")
    say(f"Gazebo results: {', '.join(str(p) for p in args.gazebo)}")
    started = time.monotonic()

    # Every decision of every drone in MuJoCo, for the tracking part.
    recorded = defaultdict(list)

    def act(scene, obs):
        actions = policy.model.predict(obs, deterministic=True)[0]
        for row, i in enumerate(np.flatnonzero(scene.present)):
            key = (scene.k, scene.n_obstacles, scene.scene_seed, int(i))
            command = policy.velocity(maze.clip_action(actions[row]))
            recorded[key].append((len(recorded[key]) * decision_dt,
                                  scene.pos[i].copy(), scene.vel[i].copy(), command))
        return actions

    rows = []
    say()
    say("1. Outcomes, the same drone in the same scenario")
    say()
    say(f"    {'k':>2} {'drones':>7} {'both':>6} {'Gazebo only':>12} {'MuJoCo only':>12}"
        f" {'neither':>8}   arrived Gazebo / MuJoCo   McNemar p")
    differ = []
    totals = Counter()
    for k in sorted(gazebo):
        scenarios = sorted({(n, s) for n, s, _ in gazebo[k]})
        scene = Scene(k, dynamics=args.dynamics, respawn=False, lidar=policy.lidar,
                      neighbours=policy.neighbours, max_decisions=limit)
        infos = fly_scenarios(scene, act, scenarios)
        here = {(i["n_obstacles"], i["scene_seed"], i["drone"]): i for i in infos}
        count = Counter()
        for key, g in sorted(gazebo[k].items()):
            m = here.get(key)
            if m is None:
                continue
            g_ok = g["outcome"] == "reached"
            m_ok = m["outcome"] == "reached"
            pair = {(True, True): "both", (True, False): "gazebo_only",
                    (False, True): "mujoco_only", (False, False): "neither"}[(g_ok, m_ok)]
            count[pair] += 1
            v, p = tracking(recorded[(k, *key)])
            rows.append({
                "k_drones": k, "n_obstacles": key[0], "seed": key[1], "drone": key[2],
                "gazebo": g["outcome"], "gazebo_clearance": g["min_clearance"],
                "mujoco": m["outcome"], "mujoco_clearance": f"{m['min_clearance']:.3f}",
                "pair": pair,
                "velocity_error_median": f"{np.median(v):.3f}" if v else "",
                "position_error_1s_median": f"{np.median(p):.3f}" if p else "",
            })
            if pair in ("gazebo_only", "mujoco_only"):
                differ.append(f"n{key[0]}_k{k}_s{key[1]} drone {key[2]}: Gazebo {g['outcome']} "
                              f"(closest {float(g['min_clearance']):.2f} m), MuJoCo {m['outcome']} "
                              f"(closest {m['min_clearance']:.2f} m)")
        totals.update(count)
        drones = sum(count.values())
        say(f"    {k:>2} {drones:>7} {count['both']:>6} {count['gazebo_only']:>12} "
            f"{count['mujoco_only']:>12} {count['neither']:>8}   "
            f"{count['both'] + count['gazebo_only']:>7} / {count['both'] + count['mujoco_only']:<7}"
            f"          {mcnemar(count['gazebo_only'], count['mujoco_only']):.3f}")
    drones = sum(totals.values())
    say(f"    {'all':>2} {drones:>7} {totals['both']:>6} {totals['gazebo_only']:>12} "
        f"{totals['mujoco_only']:>12} {totals['neither']:>8}   "
        f"{totals['both'] + totals['gazebo_only']:>7} / {totals['both'] + totals['mujoco_only']:<7}"
        f"          {mcnemar(totals['gazebo_only'], totals['mujoco_only']):.3f}")
    say()
    if differ:
        say("    drones that differ:")
        for line in differ:
            say(f"      {line}")
    else:
        say("    every drone had the same outcome in both")

    say()
    say("2. Following the commands (velocity flown between two decisions against the")
    say(f"   velocity asked for, and the position error after {LOOKAHEAD_S:.0f} s)")
    say()
    v_all, p_all = [], []
    for track in recorded.values():
        v, p = tracking(track)
        v_all.extend(v)
        p_all.extend(p)
    describe_errors("MuJoCo", v_all, p_all)
    for batch in args.gazebo_batch:
        g_v, g_p = [], []
        tracks = read_gazebo_decisions(batch)
        periods = []
        for track in tracks.values():
            v, p = tracking(track)
            g_v.extend(v)
            g_p.extend(p)
            if len(track) > 1:
                periods.append((track[-1][0] - track[0][0]) / (len(track) - 1))
        describe_errors("Gazebo", g_v, g_p)
        if periods:
            say(f"    {'':<8} {len(tracks)} drones from {batch}, decision every "
                f"{np.median(periods):.3f} s (median, {min(periods):.3f} to {max(periods):.3f})")

    say()
    say(f"took {(time.monotonic() - started) / 60:.1f} min")
    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / f"compare_{args.name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["k_drones"])
        w.writeheader()
        w.writerows(rows)
    (logs / f"compare_{args.name}.txt").write_text("\n".join(LINES) + "\n")
    print(f"report written to {logs / f'compare_{args.name}.txt'}")


if __name__ == "__main__":
    main()
