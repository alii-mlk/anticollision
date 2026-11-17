#!/usr/bin/env python3
"""Compare policies on many mazes that none of them has seen.

The sets used during training are small (25 and 100 flights), so the score of
a policy on them moves by several points from one checkpoint to the next, and
the best checkpoint was chosen on one of them. This script flies every policy,
and flying straight, on the same larger set of new mazes and reports each rate
with a 95% confidence interval, so two policies can be told apart and the
numbers can be quoted.

The mazes: maze i has HELDOUT_N[i % 5] obstacles and seed FIRST_SEED + i, for
each number of drones asked for. These seeds are used nowhere else (the Gazebo
sweep uses 1 to 5, the held-out sets of rl/train.py 101 to 120). A maze the
generator cannot build for that many drones is left out for every policy and
counted.

Reported per policy and number of drones:

  reached    drones that arrived
  crashed    drones that hit an obstacle, another drone or the ground, or left
             the workspace
  timeout    drones still flying after a minute, without having crashed
  hit drone  the part of crashed that was another drone
  ratio      travelled / straight-line distance, over the drones that arrived
  clearance  median closest approach to anything, over the drones that arrived

The intervals are Wilson intervals over drone flights. Drones in the same maze
are not independent, so with several drones the true uncertainty is somewhat
larger than the interval shown.

Every flight is also written to rl/logs/large_<name>.csv.

Usage (in the VM, after  source rl/env.sh):
  python rl/evaluate_large.py rl/runs/mujoco_1 rl/runs/swarm_1 \\
      rl/runs/swarm_1/final.zip --drones 1 3 --mazes 600 --name compare_1
"""

import argparse
import csv
import math
import multiprocessing
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from swarm import Scene, fly_scenarios  # noqa: E402
from train import HELDOUT_N, straight_policy  # noqa: E402

FIRST_SEED = 100_000
CRASHES = ("collided", "hit_drone", "ground", "out_of_bounds")
LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(text)


def wilson(successes, n, z=1.96):
    """95% confidence interval for a proportion, as (low, high)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def label_of(path):
    """mujoco_1 for a run directory, swarm_1/final for a file inside one."""
    path = Path(path)
    return path.name if path.is_dir() else f"{path.parent.name}/{path.stem}"


def fly_chunk(task):
    """One process: fly a list of mazes with one policy and k drones."""
    label, path, k, dynamics, scenarios = task
    if path is None:
        scene = Scene(k, dynamics=dynamics, respawn=False)
        act = straight_policy
    else:
        from policy import AvoidancePolicy
        policy = AvoidancePolicy(path)
        scene = Scene(k, dynamics=dynamics, respawn=False, lidar=policy.lidar,
                      neighbours=policy.neighbours)

        def act(_scene, obs):
            return policy.model.predict(obs, deterministic=True)[0]

    return label, k, fly_scenarios(scene, act, scenarios)


def summarise(infos):
    n = len(infos)
    counts = Counter(i["outcome"] for i in infos)
    arrived = [i for i in infos if i["outcome"] == "reached"]
    crashed = sum(counts[c] for c in CRASHES)
    return {
        "flights": n,
        "mazes": len({(i["n_obstacles"], i["scene_seed"]) for i in infos}),
        "reached": counts["reached"], "crashed": crashed,
        "timeout": counts["timeout"], "hit_drone": counts["hit_drone"],
        "ratio": float(np.mean([i["travelled"] / i["euclidean"] for i in arrived]))
                 if arrived else float("nan"),
        "clearance": float(np.median([i["min_clearance"] for i in arrived]))
                     if arrived else float("nan"),
    }


def rate(successes, n):
    low, high = wilson(successes, n)
    return f"{100 * successes / n:5.1f}% [{100 * low:4.1f}, {100 * high:4.1f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policies", type=Path, nargs="*",
                    help="run directories (their best.zip) or model .zip files inside one")
    ap.add_argument("--drones", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--mazes", type=int, default=600, help="mazes per number of drones")
    ap.add_argument("--dynamics", choices=["mujoco", "point"], default="mujoco")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--first-seed", type=int, default=FIRST_SEED,
                    help="use another block of mazes, e.g. to choose a checkpoint "
                         "on one block and report it on a different one")
    ap.add_argument("--no-straight", action="store_true",
                    help="do not fly the straight-line baseline")
    args = ap.parse_args()

    scenarios = [(HELDOUT_N[i % len(HELDOUT_N)], args.first_seed + i)
                 for i in range(args.mazes)]
    contenders = [(label_of(p), p) for p in args.policies]
    if not args.no_straight:
        contenders.insert(0, ("straight", None))
    if not contenders:
        ap.error("nothing to fly: give at least one policy, or drop --no-straight")
    size = math.ceil(len(scenarios) / args.workers)
    chunks = [scenarios[i:i + size] for i in range(0, len(scenarios), size)]
    tasks = [(label, path, k, args.dynamics, chunk)
             for k in args.drones for label, path in contenders for chunk in chunks]

    say(f"large evaluation {time.strftime('%Y-%m-%d %H:%M:%S')}, {args.dynamics} flight "
        f"model, {args.mazes} mazes per number of drones, seeds from {args.first_seed}")
    started = time.monotonic()
    results = {}
    # fork, and one task per process, so every task starts from a clean
    # process and its simulators are freed when it ends.
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for done, (label, k, infos) in enumerate(pool.imap_unordered(fly_chunk, tasks), 1):
            results.setdefault((k, label), []).extend(infos)
            print(f"  {done}/{len(tasks)} parts done, "
                  f"{(time.monotonic() - started) / 60:.1f} min", flush=True)

    say()
    header = (f"{'k':>2}  {'policy':<22}{'flights':>8}  {'reached':<21}{'crashed':<21}"
              f"{'timeout':>8}{'hit drone':>10}{'ratio':>7}{'clearance':>10}")
    say(header)
    say("-" * len(header))
    for k in args.drones:
        for label, _ in contenders:
            s = summarise(results.get((k, label), []))
            if not s["flights"]:
                say(f"{k:>2}  {label:<22} no flights")
                continue
            n = s["flights"]
            say(f"{k:>2}  {label:<22}{n:>8}  {rate(s['reached'], n):<21}"
                f"{rate(s['crashed'], n):<21}{100 * s['timeout'] / n:>7.1f}%"
                f"{s['hit_drone']:>10}{s['ratio']:>7.3f}{s['clearance']:>9.2f}m")
        built = summarise(results.get((k, contenders[0][0]), []))["mazes"]
        if built < args.mazes:
            say(f"    ({args.mazes - built} of {args.mazes} mazes could not be built "
                f"with {k} drones and were left out)")
        say()
    say(f"took {(time.monotonic() - started) / 60:.1f} min")

    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fields = ["policy", "k_drones", "n_obstacles", "scene_seed", "drone", "outcome",
              "decisions", "euclidean", "travelled", "min_clearance", "max_altitude"]
    with open(logs / f"large_{args.name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for (k, label), infos in sorted(results.items()):
            for info in infos:
                w.writerow({"policy": label, **info})
    (logs / f"large_{args.name}.txt").write_text("\n".join(LINES) + "\n")
    print(f"report written to {logs / f'large_{args.name}.txt'}")


if __name__ == "__main__":
    main()
