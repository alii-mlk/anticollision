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

--decision-every changes how many control steps (1/48 s each) one decision is
held for. Training used 5 (0.104 s). Until 2026-11-19 the Gazebo mission node
decided every 0.15 s, and 7 (0.146 s) is the closest MuJoCo can fly to that.
The time limit stays about a minute.

--shield flies every policy a second time with the safety layer of
rl/shield.py (a JSON settings file, or "default"), labelled "<policy> +
shield", on the same mazes. For those rows the report also gives how often the
shield changed the policy's command, how often it made the drone hover, and
how often it found no safe command. --shield-only skips the plain policies.
--recovery adds the recovery of rl/recovery.py to that arm (a stuck drone
climbs over the boxes), labelled "<policy> + shield + recovery"; the report
then says how many drones needed it and how many of them arrived.

--redraw is for many drones. The generator keeps 2.5 m clear around every
start and target, so with 15 or 20 drones it often cannot fit 16 or 20 boxes,
and leaving those mazes out would keep mostly the easy ones. With --redraw a
maze that cannot be built for k drones gets seed + 1,000,000 j instead, for
the first j (up to 2,000) that builds, with the same number of boxes. The
report says how many mazes were redrawn. The redrawn mazes are layouts where
the boxes fit, so a little less cluttered than an unrestricted draw.

--mazes-from flies only the mazes where some drone did not arrive in earlier
evaluations (their per-flight CSV files, .csv or .csv.gz, any policy), for
each number of drones in --drones, instead of the block of new mazes. That is
the quick check on the known hard cases, before a full evaluation.

Usage (in the VM, after  source rl/env.sh):
  python rl/evaluate_large.py rl/runs/mujoco_1 rl/runs/swarm_1 \\
      rl/runs/swarm_1/final.zip --drones 1 3 --mazes 600 --name compare_1
  python rl/evaluate_large.py results/2026-11-15_swarm_sweep/model/final.zip \\
      --shield default --no-straight --drones 1 4 10 --mazes 200 --name shield_1
  python rl/evaluate_large.py results/2026-11-15_swarm_sweep/model/final.zip \\
      --shield default --recovery --no-straight --drones 15 20 --mazes 400 \\
      --first-seed 300000 --redraw --name crowd_recovery
"""

import argparse
import csv
import math
import multiprocessing
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402
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
    label, path, k, dynamics, scenarios, decision_every, shield_settings, recovery = task
    # Each task runs in its own process, so changing the module's setting
    # here affects only this task.
    limit = round(maze.MAX_DECISIONS * maze.DECISION_EVERY / decision_every)
    maze.DECISION_EVERY = decision_every
    counts = None
    if path is None:
        scene = Scene(k, dynamics=dynamics, respawn=False, max_decisions=limit)
        act = straight_policy
    else:
        from policy import AvoidancePolicy
        policy = AvoidancePolicy(path)
        scene = Scene(k, dynamics=dynamics, respawn=False, lidar=policy.lidar,
                      neighbours=policy.neighbours, max_decisions=limit)

        def act(_scene, obs):
            return policy.model.predict(obs, deterministic=True)[0]

        if shield_settings is not None:
            from shield import Shield, ShieldSettings, shield_scene
            shield = Shield(ShieldSettings(**shield_settings),
                            decision_dt=decision_every / maze.CTRL_FREQ,
                            speed_xy=policy.speed_xy, speed_z=policy.speed_z)
            counts = defaultdict(Counter)
            recoveries = {} if recovery else None
            plain = act

            def act(sc, obs):
                return shield_scene(shield, sc, plain(sc, obs), counts, recoveries)

    infos = fly_scenarios(scene, act, scenarios)
    if counts is not None:
        for info in infos:
            c = counts.get((info["n_obstacles"], info["scene_seed"], info["drone"]), Counter())
            for what in ("changed", "hover", "no_safe", "recovery"):
                info[f"shield_{what}"] = c[what]
    return label, k, infos


REDRAW_STEP = 1_000_000
REDRAW_TRIES = 2_000


def buildable(task):
    """(n, seed) itself if the generator can build it for k drones, else the
    first seed + REDRAW_STEP * j that it can build, else None."""
    from swarm import scenario_gen
    k, n, seed = task
    for j in range(REDRAW_TRIES + 1):
        s = seed + REDRAW_STEP * j
        rng = random.Random(s)
        try:
            drones = scenario_gen.generate_drones(k, rng)
            scenario_gen.generate_obstacles(n, drones, rng)
        except SystemExit:
            continue
        return n, s
    return None


def mazes_with_trouble(paths, drones):
    """{k: [(obstacles, seed), ...]}: mazes where some drone did not arrive in these files."""
    import gzip
    found = defaultdict(set)
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as f:
            for r in csv.DictReader(f):
                k = int(r["k_drones"])
                if k in drones and r["outcome"] != "reached":
                    found[k].add((int(r["n_obstacles"]), int(r["scene_seed"])))
    return {k: sorted(found[k]) for k in drones}


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
        "decisions": sum(i["decisions"] for i in infos),
        "shielded": any("shield_changed" in i for i in infos),
        **{f"shield_{what}": sum(i.get(f"shield_{what}", 0) for i in infos)
           for what in ("changed", "hover", "no_safe")},
        "recovered": sum(1 for i in infos if i.get("shield_recovery", 0) > 0),
        "recovered_arrived": sum(1 for i in infos if i.get("shield_recovery", 0) > 0
                                 and i["outcome"] == "reached"),
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
    ap.add_argument("--decision-every", type=int, default=maze.DECISION_EVERY,
                    help="control steps per decision (5 = 0.104 s, as in training)")
    ap.add_argument("--no-straight", action="store_true",
                    help="do not fly the straight-line baseline")
    ap.add_argument("--shield", default=None,
                    help="also fly every policy with the safety layer: a JSON settings "
                         "file, or 'default'")
    ap.add_argument("--shield-only", action="store_true",
                    help="with --shield, fly the policies only with the safety layer")
    ap.add_argument("--recovery", action="store_true",
                    help="with --shield, add the recovery (a stuck drone climbs over the boxes)")
    ap.add_argument("--redraw", action="store_true",
                    help="give a maze the generator cannot build for k drones a new seed, "
                         "with the same number of boxes")
    ap.add_argument("--mazes-from", type=Path, nargs="+", default=None,
                    help="fly only the mazes where a drone did not arrive in these "
                         "per-flight CSV files")
    args = ap.parse_args()

    shield_settings = None
    if args.shield is not None:
        from dataclasses import asdict
        from shield import ShieldSettings
        settings = ShieldSettings.load(args.shield)
        shield_settings = asdict(settings)
    elif args.shield_only or args.recovery:
        ap.error("--shield-only and --recovery need --shield")

    if args.mazes_from:
        scenarios = mazes_with_trouble(args.mazes_from, args.drones)
        where = (f"the mazes where a drone did not arrive in "
                 f"{', '.join(p.name for p in args.mazes_from)}: "
                 + ", ".join(f"{len(v)} with {k} drones" for k, v in scenarios.items()))
    else:
        block = [(HELDOUT_N[i % len(HELDOUT_N)], args.first_seed + i) for i in range(args.mazes)]
        scenarios = {k: block for k in args.drones}
        where = f"{args.mazes} mazes per number of drones, seeds from {args.first_seed}"
    redrawn = {}
    if args.redraw:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for k in args.drones:
                found = pool.map(buildable, [(k, n, seed) for n, seed in scenarios[k]])
                redrawn[k] = sum(f is not None and f != m for f, m in zip(found, scenarios[k]))
                missing = found.count(None)
                scenarios[k] = [f for f in found if f is not None]
                print(f"  {k} drones: {redrawn[k]} mazes redrawn, {missing} not found",
                      flush=True)
                if missing:
                    where += f"; {missing} mazes with {k} drones not found even redrawn"
    shield_label = " + shield + recovery" if args.recovery else " + shield"
    contenders = []
    for p in args.policies:
        if not args.shield_only:
            contenders.append((label_of(p), p, None))
        if shield_settings is not None:
            contenders.append((f"{label_of(p)}{shield_label}", p, shield_settings))
    if not args.no_straight:
        contenders.insert(0, ("straight", None, None))
    if not contenders:
        ap.error("nothing to fly: give at least one policy, or drop --no-straight")
    tasks = []
    for k in args.drones:
        mazes = scenarios[k]
        if not mazes:
            continue
        size = math.ceil(len(mazes) / args.workers)
        for label, path, settings_k in contenders:
            for i in range(0, len(mazes), size):
                tasks.append((label, path, k, args.dynamics, mazes[i:i + size],
                              args.decision_every, settings_k,
                              args.recovery and settings_k is not None))

    say(f"large evaluation {time.strftime('%Y-%m-%d %H:%M:%S')}, {args.dynamics} flight "
        f"model, {where}, decision every {args.decision_every / maze.CTRL_FREQ:.3f} s")
    if shield_settings is not None:
        say(f"shield: {settings.describe()} ({args.shield})"
            + (", with the recovery" if args.recovery else ""))
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
    header = (f"{'k':>2}  {'policy':<32}{'flights':>8}  {'reached':<21}{'crashed':<21}"
              f"{'timeout':>8}{'hit drone':>10}{'ratio':>7}{'clearance':>10}")
    say(header)
    say("-" * len(header))
    for k in args.drones:
        for label, _, _ in contenders:
            s = summarise(results.get((k, label), []))
            if not s["flights"]:
                say(f"{k:>2}  {label:<32} no flights")
                continue
            n = s["flights"]
            say(f"{k:>2}  {label:<32}{n:>8}  {rate(s['reached'], n):<21}"
                f"{rate(s['crashed'], n):<21}{100 * s['timeout'] / n:>7.1f}%"
                f"{s['hit_drone']:>10}{s['ratio']:>7.3f}{s['clearance']:>9.2f}m")
            if s["shielded"] and s["decisions"]:
                d = s["decisions"]
                say(f"    shield: changed {100 * s['shield_changed'] / d:.1f}% of "
                    f"{d:,} decisions, hover {100 * s['shield_hover'] / d:.1f}%, "
                    f"no safe command {s['shield_no_safe']} times")
            if args.recovery and s["shielded"]:
                say(f"    recovery: needed by {s['recovered']} drones, "
                    f"{s['recovered_arrived']} of them arrived")
        wanted = len(scenarios[k])
        built = summarise(results.get((k, contenders[0][0]), []))["mazes"]
        if redrawn.get(k):
            say(f"    ({redrawn[k]} of {wanted} mazes redrawn with a new seed and the "
                f"same number of boxes, see --redraw)")
        if built < wanted:
            say(f"    ({wanted - built} of {wanted} mazes could not be built "
                f"with {k} drones and were left out)")
        say()
    say(f"took {(time.monotonic() - started) / 60:.1f} min")

    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fields = ["policy", "k_drones", "n_obstacles", "scene_seed", "drone", "outcome",
              "decisions", "euclidean", "travelled", "min_clearance", "max_altitude",
              "shield_changed", "shield_hover", "shield_no_safe", "shield_recovery"]
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
