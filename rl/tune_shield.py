#!/usr/bin/env python3
"""Choose the safety layer's settings with an evolution strategy (stage 5).

The April idea of the thesis was an avoidance algorithm whose parameters are
chosen by black-box optimisation, with OpenAI's evolution strategy (Salimans
et al. 2017, "Evolution Strategies as a Scalable Alternative to Reinforcement
Learning", the paper the professor sent), a penalty for collisions and the
time kept short. Here the parameters are the four numbers of the safety layer
(rl/shield.py), and the method is the paper's Algorithm 1 with the two
refinements it uses itself: mirrored sampling and rank-based fitness shaping.

What is optimised

  theta    the four settings, each scaled to [0, 1] over the range it may
           take (RANGES below): 0 is the low end, 1 the high end
  F(theta) the final policy flies a fixed set of MuJoCo mazes with the layer
           set to theta. Every flight scores 1 if the drone arrives, -10 if
           it crashes, and -0.01 for every second it flew, so a drone that
           times out after 62.5 s costs about 1.6 against one that arrives.
           F is the mean over the flights.

The mazes have seeds from 400000, used nowhere else (the policy was chosen on
200000 and the final results are on 300000). Every setting flies the same
mazes, and the flights are deterministic, so two settings differ only because
of the settings.

On ordinary mazes there is little to learn: the first try, 24 mazes each for
2, 4 and 8 drones (336 flights), had no crash and one or two timeouts for every
setting, and the scores of the 12 tried settings differed by less than 0.01.
So the tuning mazes are chosen with --mazes-from: evaluate_large.py first
flies the policy with and without the layer on 120 mazes of the tuning block,
and the tuning set takes the mazes where something went wrong (a crash without
the layer, or a drone that did not arrive with it), plus a few where nothing
did, so that a setting which breaks ordinary flights still loses points.
These hard mazes come from the tuning block too, never from the final test
mazes or the Gazebo scenarios, which would make the final results look better
than they are.

One iteration

  1. draw n directions eps_1 ... eps_n from a normal distribution
  2. fly theta + sigma eps_i and theta - sigma eps_i for every i (2n
     settings, the mirrored pairs), and theta itself, which is only used to
     follow the progress
  3. rank the 2n scores and turn the ranks into weights from -0.5 (worst) to
     +0.5 (best), so a lucky outlier cannot dominate the step
  4. theta <- theta + alpha / (2 n sigma) * sum_j weight_j * eps_j
     (eps_j is -eps_i for the mirrored half), then clip to [0, 1]

Output in rl/runs/<name>/:

  progress.csv    one row per iteration: theta's settings, its score and what
                  happened in its flights (the curve for the thesis)
  candidates.csv  every setting flown, with its score
  best.json       the settings whose own score was the best so far
  final.json      the settings after the last iteration
  state.json      where the run stands, for --resume

and a report in rl/logs/tune_<name>.txt, rewritten after every iteration.

Usage (in the VM, after  source rl/env.sh):
  python rl/evaluate_large.py results/2026-11-15_swarm_sweep/model/final.zip \\
      --shield default --no-straight --drones 2 4 8 --mazes 120 \\
      --first-seed 400000 --name mine_400000
  python rl/tune_shield.py --mazes-from rl/logs/large_mine_400000.csv --quick
  python rl/tune_shield.py --mazes-from rl/logs/large_mine_400000.csv
  python rl/tune_shield.py --mazes-from rl/logs/large_mine_400000.csv --resume
"""

import argparse
import csv
import json
import math
import multiprocessing
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402
from shield import ShieldSettings  # noqa: E402
from train import HELDOUT_N  # noqa: E402

DEFAULT_POLICY = HERE.parent / "results" / "2026-11-15_swarm_sweep" / "model" / "final.zip"
RANGES = {
    "margin": (0.1, 0.8),          # m kept from boxes, the ground and the edges
    "drone_margin": (0.1, 1.0),    # m kept from other drones
    "lookahead": (0.1, 1.5),       # s; the layer never uses less than one decision
    "lag": (0.2, 1.0),             # s
}
NAMES = list(RANGES)
CRASHES = ("collided", "hit_drone", "ground", "out_of_bounds")
LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(text)


# --- settings and theta ------------------------------------------------------

def to_settings(theta):
    """theta in [0, 1]^4 (clipped) to ShieldSettings."""
    theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
    return ShieldSettings(**{name: lo + (hi - lo) * float(x)
                             for (name, (lo, hi)), x in zip(RANGES.items(), theta)})


def to_theta(settings):
    return np.array([(getattr(settings, name) - lo) / (hi - lo)
                     for name, (lo, hi) in RANGES.items()], dtype=np.float64)


def centered_ranks(scores):
    """Ranks scaled to [-0.5, 0.5], the best +0.5; equal scores share a rank."""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    for value in np.unique(scores):
        same = np.flatnonzero(scores == value)
        if len(same) > 1:
            ranks[same] = ranks[same].mean()
    return ranks / max(len(scores) - 1, 1) - 0.5


# --- flying ---------------------------------------------------------------

def fly_task(task):
    """One process: one setting, one number of drones, its mazes."""
    tag, settings, k, scenarios, policy_path, dynamics, recovery = task
    from policy import AvoidancePolicy
    from shield import Shield, shield_scene
    from swarm import Scene, fly_scenarios
    policy = AvoidancePolicy(policy_path)
    shield = Shield(ShieldSettings(**settings), decision_dt=maze.DECISION_EVERY / maze.CTRL_FREQ,
                    speed_xy=policy.speed_xy, speed_z=policy.speed_z)
    scene = Scene(k, dynamics=dynamics, respawn=False, lidar=policy.lidar,
                  neighbours=policy.neighbours)
    counts = defaultdict(Counter)
    recoveries = {} if recovery else None

    def act(sc, obs):
        return shield_scene(shield, sc, policy.model.predict(obs, deterministic=True)[0],
                            counts, recoveries)

    infos = fly_scenarios(scene, act, scenarios)
    for info in infos:
        c = counts.get((info["n_obstacles"], info["scene_seed"], info["drone"]), Counter())
        info.update({f"shield_{what}": c[what]
                     for what in ("changed", "hover", "no_safe", "recovery")})
    return tag, infos


def score(infos, crash_penalty, time_cost):
    """F and what happened in the flights of one setting."""
    dt = maze.DECISION_EVERY / maze.CTRL_FREQ
    n = len(infos)
    if not n:
        # Same keys as below, so every row of the CSV files has the same columns.
        return {"score": -math.inf, "flights": 0, "reached": 0, "crashed": 0, "hit_drone": 0,
                "timeout": 0, "time_arrived": math.nan, "changed": math.nan,
                "hover": math.nan, "no_safe": 0, "recovered": 0, "closest": math.nan}
    outcomes = Counter(i["outcome"] for i in infos)
    crashed = sum(outcomes[c] for c in CRASHES)
    seconds = [i["decisions"] * dt for i in infos]
    total = sum((1.0 if i["outcome"] == "reached" else 0.0)
                - (crash_penalty if i["outcome"] in CRASHES else 0.0)
                - time_cost * t for i, t in zip(infos, seconds))
    arrived = [t for i, t in zip(infos, seconds) if i["outcome"] == "reached"]
    decisions = sum(i["decisions"] for i in infos)
    return {
        "score": total / n,
        "flights": n,
        "reached": outcomes["reached"],
        "crashed": crashed,
        "hit_drone": outcomes["hit_drone"],
        "timeout": outcomes["timeout"],
        "time_arrived": float(np.mean(arrived)) if arrived else float("nan"),
        "changed": sum(i["shield_changed"] for i in infos) / max(decisions, 1),
        "hover": sum(i["shield_hover"] for i in infos) / max(decisions, 1),
        "no_safe": sum(i["shield_no_safe"] for i in infos),
        "recovered": sum(1 for i in infos if i["shield_recovery"] > 0),
        "closest": min(i["min_clearance"] for i in infos),
    }


def choose_mazes(args):
    """The tuning mazes, {drones: [(obstacles, seed), ...]}, and lines describing them.

    Without --mazes-from: the first --mazes seeds of the block, the same for
    every number of drones. With it: the hard mazes of a run of
    evaluate_large.py on the tuning block (made with --shield, so it has both
    arms), those where the policy alone crashed a drone or a drone did not
    arrive with the layer, up to --hard-per-k of them, plus --easy-per-k mazes
    where nothing went wrong, so that a setting which breaks the ordinary
    cases still loses points.
    """
    if args.mazes_from is None:
        mazes = [(HELDOUT_N[i % len(HELDOUT_N)], args.first_seed + i) for i in range(args.mazes)]
        return ({k: mazes for k in args.drones},
                [f"{args.mazes} mazes per number of drones, seeds from {args.first_seed}"])
    flights = defaultdict(list)
    with open(args.mazes_from) as f:
        for r in csv.DictReader(f):
            key = (int(r["k_drones"]), int(r["n_obstacles"]), int(r["scene_seed"]))
            flights[key].append((r["policy"], r["outcome"]))

    def went_wrong(key):
        for label, outcome in flights[key]:
            if label.endswith("+ shield"):
                if outcome != "reached":
                    return True
            elif outcome in CRASHES:
                return True
        return False

    rng = random.Random(args.seed)
    chosen, lines = {}, [f"tuning mazes from {args.mazes_from}"]
    for k in args.drones:
        pool = sorted(key for key in flights if key[0] == k)
        hard = [key for key in pool if went_wrong(key)]
        easy = [key for key in pool if not went_wrong(key)]
        picked_hard = rng.sample(hard, min(args.hard_per_k, len(hard)))
        picked_easy = rng.sample(easy, min(args.easy_per_k, len(easy)))
        chosen[k] = sorted((n, seed) for _, n, seed in picked_hard + picked_easy)
        lines.append(f"  {k} drones: {len(picked_hard)} hard mazes (of {len(hard)} found) and "
                     f"{len(picked_easy)} without trouble, from {len(pool)} mazes")
    return chosen, lines


def fly_all(pool, candidates, mazes_per_k, args):
    """{tag: score dict} for every (tag, theta) in candidates."""
    tasks = [(tag, asdict(to_settings(theta)), k, mazes_per_k[k], str(args.policy), args.dynamics,
              args.recovery)
             for tag, theta in candidates for k in args.drones if mazes_per_k[k]]
    # The longest tasks (most drones) first, so the workers finish together.
    tasks.sort(key=lambda t: -t[2])
    infos = defaultdict(list)
    for tag, result in pool.imap_unordered(fly_task, tasks):
        infos[tag].extend(result)
    return {tag: score(infos[tag], args.crash_penalty, args.time_cost) for tag, _ in candidates}


# --- reporting ----------------------------------------------------------------

def describe(s):
    return (f"margin {s.margin:.3f} m, drone margin {s.drone_margin:.3f} m, "
            f"look-ahead {s.lookahead:.3f} s, lag {s.lag:.3f} s")


def outcome_line(r):
    return (f"score {r['score']:+.4f}: arrived {r['reached']}/{r['flights']}, crashed "
            f"{r['crashed']}, timeout {r['timeout']}, time {r['time_arrived']:.2f} s, "
            f"changed {100 * r['changed']:.2f}%, recovered {r['recovered']}, "
            f"closest {r['closest']:.3f} m")


def append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--name", default=None,
                    help="default: shield_es, or shield_es_hard with --mazes-from")
    ap.add_argument("--start", default="default",
                    help="settings to start from: 'default' or a JSON file")
    ap.add_argument("--iterations", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=6, help="mirrored pairs per iteration")
    ap.add_argument("--sigma", type=float, default=0.1, help="perturbation size, in [0, 1] units")
    ap.add_argument("--alpha", type=float, default=0.02, help="step size")
    ap.add_argument("--drones", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--mazes", type=int, default=24, help="mazes per number of drones")
    ap.add_argument("--first-seed", type=int, default=400_000)
    ap.add_argument("--mazes-from", type=Path, default=None,
                    help="per-flight CSV of evaluate_large.py --shield on the tuning block: "
                         "tune on its hard mazes plus a few normal ones")
    ap.add_argument("--hard-per-k", type=int, default=12,
                    help="with --mazes-from: hard mazes per number of drones, at most")
    ap.add_argument("--easy-per-k", type=int, default=6,
                    help="with --mazes-from: mazes without trouble per number of drones")
    ap.add_argument("--recovery", action="store_true",
                    help="tune the layer together with the recovery (rl/recovery.py)")
    ap.add_argument("--crash-penalty", type=float, default=10.0)
    ap.add_argument("--time-cost", type=float, default=0.01, help="per second flown")
    ap.add_argument("--seed", type=int, default=0, help="for the perturbations")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dynamics", choices=["mujoco", "point"], default="mujoco")
    ap.add_argument("--resume", action="store_true", help="continue rl/runs/<name>")
    ap.add_argument("--quick", action="store_true",
                    help="one small iteration (2 pairs, 3 mazes, 2 and 4 drones), to check it runs")
    args = ap.parse_args()
    if args.name is None:
        args.name = "shield_es_hard" if args.mazes_from else "shield_es"
    if args.quick:
        args.iterations, args.pairs, args.mazes, args.drones = 1, 2, 3, [2, 4]
        args.hard_per_k, args.easy_per_k = 2, 1
        args.name = args.name + "_quick"
    mazes_per_k, maze_lines = choose_mazes(args)

    run_dir = HERE / "runs" / args.name
    state_file = run_dir / "state.json"
    log_file = HERE / "logs" / f"tune_{args.name}.txt"
    if args.resume:
        if not state_file.exists():
            sys.exit(f"nothing to resume: {state_file} does not exist")
        state = json.loads(state_file.read_text())
        theta = np.array(state["theta"])
        first_iteration = state["iteration"]
        best = state["best"]
        LINES.extend(state.get("lines", []))
        say(f"resumed at iteration {first_iteration}")
    else:
        if args.quick and run_dir.exists():
            # Only the test run's own folder, so --quick can be repeated.
            shutil.rmtree(run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            sys.exit(f"{run_dir} already has a run; use --resume or another --name")
        run_dir.mkdir(parents=True, exist_ok=True)
        theta = to_theta(ShieldSettings.load(args.start))
        first_iteration = 0
        best = None
        say(f"ES tuning of the safety layer, {time.strftime('%Y-%m-%d %H:%M:%S')}")
        say(f"policy {args.policy}")
        say("ranges: " + ", ".join(f"{n} {lo}-{hi}" for n, (lo, hi) in RANGES.items()))
        say(f"start: {describe(to_settings(theta))} ({args.start})")
        say(f"{args.iterations} iterations of {args.pairs} mirrored pairs, sigma {args.sigma}, "
            f"alpha {args.alpha}; score per flight: 1 if it arrives, -{args.crash_penalty:g} "
            f"if it crashes, -{args.time_cost:g} per second"
            + ("; with the recovery" if args.recovery else ""))
        for line in maze_lines:
            say(line)
        say(f"{sum(k * len(m) for k, m in mazes_per_k.items())} flights per setting, "
            f"drones {args.drones}")
        (run_dir / "settings.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
        (run_dir / "mazes.json").write_text(json.dumps(
            {str(k): m for k, m in mazes_per_k.items()}, indent=1) + "\n")

    ctx = multiprocessing.get_context("fork")
    started = time.monotonic()
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for it in range(first_iteration, args.iterations):
            t0 = time.monotonic()
            rng = np.random.default_rng([args.seed, it])
            eps = rng.standard_normal((args.pairs, len(NAMES)))
            candidates = [("theta", theta.copy())]
            candidates += [(f"+{i}", theta + args.sigma * eps[i]) for i in range(args.pairs)]
            candidates += [(f"-{i}", theta - args.sigma * eps[i]) for i in range(args.pairs)]
            results = fly_all(pool, candidates, mazes_per_k, args)

            # The step, from the mirrored pairs only.
            scores = [results[f"+{i}"]["score"] for i in range(args.pairs)] + \
                     [results[f"-{i}"]["score"] for i in range(args.pairs)]
            weights = centered_ranks(scores)
            directions = np.vstack([eps, -eps])
            step = args.alpha / (2 * args.pairs * args.sigma) * (weights @ directions)

            here = results["theta"]
            settings_here = to_settings(theta)
            if best is None or here["score"] > best["score"]:
                best = {"score": here["score"], "iteration": it, "settings": asdict(settings_here)}
                settings_here.save(run_dir / "best.json")
            append_csv(run_dir / "progress.csv", {
                "iteration": it, **{n: f"{v:.4f}" for n, v in asdict(settings_here).items()},
                **{key: (f"{v:.5f}" if isinstance(v, float) else v) for key, v in here.items()},
                "best_perturbed": f"{max(scores):.5f}", "worst_perturbed": f"{min(scores):.5f}",
                "step_size": f"{float(np.linalg.norm(step)):.5f}",
                "minutes": f"{(time.monotonic() - t0) / 60:.2f}"})
            for tag, th in candidates:
                r = results[tag]
                append_csv(run_dir / "candidates.csv", {
                    "iteration": it, "tag": tag,
                    **{n: f"{v:.4f}" for n, v in asdict(to_settings(th)).items()},
                    **{key: (f"{v:.5f}" if isinstance(v, float) else v) for key, v in r.items()}})

            say()
            say(f"iteration {it}: {describe(settings_here)}")
            say(f"    {outcome_line(here)}")
            say(f"    perturbed settings: best {max(scores):+.4f}, worst {min(scores):+.4f}; "
                f"step {float(np.linalg.norm(step)):.4f}; {(time.monotonic() - t0) / 60:.1f} min")

            theta = np.clip(theta + step, 0.0, 1.0)
            to_settings(theta).save(run_dir / "final.json")
            state_file.write_text(json.dumps({"iteration": it + 1, "theta": theta.tolist(),
                                              "best": best, "lines": LINES}, indent=1))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("\n".join(LINES) + "\n")
            done = it + 1 - first_iteration
            left = args.iterations - it - 1
            if left:
                print(f"    about {left * (time.monotonic() - started) / done / 60:.0f} min left",
                      flush=True)

    say()
    say(f"final: {describe(to_settings(theta))}")
    if best is not None:
        say(f"best (iteration {best['iteration']}, score {best['score']:+.4f}): "
            f"{describe(ShieldSettings(**best['settings']))}")
    say(f"took {(time.monotonic() - started) / 60:.1f} min; files in {run_dir}")
    log_file.write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
