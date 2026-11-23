#!/usr/bin/env python3
"""Checks for rl/shield.py, the safety layer.

  1. hand-worked cases  with the default settings, situations whose answer can
                        be worked out on paper:
                        a far box, a wall at two distances, a box beside the
                        path, the ground, the edge of the area, a drone coming
                        head on, a drone that is already too close, a drone
                        boxed in. One fact makes them easy: a drone at rest
                        that holds a velocity c for h seconds and then brakes
                        travels exactly c * h in the model, so with the
                        default look-ahead of 0.5 s a full-speed command
                        (1.04 m/s) stops 0.52 m further on.
  2. the guarantee      drones flown with the model itself, in random mazes,
                        by three made-up policies: one that flies to the
                        target, one that picks random directions, and one
                        that always aims at the nearest box. With the shield
                        none of them may come closer than the margin to a
                        box, the ground or the edge, apart from the 2.5 cm the
                        sampling of the paths allows. Flown once with the
                        decision period of training and once with the period
                        varying by up to 12%, as measured in Gazebo.
  3. model and drones   how well the shield's model predicts real flights: the
                        position predicted 0.5 s and 1 s ahead from the
                        commands actually sent, against where the drone was.
                        MuJoCo flights of the policy on the Gazebo scenarios,
                        and Gazebo flights from the decisions.csv files of the
                        batches given with --gazebo-batch. This is the error
                        e that the margin has to be larger than.
  4. with the shield    the policy on the 100 Gazebo scenarios in MuJoCo (1
                        to 4 drones), without the shield, with it, and with
                        it and the recovery of rl/recovery.py: outcomes, how
                        often the shield changed the command, and its cost
                        per decision.

The report goes to rl/logs/check_shield.txt.

Usage (in the VM, after  source rl/env.sh):
  python rl/check_shield.py
  python rl/check_shield.py --gazebo-batch px4/runs3d/batch_20261119_210839_policy
  python rl/check_shield.py --settings rl/runs/shield_es/best.json
"""

import argparse
import math
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402
from shield import (AREA_LIMIT, CEILING, EARLY_LATE, Shield, ShieldSettings,  # noqa: E402
                    shield_scene)

LOG = HERE / "logs" / "check_shield.txt"
DEFAULT_POLICY = HERE.parent / "results" / "2026-11-15_swarm_sweep" / "model" / "final.zip"
GAZEBO_SCENARIOS = [(n, s) for n in (4, 8, 12, 16, 20) for s in (1, 2, 3, 4, 5)]
LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(text)


# --- the model, written out independently of shield.py --------------------

def step_model(pos, vel, command, dt, lag):
    """Exact first-order step: velocity moves towards the command."""
    e = math.exp(-dt / lag)
    new_pos = pos + command * dt + (vel - command) * lag * (1.0 - e)
    new_vel = command + (vel - command) * e
    return new_pos, new_vel


def clearance(p, boxes, radius=maze.DRONE_RADIUS):
    """Gap to the nearest box, the ground, the ceiling or the edge of the area."""
    c = min(p[2] - radius, CEILING - p[2], AREA_LIMIT - abs(p[0]), AREA_LIMIT - abs(p[1]))
    if len(boxes):
        c = min(c, maze.clearance_to_boxes(p, boxes, radius))
    return c


def closest_to_drones(shield, pos, vel, command_velocity, others_pos, others_vel):
    """Smallest predicted distance between centres along the checked paths."""
    P = shield.paths(pos, vel, np.asarray(command_velocity)[None])[0]
    Q = np.asarray(others_pos)[None] + shield.t[:, None, None] * np.asarray(others_vel)[None]
    return float(np.linalg.norm(P[:, None, :] - Q, axis=-1).min())


# --- 1 hand-worked cases ---------------------------------------------------

def wall_ahead(face_x):
    return np.array([[face_x, face_x + 1.0, -5.0, 5.0, 0.0, 10.0]])


def hand_cases(settings):
    shield = Shield(settings)
    m = settings.margin
    r = maze.DRONE_RADIUS
    still = np.zeros(3)
    at = np.array([0.0, 0.0, 3.0])
    ahead = np.array([1.0, 0.0, 0.0])
    stop = settings.lookahead * maze.FLOWN_SPEED_XY      # 0.52 m with the defaults
    failed = []

    def check(title, ok, detail):
        say(f"    {'ok  ' if ok else 'FAIL'} {title}: {detail}")
        if not ok:
            failed.append(title)

    def fmt(a):
        return "(" + ", ".join(f"{x:+.2f}" for x in a) + ")"

    a, what = shield.filter(ahead, at, still, np.array([[5.0, 6.0, -1.0, 1.0, 0.0, 10.0]]))
    check("nothing near", what == "unchanged", f"{what} {fmt(a)}")

    face = stop + r + m + 0.05
    a, what = shield.filter(ahead, at, still, wall_ahead(face))
    check(f"wall {face:.2f} m ahead, stops {stop:.2f} m on, {face - stop - r:.2f} m left",
          what == "unchanged", f"{what} {fmt(a)}")

    face = stop + r + m - 0.07
    a, what = shield.filter(ahead, at, still, wall_ahead(face))
    slower = 0.75 * stop
    check(f"wall {face:.2f} m ahead, full speed would leave {face - stop - r:.2f} m; "
          f"at 0.75 it stops {slower:.2f} m on and leaves {face - slower - r:.2f} m",
          what == "changed" and np.allclose(a, [0.75, 0.0, 0.0]), f"{what} {fmt(a)}")

    moving = np.array([maze.FLOWN_SPEED_XY, 0.0, 0.0])
    beside = np.array([[-5.0, 5.0, r + m + 0.1, 2.0, 0.0, 10.0]])
    a, what = shield.filter(ahead, at, moving, beside)
    check("box beside the path, 0.40 m from the drone", what == "unchanged", f"{what} {fmt(a)}")

    low = np.array([0.0, 0.0, 1.2])
    a, what = shield.filter([0.0, 0.0, -1.0], low, still, np.zeros((0, 6)))
    drop = settings.lookahead * maze.FLOWN_SPEED_Z
    check(f"straight down from 1.2 m: full speed ends {1.2 - drop - r:.3f} m above the "
          f"ground, at 0.75 it ends {1.2 - 0.75 * drop - r:.3f} m above",
          what == "changed" and np.allclose(a, [0.0, 0.0, -0.75]), f"{what} {fmt(a)}")

    for x, expect in ((14.0, "unchanged"), (14.3, "changed")):
        a, what = shield.filter(ahead, [x, 0.0, 3.0], still, np.zeros((0, 6)))
        ok = what == expect and (expect == "unchanged" or np.allclose(a, [0.75, 0.0, 0.0]))
        check(f"edge of the area at {AREA_LIMIT:.0f} m, from x = {x}: stops at "
              f"{x + stop:.2f}", ok, f"{what} {fmt(a)}")

    other, other_vel = np.array([[4.0, 0.0, 3.0]]), -moving[None]
    a, what = shield.filter(ahead, at, moving, np.zeros((0, 6)), other, other_vel)
    before = closest_to_drones(shield, at, moving, ahead * shield.scale, other, other_vel)
    after = closest_to_drones(shield, at, moving, a * shield.scale, other, other_vel)
    check("drone coming head on from 4 m: the command changes and the predicted pass "
          "is wider", what != "unchanged" and after > before,
          f"{what} {fmt(a)}, closest {before:.2f} m -> {after:.2f} m")

    a, what = shield.filter(ahead, at, moving, np.zeros((0, 6)),
                            np.array([[0.0, 5.0, 3.0]]), moving[None])
    check("drone flying alongside 5 m away", what == "unchanged", f"{what} {fmt(a)}")

    face = r + 0.10
    a, what = shield.filter(ahead, at, still, wall_ahead(face))
    P = shield.paths(at, still, (a * shield.scale)[None])[0]
    worst = min(clearance(p, wall_ahead(face)) for p in P)
    check("already 0.10 m from a wall: nothing is safe, so it must not get closer, "
          "and hovering is the closest such command",
          what == "no_safe" and not a.any() and worst >= 0.10 - 1e-9,
          f"{what} {fmt(a)}, closest {worst:.3f} m")

    g = r + m + 0.05
    cage = np.array([
        [g, g + 1.0, -2.0, 2.0, 0.0, 6.0], [-g - 1.0, -g, -2.0, 2.0, 0.0, 6.0],
        [-2.0, 2.0, g, g + 1.0, 0.0, 6.0], [-2.0, 2.0, -g - 1.0, -g, 0.0, 6.0],
        [-2.0, 2.0, -2.0, 2.0, 0.0, 3.0 - g], [-2.0, 2.0, -2.0, 2.0, 3.0 + g, 6.0]])
    a, what = shield.filter(ahead, at, still, cage)
    check("boxed in with 0.05 m to spare on every side: only hovering is safe",
          what == "hover" and not a.any(), f"{what} {fmt(a)}")

    face = maze.FLOWN_SPEED_XY * settings.lag + r + m + 0.01
    a, what = shield.filter(ahead, at, moving, wall_ahead(face))
    P = shield.paths(at, moving, (a * shield.scale)[None])[0]
    worst = min(clearance(p, wall_ahead(face)) for p in P)
    check(f"flying at a wall {face:.2f} m ahead, just far enough to brake: a safe "
          f"command is found", what in ("changed", "hover") and worst >= m - 1e-6,
          f"{what} {fmt(a)}, closest {worst:.3f} m")

    if failed:
        raise AssertionError(f"{len(failed)} case(s) failed")


# --- 2 the guarantee, in the model ------------------------------------------

def random_scene(rng, n):
    for _ in range(50):
        try:
            drones = maze.scenario_gen.generate_drones(1, rng)
            obstacles = maze.scenario_gen.generate_obstacles(n, drones, rng)
        except SystemExit:
            continue
        boxes = np.array([[*o["rect"], 0.0, o["height"]] for o in obstacles],
                         dtype=np.float64).reshape(-1, 6)
        return drones[0], boxes
    raise RuntimeError("could not draw a scene")


def nearest_box_direction(p, boxes):
    lo, hi = boxes[:, [0, 2, 4]], boxes[:, [1, 3, 5]]
    closest = np.clip(p, lo, hi)
    d = np.linalg.norm(closest - p, axis=1)
    j = int(np.argmin(d))
    return (closest[j] - p) / max(float(d[j]), 1e-9)


def model_guarantee(settings, flights, jitter):
    shield = Shield(settings)
    dt, lag, m = shield.decision_dt, settings.lag, settings.margin
    rng = random.Random(7 if jitter else 3)
    nrng = np.random.default_rng(11 if jitter else 5)
    worst = math.inf
    reached = 0
    for f in range(flights):
        drone, boxes = random_scene(rng, (4, 8, 12, 16, 20)[f % 5])
        p = np.array([drone["start"][0], drone["start"][1], maze.TAKEOFF_ALTITUDE])
        v = np.zeros(3)
        target = np.array(drone["target"], dtype=np.float64)
        mode = f % 3
        wander = None
        for step in range(300):
            if mode == 0:
                a = (target - p) / max(float(np.linalg.norm(target - p)), 1e-9)
            elif mode == 1:
                if wander is None or step % 10 == 0:
                    wander = nrng.normal(size=3)
                    wander /= np.linalg.norm(wander)
                a = wander
            else:
                a = nearest_box_direction(p, boxes) if len(boxes) else np.array([1.0, 0.0, 0.0])
            sent, _ = shield.filter(a, p, v, boxes)
            c = sent * shield.scale
            period = dt * (1.0 + nrng.uniform(-0.12, 0.12)) if jitter else dt
            for s in np.linspace(0.0, period, 21)[1:]:
                q, _ = step_model(p, v, c, s, lag)
                worst = min(worst, clearance(q, boxes))
            p, v = step_model(p, v, c, period, lag)
            if mode == 0 and np.linalg.norm(target - p) <= maze.TOLERANCE:
                reached += 1
                break
    c = shield.counts
    total = sum(c.values())
    to_target = sum(1 for f in range(flights) if f % 3 == 0)
    say(f"    {flights} flights ({to_target} to the target, of which {reached} arrived; "
        f"the rest random or aiming at the nearest box), {total:,} decisions")
    say(f"    commands changed {c['changed']}, hover {c['hover']}, no safe command {c['no_safe']}")
    say(f"    closest any drone came to a box, the ground or the edge: {worst:.3f} m "
        f"(margin {m:.2f} m)")
    if worst < m - 0.05:
        raise AssertionError(f"a drone came within {worst:.3f} m, below the margin by more "
                             f"than the sampling allows")


# --- 3 the model against flights --------------------------------------------

def record_mujoco(policy, dynamics, ks=(1, 3)):
    """(t, position, velocity, commanded velocity) at every decision of every drone."""
    from swarm import Scene, fly_scenarios
    dt = maze.DECISION_EVERY / maze.CTRL_FREQ
    tracks = defaultdict(list)
    for k in ks:
        scene = Scene(k, dynamics=dynamics, respawn=False, lidar=policy.lidar,
                      neighbours=policy.neighbours)

        def act(sc, obs, k=k):
            actions = policy.model.predict(obs, deterministic=True)[0]
            for row, i in enumerate(np.flatnonzero(sc.present)):
                key = (k, sc.n_obstacles, sc.scene_seed, int(i))
                command = policy.velocity(maze.clip_action(actions[row]))
                tracks[key].append((len(tracks[key]) * dt, sc.pos[i].copy(),
                                    sc.vel[i].copy(), command))
            return actions
        fly_scenarios(scene, act, GAZEBO_SCENARIOS)
    return list(tracks.values())


def prediction_errors(tracks, lag, horizon):
    """Position predicted `horizon` seconds ahead from the commands sent, against the track."""
    errors = []
    for rows in tracks:
        t = [r[0] for r in rows]
        for i in range(len(rows)):
            j = i
            while j + 1 < len(rows) and t[j + 1] - t[i] < horizon - 1e-6:
                j += 1
            if j + 1 >= len(rows):
                break
            p, v = rows[i][1].copy(), rows[i][2].copy()
            for n in range(i, j + 1):
                p, v = step_model(p, v, rows[n][3], t[n + 1] - t[n], lag)
            errors.append(float(np.linalg.norm(p - rows[j + 1][1])))
    return np.array(errors)


def model_against_flights(settings, policy_path, dynamics, gazebo_batches):
    from compare_simulators import read_gazebo_decisions
    from policy import AvoidancePolicy
    sources = []
    if policy_path is not None and Path(policy_path).exists():
        policy = AvoidancePolicy(policy_path)
        sources.append((f"{dynamics}, 1 and 3 drones", record_mujoco(policy, dynamics)))
    for batch in gazebo_batches:
        tracks = [rows for rows in read_gazebo_decisions(batch).values() if len(rows) > 2]
        sources.append((f"Gazebo, {Path(batch).name}", tracks))
    if not sources:
        say("    nothing to compare: no policy and no Gazebo batch given")
        return
    lags = sorted({0.2, 0.3, 0.4, 0.5, 0.6, round(settings.lag, 3)})
    for name, tracks in sources:
        say(f"    {name}: {len(tracks)} drones, {sum(len(t) for t in tracks):,} decisions")
        say(f"      {'lag':>5}   after 0.5 s: median   95%    max   after 1 s: median   95%    max")
        for lag in lags:
            cells = []
            for horizon in (0.5, 1.0):
                e = prediction_errors(tracks, lag, horizon)
                cells.append(f"{np.median(e):10.3f} {np.percentile(e, 95):6.3f} {e.max():6.3f}")
            mark = "  <- settings" if abs(lag - settings.lag) < 1e-9 else ""
            say(f"      {lag:5.2f}   {cells[0]}         {cells[1]}{mark}")
    say(f"    (metres; the guarantee needs the error to stay below the margin, "
        f"{settings.margin:.2f} m)")


# --- 4 the policy with the shield, in MuJoCo --------------------------------

def mujoco_with_shield(settings, policy_path, dynamics, ks=(1, 2, 3, 4)):
    from policy import AvoidancePolicy
    from swarm import Scene, fly_scenarios
    policy = AvoidancePolicy(policy_path)
    shield = Shield(settings, decision_dt=maze.DECISION_EVERY / maze.CTRL_FREQ,
                    speed_xy=policy.speed_xy, speed_z=policy.speed_z)
    say(f"    policy {policy.describe()}")
    say(f"    shield {shield.describe()}")

    def plain(sc, obs):
        return policy.model.predict(obs, deterministic=True)[0]

    say(f"    {'k':>2}  {'':<16}{'drones':>7}{'reached':>8}{'box':>5}{'drone':>6}{'other':>6}"
        f"{'timeout':>8}  {'changed':>8}{'hover':>7}{'no safe':>8}  {'time (s)':>8}  {'ms/check':>8}")
    for k in ks:
        for arm in ("policy", "policy + shield", "+ recovery"):
            scene = Scene(k, dynamics=dynamics, respawn=False, lidar=policy.lidar,
                          neighbours=policy.neighbours)
            counts = defaultdict(Counter)
            recoveries = {} if arm == "+ recovery" else None
            spent = [0.0, 0]
            if arm == "policy":
                act = plain
            else:
                def act(sc, obs):
                    actions = plain(sc, obs)
                    start = time.perf_counter()
                    out = shield_scene(shield, sc, actions, counts, recoveries)
                    spent[0] += time.perf_counter() - start
                    spent[1] += len(out)
                    return out
            infos = fly_scenarios(scene, act, GAZEBO_SCENARIOS)
            o = Counter(i["outcome"] for i in infos)
            other = sum(v for key, v in o.items()
                        if key not in ("reached", "collided", "hit_drone", "timeout"))
            what = Counter()
            for c in counts.values():
                what.update(c)
            decisions = sum(what[w] for w in ("unchanged", "changed", "hover", "no_safe"))
            arrived = [i for i in infos if i["outcome"] == "reached"]
            mean_time = (np.mean([i["decisions"] for i in arrived]) * maze.DECISION_EVERY
                         / maze.CTRL_FREQ) if arrived else float("nan")
            share = (lambda x: f"{100 * x / decisions:7.1f}%") if decisions else (lambda x: "      -")
            cost = f"{1000 * spent[0] / spent[1]:8.2f}" if spent[1] else "       -"
            say(f"    {k:>2}  {arm:<16}{len(infos):>7}{o['reached']:>8}{o['collided']:>5}"
                f"{o['hit_drone']:>6}{other:>6}{o['timeout']:>8}  {share(what['changed'])}"
                f"{share(what['hover'])}{what['no_safe']:>8}  {mean_time:8.1f}  {cost}")
    say("    (box, drone, other: crashes into a box, into another drone, and into the "
        "ground or out of the area; time: mean flight time of the drones that arrived)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--settings", default="default",
                    help="shield settings, a JSON file, or 'default'")
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY,
                    help="policy for checks 3 and 4")
    ap.add_argument("--gazebo-batch", type=Path, nargs="*", default=[],
                    help="Gazebo batches with decisions.csv files, for check 3")
    ap.add_argument("--dynamics", choices=["mujoco", "point"], default="mujoco")
    ap.add_argument("--model-flights", type=int, default=60,
                    help="flights per run of check 2")
    ap.add_argument("--no-flights", action="store_true",
                    help="only checks 1 and 2, which need no policy and no MuJoCo")
    args = ap.parse_args()

    settings = ShieldSettings.load(args.settings)
    say(f"shield checks {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"settings: {settings.describe()} ({args.settings})")
    steps = [
        # The answers of the hand-worked cases were worked out for the
        # default settings, so this check always uses them.
        ("1 hand-worked cases (default settings)", lambda: hand_cases(ShieldSettings())),
        ("2 the guarantee in the model, decision every 0.104 s",
         lambda: model_guarantee(settings, args.model_flights, jitter=False)),
        (f"2 the guarantee in the model, decision period varying by up to 12% "
         f"(checked for {100 * EARLY_LATE:.0f}%)",
         lambda: model_guarantee(settings, args.model_flights, jitter=True)),
    ]
    if not args.no_flights:
        steps.append(("3 the model against MuJoCo and Gazebo flights",
                      lambda: model_against_flights(settings, args.policy, args.dynamics,
                                                    args.gazebo_batch)))
        if args.policy.exists():
            steps.append(("4 the policy without and with the shield, Gazebo scenarios in "
                          f"{args.dynamics}",
                          lambda: mujoco_with_shield(settings, args.policy, args.dynamics)))
        else:
            say(f"(check 4 skipped: no policy at {args.policy})")

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
