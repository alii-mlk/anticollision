#!/usr/bin/env python3
"""Where and why does a trained policy still hit obstacles?

Flies one drone with the policy on the mazes of rl/evaluate_large.py and, for
every flight that ends against an obstacle, looks at two things.

Where on the box the drone hit: a face, an edge or a corner, and whether the
top of the box was involved (the drone was passing over it).

What the lidar showed of that box before the crash. The lidar has 82 rays,
22.5 degrees apart around the drone and 30 degrees apart in elevation, so at
2 m two neighbouring rays are about 0.8 m apart, more than the drone's radius.
A box can be closer than any ray reports when its nearest point lies between
rays. For the moments 2 s, 1 s, 0.5 s and one decision before the crash this
prints how many rays were on the box, and how much closer the box was
than the nearest of them ("hidden": true distance to the box subtracted from
the shortest ray on it, so 0 means the lidar saw the nearest point).

If most crashes are on edges and corners with a large hidden distance, the
lidar is too coarse for the policy to see what it hits. If the box was well in
view, the policy saw it and flew too close anyway, which is a matter of the
reward.

The report also goes to rl/logs/crashes_<run>.txt, one row per crash to
rl/logs/crashes_<run>.csv.

Usage (in the VM, after  source rl/env.sh):
  python rl/diagnose_crashes.py rl/runs/swarm_1
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402
from evaluate_large import FIRST_SEED, label_of  # noqa: E402
from lidar import cast  # noqa: E402
from policy import AvoidancePolicy  # noqa: E402
from train import HELDOUT_N  # noqa: E402

LOOKBACK = (("2 s", 20), ("1 s", 10), ("0.5 s", 5), ("last decision", 1))
PLACE = {0: "inside", 1: "face", 2: "edge", 3: "corner"}
LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(text)


def view_of_box(lidar, pos, box, boxes):
    """Rays on this box that nothing else blocks, and the shortest of them."""
    alone = cast(pos, lidar.directions, box[None], lidar.max_range, ground=False)
    everything = lidar.ranges(pos, boxes)
    seen = (alone < lidar.max_range) & (alone <= everything + 1e-9)
    nearest = float(alone[seen].min()) if seen.any() else float("inf")
    true = maze.clearance_to_boxes(pos, box[None]) + maze.DRONE_RADIUS
    return int(seen.sum()), nearest, true


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="run directory, or a model .zip inside one")
    ap.add_argument("--mazes", type=int, default=600)
    ap.add_argument("--first-seed", type=int, default=FIRST_SEED)
    args = ap.parse_args()

    policy = AvoidancePolicy(args.run)
    env = maze.MazeEnv(dynamics="mujoco", lidar=policy.lidar,
                       neighbours=policy.neighbours)
    lidar = policy.lidar
    name = label_of(args.run).replace("/", "_")

    say(f"crash diagnosis {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"policy {policy.describe()}")
    crashes, outcomes = [], {}
    started = time.monotonic()
    for i in range(args.mazes):
        n, seed = HELDOUT_N[i % len(HELDOUT_N)], args.first_seed + i
        try:
            obs, _ = env.reset(options={"n": n, "seed": seed})
        except RuntimeError:
            continue
        track = []
        while True:
            track.append(env.pos.copy())
            action = policy.model.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.mazes} mazes, {(time.monotonic() - started) / 60:.1f} min",
                  flush=True)
        if info["outcome"] != "collided":
            continue

        p = env.pos
        lo, hi = env.boxes[:, [0, 2, 4]], env.boxes[:, [1, 3, 5]]
        gap = np.sqrt((np.maximum(np.maximum(lo - p, 0.0), p - hi) ** 2).sum(axis=1))
        b = int(gap.argmin())
        box = env.boxes[b]
        outside = (p < lo[b]) | (p > hi[b])
        row = {"n": n, "seed": seed, "decisions": info["decisions"],
               "place": PLACE[int(outside.sum())], "top": bool(p[2] > hi[b][2]),
               "height_above_top": float(p[2] - hi[b][2]),
               "speed": float(np.linalg.norm(env.vel))}
        for label, back in LOOKBACK:
            key = label.replace(" ", "_")
            if len(track) >= back:
                rays, nearest, true = view_of_box(lidar, track[-back], box, env.boxes)
                row[f"rays_{key}"] = rays
                row[f"true_{key}"] = true
                row[f"hidden_{key}"] = nearest - true if rays else float("inf")
            else:
                row[f"rays_{key}"] = row[f"true_{key}"] = row[f"hidden_{key}"] = ""
        crashes.append(row)

    flights = sum(outcomes.values())
    say()
    say(f"{flights} flights: {outcomes}")
    say(f"{len(crashes)} ended against an obstacle")
    if crashes:
        say()
        say("where on the box:")
        for place in ("face", "edge", "corner", "inside"):
            count = sum(c["place"] == place for c in crashes)
            if count:
                say(f"    {place:<7}{count:>4}  ({100 * count / len(crashes):.0f}%)")
        over = sum(c["top"] for c in crashes)
        say(f"    the top of the box was involved in {over} "
            f"({100 * over / len(crashes):.0f}%), the others were on its sides")
        say(f"    speed at the crash: median "
            f"{np.median([c['speed'] for c in crashes]):.2f} m/s")
        say()
        say("what the lidar showed of that box:")
        say(f"    {'before the crash':<18}{'crashes':>8}{'true distance':>15}"
            f"{'rays on box':>13}{'no ray':>8}{'hidden > 0.35 m':>17}{'median hidden':>15}")
        for label, _ in LOOKBACK:
            key = label.replace(" ", "_")
            rows = [c for c in crashes if c[f"rays_{key}"] != ""]
            if not rows:
                continue
            rays = np.array([c[f"rays_{key}"] for c in rows])
            hidden = np.array([c[f"hidden_{key}"] for c in rows])
            true = np.array([c[f"true_{key}"] for c in rows])
            finite = hidden[np.isfinite(hidden)]
            med = f"{np.median(finite):.2f} m" if len(finite) else "-"
            say(f"    {label:<18}{len(rows):>8}{np.median(true):>13.2f} m"
                f"{np.median(rays):>13.0f}{100 * (rays == 0).mean():>7.0f}%"
                f"{100 * (hidden > maze.DRONE_RADIUS).mean():>16.0f}%{med:>15}")
        say("    (true distance: from the drone's centre to the box, median. hidden > 0.35 m:")
        say("     the box was closer than the lidar showed by more than the drone's radius,")
        say("     counting the crashes with no ray on the box)")

    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if crashes:
        with open(logs / f"crashes_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(crashes[0].keys()))
            w.writeheader()
            w.writerows(crashes)
    (logs / f"crashes_{name}.txt").write_text("\n".join(LINES) + "\n")
    print(f"report written to {logs / f'crashes_{name}.txt'}")


if __name__ == "__main__":
    main()
