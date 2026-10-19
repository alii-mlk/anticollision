#!/usr/bin/env python3
"""Compare a sweep flown straight with the same sweep flown by the policy.

Both batches must come from run_batch_3d.py with the same n, k and seeds, so
every run, and every drone inside it, has a counterpart flying the identical
scenario. The report has three parts:

  per configuration   reached, collisions and ratio+ side by side, with the
                      mission time and the controller's CPU time
  per drone           which drones the policy saved (crashed straight, reached
                      with the policy) and which it lost (the other way round)
  separation          clearance to obstacles of the drones that arrived, the
                      closest approach between drones, and for every crash
                      under the policy how far the nearest other drone was

Usage:
  python3 compare_sweeps_3d.py runs3d/batch_A_straight runs3d/batch_B_policy
  python3 compare_sweeps_3d.py A B --out ../results/2026-10-19_rl_sweep
"""

import argparse
import bisect
import csv
import math
import re
import statistics
from pathlib import Path

from compute_metrics_3d import collect, mean

DRONE_LINE = re.compile(
    r"\{id: (\d+), outcome: (\w+), euclidean: ([\d.]+), travelled: ([\d.]+), "
    r"time: ([\d.]+), min_clearance: ([-\d.]+), collided_with: ([^}]+)\}")

# Below this height a drone is on the ground (before takeoff or after it was
# destroyed), so it is left out of the drone-to-drone distances.
AIRBORNE_Z = 1.0
# Seconds before a crash in which to look for another drone nearby.
CRASH_WINDOW = 5.0


def drones(batch):
    """{(run, drone id): outcome, distances, time, clearance} for a batch."""
    out = {}
    for run_dir in sorted(batch.iterdir()):
        summary = run_dir / "run" / "summary.yaml"
        if not summary.exists():
            continue
        for m in DRONE_LINE.finditer(summary.read_text()):
            i, outcome, eu, tr, t, clearance, hit = m.groups()
            out[(run_dir.name, int(i))] = {
                "outcome": outcome, "euclidean": float(eu),
                "travelled": float(tr), "time": float(t),
                "clearance": float(clearance), "hit": hit.strip()}
    return out


def trajectory(batch, run, i):
    path = batch / run / "run" / f"drone{i}" / "trajectory.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return [tuple(map(float, row)) for row in list(csv.reader(f))[1:]]


def closest_pair(tracks, pairs, t_from=-math.inf, t_to=math.inf):
    """Smallest airborne distance between the given pairs of drones.

    Samples are matched by time; the logs are written at the same rate, so
    the nearest sample of the other drone is at most one period away. In some
    logs the time restarts from zero once while the drone is still on the
    ground, so the other track is sorted before searching it.
    Returns (distance, time, other drone)."""
    best = (math.inf, None, None)
    for a, b in pairs:
        other = sorted(tracks[b])
        if not other:
            continue
        times = [row[0] for row in other]
        for row in tracks[a]:
            if not (t_from <= row[0] <= t_to) or row[3] < AIRBORNE_Z:
                continue
            j = bisect.bisect_left(times, row[0])
            j = min((c for c in (j - 1, j) if 0 <= c < len(times)),
                    key=lambda c: abs(times[c] - row[0]))
            o = other[j]
            if abs(o[0] - row[0]) > 0.2 or o[3] < AIRBORNE_Z:
                continue
            d = math.dist(row[1:], o[1:])
            if d < best[0]:
                best = (d, row[0], b)
    return best


def k_of(run):
    return int(re.match(r"n\d+_k(\d+)_s\d+$", run).group(1))


def config_table(straight_rows, policy_rows):
    def group(rows):
        g = {}
        for r in rows:
            if r["status"] == "OK":
                g.setdefault((r["n_obstacles"], r["k_drones"]), []).append(r)
        return g

    s, p = group(straight_rows), group(policy_rows)
    header = (f"{'n':>3} {'k':>3}   {'reached':>15}   {'collisions/run':>14}   "
              f"{'ratio+':>13}   {'mission(s)':>11}   {'ctrl cpu(s)':>11}")
    lines = ["Per configuration, straight -> policy (5 seeds each)", "",
             header, "-" * len(header)]
    tot = {"drones": 0, "s_reached": 0, "p_reached": 0, "s_coll": 0, "p_coll": 0}
    for key in sorted(s):
        rs, rp = s[key], p.get(key, [])
        drones_total = sum(r["k_drones"] for r in rs)
        sr, pr = sum(r["reached"] for r in rs), sum(r["reached"] for r in rp)
        sc, pc = sum(r["collisions"] for r in rs), sum(r["collisions"] for r in rp)
        tot["drones"] += drones_total
        tot["s_reached"] += sr
        tot["p_reached"] += pr
        tot["s_coll"] += sc
        tot["p_coll"] += pc

        def rplus(rows):
            return mean([r["ratio_reached"] for r in rows if r["ratio_reached"] > 0])

        def avg(rows, field):
            return mean([r[field] for r in rows])

        reached = f"{sr}/{drones_total} -> {pr}/{drones_total}"
        coll = f"{sc / len(rs):.1f} -> {pc / len(rp):.1f}"
        ratio = f"{rplus(rs):.3f} -> {rplus(rp):.3f}"
        mission = f"{avg(rs, 'sim_time'):.0f} -> {avg(rp, 'sim_time'):.0f}"
        ctrl = (f"{avg(rs, 'controller_cpu_time'):.1f} -> "
                f"{avg(rp, 'controller_cpu_time'):.1f}")
        lines.append(f"{key[0]:>3} {key[1]:>3}   {reached:>15}   {coll:>14}   "
                     f"{ratio:>13}   {mission:>11}   {ctrl:>11}")
    lines.append("-" * len(header))
    d = tot["drones"]
    lines.append(
        f"all drones: reached {tot['s_reached']}/{d} -> {tot['p_reached']}/{d} "
        f"({100 * tot['s_reached'] / d:.0f}% -> {100 * tot['p_reached'] / d:.0f}%), "
        f"destroyed {tot['s_coll']} -> {tot['p_coll']}")
    return lines


def drone_report(straight, policy, S, P):
    lines = ["", "Per drone (same drone, same scenario)", ""]
    for k in sorted({k_of(run) for run, _ in S}):
        keys = [key for key in S if k_of(key[0]) == k]
        sc = sum(S[x]["outcome"] == "destroyed" for x in keys)
        pc = sum(P[x]["outcome"] == "destroyed" for x in keys)
        lines.append(f"  k={k}: destroyed {sc}/{len(keys)} ({100 * sc / len(keys):.0f}%) "
                     f"-> {pc}/{len(keys)} ({100 * pc / len(keys):.0f}%)")

    saved = [x for x in S if S[x]["outcome"] == "destroyed" and P[x]["outcome"] == "reached"]
    lost = [x for x in S if S[x]["outcome"] == "reached" and P[x]["outcome"] != "reached"]
    both = [x for x in S if S[x]["outcome"] == "destroyed" and P[x]["outcome"] == "destroyed"]
    lines.append(f"  saved {len(saved)}, lost {len(lost)}, destroyed both ways {len(both)}")
    for run, i in lost:
        lines.append(f"    lost: {run} drone {i} ({P[(run, i)]['outcome']})")

    def clearance_line(label, keys, D):
        if not keys:
            return f"  {label}: none"
        c = sorted(D[x]["clearance"] for x in keys)
        speed = mean([D[x]["travelled"] / D[x]["time"] for x in keys if D[x]["time"] > 0])
        return (f"  {label:30s} {len(keys):3d} drones, clearance median {statistics.median(c):.2f} m, "
                f"min {c[0]:.2f} m, under 0.5 m: {sum(v < 0.5 for v in c):2d}, "
                f"mean speed {speed:.2f} m/s")

    arrived_both = [x for x in S if S[x]["outcome"] == "reached" and P[x]["outcome"] == "reached"]
    lines += ["", "Closest approach to an obstacle, drones that arrived", ""]
    lines.append(clearance_line("straight, arrived both ways", arrived_both, S))
    lines.append(clearance_line("policy, arrived both ways", arrived_both, P))
    lines.append(clearance_line("policy, saved drones", saved, P))
    return lines


def separation_report(straight, policy, P):
    lines = ["", "Closest approach between drones in the air, runs with k > 1", ""]
    for label, batch in (("straight", straight), ("policy", policy)):
        seps = []
        for run_dir in sorted(batch.iterdir()):
            if not re.match(r"n\d+_k\d+_s\d+$", run_dir.name) or k_of(run_dir.name) < 2:
                continue
            k = k_of(run_dir.name)
            tracks = {i: trajectory(batch, run_dir.name, i) for i in range(k)}
            pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
            seps.append((closest_pair(tracks, pairs)[0], run_dir.name))
        seps.sort()
        lines.append(f"  {label:8s} {len(seps)} runs, under 1 m: {sum(d < 1 for d, _ in seps)}, "
                     f"under 2 m: {sum(d < 2 for d, _ in seps)}, closest: "
                     + ", ".join(f"{r} {d:.2f} m" for d, r in seps[:3]))

    lines += ["", f"Crashes under the policy, nearest other drone in the last "
                  f"{CRASH_WINDOW:.0f} s before the crash", ""]
    for (run, i), v in sorted(P.items()):
        if v["outcome"] != "destroyed":
            continue
        k = k_of(run)
        if k == 1:
            note = "flying alone"
        else:
            tracks = {j: trajectory(policy, run, j) for j in range(k)}
            d, _, j = closest_pair(tracks, [(i, j) for j in range(k) if j != i],
                                   v["time"] - CRASH_WINDOW, v["time"])
            note = f"{d:.1f} m (drone {j})" if j is not None else "no other drone airborne"
        lines.append(f"  {run:11s} drone {i}  at {v['time']:5.1f} s  hit {v['hit']:12s} {note}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("straight", type=Path)
    ap.add_argument("policy", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="also write comparison.txt into this directory")
    args = ap.parse_args()

    S, P = drones(args.straight), drones(args.policy)
    missing = sorted(set(S) ^ set(P))
    if missing:
        raise SystemExit(f"the batches do not match, e.g. {missing[:3]}")

    lines = config_table(collect(args.straight), collect(args.policy))
    lines += drone_report(args.straight, args.policy, S, P)
    lines += separation_report(args.straight, args.policy, P)
    text = "\n".join(lines) + "\n"
    print(text, end="")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "comparison.txt").write_text(
            f"straight: {args.straight.name}\npolicy:   {args.policy.name}\n\n" + text)
        print(f"\n-> {args.out / 'comparison.txt'}")


if __name__ == "__main__":
    main()
