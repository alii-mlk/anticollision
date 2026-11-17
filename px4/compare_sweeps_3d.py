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

Each drone's track counts time from the start of its own mission node, and
the nodes of one run start up to about a second apart. Runs recorded before
2026-11-17 did not save those starts, so for them the distances between
drones are lower bounds, see separations(). Newer runs record the distance in
the collision monitors, which compare live positions.

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

from compute_metrics_3d import DRONE_LINE, collect, mean

# Below this height a drone is on the ground (before takeoff or after it was
# destroyed), so it is left out of the drone-to-drone distances.
AIRBORNE_Z = 1.0
# Seconds before a crash in which to look for another drone nearby.
CRASH_WINDOW = 5.0
# For runs that did not record when each drone's clock started: the largest
# offset between two drones' clocks that the distances allow for. On the four
# runs where two drones touched, the offsets were 0.2 to 0.94 s.
MAX_CLOCK_OFFSET = 2.0


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


def raw_trajectory(batch, run, i):
    """(t, x, y, z) rows of one drone, t on the drone's own clock.

    Until a node receives the simulator's clock it counts its own time, and
    when the clock arrives the count starts again from zero, on the ground.
    Only the rows after that restart are kept. Also returns how many seconds
    the node had counted before it, 0 when there was none."""
    path = batch / run / "run" / f"drone{i}" / "trajectory.csv"
    if not path.exists():
        return [], 0.0
    with open(path) as f:
        rows = [tuple(map(float, row)) for row in list(csv.reader(f))[1:]]
    restart = max((j for j in range(1, len(rows)) if rows[j][0] < rows[j - 1][0]),
                  default=0)
    before = rows[restart - 1][0] if restart else 0.0
    return rows[restart:], before


def clock_starts(batch, run, k):
    """The simulator time at which each drone's clock started.

    Each mission node counts time from its own start, and the nodes of one run
    start up to about a second apart. Runs recorded since 2026-11-17 write
    that start into mission.yaml (sim_start). For older runs it is not known:
    estimating it from the times the nodes logged their start was tried and
    missed by up to 0.9 s on the runs where two drones touched.
    Returns {drone: start}, or None if a drone did not record it."""
    starts = {}
    for i in range(k):
        mission = batch / run / "run" / f"drone{i}" / "mission.yaml"
        if mission.exists():
            m = re.search(r"^sim_start: ([\d.]+)$", mission.read_text(), re.M)
            if m:
                starts[i] = float(m.group(1))
    return starts if len(starts) == k else None


def monitor_distance(batch, run, k):
    """Closest distance between two drones in the air, from the monitors.

    The collision monitors compare live positions, so this needs no clock
    alignment. Recorded since 2026-11-17; None for older runs. inf if no two
    drones were in the air at the same time."""
    found = []
    for i in range(k):
        outcome = batch / run / "run" / f"drone{i}" / "outcome.yaml"
        if not outcome.exists():
            return None
        m = re.search(r"^min_drone_distance: (\S+)$", outcome.read_text(), re.M)
        if not m:
            return None
        found.append(math.inf if m.group(1) == "null" else float(m.group(1)))
    return min(found)


def run_tracks(batch, run, k):
    """{drone: [(t, x, y, z)]} with every t on the run's common clock.

    If the clocks cannot be aligned the tracks stay on their own clocks and
    the second value is False."""
    starts = clock_starts(batch, run, k)
    tracks = {}
    for i in range(k):
        rows, _ = raw_trajectory(batch, run, i)
        shift = starts[i] - min(starts.values()) if starts else 0.0
        tracks[i] = [(t + shift, x, y, z) for t, x, y, z in rows]
    return tracks, starts is not None


def trajectory(batch, run, i):
    return raw_trajectory(batch, run, i)[0]


def closest_pair(tracks, pairs, t_from=-math.inf, t_to=math.inf):
    """Smallest airborne distance between the given pairs of drones.

    Samples are matched by time, so the tracks must be on one clock (see
    run_tracks). The logs are written at the same rate, so the nearest sample
    of the other drone is at most one period away.
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


def closest_pair_bound(tracks, pairs, max_shift, t_from=-math.inf, t_to=math.inf):
    """Like closest_pair, for tracks whose clocks may be up to max_shift apart.

    Every sample of one drone is compared with every sample of the other that
    lies within max_shift seconds of it, so whatever the true offset between
    the two clocks is, as long as it is within max_shift, the true closest
    approach is at least the distance returned. Returns (distance, other)."""
    best = (math.inf, None)
    for a, b in pairs:
        other = sorted(row for row in tracks[b] if row[3] >= AIRBORNE_Z)
        times = [row[0] for row in other]
        for row in tracks[a]:
            if not (t_from <= row[0] <= t_to) or row[3] < AIRBORNE_Z:
                continue
            lo = bisect.bisect_left(times, row[0] - max_shift)
            hi = bisect.bisect_right(times, row[0] + max_shift)
            for o in other[lo:hi]:
                d = math.dist(row[1:], o[1:])
                if d < best[0]:
                    best = (d, b)
    return best


def separations(batch):
    """Closest approach between two drones in the air, for every run with k > 1.

    Returns (distance, run, how) sorted by distance; the distance is between
    the drones' centres. How it was found, best first:
      monitor   recorded by the collision monitors (runs since 2026-11-17)
      tracks    from the tracks, put on one clock with sim_start
      at least  older runs, whose clocks are not known: the smallest distance
                for any clock offset up to MAX_CLOCK_OFFSET, so the true
                closest approach is at least this
    """
    seps = []
    for run_dir in sorted(batch.iterdir()):
        if not re.match(r"n\d+_k\d+_s\d+$", run_dir.name) or k_of(run_dir.name) < 2:
            continue
        run, k = run_dir.name, k_of(run_dir.name)
        pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        d = monitor_distance(batch, run, k)
        if d is not None:
            seps.append((d, run, "monitor"))
            continue
        tracks, aligned = run_tracks(batch, run, k)
        if aligned:
            seps.append((closest_pair(tracks, pairs)[0], run, "tracks"))
        else:
            seps.append((closest_pair_bound(tracks, pairs, MAX_CLOCK_OFFSET)[0],
                         run, "at least"))
    return sorted(seps)


def separation_report(straight, policy, S, P, seps):
    lines = ["", "Closest approach between drones in the air, runs with k > 1", ""]
    for label, D in (("straight", S), ("policy", P)):
        found = seps[label]
        contacts = sum(v["hit"] == "another drone" for (run, _), v in D.items()
                       if k_of(run) > 1)
        bound = any(how == "at least" for _, _, how in found)
        word = "at least " if bound else ""
        lines.append(f"  {label:8s} {len(found)} runs, drones destroyed by touching another "
                     f"drone: {contacts}, closest: "
                     + ", ".join(f"{r} {word}{d:.2f} m" for d, r, _ in found[:3]))
    if any(how == "at least" for found in seps.values() for _, _, how in found):
        lines += ["",
                  "  These runs did not record when each drone's clock started (each",
                  "  mission node counts from its own start, up to about a second apart),",
                  f"  so the distances are lower bounds: the closest the two drones could",
                  f"  have been for any offset between their clocks up to {MAX_CLOCK_OFFSET:.0f} s.",
                  "  Contacts are counted by the collision monitor from live positions."]

    lines += ["", f"Crashes under the policy, nearest other drone in the last "
                  f"{CRASH_WINDOW:.0f} s before the crash", ""]
    for (run, i), v in sorted(P.items()):
        if v["outcome"] != "destroyed":
            continue
        k = k_of(run)
        if k == 1:
            note = "flying alone"
        else:
            tracks, aligned = run_tracks(policy, run, k)
            # The crash is the end of the drone's track.
            end = tracks[i][-1][0] if tracks[i] else v["time"]
            others = [(i, j) for j in range(k) if j != i]
            if aligned:
                d, _, j = closest_pair(tracks, others, end - CRASH_WINDOW, end)
                word = ""
            else:
                d, j = closest_pair_bound(tracks, others, MAX_CLOCK_OFFSET,
                                          end - CRASH_WINDOW, end)
                word = "at least "
            note = (f"{word}{d:.1f} m (drone {j})" if j is not None
                    else "no other drone airborne")
        lines.append(f"  {run:11s} drone {i}  at {v['time']:5.1f} s  hit {v['hit']:12s} {note}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("straight", type=Path)
    ap.add_argument("policy", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="also write comparison.txt into this directory")
    ap.add_argument("--separation", type=Path, default=None,
                    help="also write the closest approach between drones of "
                         "every run with k > 1 to this CSV file")
    args = ap.parse_args()

    S, P = drones(args.straight), drones(args.policy)
    missing = sorted(set(S) ^ set(P))
    if missing:
        raise SystemExit(f"the batches do not match, e.g. {missing[:3]}")

    seps = {"straight": separations(args.straight),
            "policy": separations(args.policy)}
    lines = config_table(collect(args.straight), collect(args.policy))
    lines += drone_report(args.straight, args.policy, S, P)
    lines += separation_report(args.straight, args.policy, S, P, seps)
    text = "\n".join(lines) + "\n"
    print(text, end="")

    if args.separation:
        args.separation.parent.mkdir(parents=True, exist_ok=True)
        with open(args.separation, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["controller", "batch", "run", "k_drones",
                             "closest_m", "measured", "touched"])
            for label, batch, D in (("straight", args.straight, S),
                                    ("policy", args.policy, P)):
                for d, run, how in sorted(seps[label], key=lambda s: s[1]):
                    touched = sum(v["hit"] == "another drone"
                                  for (r, _), v in D.items() if r == run)
                    writer.writerow([label, batch.name, run, k_of(run),
                                     f"{d:.3f}" if math.isfinite(d) else "",
                                     how, touched])
        print(f"-> {args.separation}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "comparison.txt").write_text(
            f"straight: {args.straight.name}\npolicy:   {args.policy.name}\n\n" + text)
        print(f"\n-> {args.out / 'comparison.txt'}")


if __name__ == "__main__":
    main()
