#!/usr/bin/env python3
"""Aggregate the results of a 3D (n, k) sweep into a table and a CSV.

Reads every run directory produced by run_batch_3d.py and reports, for each
(n obstacles, k drones) configuration, the metrics from the evaluation plan:

  collisions        number of drones destroyed (obstacles and drone-drone)
  euclidean         summed start-to-target distance over the drones
  travelled         summed distance actually flown
  ratio             travelled / euclidean, over every drone
  ratio+            the same, over only the drones that reached their target
  sim time          simulation time of the run
  cpu time          processor time used by the simulator and autopilots

Values are averaged over the seeds of a configuration.

Simulation time and wall-clock time are reported separately, along with the
real time factor between them. They are equal only while the machine can run
the simulation at full speed, which stops being true as the number of drones
grows; simulation time is the one that describes the flight.

Usage:  python3 compute_metrics_3d.py runs3d/batch_20260827_120000
"""

import csv
import re
import sys
from pathlib import Path


def read_scalars(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        m = re.match(r"^(\w+):\s*([^{].*)$", line.strip())
        if m:
            values[m.group(1)] = m.group(2).strip()
    return values


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect(batch_dir):
    rows = []
    for run_dir in sorted(batch_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = re.match(r"n(\d+)_k(\d+)_s(\d+)$", run_dir.name)
        if not m:
            continue
        n, k, seed = (int(g) for g in m.groups())

        status_file = run_dir / "status.txt"
        status = status_file.read_text().strip() if status_file.exists() else "MISSING"
        summary = read_scalars(run_dir / "run" / "summary.yaml")

        rows.append({
            "run": run_dir.name,
            "n_obstacles": n,
            "k_drones": k,
            "seed": seed,
            "status": status,
            "reached": int(to_float(summary.get("reached"))),
            "collisions": int(to_float(summary.get("collisions"))),
            "collisions_obstacles": int(
                to_float(summary.get("collisions_with_obstacles"))),
            "collisions_drones": int(
                to_float(summary.get("collisions_between_drones"))),
            "euclidean": to_float(summary.get("euclidean_distance_total")),
            "travelled": to_float(summary.get("travelled_distance_total")),
            "ratio": to_float(summary.get("ratio")),
            "ratio_reached": to_float(summary.get("ratio_reached")),
            "sim_time": to_float(summary.get("simulation_time")),
            "real_time_factor": to_float(summary.get("real_time_factor")),
            "cpu_time": to_float(summary.get("cpu_time")),
            "wall_time": to_float(summary.get("wall_time")),
        })
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["n_obstacles"], r["k_drones"]), []).append(r)

    header = (f"{'n':>3} {'k':>3} {'runs':>5} {'reached':>9} {'coll':>5} "
              f"{'(obst/drone)':>13} {'euclid(m)':>10} {'travel(m)':>10} "
              f"{'ratio':>7} {'ratio+':>7} {'sim(s)':>8} {'wall(s)':>8} "
              f"{'rtf':>6} {'cpu(s)':>8}")
    lines = [header, "-" * len(header)]

    for (n, k) in sorted(groups):
        rs = groups[(n, k)]
        ok = [r for r in rs if r["status"] == "OK"]
        if not ok:
            lines.append(f"{n:>3} {k:>3} {len(rs):>5} {'no valid runs':>9}")
            continue

        total_drones = sum(r["k_drones"] for r in ok)
        total_reached = sum(r["reached"] for r in ok)
        reached_str = f"{total_reached}/{total_drones}"

        obst = mean([r["collisions_obstacles"] for r in ok])
        drone = mean([r["collisions_drones"] for r in ok])
        breakdown = f"{obst:.1f}/{drone:.1f}"

        lines.append(
            f"{n:>3} {k:>3} {len(ok):>5} {reached_str:>9} "
            f"{mean([r['collisions'] for r in ok]):>5.1f} "
            f"{breakdown:>13} "
            f"{mean([r['euclidean'] for r in ok]):>10.1f} "
            f"{mean([r['travelled'] for r in ok]):>10.1f} "
            f"{mean([r['ratio'] for r in ok]):>7.3f} "
            # Runs where nothing arrived have no such ratio; averaging their
            # zeroes in would drag the column down for the wrong reason.
            f"{mean([r['ratio_reached'] for r in ok if r['ratio_reached'] > 0]):>7.3f} "
            f"{mean([r['sim_time'] for r in ok]):>8.1f} "
            f"{mean([r['wall_time'] for r in ok]):>8.1f} "
            f"{mean([r['real_time_factor'] for r in ok]):>6.2f} "
            f"{mean([r['cpu_time'] for r in ok]):>8.0f}")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <batch_dir>")
    batch_dir = Path(sys.argv[1])
    if not batch_dir.is_dir():
        sys.exit(f"not a directory: {batch_dir}")

    rows = collect(batch_dir)
    if not rows:
        sys.exit("no runs found (expected subdirs like n8_k2_s1/)")

    csv_path = batch_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    table = summarize(rows)
    (batch_dir / "summary.txt").write_text(table + "\n")

    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1

    print(f"{len(rows)} runs -> {csv_path}")
    print("statuses: " + ", ".join(f"{v}x {k}" for k, v in sorted(statuses.items())))
    print()
    print(table)


if __name__ == "__main__":
    main()
