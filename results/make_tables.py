#!/usr/bin/env python3
"""The tables of the thesis, computed from the curated result folders.

Reads the per-run and per-drone files that px4/compute_metrics_3d.py wrote
for the Gazebo sweeps, and the per-flight files of rl/evaluate_large.py for
the MuJoCo evaluation. Writes every table as text into
results/thesis/tables.txt and as CSV into results/thesis/tables/.

How the numbers are computed:

  share that arrived  drones that reached their target / drones flown, with
                      the 95% Wilson interval. An arrival is a yes or a no,
                      so it gets an interval and no standard deviation. The
                      interval treats the flights as independent, which the
                      drones of one scenario are not completely.
  gain                for one scenario, the share of its drones that arrived
                      with one controller minus the share that arrived with
                      the other, in percentage points. Mean and standard
                      deviation over the scenarios.
  mean, std           over the drones or runs named in the table. The standard
                      deviation is the sample one, with n - 1.

The three controllers:

  straight        no avoidance, every drone flies the straight line
  trained alone   the October policy, trained with one drone in the maze
  final policy    trained with 4 to 8 drones and the neighbour inputs

Plain Python without packages, so it runs on any machine.

Usage:  python3 results/make_tables.py
"""

import csv
import gzip
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OCTOBER = HERE / "2026-10-19_rl_sweep"
NOVEMBER = HERE / "2026-11-15_swarm_sweep"
OUT = HERE / "thesis"

# Where the Gazebo runs of each controller are. The sweep has 1 to 3 drones;
# the runs with 4 drones are batches of their own, and the policy trained
# alone was not flown with 4.
ARMS = {
    "straight": [OCTOBER / "straight", NOVEMBER / "k4_straight"],
    "trained alone": [OCTOBER / "policy"],
    "final policy": [NOVEMBER / "policy", NOVEMBER / "k4_policy"],
}
MEMORY = {
    "straight": NOVEMBER / "ram_straight",
    "final policy": NOVEMBER / "ram_policy",
}
MEMORY_PARTS = ("gazebo", "px4", "mavros", "helpers", "mission", "monitor")
MUJOCO_FILES = [NOVEMBER / "mujoco" / "large_swarm_scaling.csv.gz",
                NOVEMBER / "mujoco" / "large_swarm_scaling_2.csv.gz"]
# Names of the policies in those files, and what they are called here.
MUJOCO_NAMES = {"straight": "straight", "mujoco_1": "trained alone",
                "swarm_4/final": "final policy"}
MUJOCO_CRASHES = ("collided", "hit_drone", "ground", "out_of_bounds")
TRAINING_EVAL = NOVEMBER / "model" / "eval.csv"
# The evaluations during training: the mazes and number of drones of each.
TRAINING_SETS = (("heldout", "1 drone"), ("heldout_k3", "3 drones"),
                 ("heldout_k8", "8 drones"))
# Closest approach between two drones of a run, per controller. Each file has
# a straight and a policy part, see px4/compare_sweeps_3d.py --separation.
SEPARATION = {
    "straight": [(OCTOBER / "separation.csv", "straight"),
                 (NOVEMBER / "separation_k4.csv", "straight")],
    "trained alone": [(OCTOBER / "separation.csv", "policy")],
    "final policy": [(NOVEMBER / "separation.csv", "policy"),
                     (NOVEMBER / "separation_k4.csv", "policy")],
}
# Two drones touch when their centres are this close (twice the radius).
TOUCHING_M = 0.7

# Groups of drone counts a table has rows for. "1 to 3" is the whole sweep.
K_GROUPS = (1, 2, 3, "1 to 3", 4)


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def wilson(successes, n, z=1.96):
    """95% confidence interval for a proportion, as (low, high)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mean_std(values):
    """(mean, sample standard deviation); the deviation of one value is 0."""
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.fmean(values), std


def num(x, digits=2):
    return "-" if x is None or math.isnan(x) else f"{x:.{digits}f}"


def interval(successes, n):
    low, high = wilson(successes, n)
    return f"{100 * low:.1f} to {100 * high:.1f}"


def in_group(k, group):
    return k in (1, 2, 3) if group == "1 to 3" else k == group


# ---------------------------------------------------------------- loading

def gazebo_runs(arm):
    """{run name: row of metrics.csv} for the runs that completed."""
    rows = {}
    for folder in ARMS[arm]:
        for r in read(folder / "metrics.csv"):
            if r["status"] == "OK":
                rows[r["run"]] = r
    return rows


def gazebo_drones(arm):
    """{(run name, drone): row of drones.csv}, numbers converted."""
    rows = {}
    for folder in ARMS[arm]:
        for r in read(folder / "drones.csv"):
            for field in ("euclidean", "travelled", "time", "min_clearance"):
                r[field] = float(r[field])
            r["k"] = int(r["k_drones"])
            r["n"] = int(r["n_obstacles"])
            r["arrived"] = r["outcome"] == "reached"
            rows[(r["run"], int(r["drone"]))] = r
    return rows


def training_curve():
    """The evaluations made during training, per set of mazes.

    Returns ({set: [(decisions, reached, flights)]} for the checkpoints,
    {set: (decisions, reached, flights)} for the final policy,
    {set: (reached, flights)} for flying straight)."""
    checkpoints, final, straight = {}, {}, {}
    for r in read(TRAINING_EVAL):
        point = (int(r["decisions"]), int(r["reached"]), int(r["episodes"]))
        if r["policy"] == "policy":
            checkpoints.setdefault(r["set"], []).append(point)
        elif r["policy"] == "final":
            final[r["set"]] = point
        elif r["policy"] == "straight":
            straight[r["set"]] = point[1:]
    return ({s: sorted(points) for s, points in checkpoints.items()},
            final, straight)


def separation():
    """{controller: [(drones, closest distance, how it was measured)]}.

    One value per run with more than one drone. A run in which no two drones
    were in the air at the same time has no value and is left out."""
    out = {}
    for arm, sources in SEPARATION.items():
        out[arm] = []
        for path, part in sources:
            for r in read(path):
                if r["controller"] == part and r["closest_m"]:
                    out[arm].append((int(r["k_drones"]), float(r["closest_m"]),
                                     r["measured"]))
    return out


def arrived_with_all(drones, k_values=(1, 2, 3)):
    """Drones that arrived with every controller, in the runs all three flew."""
    keys = set.intersection(*(set(d) for d in drones.values()))
    return sorted(key for key in keys
                  if drones["straight"][key]["k"] in k_values
                  and all(drones[arm][key]["arrived"] for arm in drones))


def quartiles(values):
    """(first quartile, median, third quartile)."""
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[0], q[1], q[2]


def mujoco_flights():
    """{controller: [flights]} of the large MuJoCo evaluation."""
    flights = {name: [] for name in MUJOCO_NAMES.values()}
    for path in MUJOCO_FILES:
        for r in read(path):
            name = MUJOCO_NAMES.get(r["policy"])
            if name is None:
                continue
            r["k"] = int(r["k_drones"])
            for field in ("euclidean", "travelled", "min_clearance"):
                r[field] = float(r[field])
            flights[name].append(r)
    return flights


# ----------------------------------------------------------------- tables

def table_training():
    checkpoints, final, straight = training_curve()
    rows = []
    decisions = [d for d, _, _ in checkpoints[TRAINING_SETS[0][0]]]
    for i, d in enumerate(decisions):
        row = [f"{d / 1e6:.1f} M", "checkpoint"]
        for name, _ in TRAINING_SETS:
            _, reached, flights = checkpoints[name][i]
            row.append(f"{reached}/{flights}")
        rows.append(row)
    rows.append([f"{final[TRAINING_SETS[0][0]][0] / 1e6:.1f} M", "final.zip"]
                + [f"{final[name][1]}/{final[name][2]}"
                   for name, _ in TRAINING_SETS])
    rows.append(["-", "flying straight"]
                + [f"{straight[name][0]}/{straight[name][1]}"
                   for name, _ in TRAINING_SETS])
    return {
        "name": "training",
        "title": "MuJoCo: the final policy during training, drones that "
                 "arrived on the evaluation mazes",
        "columns": ["decisions", "policy"] + [title for _, title in TRAINING_SETS],
        "rows": rows,
        "notes": ["Training ran 4 million decisions in 59 minutes. Every "
                  "400,000 decisions the policy flew the same mazes, which "
                  "were never used for training.",
                  "1 drone: 40 mazes. 3 drones: 20 mazes. 8 drones: 10 "
                  "mazes. The policy used everywhere else is final.zip."],
    }


def table_arrivals(drones):
    rows = []
    for group in K_GROUPS:
        for arm in ARMS:
            ds = [d for d in drones[arm].values() if in_group(d["k"], group)]
            if not ds:
                continue
            reached = sum(d["arrived"] for d in ds)
            obstacle = sum(d["collided_with"] == "an obstacle" for d in ds)
            drone = sum(d["collided_with"] == "another drone" for d in ds)
            rows.append([group, arm, len(ds), reached,
                         num(100 * reached / len(ds), 1),
                         interval(reached, len(ds)), obstacle, drone,
                         len(ds) - reached - obstacle - drone])
        rows.append(None)
    return {
        "name": "gazebo_arrivals",
        "title": "Gazebo: drones that arrived, by number of drones",
        "columns": ["drones", "controller", "flights", "arrived", "%",
                    "95% interval", "hit obstacle", "hit drone", "other end"],
        "rows": rows,
        "notes": ["Each row of 1, 2 or 3 drones is 25 scenarios (5 obstacle "
                  "counts, 5 seeds); 4 drones is 25 more.",
                  "With 4 drones the VM ran at its limit, see the cost table."],
    }


def table_arrivals_by_obstacles(drones):
    rows = []
    for k in (1, 2, 3, 4):
        for n in (4, 8, 12, 16, 20):
            row = [k, n]
            for arm in ARMS:
                ds = [d for d in drones[arm].values()
                      if d["k"] == k and d["n"] == n]
                row.append(f"{sum(d['arrived'] for d in ds)}/{len(ds)}"
                           if ds else "-")
            rows.append(row)
        rows.append(None)
    return {
        "name": "gazebo_arrivals_by_obstacles",
        "title": "Gazebo: drones that arrived, by number of drones and of "
                 "obstacles",
        "columns": ["drones", "obstacles"] + list(ARMS),
        "rows": rows,
        "notes": ["Arrived / flown, 5 seeds per cell."],
    }


def table_gain(runs):
    def share(arm):
        return {name: int(r["reached"]) / int(r["k_drones"])
                for name, r in runs[arm].items()}

    rows = []
    for better, worse in (("final policy", "straight"),
                          ("trained alone", "straight"),
                          ("final policy", "trained alone")):
        a, b = share(better), share(worse)
        for group in K_GROUPS:
            names = [name for name in a if name in b
                     and in_group(int(runs[better][name]["k_drones"]), group)]
            if not names:
                continue
            gains = [100 * (a[name] - b[name]) for name in names]
            mean, std = mean_std(gains)
            rows.append([f"{better} over {worse}", group, len(gains),
                         num(mean, 1), num(std, 1),
                         num(std / math.sqrt(len(gains)), 1),
                         sum(g > 0 for g in gains), sum(g == 0 for g in gains),
                         sum(g < 0 for g in gains)])
        rows.append(None)
    return {
        "name": "gazebo_gain",
        "title": "Gazebo: gain per scenario, in percentage points of the "
                 "scenario's drones that arrived",
        "columns": ["comparison", "drones", "scenarios", "mean gain", "std",
                    "std error", "better", "equal", "worse"],
        "rows": rows,
        "notes": ["Both controllers flew the same scenario, so the gain is a "
                  "difference within a scenario.",
                  "Std error is std / sqrt(scenarios). Better, equal and worse "
                  "count scenarios."],
    }


def table_paired_drones(drones):
    rows = []
    for better, worse in (("final policy", "straight"),
                          ("trained alone", "straight")):
        a, b = drones[better], drones[worse]
        for group in K_GROUPS:
            keys = [key for key in a if key in b and in_group(a[key]["k"], group)]
            if not keys:
                continue
            both = sum(a[x]["arrived"] and b[x]["arrived"] for x in keys)
            saved = sum(a[x]["arrived"] and not b[x]["arrived"] for x in keys)
            lost = sum(not a[x]["arrived"] and b[x]["arrived"] for x in keys)
            rows.append([f"{better} against {worse}", group, len(keys), both,
                         saved, lost, len(keys) - both - saved - lost])
        rows.append(None)
    return {
        "name": "gazebo_drone_by_drone",
        "title": "Gazebo: the same drone in the same scenario, with and "
                 "without the policy",
        "columns": ["comparison", "drones", "flights", "arrived both ways",
                    "saved", "lost", "arrived neither way"],
        "rows": rows,
        "notes": ["Saved: destroyed flying straight, arrived with the policy. "
                  "Lost: the other way round."],
    }


def table_flight(drones):
    rows = []
    for arm in ARMS:
        for group in K_GROUPS:
            ds = [d for d in drones[arm].values()
                  if d["arrived"] and in_group(d["k"], group)]
            if not ds:
                continue
            ratio = mean_std(d["travelled"] / d["euclidean"] for d in ds)
            time = mean_std(d["time"] for d in ds)
            speed = statistics.fmean(d["travelled"] / d["time"] for d in ds)
            clearance = [d["min_clearance"] for d in ds]
            c_mean, c_std = mean_std(clearance)
            rows.append([arm, group, len(ds), num(ratio[0], 3), num(ratio[1], 3),
                         num(time[0], 1), num(time[1], 1), num(speed),
                         num(statistics.median(clearance)), num(c_mean),
                         num(c_std), num(min(clearance)),
                         sum(c < 0.5 for c in clearance)])
        rows.append(None)
    return {
        "name": "gazebo_flight",
        "title": "Gazebo: path, time and clearance of the drones that arrived",
        "columns": ["controller", "drones", "arrived", "ratio mean",
                    "ratio std", "time mean (s)", "time std", "speed (m/s)",
                    "clearance median (m)", "clearance mean", "clearance std",
                    "clearance min", "under 0.5 m"],
        "rows": rows,
        "notes": ["Ratio: distance flown / straight-line distance. Time: from "
                  "the start of the mission to the target, with the climb.",
                  "Clearance: the smallest distance between the drone's hull "
                  "and an obstacle during the flight.",
                  "The drones that arrive flying straight are the ones with "
                  "nothing in their way, so these rows do not compare like "
                  "with like. The next table does."],
    }


def table_flight_by_obstacles(drones):
    rows = []
    for arm in ARMS:
        for n in (4, 8, 12, 16, 20):
            ds = [d for d in drones[arm].values()
                  if d["arrived"] and d["n"] == n and d["k"] in (1, 2, 3)]
            ratio = mean_std(d["travelled"] / d["euclidean"] for d in ds)
            time = mean_std(d["time"] for d in ds)
            rows.append([arm, n, len(ds), num(ratio[0], 3), num(ratio[1], 3),
                         num(time[0], 1), num(time[1], 1)])
        rows.append(None)
    return {
        "name": "gazebo_flight_by_obstacles",
        "title": "Gazebo: path and time of the drones that arrived, by number "
                 "of obstacles",
        "columns": ["controller", "obstacles", "arrived", "ratio mean",
                    "ratio std", "time mean (s)", "time std"],
        "rows": rows,
        "notes": ["1 to 3 drones, 30 flights per row. With 4 drones only two "
                  "controllers were flown, so they are left out here.",
                  "As in the table by number of drones, the drones that "
                  "arrive flying straight are the ones with nothing in their "
                  "way."],
    }


def table_clearance(drones):
    keys = arrived_with_all(drones)
    rows = []
    for arm in ARMS:
        clearance = [drones[arm][key]["min_clearance"] for key in keys]
        q1, median, q3 = quartiles(clearance)
        rows.append([arm, len(clearance), num(min(clearance)), num(q1),
                     num(median), num(q3), num(max(clearance)),
                     sum(c < 0.5 for c in clearance),
                     sum(c < 1.0 for c in clearance)])
    return {
        "name": "gazebo_clearance",
        "title": "Gazebo: closest approach to an obstacle of the drones that "
                 "arrived with all three controllers",
        "columns": ["controller", "drones", "min (m)", "first quartile",
                    "median", "third quartile", "max", "under 0.5 m",
                    "under 1 m"],
        "rows": rows,
        "notes": ["The same drones in the same scenarios for every "
                  "controller, 1 to 3 drones. Distance from the drone's hull "
                  "to the nearest obstacle."],
    }


def table_separation(drones):
    seps = separation()
    rows = []
    lower_bound = False
    for k in (2, 3, 4, "all"):
        for arm in ARMS:
            found = [(d, how) for kk, d, how in seps[arm] if k == "all" or kk == k]
            if not found:
                continue
            values = [d for d, _ in found]
            lower_bound |= any(how == "at least" for _, how in found)
            touched = sum(d["collided_with"] == "another drone"
                          for d in drones[arm].values()
                          if d["k"] > 1 and (k == "all" or d["k"] == k))
            rows.append([k, arm, len(values), touched, num(min(values)),
                         num(statistics.median(values))])
        rows.append(None)
    notes = ["Distance between the centres of the two closest drones while "
             "both were higher than 1 m, one value per run; the closest and "
             "median columns are over the runs.",
             "Touched: drones destroyed by touching another drone (centres "
             f"{TOUCHING_M} m apart), counted by the collision monitors from "
             "live positions."]
    if lower_bound:
        notes += ["These runs did not record when each drone's clock started. "
                  "Each mission node counts time from its own start and the "
                  "nodes of a run start up to about a second apart, so the "
                  "distances are lower bounds: the closest two drones could "
                  "have been for any offset between their clocks up to 2 s."]
    return {
        "name": "gazebo_drone_separation",
        "title": "Gazebo: closest approach between two drones in the air",
        "columns": ["drones", "controller", "runs", "touched",
                    "closest, at least (m)" if lower_bound else "closest (m)",
                    "median, at least (m)" if lower_bound else "median (m)"],
        "rows": rows,
        "notes": notes,
    }


def table_cost_of_avoiding(drones):
    a, b = drones["final policy"], drones["straight"]
    rows = []
    for group in K_GROUPS:
        keys = [key for key in a if key in b and in_group(a[key]["k"], group)
                and a[key]["arrived"] and b[key]["arrived"]]
        if not keys:
            continue
        extra_path = mean_std(
            100 * (a[x]["travelled"] - b[x]["travelled"]) / a[x]["euclidean"]
            for x in keys)
        extra_time = mean_std(a[x]["time"] - b[x]["time"] for x in keys)
        rows.append([
            group, len(keys),
            num(statistics.fmean(b[x]["travelled"] / b[x]["euclidean"] for x in keys), 3),
            num(statistics.fmean(a[x]["travelled"] / a[x]["euclidean"] for x in keys), 3),
            num(extra_path[0], 1), num(extra_path[1], 1),
            num(statistics.fmean(b[x]["time"] for x in keys), 1),
            num(statistics.fmean(a[x]["time"] for x in keys), 1),
            num(extra_time[0], 1), num(extra_time[1], 1),
            num(statistics.median(b[x]["min_clearance"] for x in keys)),
            num(statistics.median(a[x]["min_clearance"] for x in keys))])
    return {
        "name": "gazebo_cost_of_avoiding",
        "title": "Gazebo: what the final policy costs a drone that would have "
                 "arrived anyway",
        "columns": ["drones", "flights", "ratio straight", "ratio policy",
                    "extra path mean (%)", "extra path std",
                    "time straight (s)", "time policy (s)",
                    "extra time mean (s)", "extra time std",
                    "clearance straight (m)", "clearance policy (m)"],
        "rows": rows,
        "notes": ["Only the drones that arrived both flying straight and with "
                  "the final policy, same drone and same scenario.",
                  "Extra path is in percent of the straight-line distance. "
                  "Clearance is the median."],
    }


def table_cpu(runs, drones):
    rows = []
    for arm in ARMS:
        for k in (1, 2, 3, 4):
            rs = [r for r in runs[arm].values() if int(r["k_drones"]) == k]
            if not rs:
                continue
            cpu = mean_std(float(r["cpu_time"]) for r in rs)
            cores = mean_std(float(r["cpu_time"]) / float(r["wall_time_measured"])
                             for r in rs)
            rtf = [float(r["real_time_factor"]) for r in rs]
            ds = [d for d in drones[arm].values()
                  if d["k"] == k and d["controller_cpu_time"] and d["time"] > 0]
            controller = mean_std(100 * float(d["controller_cpu_time"]) / d["time"]
                                  for d in ds)
            rows.append([arm, k, len(rs),
                         num(statistics.fmean(float(r["sim_time"]) for r in rs), 1),
                         num(statistics.fmean(rtf)), num(min(rtf)),
                         num(cpu[0], 0), num(cpu[1], 0),
                         num(cores[0]), num(cores[1]),
                         num(controller[0], 1), num(controller[1], 1)])
        rows.append(None)
    return {
        "name": "gazebo_cpu",
        "title": "Gazebo: processor time",
        "columns": ["controller", "drones", "runs", "mission (s)",
                    "real time factor", "lowest", "simulator CPU mean (s)",
                    "std", "cores busy mean", "std",
                    "controller per drone (% of a core)", "std"],
        "rows": rows,
        "notes": ["Simulator CPU: processor seconds of Gazebo, the PX4 "
                  "instances and the MAVROS nodes during the mission.",
                  "Cores busy: those seconds / the wall-clock seconds they "
                  "were measured over.",
                  "Controller: processor seconds of one drone's mission node "
                  "/ simulated seconds of its flight. With a policy this "
                  "holds the lidar and the network.",
                  "A real time factor under 1 means the machine could not "
                  "keep up with the simulation."],
    }


def table_memory():
    rows = []
    for arm, folder in MEMORY.items():
        runs = [r for r in read(folder / "metrics.csv") if r["status"] == "OK"]
        for k in (1, 2, 3, 4):
            rs = [r for r in runs if int(r["k_drones"]) == k]
            if not rs:
                continue
            parts = [statistics.fmean(float(r[f"memory_pss_{p}_mb"]) for r in rs)
                     for p in MEMORY_PARTS]
            totals = [sum(float(r[f"memory_pss_{p}_mb"]) for p in MEMORY_PARTS)
                      for r in rs]
            resident = statistics.fmean(
                float(r["memory_simulation_mb"]) + float(r["memory_controller_mb"])
                for r in rs)
            drop = statistics.fmean(
                float(r["memory_free_before_mb"]) - float(r["memory_free_min_mb"])
                for r in rs)
            rows.append([arm, k, len(rs)] + [num(p, 0) for p in parts]
                        + [num(statistics.fmean(totals), 0), num(min(totals), 0),
                           num(max(totals), 0), num(resident, 0), num(drop, 0)])
        rows.append(None)
    return {
        "name": "gazebo_memory",
        "title": "Gazebo: memory in MB",
        "columns": ["controller", "drones", "runs"] + list(MEMORY_PARTS)
                   + ["total", "lowest", "highest", "resident sum",
                      "free memory drop"],
        "rows": rows,
        "notes": ["One run per obstacle count (4 to 20), averaged. Lowest and "
                  "highest total are the runs with 4 and with 20 obstacles.",
                  "The parts are proportional memory: a page shared by "
                  "several processes is divided among them, so the parts add "
                  "up to the total.",
                  "Resident sum counts shared pages once per process and is "
                  "an upper bound. Free memory drop is the machine's "
                  "available memory before the stack started minus its "
                  "lowest value in the flight.",
                  "Helpers: clock bridge, ground station heartbeats and the "
                  "wrappers that start the nodes."],
    }


def table_mujoco(flights):
    rows = []
    for k in (1, 2, 3, 4, 6, 8, 10, "all"):
        for name, all_flights in flights.items():
            fs = [f for f in all_flights if k == "all" or f["k"] == k]
            if not fs:
                continue
            n = len(fs)
            reached = sum(f["outcome"] == "reached" for f in fs)
            crashed = sum(f["outcome"] in MUJOCO_CRASHES for f in fs)
            hit_drone = sum(f["outcome"] == "hit_drone" for f in fs)
            arrived = [f for f in fs if f["outcome"] == "reached"]
            ratio = mean_std(f["travelled"] / f["euclidean"] for f in arrived)
            clearance = [f["min_clearance"] for f in arrived]
            rows.append([k, name, n, num(100 * reached / n, 1),
                         interval(reached, n), num(100 * crashed / n, 1),
                         hit_drone, num(100 * hit_drone / n, 2),
                         num(100 * wilson(hit_drone, n)[1], 2),
                         num(100 * (n - reached - crashed) / n, 1),
                         num(ratio[0], 3), num(ratio[1], 3),
                         num(statistics.median(clearance))])
        rows.append(None)
    return {
        "name": "mujoco_evaluation",
        "title": "MuJoCo: 400 new mazes per number of drones",
        "columns": ["drones", "controller", "flights", "arrived %",
                    "95% interval", "crashed %", "hit drone", "hit drone %",
                    "at most %", "timeout %", "ratio mean", "ratio std",
                    "clearance median (m)"],
        "rows": rows,
        "notes": ["The mazes have 0, 5, 10, 15 or 20 obstacles and were not "
                  "used to train or to choose any policy.",
                  "Crashed: hit an obstacle, another drone or the ground, or "
                  "left the area. Hit drone is the part of crashed that was "
                  "another drone.",
                  "At most: the upper end of the 95% interval of the hit "
                  "drone share."],
    }


def table_simulators(drones):
    """The same scenarios in MuJoCo (from training's evaluations) and Gazebo."""
    evaluations = read(TRAINING_EVAL)
    rows = []
    for arm, policy in (("straight", "straight"), ("final policy", "final")):
        for k, name in ((1, "gazebo"), (3, "gazebo_k3")):
            mujoco = [e for e in evaluations
                      if e["policy"] == policy and e["set"] == name][-1]
            ds = [d for d in drones[arm].values() if d["k"] == k]
            rows.append([arm, k, len(ds), int(mujoco["reached"]),
                         sum(d["arrived"] for d in ds)])
            assert int(mujoco["episodes"]) == len(ds)
    return {
        "name": "mujoco_against_gazebo",
        "title": "The 25 sweep scenarios per drone count, flown in both "
                 "simulators",
        "columns": ["controller", "drones", "flights", "arrived in MuJoCo",
                    "arrived in Gazebo"],
        "rows": rows,
        "notes": ["MuJoCo is the simulator the policy was trained in, Gazebo "
                  "with PX4 the one it is tested in.",
                  "For 2 drones see mujoco/check_swarm_swarm_4.txt."],
    }


# ---------------------------------------------------------------- writing

def as_text(table):
    cells = [[str(c) for c in row] for row in table["rows"] if row is not None]
    columns = table["columns"]
    widths = [max(len(columns[i]), *(len(row[i]) for row in cells))
              for i in range(len(columns))]
    # A column of words is set to the left, a column of numbers to the right.
    left = [any(ch.isalpha() for row in cells for ch in row[i])
            for i in range(len(columns))]

    def line(values):
        return "  ".join(v.ljust(w) if l else v.rjust(w)
                         for v, w, l in zip(values, widths, left)).rstrip()

    out = [table["title"], "", line(columns), "-" * len(line(columns))]
    rows = list(table["rows"])
    while rows and rows[-1] is None:
        rows.pop()
    for row in rows:
        out.append("" if row is None else line([str(c) for c in row]))
    out.append("")
    out += table["notes"]
    return "\n".join(out)


def main():
    runs = {arm: gazebo_runs(arm) for arm in ARMS}
    drones = {arm: gazebo_drones(arm) for arm in ARMS}
    tables = [
        table_training(),
        table_arrivals(drones),
        table_arrivals_by_obstacles(drones),
        table_gain(runs),
        table_paired_drones(drones),
        table_flight(drones),
        table_flight_by_obstacles(drones),
        table_cost_of_avoiding(drones),
        table_clearance(drones),
        table_separation(drones),
        table_cpu(runs, drones),
        table_memory(),
        table_mujoco(mujoco_flights()),
        table_simulators(drones),
    ]

    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    text = []
    for number, table in enumerate(tables, start=1):
        text.append(f"Table {number}. " + as_text(table))
        with open(OUT / "tables" / f"{number:02d}_{table['name']}.csv", "w",
                  newline="") as f:
            writer = csv.writer(f)
            writer.writerow(table["columns"])
            writer.writerows(row for row in table["rows"] if row is not None)
    (OUT / "tables.txt").write_text("\n\n\n".join(text) + "\n")
    print("\n\n\n".join(text))
    print(f"\n-> {OUT / 'tables.txt'} and {len(tables)} CSV files in "
          f"{OUT / 'tables'}")


if __name__ == "__main__":
    main()
