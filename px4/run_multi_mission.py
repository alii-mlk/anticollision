#!/usr/bin/env python3
"""Run a whole multi-drone scenario: k missions and k collision monitors.

Starts one collision monitor and one mission runner per drone, waits until
every drone has either reached its target or been destroyed, then stops the
monitors and writes a summary with the metrics from the evaluation plan:
number of collisions, summed Euclidean distance, summed travelled distance,
their ratio, and the simulation time.

Assumes the drones are already flying, i.e. start_px4_multi.sh has been run
for the same scenario and the obstacles have been spawned.

Usage:
  ./run_multi_mission.py --scenario scenarios3d/s11_n6_k2
  ./run_multi_mission.py --scenario <dir> --out runs/run1 --timeout 240
"""

import argparse
import math
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MONITOR = HERE / "collision_monitor_3d.py"
MISSION = HERE / "run_mission.py"
READY_MARKER = HERE / "logs" / "stack_ready"


def wait_for_stack(scenario, timeout=600.0):
    """Block until start_px4_multi.sh reports the stack is fully up.

    Bringing k drones up takes minutes, and its last step (reading every SITL
    parameter back from each autopilot) prints nothing for most of that time.
    Flying before it finishes looks like it works: the drones arm and take off,
    then the battery failsafe lands them mid-mission because SIM_BAT_MIN_PCT
    was applied too late. The run is wasted and the cause is not obvious from
    the results, so refuse to start rather than produce numbers that mean
    nothing.
    """
    deadline = time.time() + timeout
    announced = False
    while time.time() < deadline:
        if READY_MARKER.exists():
            marker = READY_MARKER.read_text()
            if str(scenario) in marker:
                return True
            print(f"  the running stack is for another scenario:\n{marker}")
            return False
        if not announced:
            print("Waiting for start_px4_multi.sh to finish bringing the "
                  "stack up ...")
            announced = True
        time.sleep(5)
    print("The stack never reported ready. Start it with "
          "./start_px4_multi.sh --scenario <dir> and wait for READY.")
    return False


def parse_drones(path):
    text = path.read_text()
    pattern = (r"start:\s*\{x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*z:\s*([-\d.]+)\}\s*"
               r"target:\s*\{x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*z:\s*([-\d.]+)\}")
    drones = []
    for m in re.finditer(pattern, text):
        v = [float(x) for x in m.groups()]
        drones.append({"start": tuple(v[0:3]), "target": tuple(v[3:6])})
    return drones


def read_yaml_scalars(path):
    """Small reader for the flat key: value files the nodes write."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        m = re.match(r"^(\w+):\s*(.+)$", line.strip())
        if m:
            values[m.group(1)] = m.group(2).strip()
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: <scenario>/run)")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-drone mission timeout in SIMULATED seconds")
    ap.add_argument("--wall-timeout", type=float, default=0.0,
                    help="wall-clock safety limit per drone in seconds "
                         "(default: 5 x --timeout). Runs with several drones "
                         "on a machine that cannot simulate them at full "
                         "speed take proportionally longer in wall time; this "
                         "only catches a simulation that has stopped.")
    ap.add_argument("--no-wait", action="store_true",
                    help="start even if the stack has not reported ready")
    ap.add_argument("--policy", type=Path, default=None,
                    help="fly every drone with this trained policy "
                         "(an rl/runs/<name> directory); default: straight")
    args = ap.parse_args()

    if args.policy is not None:
        # The missions run with this same interpreter, so check here, once,
        # rather than have every drone fail on its own.
        try:
            import stable_baselines3  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            sys.exit("--policy needs the RL venv. Run  source rl/env.sh  first.")
        args.policy = args.policy.expanduser().resolve()
        if not args.policy.exists():
            sys.exit(f"policy not found: {args.policy}")

    scenario = args.scenario.resolve()
    scenario_file = scenario / "scenario.yaml"
    if not scenario_file.exists():
        sys.exit(f"not found: {scenario_file}")

    drones = parse_drones(scenario_file)
    if not drones:
        sys.exit(f"no drones found in {scenario_file}")

    if not args.no_wait and not wait_for_stack(scenario):
        sys.exit(1)

    out_root = (args.out or (scenario / "run")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    k = len(drones)
    wall_timeout = args.wall_timeout or args.timeout * 5.0

    print(f"Scenario: {scenario}")
    print(f"Drones:   {k}")
    print(f"Output:   {out_root}")
    print(f"Timeout:  {args.timeout:.0f} s simulated, "
          f"{wall_timeout:.0f} s wall clock")
    print(f"Flying:   {'policy ' + str(args.policy) if args.policy else 'straight (no avoidance)'}")
    print()

    monitors, missions, out_dirs = [], [], []
    started = time.time()

    # Results from a previous run of the same scenario are removed first. A
    # drone that fails to start writes nothing, and the aggregation below would
    # then happily read the last run's numbers and report them as this run's.
    for stale in list(out_root.glob("drone*/*.yaml")) + [out_root / "summary.yaml"]:
        stale.unlink(missing_ok=True)

    # Monitors first, so nothing is missed while the missions come up.
    for i in range(k):
        out_dir = out_root / f"drone{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.append(out_dir)

        monitors.append(subprocess.Popen(
            [sys.executable, str(MONITOR),
             "--scenario", str(scenario),
             "--drone-id", str(i),
             "--mavros-ns", f"/uav{i}",
             "--out", str(out_dir)],
            stdout=open(out_dir / "monitor.log", "w"),
            stderr=subprocess.STDOUT))
        print(f"  monitor {i} started")

    time.sleep(3)

    policy_args = ["--policy", str(args.policy)] if args.policy else []
    for i in range(k):
        missions.append(subprocess.Popen(
            [sys.executable, str(MISSION),
             "--scenario", str(scenario),
             "--drone-id", str(i),
             "--mavros-ns", f"/uav{i}",
             "--timeout", str(args.timeout),
             "--wall-timeout", str(wall_timeout),
             "--out", str(out_dirs[i]), *policy_args],
            stdout=open(out_dirs[i] / "mission.log", "w"),
            stderr=subprocess.STDOUT))
        print(f"  mission {i} started")

    print()
    print("Flying. The run ends when every drone has reached its target or "
          "been destroyed.")

    try:
        # The run is over when all missions have exited. Each mission enforces
        # its own wall-clock limit, so this only has to outlast that.
        for i, proc in enumerate(missions):
            proc.wait(timeout=wall_timeout + 120)
            print(f"  drone {i} finished")
    except subprocess.TimeoutExpired:
        print("  a mission overran its timeout; stopping it")
        for proc in missions:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
    except KeyboardInterrupt:
        print("  interrupted; stopping")
        for proc in missions:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)

    # Give the missions a moment to write their results, then stop monitors.
    time.sleep(2)
    for proc in monitors:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
    for proc in monitors:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    wall_time = time.time() - started

    # --- aggregate ---
    rows = []
    for i, out_dir in enumerate(out_dirs):
        mission = read_yaml_scalars(out_dir / "mission.yaml")
        outcome = read_yaml_scalars(out_dir / "outcome.yaml")
        rows.append({
            "id": i,
            "outcome": mission.get("outcome", "missing"),
            "euclidean": float(mission.get("euclidean_distance", 0.0)),
            "travelled": float(mission.get("travelled_distance", 0.0)),
            "time": float(mission.get("simulation_time", 0.0)),
            "wall": float(mission.get("wall_time", 0.0)),
            "cpu": float(mission.get("cpu_time", 0.0)),
            "destroyed": outcome.get("destroyed", "unknown") == "true",
            "min_clearance": outcome.get("min_clearance", "null"),
            "collided_with": outcome.get("collided_with", "-"),
        })

    total_euclid = sum(r["euclidean"] for r in rows)
    total_travelled = sum(r["travelled"] for r in rows)
    collisions = sum(1 for r in rows if r["destroyed"])
    reached = sum(1 for r in rows if r["outcome"] == "reached")
    ratio = (total_travelled / total_euclid) if total_euclid > 0 else 0.0

    # The ratio above is over every drone, which is the metric as specified,
    # but on its own it misleads once drones start colliding: a destroyed drone
    # contributes the distance it managed against the full distance it was
    # asked to cover, so the ratio drops below 1 and reads like a shortcut.
    # This second ratio covers only the drones that actually arrived, and is
    # the one that says how efficient a completed path was.
    arrived = [r for r in rows if r["outcome"] == "reached"]
    euclid_arrived = sum(r["euclidean"] for r in arrived)
    ratio_reached = (sum(r["travelled"] for r in arrived) / euclid_arrived
                     if euclid_arrived > 0 else 0.0)

    hit_drone = sum(1 for r in rows if r["collided_with"] == "another drone")
    hit_obstacle = sum(1 for r in rows if r["collided_with"] == "an obstacle")

    # The run lasts as long as its slowest drone, so that is the run's
    # simulation time. Comparing it against the wall clock gives the real time
    # factor the machine actually achieved, which is the number that says
    # whether this many drones can be simulated here at full speed.
    sim_time = max((r["time"] for r in rows), default=0.0)
    # Processor time of the mission nodes themselves, i.e. the controller:
    # negligible flying straight, the neural network's cost with the policy.
    controller_cpu = sum(r["cpu"] for r in rows)
    slowest_wall = max((r["wall"] for r in rows), default=0.0)
    rtf = (sim_time / slowest_wall) if slowest_wall > 0 else 0.0

    print()
    print(f"{'drone':>5} {'outcome':>10} {'euclid(m)':>10} "
          f"{'travelled(m)':>13} {'time(s)':>8} {'clearance(m)':>13} "
          f"{'hit':>14}")
    print("-" * 79)
    for r in rows:
        print(f"{r['id']:>5} {r['outcome']:>10} {r['euclidean']:>10.2f} "
              f"{r['travelled']:>13.2f} {r['time']:>8.1f} "
              f"{r['min_clearance']:>13} {r['collided_with']:>14}")
    print("-" * 79)
    print(f"reached {reached}/{len(rows)}, collisions {collisions} "
          f"({hit_obstacle} with obstacles, {hit_drone} between drones), "
          f"summed euclidean {total_euclid:.2f} m, "
          f"summed travelled {total_travelled:.2f} m, ratio {ratio:.3f} "
          f"(over the {len(arrived)} that arrived: {ratio_reached:.3f})")
    print(f"simulation time {sim_time:.1f} s, wall time {wall_time:.1f} s, "
          f"real time factor {rtf:.2f}")

    (out_root / "summary.yaml").write_text(
        f"controller: {'policy' if args.policy else 'straight'}\n"
        f"policy: {args.policy or '-'}\n"
        f"k_drones: {len(rows)}\n"
        f"reached: {reached}\n"
        f"collisions: {collisions}\n"
        f"collisions_with_obstacles: {hit_obstacle}\n"
        f"collisions_between_drones: {hit_drone}\n"
        f"euclidean_distance_total: {total_euclid:.3f}\n"
        f"travelled_distance_total: {total_travelled:.3f}\n"
        f"ratio: {ratio:.3f}\n"
        f"ratio_reached: {ratio_reached:.3f}\n"
        f"simulation_time: {sim_time:.2f}\n"
        f"real_time_factor: {rtf:.3f}\n"
        f"wall_time: {wall_time:.2f}\n"
        f"controller_cpu_time: {controller_cpu:.2f}\n"
        "drones:\n" +
        "".join(
            f"  - {{id: {r['id']}, outcome: {r['outcome']}, "
            f"euclidean: {r['euclidean']:.3f}, travelled: {r['travelled']:.3f}, "
            f"time: {r['time']:.2f}, min_clearance: {r['min_clearance']}, "
            f"collided_with: {r['collided_with']}}}\n"
            for r in rows)
    )
    print()
    print(f"Summary written to {out_root}/summary.yaml")


if __name__ == "__main__":
    main()
