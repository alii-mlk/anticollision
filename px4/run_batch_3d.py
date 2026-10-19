#!/usr/bin/env python3
"""Unattended sweep over (n obstacles, k drones) configurations.

For every combination of obstacle count, drone count and seed it generates a
scenario, brings up the simulator and k PX4 instances, spawns the obstacles,
flies all drones to their targets, records the results, and tears everything
down before the next run. A run that fails is recorded and the sweep
continues.

For each configuration it reports the metrics from the evaluation plan:
number of collisions, Euclidean distance, travelled distance, the ratio
between them, simulation time, and CPU time.

CPU time is the processor time consumed by the simulator and autopilots
during the run, summed over the Gazebo, PX4 and MAVROS processes. It is a
cost measure, and it grows with the number of drones.

Usage:
  ./run_batch_3d.py
  ./run_batch_3d.py --n 4 8 --k 1 2 --seeds 1 2 3
  ./run_batch_3d.py --n 8 --k 3 --seeds 1 --timeout 240
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE / "gen_scenario_3d.py"
START = HERE / "start_px4_multi.sh"
SPAWN = HERE / "spawn_obstacles.py"
MULTI = HERE / "run_multi_mission.py"
STOP = HERE / "stop_px4_sitl.sh"

# Processes whose CPU time counts as the cost of running a scenario.
CPU_PATTERNS = ("px4", "gz sim", "ruby", "mavros_node")


def sh(cmd, **kwargs):
    return subprocess.run(cmd, shell=isinstance(cmd, str), **kwargs)


def cpu_seconds():
    """Total CPU seconds used so far by the simulator and autopilot processes."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "cputimes,args", "--no-headers"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return 0.0

    total = 0.0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        secs, args = parts
        if not secs.isdigit():
            continue
        if any(p in args for p in CPU_PATTERNS):
            total += float(secs)
    return total


def read_scalars(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        m = re.match(r"^(\w+):\s*(.+)$", line.strip())
        if m:
            values[m.group(1)] = m.group(2).strip()
    return values


PX4_BUILD_DIR = Path.home() / "px4" / "PX4-Autopilot" / "build"


def teardown():
    sh([str(STOP)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    # PX4's own flight logs. start_px4_multi.sh switches them off, but an
    # instance only reads that setting when it boots, so the first run of a
    # sweep can still write some. Nothing here uses them, and left alone they
    # would fill the disk over a long sweep.
    for ulg in PX4_BUILD_DIR.rglob("*.ulg"):
        ulg.unlink(missing_ok=True)


def wait_for_ready(log_dir, k, timeout=120):
    """Wait until every PX4 instance reports it is ready to fly."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready = 0
        for i in range(k):
            log = log_dir / f"PX4_instance_{i}.txt"
            if log.exists() and "Ready for takeoff" in log.read_text(
                    errors="replace"):
                ready += 1
        if ready == k:
            return True
        time.sleep(3)
    return False


def run_one(n, k, seed, batch_dir, timeout, wall_timeout, policy=None):
    run_dir = batch_dir / f"n{n}_k{k}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    scenario = run_dir / "scenario"

    print(f"=== n={n} k={k} seed={seed} -> {run_dir.name}")

    gen = sh([sys.executable, str(GEN), "--n-obstacles", str(n),
              "--k-drones", str(k), "--seed", str(seed),
              "--out", str(scenario)],
             capture_output=True, text=True)
    if gen.returncode != 0:
        (run_dir / "status.txt").write_text("GEN_FAIL\n")
        print(f"    scenario generation failed: {gen.stderr.strip()}")
        return

    teardown()

    env_log_dir = HERE / "logs"
    for f in env_log_dir.glob("PX4_instance_*.txt"):
        f.unlink(missing_ok=True)

    start = sh(f"BATCH=1 {START} --scenario {scenario}",
               capture_output=True, text=True)
    if start.returncode != 0:
        (run_dir / "status.txt").write_text("START_FAIL\n")
        print("    stack failed to start")
        teardown()
        return

    if not wait_for_ready(env_log_dir, k):
        (run_dir / "status.txt").write_text("NOT_READY\n")
        print("    not all drones became ready")
        teardown()
        return

    spawn = sh([sys.executable, str(SPAWN), "--scenario", str(scenario)],
               capture_output=True, text=True)
    if spawn.returncode != 0:
        (run_dir / "status.txt").write_text("SPAWN_FAIL\n")
        print(f"    obstacle spawn failed: {spawn.stdout.strip()}")
        teardown()
        return

    cpu_before = cpu_seconds()
    wall_before = time.time()

    mission = sh([sys.executable, str(MULTI), "--scenario", str(scenario),
                  "--out", str(run_dir / "run"), "--timeout", str(timeout),
                  "--wall-timeout", str(wall_timeout),
                  *(["--policy", str(policy)] if policy else [])],
                 capture_output=True, text=True)
    (run_dir / "multi_mission.log").write_text(
        mission.stdout + "\n" + mission.stderr)

    # Read CPU before teardown, while the processes still exist.
    cpu_used = max(0.0, cpu_seconds() - cpu_before)
    wall_used = time.time() - wall_before

    summary = read_scalars(run_dir / "run" / "summary.yaml")
    if summary:
        summary_path = run_dir / "run" / "summary.yaml"
        with open(summary_path, "a") as f:
            f.write(f"cpu_time: {cpu_used:.1f}\n")
            f.write(f"wall_time_measured: {wall_used:.1f}\n")
        (run_dir / "status.txt").write_text("OK\n")
        print(f"    reached {summary.get('reached', '?')}/{k}, "
              f"collisions {summary.get('collisions', '?')}, "
              f"ratio {summary.get('ratio', '?')}, "
              f"sim {summary.get('simulation_time', '?')}s, "
              f"rtf {summary.get('real_time_factor', '?')}, "
              f"cpu {cpu_used:.0f}s")
    else:
        (run_dir / "status.txt").write_text("NO_SUMMARY\n")
        print("    run produced no summary")

    teardown()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, nargs="+", default=[4, 8, 12],
                    help="obstacle counts to sweep")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3],
                    help="drone counts to sweep")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="random seeds per configuration")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-drone mission timeout in SIMULATED seconds")
    ap.add_argument("--wall-timeout", type=float, default=0.0,
                    help="wall-clock safety limit per drone in seconds "
                         "(default: 5 x --timeout). Raise it when the "
                         "simulation runs well below real time.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--policy", type=Path, default=None,
                    help="fly every drone with this trained policy "
                         "(an rl/runs/<name> directory); default: straight. "
                         "Needs the RL venv: source rl/env.sh")
    args = ap.parse_args()

    if args.policy is not None:
        args.policy = args.policy.expanduser().resolve()
        try:
            import stable_baselines3  # noqa: F401
        except ImportError:
            sys.exit("--policy needs the RL venv. Run  source rl/env.sh  first.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "policy" if args.policy else "straight"
    batch_dir = (args.out or (HERE / "runs3d" / f"batch_{stamp}_{label}")).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.n) * len(args.k) * len(args.seeds)
    print(f"Flying: {'policy ' + str(args.policy) if args.policy else 'straight (no avoidance)'}")
    print(f"Sweep: n in {args.n}, k in {args.k}, seeds {args.seeds}")
    print(f"{total} runs -> {batch_dir}")
    print()

    wall_timeout = args.wall_timeout or args.timeout * 5.0

    started = time.time()
    try:
        for n in args.n:
            for k in args.k:
                for seed in args.seeds:
                    run_one(n, k, seed, batch_dir, args.timeout, wall_timeout,
                            args.policy)
    except KeyboardInterrupt:
        print("\ninterrupted; cleaning up")
        teardown()

    mins = (time.time() - started) / 60
    print()
    print(f"Sweep finished in {mins:.0f} min.")
    print(f"Metrics: python3 compute_metrics_3d.py {batch_dir}")


if __name__ == "__main__":
    main()
