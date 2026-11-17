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

Memory is sampled every two seconds while the drones fly, for the simulation
side (Gazebo, the PX4 instances, the MAVROS nodes and the small helpers around
them) and separately for the mission and monitor nodes, which is where the
policy's network lives. Three figures are recorded:

  resident      the peak of the summed resident memory of each side. A page
                shared between processes is counted once per process, so this
                is an upper bound.
  proportional  the same processes with every shared page divided among the
                processes that share it, per part of the stack. These add up
                without counting anything twice.
  free          the machine's available memory before the stack is started
                and its lowest value during the flight.

Every counted process is listed in memory.txt in the run's directory.

Usage:
  ./run_batch_3d.py
  ./run_batch_3d.py --n 4 8 --k 1 2 --seeds 1 2 3
  ./run_batch_3d.py --n 8 --k 3 --seeds 1 --timeout 240
"""

import argparse
import importlib.util
import os
import re
import signal
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
# The scripts that only start and wait for the others. Their paths contain
# "px4", so without this they would be counted as part of the simulator.
LAUNCHERS = ("run_batch_3d.py", "run_multi_mission.py")
# The parts of the stack whose memory is measured, see part_of().
SIMULATION_PARTS = ("gazebo", "px4", "mavros", "helpers")
CONTROL_PARTS = ("mission", "monitor")
MEMORY_PERIOD_S = 2.0
PROC = Path("/proc")


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
        if any(p in args for p in LAUNCHERS):
            continue
        if any(p in args for p in CPU_PATTERNS):
            total += float(secs)
    return total


def part_of(args):
    """The part of the stack a command line belongs to, or None.

    Decided by the program that runs. Every script here lives in a folder
    called px4 and PX4 itself in ~/px4, so looking for that text anywhere in
    the command line also counts whatever else has one of these files open,
    a text editor for example.
    """
    words = args.split()
    if not words or any(p in args for p in LAUNCHERS):
        return None
    program = words[0].rsplit("/", 1)[-1]
    python = program.startswith("python")
    if python and "run_mission.py" in args:
        return "mission"
    if python and "collision_monitor_3d.py" in args:
        return "monitor"
    if program == "px4":
        return "px4"
    if program == "mavros_node":
        return "mavros"
    if "gz sim" in args:
        return "gazebo"
    # What is left are the small processes around those: the clock bridge, the
    # ground station heartbeats, the "ros2 run" wrappers and the shells of
    # start_px4_multi.sh that stay as their parents.
    if program == "parameter_bridge":
        return "helpers"
    if python and any(h in args for h in
                      ("gcs_heartbeat.py", "mavros_node", "parameter_bridge")):
        return "helpers"
    if program in ("bash", "sh") and "start_px4_multi.sh" in args:
        return "helpers"
    return None


def parse_processes(text):
    """Lines of  pid rss args  as (pid, resident MB, command line)."""
    found = []
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            found.append((int(parts[0]), int(parts[1]) / 1024.0, parts[2]))
    return found


def processes():
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,rss,args", "--no-headers"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    return parse_processes(out)


def stack_processes(found=None):
    """The stack's own processes: (part, pid, resident MB, command line)."""
    found = processes() if found is None else found
    tagged = [(part_of(args), pid, mb, args) for pid, mb, args in found]
    return [row for row in tagged if row[0]]


def outside_processes(found):
    """Processes the old, looser match counted and part_of() does not."""
    return [(pid, mb, args) for pid, mb, args in found
            if part_of(args) is None
            and not any(p in args for p in LAUNCHERS)
            and any(p in args for p in CPU_PATTERNS)]


def proportional_mb(pid, proc=PROC):
    """Proportional memory of one process in MB, or None if it cannot be read.

    Every page is divided by the number of processes that share it, so these
    values add up without counting a shared library once per process.
    """
    try:
        for line in (proc / str(pid) / "smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def side_mb(found, parts):
    return sum(mb for part, _, mb, _ in found if part in parts)


def write_memory_report(path, snapshot, outside):
    """One line per counted process, from the fullest sample of the flight."""
    lines = [f"{'part':<9}{'pid':>8}{'resident MB':>13}{'proportional MB':>17}"
             "  command"]
    for part, pid, mb, pss, args in sorted(snapshot, key=lambda r: (r[0], r[1])):
        shown = f"{pss:.1f}" if pss is not None else "-"
        lines.append(f"{part:<9}{pid:>8}{mb:>13.1f}{shown:>17}  {args[:140]}")
    if outside:
        lines.append("")
        lines.append("not counted, these only mention px4, gz sim, ruby or "
                     "mavros_node in their command line:")
        for pid, mb, args in outside:
            lines.append(f"{'':<9}{pid:>8}{mb:>13.1f}{'':>17}  {args[:140]}")
    path.write_text("\n".join(lines) + "\n")


def free_memory_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


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
    # The stop script asks the processes to end. One that does not answer
    # would stay under every later run of the sweep and take memory and
    # network ports with it, so whatever is still there now is killed and
    # written down.
    for part, pid, mb, args in stack_processes():
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
        note = (f"{datetime.now():%Y-%m-%d %H:%M:%S} killed leftover {part} "
                f"(pid {pid}, {mb:.0f} MB): {args[:120]}")
        print(f"    {note}")
        try:
            with open(HERE / "logs" / "leftovers.txt", "a") as f:
                f.write(note + "\n")
        except OSError:
            pass
    # PX4's own flight logs. start_px4_multi.sh switches them off, but an
    # instance only reads that setting when it boots, so the first run of a
    # sweep can still write some. Nothing here uses them, and left alone they
    # would fill the disk over a long sweep.
    for ulg in PX4_BUILD_DIR.rglob("*.ulg"):
        ulg.unlink(missing_ok=True)


def wait_for_ready(log_dir, k, timeout=120):
    """Wait until every PX4 instance reports it is ready to fly."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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
    free_before = free_memory_mb()

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
    wall_before = time.monotonic()

    # The mission runs in the background so memory can be sampled while the
    # drones fly. Its output goes straight to the log file.
    sim_peak = control_peak = fullest = 0.0
    snapshot, outside = [], []
    sample_failed = False
    free_low = free_memory_mb()
    with open(run_dir / "multi_mission.log", "w") as log:
        mission = subprocess.Popen(
            [sys.executable, str(MULTI), "--scenario", str(scenario),
             "--out", str(run_dir / "run"), "--timeout", str(timeout),
             "--wall-timeout", str(wall_timeout),
             *(["--policy", str(policy)] if policy else [])],
            stdout=log, stderr=subprocess.STDOUT, text=True)
        while mission.poll() is None:
            # Nothing in the measurement may stop a flight that is under way.
            try:
                everything = processes()
                found = stack_processes(everything)
                sim_mb = side_mb(found, SIMULATION_PARTS)
                control_mb = side_mb(found, CONTROL_PARTS)
                sim_peak = max(sim_peak, sim_mb)
                control_peak = max(control_peak, control_mb)
                # The proportional figures are read from the fullest sample.
                # Reading them costs more than ps, so it is only done again
                # once the total has grown by another percent.
                if sim_mb + control_mb > 1.01 * fullest:
                    fullest = sim_mb + control_mb
                    snapshot = [(part, pid, mb, proportional_mb(pid), args)
                                for part, pid, mb, args in found]
                    outside = outside_processes(everything)
                free_low = min(free_low, free_memory_mb())
            except Exception as e:
                if not sample_failed:
                    print(f"    memory sample failed: {e}")
                sample_failed = True
            try:
                mission.wait(timeout=MEMORY_PERIOD_S)
            except subprocess.TimeoutExpired:
                pass

    # Read CPU before teardown, while the processes still exist.
    cpu_used = max(0.0, cpu_seconds() - cpu_before)
    wall_used = time.monotonic() - wall_before

    # Proportional memory per part, when the kernel reports it.
    proportional = {}
    for part, _, _, pss, _ in snapshot:
        if pss is not None:
            proportional[part] = proportional.get(part, 0.0) + pss
    try:
        write_memory_report(run_dir / "memory.txt", snapshot, outside)
    except Exception as e:
        print(f"    memory report not written: {e}")

    summary = read_scalars(run_dir / "run" / "summary.yaml")
    if summary:
        summary_path = run_dir / "run" / "summary.yaml"
        with open(summary_path, "a") as f:
            f.write(f"cpu_time: {cpu_used:.1f}\n")
            f.write(f"wall_time_measured: {wall_used:.1f}\n")
            f.write(f"memory_simulation_mb: {sim_peak:.0f}\n")
            f.write(f"memory_controller_mb: {control_peak:.0f}\n")
            f.write(f"memory_free_before_mb: {free_before:.0f}\n")
            f.write(f"memory_free_min_mb: {free_low:.0f}\n")
            if proportional:
                for part in SIMULATION_PARTS + CONTROL_PARTS:
                    f.write(f"memory_pss_{part}_mb: "
                            f"{proportional.get(part, 0.0):.0f}\n")
        (run_dir / "status.txt").write_text("OK\n")
        shared_once = (f", {sum(proportional.values()):.0f} MB with shared "
                       "pages counted once" if proportional else "")
        print(f"    reached {summary.get('reached', '?')}/{k}, "
              f"collisions {summary.get('collisions', '?')}, "
              f"ratio {summary.get('ratio', '?')}, "
              f"sim {summary.get('simulation_time', '?')}s, "
              f"rtf {summary.get('real_time_factor', '?')}, "
              f"cpu {cpu_used:.0f}s, "
              f"ram {sim_peak:.0f}+{control_peak:.0f} MB{shared_once}")
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
        # Only looked up with find_spec. Importing it would load PyTorch into
        # this process for nothing.
        if importlib.util.find_spec("stable_baselines3") is None:
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

    started = time.monotonic()
    try:
        for n in args.n:
            for k in args.k:
                for seed in args.seeds:
                    run_one(n, k, seed, batch_dir, args.timeout, wall_timeout,
                            args.policy)
    except KeyboardInterrupt:
        print("\ninterrupted; cleaning up")
        teardown()

    mins = (time.monotonic() - started) / 60
    print()
    print(f"Sweep finished in {mins:.0f} min.")
    print(f"Metrics: python3 compute_metrics_3d.py {batch_dir}")


if __name__ == "__main__":
    main()
