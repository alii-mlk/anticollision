#!/usr/bin/env python3
"""Measure how fast the running Gazebo simulation is actually going.

The real time factor is how many seconds of simulated time the machine
produces per second of wall-clock time. At 1.0 the simulation keeps up with
reality. Below that it is running in slow motion, which is not by itself a
problem: the physics is unchanged and simulated time is still the honest
measure of a flight. It only becomes a problem when the autopilot does not
know about it, because a flight controller expecting sensor data every few
milliseconds of ITS clock will declare the sensors dead.

This is the measurement to take first when drones fall out of the sky without
touching anything, and after every change to the number of drones.

It also lists which processes are consuming the processor, which separates
"the machine is saturated" from "the machine is idle and something else is
wrong".

Usage:
  ./sim_speed.py                 # sample for 5 s and report
  ./sim_speed.py --window 20     # average over a longer window
  ./sim_speed.py --world default
"""

import argparse
import re
import subprocess
import sys
import time

# Processes worth naming in the breakdown: the simulator, the autopilots and
# the MAVLink bridges.
CPU_PATTERNS = ("gz sim", "ruby", "px4", "mavros_node", "parameter_bridge")


def gz_stats(world, timeout=6.0):
    """Return the text of one world statistics message, or None."""
    topic = f"/world/{world}/stats"
    for args in (["gz", "topic", "-e", "-t", topic, "-n", "1"],
                 ["gz", "topic", "-e", "-t", topic]):
        try:
            proc = subprocess.run(args, capture_output=True, text=True,
                                  timeout=timeout)
            if proc.stdout.strip():
                return proc.stdout
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            if out and out.strip():
                return out
        except FileNotFoundError:
            sys.exit("ERROR: 'gz' not found. Source the Gazebo environment "
                     "first, or run this from a shell that has it.")
    return None


def parse_duration(text, field):
    """Pull a seconds value out of a `field { sec: .. nsec: .. }` block."""
    match = re.search(field + r"\s*\{([^}]*)\}", text)
    if not match:
        return None
    body = match.group(1)
    sec = re.search(r"\bsec:\s*(-?\d+)", body)
    nsec = re.search(r"\bnsec:\s*(-?\d+)", body)
    if not sec:
        return None
    return int(sec.group(1)) + (int(nsec.group(1)) if nsec else 0) * 1e-9


def sample(world):
    text = gz_stats(world)
    if text is None:
        return None
    return {
        "sim": parse_duration(text, "sim_time"),
        "real": parse_duration(text, "real_time"),
        "paused": "paused: true" in text,
    }


def cpu_breakdown():
    """Per-process CPU share, busiest first."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pcpu,comm,args", "--no-headers"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []

    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pcpu, comm, args = parts
        try:
            pcpu = float(pcpu)
        except ValueError:
            continue
        if pcpu < 1.0:
            continue
        if any(p in args for p in CPU_PATTERNS):
            rows.append((pcpu, comm, args[:70]))
    rows.sort(reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", default="default",
                    help="Gazebo world name (PX4 SITL default: 'default')")
    ap.add_argument("--window", type=float, default=5.0,
                    help="seconds to average over")
    args = ap.parse_args()

    first = sample(args.world)
    if first is None or first["sim"] is None:
        print(f"No statistics on /world/{args.world}/stats.")
        print("Either no simulation is running, or the world has another "
              "name. List the worlds with:  gz topic -l | grep stats")
        return 1

    if first["paused"]:
        print("The simulation is PAUSED. Nothing will fly until it is running.")

    wall_before = time.monotonic()
    time.sleep(args.window)
    second = sample(args.world)
    wall_after = time.monotonic()

    if second is None or second["sim"] is None:
        print("The simulation stopped responding during the measurement.")
        return 1

    sim_delta = second["sim"] - first["sim"]
    wall_delta = wall_after - wall_before
    rtf = sim_delta / wall_delta if wall_delta > 0 else 0.0

    print(f"Simulated {sim_delta:.2f} s in {wall_delta:.2f} s of wall clock.")
    print(f"Real time factor: {rtf:.2f}")

    if sim_delta <= 0.01:
        print("  The simulation clock is not advancing at all.")
    elif rtf >= 0.9:
        print("  The machine is keeping up.")
    elif rtf >= 0.25:
        print("  Slow motion. Runs are valid but take "
              f"{1 / rtf:.1f}x longer in wall-clock time, and the autopilots "
              "must be following this clock rather than the wall clock.")
    else:
        print("  Very slow. Expect the autopilots to lose their sensor "
              "timing unless the simulation is deliberately slowed down "
              "with set_sim_speed.py.")

    rows = cpu_breakdown()
    if rows:
        total = sum(r[0] for r in rows)
        print()
        print(f"CPU used by the simulation processes: {total:.0f}% "
              f"(one fully busy core is 100%)")
        for pcpu, comm, argline in rows[:10]:
            print(f"  {pcpu:>6.1f}%  {comm:<16} {argline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
