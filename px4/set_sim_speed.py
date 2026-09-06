#!/usr/bin/env python3
"""Set the speed Gazebo aims to run PX4's SITL world at.

Why this exists. A machine that cannot simulate k drones at full speed does
not fail gracefully: Gazebo falls behind by an amount that varies from moment
to moment, sensor messages arrive late and unevenly, and PX4 reports
`Accel #0 fail: TIMEOUT`, loses its attitude estimate and drops the drone out
of the sky. The flight fails for reasons that have nothing to do with
navigation, which makes the run worthless.

Telling Gazebo to target a speed it can actually sustain replaces that with an
evenly slowed simulation. The physics is identical, every drone experiences
the same clock, and the run simply takes longer in wall-clock time. Simulated
time, distances, collisions and their ratio are all unaffected, so the results
mean the same thing they would on a faster machine.

This edits PX4's own world file, keeping the original alongside it. `reset`
puts it back.

Usage:
  ./set_sim_speed.py show
  ./set_sim_speed.py 0.3
  ./set_sim_speed.py 0.3 --max-step-size 0.008
  ./set_sim_speed.py reset
  PX4_DIR=~/somewhere/PX4-Autopilot ./set_sim_speed.py show
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

WORLD_RELPATH = "Tools/simulation/gz/worlds/default.sdf"
BACKUP_SUFFIX = ".px4-original"


def world_path():
    px4_dir = Path(os.environ.get("PX4_DIR", Path.home() / "px4/PX4-Autopilot"))
    path = px4_dir / WORLD_RELPATH
    if not path.is_file():
        sys.exit(f"ERROR: PX4's world file not found at {path}\n"
                 "Set PX4_DIR to the PX4-Autopilot directory.")
    return path


def current_values(text):
    rtf = re.search(r"<real_time_factor>\s*([\d.]+)\s*</real_time_factor>", text)
    step = re.search(r"<max_step_size>\s*([\d.]+)\s*</max_step_size>", text)
    return (float(rtf.group(1)) if rtf else None,
            float(step.group(1)) if step else None)


def set_element(text, tag, value):
    """Replace an element's value, or add it inside <physics> if absent."""
    pattern = rf"(<{tag}>)\s*[\d.]+\s*(</{tag}>)"
    if re.search(pattern, text):
        return re.sub(pattern, rf"\g<1>{value}\g<2>", text, count=1)

    physics = re.search(r"<physics\b[^>]*>", text)
    if not physics:
        sys.exit(f"ERROR: no <physics> element in the world file, cannot add "
                 f"<{tag}>.")
    insert_at = physics.end()
    return text[:insert_at] + f"\n      <{tag}>{value}</{tag}>" + text[insert_at:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("speed",
                    help="target real time factor (e.g. 0.3), "
                         "'show' to report the current setting, "
                         "or 'reset' to restore PX4's original world")
    ap.add_argument("--max-step-size", type=float, default=None,
                    help="physics step in seconds. Larger steps cost less "
                         "processor time but coarsen the physics; leave this "
                         "alone unless slowing the world down is not enough.")
    args = ap.parse_args()

    path = world_path()
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    text = path.read_text()

    if args.speed == "show":
        rtf, step = current_values(text)
        print(f"World file: {path}")
        print(f"  real_time_factor: {rtf if rtf is not None else 'not set (1.0)'}")
        print(f"  max_step_size:    {step if step is not None else 'not set'}")
        print(f"  original saved:   {'yes' if backup.exists() else 'no'}")
        return 0

    if args.speed == "reset":
        if not backup.exists():
            print("Nothing to restore: no original was saved, so the world "
                  "file has not been changed by this script.")
            return 0
        shutil.copyfile(backup, path)
        print(f"Restored PX4's original world file from {backup.name}.")
        return 0

    try:
        speed = float(args.speed)
    except ValueError:
        sys.exit(f"ERROR: '{args.speed}' is not a speed, 'show' or 'reset'.")
    if not 0.01 <= speed <= 10.0:
        sys.exit("ERROR: speed must be between 0.01 and 10.")

    # Keep PX4's original exactly once, so reset always goes back to the file
    # PX4 shipped rather than to whatever the previous run left behind.
    if not backup.exists():
        shutil.copyfile(path, backup)
        print(f"Saved PX4's original world file as {backup.name}.")

    text = set_element(text, "real_time_factor", f"{speed:g}")
    if args.max_step_size is not None:
        text = set_element(text, "max_step_size", f"{args.max_step_size:g}")
    path.write_text(text)

    rtf, step = current_values(text)
    print(f"World file: {path}")
    print(f"  real_time_factor -> {rtf}")
    print(f"  max_step_size    -> {step}")
    print()
    print("Restart the simulation for this to take effect. A run will now "
          f"take about {1 / speed:.1f}x longer in wall-clock time, while its "
          "simulated duration stays the same.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
