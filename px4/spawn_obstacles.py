#!/usr/bin/env python3
"""Spawn a scenario's obstacles into the running Gazebo world.

Reads scenario.yaml (from gen_scenario_3d.py) and creates one static box per
obstacle through Gazebo's entity creation service, so PX4's own world file is
left untouched and scenarios can be swapped without restarting the simulator.

Existing obstacles from a previous scenario are removed first, so this can be
run repeatedly against the same running Gazebo.

Usage:
  ./spawn_obstacles.py --scenario scenarios3d/s1_n8_k1
  ./spawn_obstacles.py --scenario <dir> --world default
  ./spawn_obstacles.py --clear --world default        # remove obstacles only
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

NAME_PREFIX = "maze_obstacle_"
SERVICE_TIMEOUT_MS = 5000


def run_gz(args, timeout=15):
    """Run a gz command, returning (ok, output)."""
    try:
        proc = subprocess.run(["gz"] + args, capture_output=True, text=True,
                              timeout=timeout)
        return proc.returncode == 0, (proc.stdout + proc.stderr).strip()
    except FileNotFoundError:
        sys.exit("ERROR: 'gz' not found. Source the Gazebo environment first.")
    except subprocess.TimeoutExpired:
        return False, "timed out"


def parse_obstacles(path):
    """Minimal parser for the obstacle list written by gen_scenario_3d.py."""
    text = path.read_text()
    pattern = (r"x_min:\s*([-\d.]+),\s*x_max:\s*([-\d.]+),\s*"
               r"y_min:\s*([-\d.]+),\s*y_max:\s*([-\d.]+),\s*"
               r"z_min:\s*([-\d.]+),\s*z_max:\s*([-\d.]+)")
    obstacles = []
    for m in re.finditer(pattern, text):
        x0, x1, y0, y1, z0, z1 = (float(v) for v in m.groups())
        obstacles.append((x0, x1, y0, y1, z0, z1))
    return obstacles


def box_sdf(name, cx, cy, cz, sx, sy, sz):
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <pose>{cx:.3f} {cy:.3f} {cz:.3f} 0 0 0</pose>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
        <material>
          <ambient>0.85 0.35 0.1 1</ambient>
          <diffuse>0.85 0.35 0.1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def existing_obstacle_names(world):
    ok, out = run_gz(["model", "--list"])
    if not ok:
        return []
    return [line.strip().lstrip("- ").strip()
            for line in out.splitlines()
            if NAME_PREFIX in line]


def clear_obstacles(world):
    names = existing_obstacle_names(world)
    for name in names:
        ok, out = run_gz([
            "service", "-s", f"/world/{world}/remove",
            "--reqtype", "gz.msgs.Entity",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", str(SERVICE_TIMEOUT_MS),
            "--req", f'name: "{name}" type: MODEL',
        ])
        if not ok:
            print(f"  warning: could not remove {name}: {out}")
    if names:
        print(f"Removed {len(names)} obstacle(s) from a previous scenario.")
    return len(names)


def spawn_obstacles(world, obstacles, sdf_dir):
    """Spawn each obstacle from an SDF file next to the scenario.

    Two details matter here. The SDF is passed by filename rather than inline,
    because an inline SDF would have to go into a protobuf text string, which
    cannot contain the newlines an SDF document needs. And the files are kept
    rather than written to a temporary directory: the create service returns as
    soon as the request is acknowledged, while Gazebo reads the file on a later
    update, so deleting the files right after the loop loses the last obstacle.
    """
    sdf_dir.mkdir(parents=True, exist_ok=True)
    acknowledged = 0
    for i, (x0, x1, y0, y1, z0, z1) in enumerate(obstacles):
        cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
        sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
        name = f"{NAME_PREFIX}{i}"

        sdf_path = (sdf_dir / f"{name}.sdf").resolve()
        sdf_path.write_text(box_sdf(name, cx, cy, cz, sx, sy, sz))

        # Absolute path: the Gazebo server runs from its own working
        # directory (the PX4 tree), so a relative path resolves to nothing.
        ok, out = run_gz([
            "service", "-s", f"/world/{world}/create",
            "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", str(SERVICE_TIMEOUT_MS),
            "--req", f'sdf_filename: "{sdf_path}"',
        ])
        if ok and "data: true" in out.lower():
            acknowledged += 1
        else:
            print(f"  warning: obstacle {i} was not acknowledged: {out}")
    return acknowledged


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", type=Path,
                    help="scenario directory containing scenario.yaml")
    ap.add_argument("--world", default="default",
                    help="Gazebo world name (PX4 SITL default: 'default')")
    ap.add_argument("--clear", action="store_true",
                    help="only remove existing obstacles, do not spawn")
    args = ap.parse_args()

    if args.clear:
        clear_obstacles(args.world)
        return

    if not args.scenario:
        sys.exit("need --scenario <dir> (or --clear)")

    args.scenario = args.scenario.resolve()
    scenario_file = args.scenario / "scenario.yaml"
    if not scenario_file.exists():
        sys.exit(f"not found: {scenario_file}")

    obstacles = parse_obstacles(scenario_file)
    if not obstacles:
        sys.exit(f"no obstacles found in {scenario_file}")

    clear_obstacles(args.world)
    print(f"Spawning {len(obstacles)} obstacle(s) into world '{args.world}' ...")
    acknowledged = spawn_obstacles(args.world, obstacles,
                                   args.scenario / "obstacles")

    # Verify against the world itself. The create service acknowledges a
    # request before Gazebo has actually built the entity, so an acknowledged
    # spawn is not proof; poll until the world agrees or we give up.
    present = 0
    for _ in range(10):
        present = len(existing_obstacle_names(args.world))
        if present == len(obstacles):
            break
        time.sleep(1)

    print(f"Acknowledged {acknowledged}/{len(obstacles)}, "
          f"{present} present in world '{args.world}'.")
    if present != len(obstacles):
        print("ERROR: obstacle count in the world does not match the scenario.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
