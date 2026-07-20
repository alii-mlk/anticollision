#!/usr/bin/env python3
"""Random scenario generator for the drone Nav2 evaluation.

Picks a random start point, a random goal point, and N randomly placed
axis-aligned box obstacles, then writes BOTH artifacts from this single
source so they can never disagree:

  <out>/world.sdf       -- Gazebo world (obstacles + drone spawned at start)
  <out>/scenario.yaml   -- machine-readable scenario (read by virtual_lidar.py,
                           hit_monitor.py, send_nav2_goal.sh, batch tooling)

Usage:
  ./gen_scenario.py --n-obstacles 4 --seed 42
  ./gen_scenario.py --n-obstacles 8 --seed 3 --out scenarios/hard_8
  ./gen_scenario.py --n-obstacles 6 --seed 1 --obstacle-speed 0.5   # moving obstacles

With --obstacle-speed > 0 each obstacle gets a random direction at that speed
(bouncing off the workspace bounds; motion is integrated by obstacle_mover.py).
Moving obstacles are NOT placed in the Gazebo world: the whole sensing chain
(virtual lidar, hit monitor) tracks the moving ground truth published on
/obstacles, and a physical model frozen at its spawn pose would let the drone
crash into a position the obstacle has virtually left. RViz shows the moving
boxes via the /obstacles markers.

No third-party dependencies (YAML is written directly); safe to run on any
machine with Python 3.

Limitations (kept in sync with virtual_lidar.py / hit_monitor.py):
obstacles are axis-aligned boxes, tall enough to intersect the flight
altitude, and static.
"""

import argparse
import math
import random
import sys
from pathlib import Path

# Workspace and placement rules. Global costmap covers [-15,15]^2, so keep
# everything comfortably inside it.
BOUNDS = 9.0                 # scenario elements within [-BOUNDS, BOUNDS]^2
MIN_START_GOAL_DIST = 8.0    # professor's metric needs a meaningful flight
CLEAR_RADIUS = 1.8           # obstacle-free zone around start and goal
OBSTACLE_GAP = 0.4           # minimum gap between obstacles
SIZE_X = (0.5, 2.0)          # obstacle footprint size ranges (m)
SIZE_Y = (0.5, 4.0)
HEIGHT = (2.0, 3.0)          # must intersect flight altitude (z = 1.0)
FLIGHT_Z = 1.0
MAX_TRIES = 2000


def rect_circle_clear(rect, cx, cy, radius):
    """True if circle (cx,cy,radius) does NOT touch rect (x0,x1,y0,y1)."""
    x0, x1, y0, y1 = rect
    nx = min(max(cx, x0), x1)
    ny = min(max(cy, y0), y1)
    return math.hypot(cx - nx, cy - ny) > radius


def rects_disjoint(a, b, gap):
    return (a[1] + gap < b[0] or b[1] + gap < a[0] or
            a[3] + gap < b[2] or b[3] + gap < a[2])


def generate(n_obstacles, rng):
    sx = rng.uniform(-BOUNDS, BOUNDS)
    sy = rng.uniform(-BOUNDS, BOUNDS)
    for _ in range(MAX_TRIES):
        gx = rng.uniform(-BOUNDS, BOUNDS)
        gy = rng.uniform(-BOUNDS, BOUNDS)
        if math.hypot(gx - sx, gy - sy) >= MIN_START_GOAL_DIST:
            break
    else:
        sys.exit("could not place goal far enough from start")

    obstacles = []
    for _ in range(n_obstacles):
        for _ in range(MAX_TRIES):
            w = rng.uniform(*SIZE_X)
            d = rng.uniform(*SIZE_Y)
            cx = rng.uniform(-BOUNDS + w / 2, BOUNDS - w / 2)
            cy = rng.uniform(-BOUNDS + d / 2, BOUNDS - d / 2)
            rect = (cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2)
            if not rect_circle_clear(rect, sx, sy, CLEAR_RADIUS):
                continue
            if not rect_circle_clear(rect, gx, gy, CLEAR_RADIUS):
                continue
            if all(rects_disjoint(rect, o["rect"], OBSTACLE_GAP) for o in obstacles):
                obstacles.append({"rect": rect, "height": rng.uniform(*HEIGHT)})
                break
        else:
            sys.exit(f"could not place obstacle {len(obstacles) + 1}; "
                     f"lower --n-obstacles or enlarge BOUNDS")
    return (sx, sy), (gx, gy), obstacles


def obstacle_sdf(i, ob):
    x0, x1, y0, y1 = ob["rect"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, d, h = x1 - x0, y1 - y0, ob["height"]
    return f"""
    <model name="obstacle_{i}">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} {h / 2:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{w:.3f} {d:.3f} {h:.3f}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{w:.3f} {d:.3f} {h:.3f}</size></box></geometry>
          <material>
            <ambient>0.9 0.3 0.0 1</ambient>
            <diffuse>0.9 0.3 0.0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def motor_plugin(i, direction):
    return f"""
      <plugin filename="gz-sim-multicopter-motor-model-system" name="gz::sim::systems::MulticopterMotorModel">
        <robotNamespace>drone_1</robotNamespace>
        <jointName>X3/rotor_{i}_joint</jointName>
        <linkName>X3/rotor_{i}</linkName>
        <turningDirection>{direction}</turningDirection>
        <timeConstantUp>0.0125</timeConstantUp>
        <timeConstantDown>0.025</timeConstantDown>
        <maxRotVelocity>800.0</maxRotVelocity>
        <motorConstant>8.54858e-06</motorConstant>
        <momentConstant>0.016</momentConstant>
        <commandSubTopic>gazebo/command/motor_speed</commandSubTopic>
        <actuator_number>{i}</actuator_number>
        <rotorDragCoefficient>8.06428e-05</rotorDragCoefficient>
        <rollingMomentCoefficient>1e-06</rollingMomentCoefficient>
        <motorSpeedPubTopic>motor_speed/{i}</motorSpeedPubTopic>
        <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
        <motorType>velocity</motorType>
      </plugin>"""


def rotor_config(i, direction):
    return f"""
          <rotor>
            <jointName>X3/rotor_{i}_joint</jointName>
            <forceConstant>8.54858e-06</forceConstant>
            <momentConstant>0.016</momentConstant>
            <direction>{direction}</direction>
          </rotor>"""


def world_sdf(start, obstacles, moving):
    sx, sy = start
    if moving:
        obstacles_xml = ("\n    <!-- Moving scenario: obstacles are virtual "
                         "(see module docstring); none are modeled here. -->")
    else:
        obstacles_xml = "\n".join(obstacle_sdf(i + 1, ob) for i, ob in enumerate(obstacles))
    motors = "".join(motor_plugin(i, d) for i, d in
                     enumerate(["ccw", "ccw", "cw", "cw"]))
    rotors = "".join(rotor_config(i, d) for i, d in
                     enumerate(["1", "1", "-1", "-1"]))
    return f"""<?xml version="1.0"?>
<!-- GENERATED by gen_scenario.py - do not edit by hand; regenerate instead. -->
<sdf version="1.6">
  <world name="drone_nav2_world">

    <!-- 2 ms steps: half the CPU cost of the original 1 ms; still plenty
         for the multicopter controller. Keeps the starved VM responsive. -->
    <physics name="2ms" type="ignored">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <!-- No sensors system: obstacle sensing is virtual_lidar.py (see README). -->

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry>
          <material>
            <ambient>0.25 0.55 0.25 1</ambient>
            <diffuse>0.25 0.65 0.25 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
{obstacles_xml}

    <include>
      <uri>https://fuel.gazebosim.org/1.0/openrobotics/models/x3 uav/6</uri>
      <name>drone_1</name>
      <pose>{sx:.3f} {sy:.3f} {FLIGHT_Z} 0 0 0</pose>
{motors}

      <plugin filename="gz-sim-multicopter-control-system" name="gz::sim::systems::MulticopterVelocityControl">
        <robotNamespace>drone_1</robotNamespace>
        <commandSubTopic>gazebo/command/twist</commandSubTopic>
        <enableSubTopic>enable</enableSubTopic>
        <comLinkName>X3/base_link</comLinkName>
        <velocityGain>2.7 2.7 2.7</velocityGain>
        <attitudeGain>2 3 0.15</attitudeGain>
        <angularRateGain>0.4 0.52 0.18</angularRateGain>
        <maximumLinearAcceleration>2 2 2</maximumLinearAcceleration>
        <rotorConfiguration>{rotors}
        </rotorConfiguration>
      </plugin>

      <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <dimensions>3</dimensions>
        <odom_frame>drone_1/odom</odom_frame>
        <robot_base_frame>drone_1/base_link</robot_base_frame>
        <odom_publish_frequency>20</odom_publish_frequency>
      </plugin>
    </include>

  </world>
</sdf>
"""


def scenario_yaml(seed, start, goal, obstacles, speed):
    sx, sy = start
    gx, gy = goal
    lines = [
        "# GENERATED by gen_scenario.py - single source of truth for this scenario.",
        f"seed: {seed}",
        f"n_obstacles: {len(obstacles)}",
        f"obstacle_speed: {speed:.3f}",
        f"bounds: {BOUNDS}",
        f"flight_altitude: {FLIGHT_Z}",
        f"start: {{x: {sx:.3f}, y: {sy:.3f}}}",
        f"goal: {{x: {gx:.3f}, y: {gy:.3f}}}",
        "obstacles:",
    ]
    for ob in obstacles:
        x0, x1, y0, y1 = ob["rect"]
        lines.append(f"  - {{x_min: {x0:.3f}, x_max: {x1:.3f}, "
                     f"y_min: {y0:.3f}, y_max: {y1:.3f}, height: {ob['height']:.3f}, "
                     f"vx: {ob['vx']:.3f}, vy: {ob['vy']:.3f}}}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-obstacles", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--obstacle-speed", type=float, default=0.0,
                    help="m/s; each obstacle moves in a random direction, "
                         "bouncing off the workspace bounds (default 0 = static)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: scenarios/s<seed>_n<N>[_v<speed>])")
    args = ap.parse_args()

    suffix = f"_v{args.obstacle_speed:g}" if args.obstacle_speed > 0 else ""
    out = args.out or (Path(__file__).parent / "scenarios"
                       / f"s{args.seed}_n{args.n_obstacles}{suffix}")
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    start, goal, obstacles = generate(args.n_obstacles, rng)
    for ob in obstacles:
        if args.obstacle_speed > 0:
            heading = rng.uniform(0, 2 * math.pi)
            ob["vx"] = args.obstacle_speed * math.cos(heading)
            ob["vy"] = args.obstacle_speed * math.sin(heading)
        else:
            ob["vx"] = ob["vy"] = 0.0

    moving = args.obstacle_speed > 0
    (out / "world.sdf").write_text(world_sdf(start, obstacles, moving))
    (out / "scenario.yaml").write_text(
        scenario_yaml(args.seed, start, goal, obstacles, args.obstacle_speed))

    dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
    print(f"scenario written to {out}/")
    print(f"  start=({start[0]:.2f},{start[1]:.2f})  goal=({goal[0]:.2f},{goal[1]:.2f})"
          f"  dist={dist:.2f}m  obstacles={len(obstacles)}")
    print(f"run it:   ./start_drone_stack.sh {out}")
    print(f"goal it:  ./send_nav2_goal.sh --scenario {out}")


if __name__ == "__main__":
    main()
