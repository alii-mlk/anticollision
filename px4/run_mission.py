#!/usr/bin/env python3
"""Fly a drone to its scenario target and record the result.

This is the baseline mission: the drone flies straight at its target using
offboard position setpoints, with no avoidance. Obstacles are in its way, so
this run is expected to collide in a dense maze. It provides the "without
avoidance" numbers the avoidance component is later compared against.

The run ends when the drone reaches its target, is destroyed by
collision_monitor_3d.py (which disarms it), or the timeout expires.

Frames: scenario coordinates are Gazebo world coordinates, while MAVROS works
in PX4's local frame whose origin is the drone's spawn point, so the target is
converted with

    local = world - start

Results are written to <out>/mission.yaml, and the flown path to
<out>/trajectory.csv, giving the travelled distance and the ratio against the
Euclidean distance.

Usage:
  ./run_mission.py --scenario scenarios3d/s1_n8_k1
  ./run_mission.py --scenario <dir> --out runs/r1 --timeout 180
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

SETPOINT_RATE_HZ = 20.0
PRESTREAM_SETPOINTS = 40      # ~2 s of setpoints before requesting OFFBOARD
TAKEOFF_ALTITUDE = 3.0        # climb here first, then head for the target
TAKEOFF_TOLERANCE = 0.8


def parse_scenario(path):
    text = path.read_text()
    drones = []
    pattern = (r"start:\s*\{x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*z:\s*([-\d.]+)\}\s*"
               r"target:\s*\{x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*z:\s*([-\d.]+)\}")
    for m in re.finditer(pattern, text):
        v = [float(x) for x in m.groups()]
        drones.append({"start": tuple(v[0:3]), "target": tuple(v[3:6])})
    return drones


class MissionRunner(Node):
    def __init__(self, start, target, tolerance, timeout, out_dir, namespace):
        super().__init__("run_mission")

        self.start = start
        self.target_world = target
        # PX4's local frame originates at the spawn point.
        self.target_local = tuple(t - s for t, s in zip(target, start))
        self.tolerance = tolerance
        self.timeout = timeout
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        prefix = f"/{namespace}" if namespace else ""
        self.state = State()
        self.local_pos = None

        self.create_subscription(State, f"{prefix}/mavros/state",
                                 self.state_cb, sensor_qos)
        self.create_subscription(PoseStamped,
                                 f"{prefix}/mavros/local_position/pose",
                                 self.pose_cb, sensor_qos)
        # The collision monitor announces destruction here. Do not infer it
        # from the armed flag: a force-disarmed drone can still report armed
        # briefly, and PX4 may refuse an ordinary disarm outright.
        self.create_subscription(Bool,
                                 f"{prefix}/collision_monitor/destroyed",
                                 self.destroyed_cb, 10)
        self.setpoint_pub = self.create_publisher(
            PoseStamped, f"{prefix}/mavros/setpoint_position/local", 10)
        self.arming_client = self.create_client(
            CommandBool, f"{prefix}/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode, f"{prefix}/mavros/set_mode")

        self.sent = 0
        self.elapsed = 0.0
        self.offboard_requested = False
        self.armed_requested = False
        self.climbed = False
        self.outcome = None
        self.was_armed = False
        self.destroyed = False

        self.trajectory = []       # (t, world x, y, z)
        self.travelled = 0.0
        self.last_world = None

        self.euclidean = math.dist(start, target)
        self.timer = self.create_timer(1.0 / SETPOINT_RATE_HZ, self.tick)

        self.get_logger().info(
            f"Mission: world target ({target[0]:.2f}, {target[1]:.2f}, "
            f"{target[2]:.2f}), local ({self.target_local[0]:.2f}, "
            f"{self.target_local[1]:.2f}, {self.target_local[2]:.2f}), "
            f"Euclidean {self.euclidean:.2f} m"
        )

    def state_cb(self, msg):
        self.state = msg
        if msg.armed:
            self.was_armed = True

    def destroyed_cb(self, msg):
        if msg.data:
            self.destroyed = True

    def pose_cb(self, msg):
        p = msg.pose.position
        self.local_pos = (p.x, p.y, p.z)
        world = (p.x + self.start[0], p.y + self.start[1], p.z + self.start[2])
        if self.last_world is not None:
            self.travelled += math.dist(self.last_world, world)
        self.last_world = world
        self.trajectory.append((self.elapsed, *world))

    def make_setpoint(self, xyz):
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = "map"
        sp.pose.position.x, sp.pose.position.y, sp.pose.position.z = xyz
        sp.pose.orientation.w = 1.0
        return sp

    def tick(self):
        # Climb straight up first, then fly to the target. Heading for the
        # target from the ground would drag the drone through obstacles at
        # ground level before it has any altitude.
        if not self.climbed:
            setpoint = (0.0, 0.0, TAKEOFF_ALTITUDE)
        else:
            setpoint = self.target_local

        # A destroyed drone is not flown any further, so stop the setpoint
        # stream before publishing anything else this tick.
        if self.destroyed and self.outcome is None:
            self.finish("destroyed")
            return

        self.setpoint_pub.publish(self.make_setpoint(setpoint))
        self.sent += 1
        self.elapsed += 1.0 / SETPOINT_RATE_HZ

        if self.sent < PRESTREAM_SETPOINTS:
            return

        if self.state.mode != "OFFBOARD" and not self.offboard_requested:
            self.request_mode("OFFBOARD")
            self.offboard_requested = True
            return

        if self.state.mode == "OFFBOARD" and not self.state.armed \
                and not self.armed_requested:
            self.request_arm()
            self.armed_requested = True
            return

        if self.local_pos is None:
            return

        if not self.climbed:
            if abs(self.local_pos[2] - TAKEOFF_ALTITUDE) <= TAKEOFF_TOLERANCE:
                self.climbed = True
                self.get_logger().info(
                    f"Reached {TAKEOFF_ALTITUDE:.1f} m, heading for the target.")
            return

        dist = math.dist(self.local_pos, self.target_local)
        if self.sent % 40 == 0:
            w = self.last_world or (0, 0, 0)
            self.get_logger().info(
                f"mode={self.state.mode} armed={self.state.armed} "
                f"world=({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}) "
                f"dist={dist:.2f} m travelled={self.travelled:.1f} m"
            )

        if dist <= self.tolerance:
            self.finish("reached")
            return

        if self.elapsed > self.timeout:
            self.finish("timeout")

    def request_mode(self, mode):
        if self.mode_client.service_is_ready():
            req = SetMode.Request()
            req.base_mode = 0
            req.custom_mode = mode
            self.mode_client.call_async(req)
            self.get_logger().info(f"Requested mode {mode}")

    def request_arm(self):
        if self.arming_client.service_is_ready():
            req = CommandBool.Request()
            req.value = True
            self.arming_client.call_async(req)
            self.get_logger().info("Requested arm")

    def finish(self, outcome):
        self.outcome = outcome
        self.write_results()
        self.get_logger().info(
            f"Mission {outcome}: travelled {self.travelled:.2f} m, "
            f"Euclidean {self.euclidean:.2f} m, "
            f"ratio {self.travelled / self.euclidean:.3f}, "
            f"time {self.elapsed:.1f} s"
        )
        raise SystemExit(0 if outcome == "reached" else 1)

    def write_results(self):
        ratio = self.travelled / self.euclidean if self.euclidean > 0 else 0.0
        w = self.last_world or (0.0, 0.0, 0.0)
        (self.out_dir / "mission.yaml").write_text(
            f"outcome: {self.outcome}\n"
            f"euclidean_distance: {self.euclidean:.3f}\n"
            f"travelled_distance: {self.travelled:.3f}\n"
            f"ratio: {ratio:.3f}\n"
            f"simulation_time: {self.elapsed:.2f}\n"
            f"final_world_position: {{x: {w[0]:.3f}, y: {w[1]:.3f}, z: {w[2]:.3f}}}\n"
            f"target_world_position: {{x: {self.target_world[0]:.3f}, "
            f"y: {self.target_world[1]:.3f}, z: {self.target_world[2]:.3f}}}\n"
        )
        with open(self.out_dir / "trajectory.csv", "w") as f:
            f.write("t,x,y,z\n")
            for t, x, y, z in self.trajectory:
                f.write(f"{t:.2f},{x:.3f},{y:.3f},{z:.3f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--drone-id", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.6)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--namespace", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scenario_file = args.scenario / "scenario.yaml"
    if not scenario_file.exists():
        sys.exit(f"not found: {scenario_file}")

    drones = parse_scenario(scenario_file)
    if args.drone_id >= len(drones):
        sys.exit(f"drone {args.drone_id} not in scenario ({len(drones)} drones)")

    d = drones[args.drone_id]
    out_dir = args.out or args.scenario

    rclpy.init()
    node = MissionRunner(d["start"], d["target"], args.tolerance,
                         args.timeout, out_dir, args.namespace)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node.outcome is None:
            node.outcome = "interrupted"
            node.write_results()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
