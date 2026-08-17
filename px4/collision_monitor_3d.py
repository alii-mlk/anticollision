#!/usr/bin/env python3
"""3D collision monitor: detects drone/obstacle contact and destroys the drone.

Per the evaluation protocol a drone that collides is destroyed rather than
counted and allowed to continue, so on contact this node disarms the vehicle
through MAVROS. The drone drops where it was hit and stays there, which also
leaves it as an obstacle for the other drones in the multi-drone phase.

Frames matter here. Obstacles in scenario.yaml are in Gazebo world
coordinates, while MAVROS reports the drone in PX4's local frame, whose origin
is the drone's spawn point. This node converts local to world by adding the
drone's start position from the scenario:

    world = local + start

The drone is modelled as a sphere of radius `drone_radius`; obstacles are
axis-aligned boxes standing on the ground.

Outcome is written continuously to <out>/outcome.yaml so a batch runner can
read it even if the node is killed.

Usage:
  ./collision_monitor_3d.py --scenario scenarios3d/s1_n8_k1
  ./collision_monitor_3d.py --scenario <dir> --drone-id 0 --out runs/xyz
"""

import argparse
import math
import re
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from mavros_msgs.srv import CommandBool, CommandLong

# PX4 refuses a normal disarm while the vehicle is airborne ("Disarming
# denied: not landed"). MAV_CMD_COMPONENT_ARM_DISARM with this magic value in
# param2 forces it, which is what "destroyed" needs to mean here.
MAV_CMD_COMPONENT_ARM_DISARM = 400
FORCE_DISARM_MAGIC = 21196.0

DRONE_RADIUS = 0.35        # x500 is roughly 0.5 m across; be slightly generous
CHECK_RATE_HZ = 20.0


def parse_scenario(path):
    """Read obstacles and the drone's start/target from scenario.yaml."""
    text = path.read_text()

    obstacles = []
    ob_pattern = (r"x_min:\s*([-\d.]+),\s*x_max:\s*([-\d.]+),\s*"
                  r"y_min:\s*([-\d.]+),\s*y_max:\s*([-\d.]+),\s*"
                  r"z_min:\s*([-\d.]+),\s*z_max:\s*([-\d.]+)")
    for m in re.finditer(ob_pattern, text):
        obstacles.append(tuple(float(v) for v in m.groups()))

    drones = []
    dr_pattern = (r"start:\s*\{x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*z:\s*([-\d.]+)\}\s*"
                  r"target:\s*\{x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*z:\s*([-\d.]+)\}")
    for m in re.finditer(dr_pattern, text):
        v = [float(x) for x in m.groups()]
        drones.append({"start": tuple(v[0:3]), "target": tuple(v[3:6])})

    return obstacles, drones


def distance_to_box(px, py, pz, box):
    """Distance from a point to an axis-aligned box (0 if inside)."""
    x0, x1, y0, y1, z0, z1 = box
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    dz = max(z0 - pz, 0.0, pz - z1)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class CollisionMonitor3D(Node):
    def __init__(self, obstacles, start, radius, out_dir, namespace):
        super().__init__("collision_monitor_3d")

        self.obstacles = obstacles
        self.start = start
        self.radius = radius
        self.out_file = out_dir / "outcome.yaml"
        self.out_file.parent.mkdir(parents=True, exist_ok=True)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        prefix = f"/{namespace}" if namespace else ""
        self.create_subscription(
            PoseStamped, f"{prefix}/mavros/local_position/pose",
            self.pose_cb, sensor_qos)
        self.arming_client = self.create_client(
            CommandBool, f"{prefix}/mavros/cmd/arming")
        self.command_client = self.create_client(
            CommandLong, f"{prefix}/mavros/cmd/command")
        # Announce destruction so the mission runner can stop immediately
        # rather than inferring it from the armed flag.
        self.destroyed_pub = self.create_publisher(
            Bool, f"{prefix}/collision_monitor/destroyed", 10)

        self.world_pos = None
        self.min_clearance = math.inf
        self.destroyed = False
        self.destroyed_at = None
        self.samples = 0

        self.timer = self.create_timer(1.0 / CHECK_RATE_HZ, self.write_outcome)

        self.get_logger().info(
            f"Collision monitor active: {len(obstacles)} obstacle(s), "
            f"drone radius {radius:.2f} m, "
            f"world offset ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})."
        )

    def pose_cb(self, msg):
        if self.destroyed:
            return

        p = msg.pose.position
        # MAVROS local frame has its origin at the drone's spawn point.
        wx = p.x + self.start[0]
        wy = p.y + self.start[1]
        wz = p.z + self.start[2]
        self.world_pos = (wx, wy, wz)
        self.samples += 1

        clearance = min(
            (distance_to_box(wx, wy, wz, b) for b in self.obstacles),
            default=math.inf,
        ) - self.radius
        self.min_clearance = min(self.min_clearance, clearance)

        if clearance <= 0.0:
            self.destroy_drone(wx, wy, wz)

    def destroy_drone(self, wx, wy, wz):
        self.destroyed = True
        self.destroyed_at = (wx, wy, wz)
        self.get_logger().error(
            f"COLLISION at world ({wx:.2f}, {wy:.2f}, {wz:.2f}). "
            f"Destroying the drone."
        )

        if self.command_client.service_is_ready():
            req = CommandLong.Request()
            req.broadcast = False
            req.command = MAV_CMD_COMPONENT_ARM_DISARM
            req.confirmation = 0
            req.param1 = 0.0                  # 0 = disarm
            req.param2 = FORCE_DISARM_MAGIC   # force, even while airborne
            self.command_client.call_async(req)
            self.get_logger().info("Sent force-disarm command.")
        else:
            self.get_logger().warn(
                "Command service not available; falling back to plain disarm "
                "(PX4 will refuse this while airborne).")
            if self.arming_client.service_is_ready():
                req = CommandBool.Request()
                req.value = False
                self.arming_client.call_async(req)

        self.write_outcome()

    def write_outcome(self):
        # Keep announcing the destroyed state: the mission runner may start or
        # reconnect after the collision happened.
        self.destroyed_pub.publish(Bool(data=self.destroyed))

        if self.samples == 0:
            return
        clearance = ("null" if math.isinf(self.min_clearance)
                     else f"{self.min_clearance:.3f}")
        lines = [
            f"destroyed: {str(self.destroyed).lower()}",
            f"min_clearance: {clearance}",
            f"samples: {self.samples}",
        ]
        if self.world_pos:
            wx, wy, wz = self.world_pos
            lines.append(f"last_world_position: {{x: {wx:.3f}, y: {wy:.3f}, "
                         f"z: {wz:.3f}}}")
        if self.destroyed_at:
            dx, dy, dz = self.destroyed_at
            lines.append(f"collision_position: {{x: {dx:.3f}, y: {dy:.3f}, "
                         f"z: {dz:.3f}}}")
        self.out_file.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--drone-id", type=int, default=0)
    ap.add_argument("--drone-radius", type=float, default=DRONE_RADIUS)
    ap.add_argument("--namespace", default="",
                    help="MAVROS namespace for multi-drone runs (e.g. uav0)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: the scenario directory)")
    args = ap.parse_args()

    scenario_file = args.scenario / "scenario.yaml"
    if not scenario_file.exists():
        sys.exit(f"not found: {scenario_file}")

    obstacles, drones = parse_scenario(scenario_file)
    if args.drone_id >= len(drones):
        sys.exit(f"drone {args.drone_id} not in scenario ({len(drones)} drones)")

    out_dir = args.out or args.scenario

    rclpy.init()
    node = CollisionMonitor3D(
        obstacles, drones[args.drone_id]["start"], args.drone_radius,
        out_dir, args.namespace)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_outcome()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
