#!/usr/bin/env python3
"""Fly the PX4 drone to a 3D target using MAVROS offboard position setpoints.

This is the control path Phase E needs: unlike AUTO.TAKEOFF (which only uses
PX4's own takeoff altitude), offboard mode lets ROS 2 choose an arbitrary
target in 3D.

PX4 will not enter OFFBOARD mode unless setpoints are already streaming at
more than 2 Hz, and it drops out of OFFBOARD if the stream stalls. So the
node starts publishing first, then requests the mode switch, then arms, and
keeps publishing for the whole flight.

Coordinates are ENU (x east, y north, z up), matching MAVROS conventions.

Usage:
  ./offboard_goto.py                      # default target (5, 0, 3)
  ./offboard_goto.py --x 5 --y -4 --z 6
  ./offboard_goto.py --x 5 --y 0 --z 3 --tolerance 0.5
"""

import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

SETPOINT_RATE_HZ = 20.0     # PX4 needs > 2 Hz; 20 Hz is the usual choice
PRESTREAM_SETPOINTS = 40    # sent before requesting OFFBOARD (~2 s at 20 Hz)


class OffboardGoto(Node):
    def __init__(self, target, tolerance, timeout):
        super().__init__("offboard_goto")

        self.target = target
        self.tolerance = tolerance
        self.timeout = timeout

        # MAVROS publishes state with BEST_EFFORT/volatile; match it or the
        # subscription silently receives nothing.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.state = State()
        self.pose = None

        self.create_subscription(State, "/mavros/state", self.state_cb, sensor_qos)
        self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self.pose_cb, sensor_qos)

        self.setpoint_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", 10)

        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")

        self.sent = 0
        self.offboard_requested = False
        self.armed_requested = False
        self.elapsed = 0.0
        self.reached = False

        self.timer = self.create_timer(1.0 / SETPOINT_RATE_HZ, self.tick)

        self.get_logger().info(
            f"Target (ENU): x={target[0]:.2f} y={target[1]:.2f} z={target[2]:.2f}, "
            f"tolerance {tolerance:.2f} m"
        )

    def state_cb(self, msg):
        self.state = msg

    def pose_cb(self, msg):
        p = msg.pose.position
        self.pose = (p.x, p.y, p.z)

    def make_setpoint(self):
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = "map"
        sp.pose.position.x = self.target[0]
        sp.pose.position.y = self.target[1]
        sp.pose.position.z = self.target[2]
        sp.pose.orientation.w = 1.0
        return sp

    def call_mode(self, mode):
        if not self.mode_client.service_is_ready():
            return
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = mode
        self.mode_client.call_async(req)
        self.get_logger().info(f"Requested mode {mode}")

    def call_arm(self):
        if not self.arming_client.service_is_ready():
            return
        req = CommandBool.Request()
        req.value = True
        self.arming_client.call_async(req)
        self.get_logger().info("Requested arm")

    def tick(self):
        # Always keep the setpoint stream alive; PX4 exits OFFBOARD without it.
        self.setpoint_pub.publish(self.make_setpoint())
        self.sent += 1
        self.elapsed += 1.0 / SETPOINT_RATE_HZ

        # Stream a couple of seconds of setpoints before asking for OFFBOARD.
        if self.sent < PRESTREAM_SETPOINTS:
            return

        if self.state.mode != "OFFBOARD" and not self.offboard_requested:
            self.call_mode("OFFBOARD")
            self.offboard_requested = True
            return

        if self.state.mode == "OFFBOARD" and not self.state.armed \
                and not self.armed_requested:
            self.call_arm()
            self.armed_requested = True
            return

        # Retry the mode request if PX4 dropped back out of OFFBOARD.
        if self.offboard_requested and self.state.mode != "OFFBOARD" \
                and self.sent % 100 == 0:
            self.call_mode("OFFBOARD")

        if self.pose is None:
            return

        dist = math.dist(self.pose, self.target)
        if self.sent % 40 == 0:
            self.get_logger().info(
                f"mode={self.state.mode} armed={self.state.armed} "
                f"pos=({self.pose[0]:.2f}, {self.pose[1]:.2f}, {self.pose[2]:.2f}) "
                f"dist={dist:.2f} m"
            )

        if dist <= self.tolerance and not self.reached:
            self.reached = True
            self.get_logger().info(
                f"Target reached in {self.elapsed:.1f} s "
                f"(distance {dist:.2f} m). Holding position; Ctrl+C to stop."
            )

        if self.elapsed > self.timeout and not self.reached:
            self.get_logger().error(
                f"Timeout after {self.timeout:.0f} s, still {dist:.2f} m away."
            )
            raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--x", type=float, default=5.0)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--tolerance", type=float, default=0.4)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    rclpy.init()
    node = OffboardGoto((args.x, args.y, args.z), args.tolerance, args.timeout)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
