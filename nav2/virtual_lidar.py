#!/usr/bin/env python3
"""Virtual 2D lidar: publishes /scan computed analytically from the drone's
odometry pose against a known list of box obstacles.

Replaces the Gazebo gpu_lidar sensor, which requires GPU rendering that is
unavailable in this VM (VMware SVGA3D is too unstable for OGRE2). From Nav2's
perspective the output is a normal sensor_msgs/LaserScan on /scan.

The obstacle list must be kept in sync with the world SDF. Each obstacle is
an axis-aligned box footprint: (x_min, x_max, y_min, y_max).
"""

import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# obstacle_1 in drone_nav2_world.sdf: 1x4x2 box centered at (4, 0, 1)
OBSTACLES = [
    (3.5, 4.5, -2.0, 2.0),
]

N_RAYS = 360
ANGLE_MIN = -math.pi
ANGLE_MAX = math.pi
RANGE_MIN = 0.2
RANGE_MAX = 20.0
PUBLISH_RATE_HZ = 10.0


def ray_box_distance(px, py, dx, dy, box):
    """Distance along ray (px,py)+t*(dx,dy) to axis-aligned box, or inf."""
    x_min, x_max, y_min, y_max = box

    t_near = -math.inf
    t_far = math.inf

    for p, d, lo, hi in ((px, dx, x_min, x_max), (py, dy, y_min, y_max)):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return math.inf
        else:
            t1 = (lo - p) / d
            t2 = (hi - p) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
            if t_near > t_far:
                return math.inf

    if t_far < 0:
        return math.inf
    return t_near if t_near >= 0 else 0.0


def quaternion_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class VirtualLidar(Node):
    def __init__(self):
        super().__init__("virtual_lidar")

        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.pose = None
        self.timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self.publish_scan)

        self.get_logger().info(
            f"Virtual lidar started: {N_RAYS} rays, {len(OBSTACLES)} obstacle(s)."
        )

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose = (p.x, p.y, quaternion_to_yaw(q.x, q.y, q.z, q.w))

    def publish_scan(self):
        if self.pose is None:
            return

        x, y, yaw = self.pose

        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = "base_link"
        scan.angle_min = ANGLE_MIN
        scan.angle_max = ANGLE_MAX
        scan.angle_increment = (ANGLE_MAX - ANGLE_MIN) / N_RAYS
        scan.range_min = RANGE_MIN
        scan.range_max = RANGE_MAX

        ranges = []
        for i in range(N_RAYS):
            angle = yaw + ANGLE_MIN + i * scan.angle_increment
            dx = math.cos(angle)
            dy = math.sin(angle)

            dist = min(
                (ray_box_distance(x, y, dx, dy, box) for box in OBSTACLES),
                default=math.inf,
            )

            if dist < RANGE_MIN or dist > RANGE_MAX:
                ranges.append(math.inf)
            else:
                ranges.append(dist)

        scan.ranges = ranges
        self.scan_pub.publish(scan)


def main():
    rclpy.init()
    node = VirtualLidar()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
