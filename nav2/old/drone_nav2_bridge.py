#!/usr/bin/env python3

import math
import subprocess
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class DroneNav2Bridge(Node):
    """
    Prototype bridge:
    - Subscribes to ROS 2 /cmd_vel, the same command topic Nav2 uses.
    - Publishes fake /odom and odom -> base_link TF for Nav2.
    - Sends the velocity command to the Gazebo X3 drone.

    This is a first prototype. It treats the drone like a 2D robot flying at fixed height.
    """

    def __init__(self):
        super().__init__("drone_nav2_bridge")

        # Gazebo topic used by your X3 drone velocity controller.
        self.gz_twist_topic = "/model/drone_1/gazebo/command/twist"

        # Enable topic for the multicopter controller.
        self.gz_enable_topic = "/model/drone_1/enable"

        # Internal estimated 2D pose for Nav2.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.z = 1.0

        # Current command from Nav2 or manual /cmd_vel.
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.last_cmd_time = self.get_clock().now()

        # Safety limits.
        self.max_vx = 0.6
        self.max_vy = 0.4
        self.max_wz = 1.0
        self.cmd_timeout = 0.5

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.cmd_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_callback,
            10
        )

        self.last_update_time = time.time()

        # Timer for odometry / tf / Gazebo command.
        self.timer = self.create_timer(0.1, self.update)

        self.enable_drone()

        self.get_logger().info("Drone Nav2 bridge started.")
        self.get_logger().info("Listening to /cmd_vel")
        self.get_logger().info(f"Sending Gazebo commands to {self.gz_twist_topic}")

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def cmd_callback(self, msg: Twist):
        self.cmd_vx = self.clamp(msg.linear.x, -self.max_vx, self.max_vx)
        self.cmd_vy = self.clamp(msg.linear.y, -self.max_vy, self.max_vy)
        self.cmd_wz = self.clamp(msg.angular.z, -self.max_wz, self.max_wz)
        self.last_cmd_time = self.get_clock().now()

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return 0.0, 0.0, qz, qw

    def enable_drone(self):
        cmd = [
            "gz", "topic",
            "-t", self.gz_enable_topic,
            "-m", "gz.msgs.Boolean",
            "-p", "data: true"
        ]

        try:
            subprocess.run(cmd, timeout=1.0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info("Sent enable command to drone.")
        except Exception as e:
            self.get_logger().warn(f"Could not enable drone yet: {e}")

    def send_gazebo_twist(self, vx, vy, wz):
        payload = (
            f"linear: {{x: {vx:.4f}, y: {vy:.4f}, z: 0.0}} "
            f"angular: {{x: 0.0, y: 0.0, z: {wz:.4f}}}"
        )

        cmd = [
            "gz", "topic",
            "-t", self.gz_twist_topic,
            "-m", "gz.msgs.Twist",
            "-p", payload
        ]

        try:
            subprocess.run(cmd, timeout=1.0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self.get_logger().warn(f"Could not send Gazebo twist: {e}")

    def publish_odom_and_tf(self, vx, vy, wz):
        now_ros = self.get_clock().now().to_msg()

        qx, qy, qz, qw = self.yaw_to_quaternion(self.yaw)

        odom = Odometry()
        odom.header.stamp = now_ros
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = self.z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz

        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = now_ros
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"

        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = self.z
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf)

    def update(self):
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

        # Stop if command is old.
        age = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if age > self.cmd_timeout:
            vx = 0.0
            vy = 0.0
            wz = 0.0
        else:
            vx = self.cmd_vx
            vy = self.cmd_vy
            wz = self.cmd_wz

        # Integrate 2D odometry estimate.
        self.x += (vx * math.cos(self.yaw) - vy * math.sin(self.yaw)) * dt
        self.y += (vx * math.sin(self.yaw) + vy * math.cos(self.yaw)) * dt
        self.yaw += wz * dt

        # Keep yaw between -pi and pi.
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        self.publish_odom_and_tf(vx, vy, wz)
        self.send_gazebo_twist(vx, vy, wz)


def main():
    rclpy.init()
    node = DroneNav2Bridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.send_gazebo_twist(0.0, 0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()