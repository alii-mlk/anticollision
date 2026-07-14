#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


class DronePoseToOdomTF(Node):
    """
    Republishes Gazebo's OdometryPublisher output (bridged from
    /model/drone_1/odometry) as /odom with Nav2's expected frame names,
    and broadcasts the odom -> base_link TF.

    This reads the drone's odometry directly from its own named gz topic,
    so there's no need to guess which entity in a pose stream is the drone.
    """

    def __init__(self):
        super().__init__("drone_pose_to_odom_tf")

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.odom_sub = self.create_subscription(
            Odometry,
            "/model/drone_1/odometry",
            self.odom_callback,
            10
        )

        self.log_counter = 0

        self.publish_static_map_to_odom()

        self.get_logger().info("Drone pose to /odom and /tf bridge started.")
        self.get_logger().info("Waiting for /model/drone_1/odometry from Gazebo...")

    def publish_static_map_to_odom(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "map"
        tf.child_frame_id = "odom"

        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0

        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 1.0

        self.static_tf_broadcaster.sendTransform(tf)
        self.get_logger().info("Published static TF: map -> odom")

    def odom_callback(self, msg: Odometry):
        now_msg = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose = msg.pose.pose
        odom.twist.twist = msg.twist.twist

        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = now_msg
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"

        tf.transform.translation.x = msg.pose.pose.position.x
        tf.transform.translation.y = msg.pose.pose.position.y
        tf.transform.translation.z = msg.pose.pose.position.z
        tf.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf)

        self.log_counter += 1
        if self.log_counter % 40 == 0:
            p = msg.pose.pose.position
            self.get_logger().info(
                f"Publishing /odom: x={p.x:.2f}, y={p.y:.2f}, z={p.z:.2f}"
            )


def main():
    rclpy.init()
    node = DronePoseToOdomTF()

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
