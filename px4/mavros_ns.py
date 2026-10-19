"""Find where MAVROS has put its topics and services.

This changed between MAVROS versions. Started as

  ros2 run mavros mavros_node --ros-args -r __ns:=/uav0

version 2.14 published its plugins' topics directly under the namespace
(/uav0/state, /uav0/local_position/pose, /uav0/cmd/arming), while 2.16, which
an apt upgrade brought in on 2026-10-11, nests them under the node's own name
(/uav0/mavros/state, ...). Its log shows the same move: plugin loggers went
from uav0.sys to uav0.mavros.sys.

Code written for one version receives nothing from the other. On the first
run after the upgrade every drone sat on the ground until its arm deadline,
with the mission reporting connected=False, and the collision monitors, which
listen to the same topics, would have missed every collision too.

So instead of assuming the layout, look it up in the ROS graph. The state
topic is the one of type mavros_msgs/msg/State inside the drone's namespace,
and every other MAVROS topic and service sits next to it.
"""

import time

STATE_TYPE = "mavros_msgs/msg/State"


def resolve(node, namespace, timeout=15.0):
    """The prefix MAVROS uses inside `namespace`, e.g. /uav0 or /uav0/mavros."""
    ns = namespace.rstrip("/") or "/"
    deadline = time.monotonic() + timeout
    while True:
        for name, types in node.get_topic_names_and_types():
            if (STATE_TYPE in types and name.endswith("/state")
                    and name.startswith(ns.rstrip("/") + "/")):
                prefix = name[: -len("/state")]
                node.get_logger().info(f"MAVROS topics found under {prefix}")
                return prefix
        if time.monotonic() >= deadline:
            break
        # Discovery runs in the background; the graph fills in within a
        # second or two of the node starting.
        time.sleep(0.2)
    node.get_logger().warn(
        f"no MAVROS state topic under {ns} after {timeout:.0f} s; "
        f"assuming {ns} and hoping for the best")
    return ns.rstrip("/")
