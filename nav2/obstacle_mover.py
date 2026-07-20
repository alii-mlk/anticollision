#!/usr/bin/env python3
"""Ground-truth publisher for (possibly moving) obstacles.

Loads the obstacle set from the `scenario_file` parameter (scenario.yaml from
gen_scenario.py; falls back to the built-in static wall of the classic world),
integrates each obstacle's motion on sim time (straight line at its scenario
velocity, bouncing elastically off the workspace bounds) and publishes the
current footprints as a visualization_msgs/MarkerArray on /obstacles at 20 Hz.

That topic is the single source of obstacle truth while the stack runs:
virtual_lidar.py and hit_monitor.py rebuild their rectangle lists from it,
and RViz renders the same markers as boxes. With a static scenario (or none)
the markers simply never move.
"""

import math

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray

DEFAULT_BOUNDS = 9.0
# Classic drone_nav2_world.sdf wall (static).
DEFAULT_OBSTACLES = [
    {"x_min": 3.5, "x_max": 4.5, "y_min": -2.0, "y_max": 2.0,
     "height": 2.0, "vx": 0.0, "vy": 0.0},
]

PUBLISH_RATE_HZ = 20.0


class ObstacleMover(Node):
    def __init__(self):
        super().__init__("obstacle_mover")

        self.declare_parameter("scenario_file", "")
        scenario_file = self.get_parameter("scenario_file").value

        self.bounds = DEFAULT_BOUNDS
        if scenario_file:
            import yaml
            with open(scenario_file) as f:
                scenario = yaml.safe_load(f)
            self.bounds = float(scenario.get("bounds", DEFAULT_BOUNDS))
            raw = scenario["obstacles"]
        else:
            raw = DEFAULT_OBSTACLES

        # Internal state: center position + half-sizes + velocity per obstacle.
        self.obstacles = []
        for ob in raw:
            self.obstacles.append({
                "cx": (ob["x_min"] + ob["x_max"]) / 2,
                "cy": (ob["y_min"] + ob["y_max"]) / 2,
                "hx": (ob["x_max"] - ob["x_min"]) / 2,
                "hy": (ob["y_max"] - ob["y_min"]) / 2,
                "h": ob.get("height", 2.0),
                "vx": ob.get("vx", 0.0),
                "vy": ob.get("vy", 0.0),
            })

        moving = sum(1 for ob in self.obstacles if ob["vx"] or ob["vy"])
        self.marker_pub = self.create_publisher(MarkerArray, "/obstacles", 10)
        self.last_time = None
        self.timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self.tick)

        self.get_logger().info(
            f"Obstacle mover active: {len(self.obstacles)} obstacle(s), "
            f"{moving} moving, workspace bounds ±{self.bounds:g} m."
        )

    def tick(self):
        now = self.get_clock().now()
        if self.last_time is not None:
            dt = (now - self.last_time).nanoseconds / 1e9
            # Guard against clock jumps (sim restarts, first ticks).
            if 0.0 < dt < 1.0:
                self.integrate(dt)
        self.last_time = now
        self.publish_markers(now)

    def integrate(self, dt):
        b = self.bounds
        for ob in self.obstacles:
            ob["cx"] += ob["vx"] * dt
            ob["cy"] += ob["vy"] * dt
            # Elastic bounce: keep the full footprint inside the bounds.
            if ob["cx"] - ob["hx"] < -b:
                ob["cx"] = -b + ob["hx"]
                ob["vx"] = abs(ob["vx"])
            elif ob["cx"] + ob["hx"] > b:
                ob["cx"] = b - ob["hx"]
                ob["vx"] = -abs(ob["vx"])
            if ob["cy"] - ob["hy"] < -b:
                ob["cy"] = -b + ob["hy"]
                ob["vy"] = abs(ob["vy"])
            elif ob["cy"] + ob["hy"] > b:
                ob["cy"] = b - ob["hy"]
                ob["vy"] = -abs(ob["vy"])

    def publish_markers(self, now):
        msg = MarkerArray()
        stamp = now.to_msg()
        for i, ob in enumerate(self.obstacles):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "map"
            m.ns = "obstacles"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = ob["cx"]
            m.pose.position.y = ob["cy"]
            m.pose.position.z = ob["h"] / 2
            m.pose.orientation.w = 1.0
            m.scale.x = 2 * ob["hx"]
            m.scale.y = 2 * ob["hy"]
            m.scale.z = ob["h"]
            m.color.r = 0.9
            m.color.g = 0.3
            m.color.b = 0.0
            m.color.a = 0.85
            msg.markers.append(m)
        self.marker_pub.publish(msg)


def main():
    rclpy.init()
    node = ObstacleMover()

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
