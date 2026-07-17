#!/usr/bin/env python3
"""Collision counter for the drone Nav2 evaluation.

Per the evaluation protocol, the simulation is NOT stopped when the drone
touches an obstacle; instead the number of hits is counted as an
effectiveness metric. This node subscribes to /odom, models the drone as a
disc of radius `drone_radius`, and checks it against the obstacle
rectangles each update:

  - a "hit" is counted once per contact episode (entering contact increments
    the counter; the episode ends when contact is fully released)
  - minimum clearance (distance from drone disc edge to nearest obstacle)
    is tracked over the whole run

State is continuously rewritten to `logs/hits_current.yaml` so batch tooling
can read the totals after a run without depending on clean node shutdown.
Hits are also logged to the console (captured by the terminal tee).

Obstacles come from the `scenario_file` parameter (scenario.yaml from
gen_scenario.py) or fall back to the built-in drone_nav2_world.sdf wall.
"""

import math
from pathlib import Path

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry

DEFAULT_OBSTACLES = [
    (3.5, 4.5, -2.0, 2.0),
]


def distance_to_rect(px, py, rect):
    """Distance from point to axis-aligned rect (0 if inside)."""
    x0, x1, y0, y1 = rect
    nx = min(max(px, x0), x1)
    ny = min(max(py, y0), y1)
    return math.hypot(px - nx, py - ny)


class HitMonitor(Node):
    def __init__(self):
        super().__init__("hit_monitor")

        self.declare_parameter("scenario_file", "")
        self.declare_parameter("drone_radius", 0.3)
        self.declare_parameter("state_file",
                               str(Path(__file__).parent / "logs" / "hits_current.yaml"))

        scenario_file = self.get_parameter("scenario_file").value
        self.radius = self.get_parameter("drone_radius").value
        self.state_file = Path(self.get_parameter("state_file").value)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        if scenario_file:
            import yaml
            with open(scenario_file) as f:
                scenario = yaml.safe_load(f)
            self.obstacles = [(o["x_min"], o["x_max"], o["y_min"], o["y_max"])
                              for o in scenario["obstacles"]]
        else:
            self.obstacles = DEFAULT_OBSTACLES

        self.hits = 0
        self.in_contact = False
        self.min_clearance = math.inf
        self.samples = 0

        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10)

        # Rewrite the state file at 2 Hz rather than every odom message.
        self.timer = self.create_timer(0.5, self.write_state)

        self.get_logger().info(
            f"Hit monitor started: {len(self.obstacles)} obstacle(s), "
            f"drone radius {self.radius} m. Counting hits, not stopping."
        )

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        clearance = min(
            (distance_to_rect(p.x, p.y, r) for r in self.obstacles),
            default=math.inf,
        ) - self.radius
        self.samples += 1
        self.min_clearance = min(self.min_clearance, clearance)

        contact = clearance <= 0.0
        if contact and not self.in_contact:
            self.hits += 1
            self.get_logger().warn(
                f"HIT #{self.hits} at x={p.x:.2f}, y={p.y:.2f}"
            )
        self.in_contact = contact

    def write_state(self):
        if self.samples == 0:
            return
        clearance = "null" if math.isinf(self.min_clearance) \
            else f"{self.min_clearance:.3f}"
        self.state_file.write_text(
            f"hits: {self.hits}\n"
            f"min_clearance: {clearance}\n"
            f"in_contact: {str(self.in_contact).lower()}\n"
            f"samples: {self.samples}\n"
        )


def main():
    rclpy.init()
    node = HitMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_state()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
