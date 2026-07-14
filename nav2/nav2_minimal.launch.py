#!/usr/bin/env python3
"""Minimal Nav2 bringup for the drone world.

Launches only the nodes this project actually uses:
controller_server, planner_server, behavior_server, bt_navigator,
plus a lifecycle manager. Avoids the extra servers in the stock
nav2_bringup navigation_launch.py (collision_monitor, route_server,
opennav_docking, ...) that need their own configuration.
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "nav2_params.yaml")

MANAGED_NODES = [
    "controller_server",
    "planner_server",
    "behavior_server",
    "bt_navigator",
]


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="screen",
            parameters=[PARAMS_FILE],
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[PARAMS_FILE],
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            output="screen",
            parameters=[PARAMS_FILE],
        ),
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            output="screen",
            parameters=[PARAMS_FILE],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "autostart": True,
                "node_names": MANAGED_NODES,
            }],
        ),
    ])
