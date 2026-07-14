#!/usr/bin/env bash

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/jazzy/setup.bash"

# ROS setup.bash references optional env vars; source it before `set -u`.
source "$ROS_SETUP"
set -u

echo "Launching Nav2 (controller/planner/behavior/bt_navigator) against $WORKDIR/nav2_params.yaml"
echo "Make sure start_drone_stack.sh is already running (Gazebo + bridges + TF) before this."

mkdir -p "$WORKDIR/logs"
ros2 launch "$WORKDIR/nav2_minimal.launch.py" 2>&1 | tee "$WORKDIR/logs/nav2.txt"
