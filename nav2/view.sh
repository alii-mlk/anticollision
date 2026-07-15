#!/usr/bin/env bash

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ROS setup.bash references optional env vars; source it before `set -u`.
source /opt/ros/jazzy/setup.bash
set -u

# RViz2 visualization of the running stack: drone pose (TF/odometry trail),
# lidar hits, Nav2 global costmap and planned path.
# If rendering fails in the VM, retry with: LIBGL_ALWAYS_SOFTWARE=1 ./view.sh
rviz2 -d "$WORKDIR/drone_view.rviz" --ros-args -p use_sim_time:=true
