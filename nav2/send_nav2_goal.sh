#!/usr/bin/env bash

# Note: ROS setup.bash references optional env vars, so it must be sourced
# before enabling `set -u`.
ROS_SETUP="/opt/ros/jazzy/setup.bash"
source "$ROS_SETUP"
set -u

# obstacle_1 is a 1x4x2 box centered at (4, 0, 1) -- spans x:[3.5,4.5], y:[-2,2].
# Drone starts at (0, 0, 1). This goal sits directly behind the obstacle, so
# a straight-line plan would drive through it; Nav2 has to route around.
GOAL_X="${1:-8.0}"
GOAL_Y="${2:-0.0}"

echo "Sending NavigateToPose goal: x=$GOAL_X, y=$GOAL_Y"

LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$LOGDIR"

ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "
pose:
  header:
    frame_id: 'map'
  pose:
    position: {x: $GOAL_X, y: $GOAL_Y, z: 0.0}
    orientation: {w: 1.0}
" --feedback 2>&1 | tee "$LOGDIR/goal_$(date +%H%M%S).txt"
