#!/usr/bin/env bash

# Note: ROS setup.bash references optional env vars, so it must be sourced
# before enabling `set -u`.
ROS_SETUP="/opt/ros/jazzy/setup.bash"
source "$ROS_SETUP"
set -u

# Usage:
#   ./send_nav2_goal.sh                      -> default goal (8, 0) for the classic world
#   ./send_nav2_goal.sh X Y                  -> explicit goal
#   ./send_nav2_goal.sh --scenario <dir>     -> goal from <dir>/scenario.yaml
if [ "${1:-}" = "--scenario" ]; then
  SCENARIO_YAML="$2/scenario.yaml"
  read -r GOAL_X GOAL_Y <<< "$(python3 -c "
import yaml
g = yaml.safe_load(open('$SCENARIO_YAML'))['goal']
print(g['x'], g['y'])
")"
else
  GOAL_X="${1:-8.0}"
  GOAL_Y="${2:-0.0}"
fi

# Arm the drone right before every goal: the enable retries in
# start_drone_stack.sh can all land while Gazebo is still loading a heavy
# world, leaving the motors disabled. Re-sending here is idempotent.
echo "Enabling drone..."
gz topic -t /drone_1/enable -m gz.msgs.Boolean -p "data: true"
sleep 1

# Zero the hit counter so hits are measured from navigation start.
ros2 topic pub --once /hit_monitor/reset std_msgs/msg/Empty > /dev/null

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

# Snapshot the hit state at goal end (hits_current.yaml keeps updating while
# the drone sits at the goal and obstacles drift around).
if [ -f "$LOGDIR/hits_current.yaml" ]; then
  echo ""
  echo "Navigation hit summary:"
  tee "$LOGDIR/hits_goal_$(date +%H%M%S).yaml" < "$LOGDIR/hits_current.yaml"
fi
