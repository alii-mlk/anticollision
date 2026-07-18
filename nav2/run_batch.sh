#!/usr/bin/env bash
# Unattended batch evaluation: sweeps scenarios over N obstacles x seeds.
#
# For each (N, seed): generate scenario -> start the full stack headless
# (background processes, no terminals) -> launch Nav2 -> arm + send the
# scenario goal with a timeout -> save all artifacts -> tear down -> next.
# A failed or hung run is recorded and the batch continues.
#
# Usage:
#   ./run_batch.sh                          # default sweep (see below)
#   N_LIST="1 4 8" SEEDS="1 2 3" ./run_batch.sh
#   GOAL_TIMEOUT=300 ./run_batch.sh
#
# Results land in runs/batch_<timestamp>/n<N>_s<seed>/ with:
#   scenario/      world.sdf + scenario.yaml
#   goal.txt       full action feedback stream (source for metrics)
#   hits.yaml      hit count + min clearance (from hit_monitor)
#   *.log          every component's output
#   status.txt     SUCCEEDED / ABORTED / TIMEOUT / STACK_FAIL / NAV2_FAIL
#
# Afterwards: python3 compute_metrics.py runs/batch_<timestamp>

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ROS setup.bash references optional env vars; source before `set -u`.
source /opt/ros/jazzy/setup.bash
set -u

N_LIST="${N_LIST:-1 2 4 6 8 10}"
SEEDS="${SEEDS:-1 2 3 4 5}"
GOAL_TIMEOUT="${GOAL_TIMEOUT:-240}"     # wall seconds per goal attempt
BATCH_DIR="$WORKDIR/runs/batch_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BATCH_DIR"
echo "Batch: N in [$N_LIST], seeds in [$SEEDS], goal timeout ${GOAL_TIMEOUT}s"
echo "Output: $BATCH_DIR"

kill_stack() {
  pkill -f "gz sim" 2>/dev/null || true
  pkill -f "ruby.*gz sim" 2>/dev/null || true
  pkill -f "ros_gz_bridge.*parameter_bridge" 2>/dev/null || true
  pkill -f "drone_pose_to_odom_tf.py" 2>/dev/null || true
  pkill -f "virtual_lidar.py" 2>/dev/null || true
  pkill -f "hit_monitor.py" 2>/dev/null || true
  pkill -f "nav2_minimal.launch.py" 2>/dev/null || true
  pkill -f "controller_server" 2>/dev/null || true
  pkill -f "planner_server" 2>/dev/null || true
  pkill -f "behavior_server" 2>/dev/null || true
  pkill -f "bt_navigator" 2>/dev/null || true
  pkill -f "nav2_lifecycle_manager" 2>/dev/null || true
  ros2 daemon stop >/dev/null 2>&1 || true
  sleep 3
}

# wait_for_line <file> <pattern> <timeout_s>
wait_for_line() {
  local file="$1" pattern="$2" timeout_s="$3" waited=0
  while ! grep -q "$pattern" "$file" 2>/dev/null; do
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge "$timeout_s" ]; then
      return 1
    fi
  done
  return 0
}

run_one() {
  local n="$1" seed="$2"
  local run_dir="$BATCH_DIR/n${n}_s${seed}"
  local scen_dir="$run_dir/scenario"
  mkdir -p "$run_dir"

  echo "=== run n=$n seed=$seed -> $run_dir"
  python3 "$WORKDIR/gen_scenario.py" --n-obstacles "$n" --seed "$seed" \
    --out "$scen_dir" > "$run_dir/gen.log" 2>&1 || {
      echo "STACK_FAIL" > "$run_dir/status.txt"
      echo "    generation failed"; return; }

  kill_stack

  # --- simulation + bridges (all background, logs into run_dir) ---
  ( cd "$WORKDIR" && gz sim -s -r "$scen_dir/world.sdf" ) \
    > "$run_dir/gz.log" 2>&1 &
  sleep 6

  ros2 run ros_gz_bridge parameter_bridge \
    /model/drone_1/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry \
    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" \
    > "$run_dir/bridge_odom.log" 2>&1 &

  ros2 run ros_gz_bridge parameter_bridge \
    /drone_1/gazebo/command/twist@geometry_msgs/msg/Twist@gz.msgs.Twist \
    --ros-args -r /drone_1/gazebo/command/twist:=/cmd_vel \
    > "$run_dir/bridge_cmd.log" 2>&1 &

  python3 "$WORKDIR/drone_pose_to_odom_tf.py" --ros-args \
    -p use_sim_time:=true \
    > "$run_dir/odom_tf.log" 2>&1 &

  python3 "$WORKDIR/virtual_lidar.py" --ros-args \
    -p use_sim_time:=true -p "scenario_file:=$scen_dir/scenario.yaml" \
    > "$run_dir/lidar.log" 2>&1 &

  python3 "$WORKDIR/hit_monitor.py" --ros-args \
    -p use_sim_time:=true -p "scenario_file:=$scen_dir/scenario.yaml" \
    -p "state_file:=$run_dir/hits.yaml" \
    > "$run_dir/monitor.log" 2>&1 &

  # readiness: odometry flowing means gz + bridge + tf node are alive
  if ! timeout 40 ros2 topic echo /odom --once > /dev/null 2>&1; then
    echo "STACK_FAIL" > "$run_dir/status.txt"
    echo "    stack failed (no /odom)"; kill_stack; return
  fi

  # --- Nav2 ---
  ros2 launch "$WORKDIR/nav2_minimal.launch.py" \
    > "$run_dir/nav2.log" 2>&1 &

  if ! wait_for_line "$run_dir/nav2.log" "Managed nodes are active" 90; then
    echo "NAV2_FAIL" > "$run_dir/status.txt"
    echo "    Nav2 failed to activate"; kill_stack; return
  fi

  # --- arm + goal ---
  for _ in 1 2 3; do
    gz topic -t /drone_1/enable -m gz.msgs.Boolean -p "data: true" >/dev/null 2>&1
    sleep 1
  done

  read -r GX GY <<< "$(python3 -c "
import yaml
g = yaml.safe_load(open('$scen_dir/scenario.yaml'))['goal']
print(g['x'], g['y'])
")"

  timeout "$GOAL_TIMEOUT" ros2 action send_goal /navigate_to_pose \
    nav2_msgs/action/NavigateToPose "
pose:
  header:
    frame_id: 'map'
  pose:
    position: {x: $GX, y: $GY, z: 0.0}
    orientation: {w: 1.0}
" --feedback > "$run_dir/goal.txt" 2>&1

  local status
  status="$(grep -oE 'Goal finished with status: [A-Z]+' "$run_dir/goal.txt" \
            | tail -1 | awk '{print $NF}')"
  if [ -z "$status" ]; then
    status="TIMEOUT"
  fi
  echo "$status" > "$run_dir/status.txt"
  echo "    result: $status"

  kill_stack
}

trap 'echo "interrupted -- cleaning up"; kill_stack; exit 130' INT TERM

START_TS=$(date +%s)
TOTAL=0
for n in $N_LIST; do
  for seed in $SEEDS; do
    run_one "$n" "$seed"
    TOTAL=$((TOTAL + 1))
  done
done

echo ""
echo "Batch finished: $TOTAL runs in $(( ($(date +%s) - START_TS) / 60 )) min."
echo "Statuses:"
cat "$BATCH_DIR"/*/status.txt | sort | uniq -c
echo ""
echo "Metrics: python3 compute_metrics.py $BATCH_DIR"
