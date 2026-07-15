#!/usr/bin/env bash

set -u

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/jazzy/setup.bash"

LOGDIR="$WORKDIR/logs"
mkdir -p "$LOGDIR"

open_term() {
  local title="$1"
  local cmd="$2"
  local logfile="$LOGDIR/$(echo "$title" | tr ' /' '__').txt"

  gnome-terminal --title="$title" -- bash -lc "
    echo '===== $title ====='
    ( $cmd ) 2>&1 | tee '$logfile'
  "
}

echo "Stopping old drone/Nav2 bridge processes..."

pkill -f "ros2 topic pub /cmd_vel" || true
pkill -f "drone_pose_to_odom_tf.py" || true
pkill -f "ros_gz_bridge.*parameter_bridge" || true
pkill -f "gz sim.*drone_nav2_world.sdf" || true
pkill -f "ruby.*gz sim.*drone_nav2_world.sdf" || true
pkill -f "virtual_lidar.py" || true

sleep 2

echo "Starting Gazebo + bridges..."

open_term "1 Gazebo Drone World (server, no GUI)" "
  cd '$WORKDIR' &&
  gz sim -s -r drone_nav2_world.sdf
"

sleep 6

open_term "2 Gazebo Odometry + Clock Bridge" "
  source '$ROS_SETUP' &&
  ros2 run ros_gz_bridge parameter_bridge \
  /model/drone_1/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry \
  /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock
"

sleep 2

open_term "3 Drone Odom TF Bridge" "
  cd '$WORKDIR' &&
  source '$ROS_SETUP' &&
  python3 drone_pose_to_odom_tf.py --ros-args -p use_sim_time:=true
"

sleep 2

open_term "4 ROS CmdVel to Gazebo Bridge" "
  source '$ROS_SETUP' &&
  ros2 run ros_gz_bridge parameter_bridge \
  /drone_1/gazebo/command/twist@geometry_msgs/msg/Twist@gz.msgs.Twist \
  --ros-args -r /drone_1/gazebo/command/twist:=/cmd_vel
"

sleep 2

open_term "5 Virtual Lidar" "
  cd '$WORKDIR' &&
  source '$ROS_SETUP' &&
  python3 virtual_lidar.py --ros-args -p use_sim_time:=true
"

sleep 2

open_term "6 Enable Drone" "
  for i in \$(seq 1 10); do
    sleep 2
    gz topic -t /drone_1/enable -m gz.msgs.Boolean -p 'data: true' &&
    echo \"Enable command \$i/10 sent.\"
  done
  echo 'Drone enable retries finished.'
"

echo "All bridge terminals started."
echo "Next steps:"
echo "  ./launch_nav2.sh        # start Nav2, wait for 'Managed nodes are active'"
echo "  ./send_nav2_goal.sh     # send goal (8,0); or: ./send_nav2_goal.sh X Y"
echo "Optional:"
echo "  ./view.sh               # RViz: costmap, path, lidar, odom trail"
echo "  ./view_gazebo.sh        # Gazebo GUI (needs 3D accel OFF in VMware)"
echo "  ./test_move_drone.sh    # manual /cmd_vel sanity check, bypasses Nav2"
echo "To stop everything: ./stop_drone_stack.sh"