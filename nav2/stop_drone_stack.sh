#!/usr/bin/env bash

echo "Stopping drone/Nav2 bridge processes..."

pkill -f "ros2 topic pub /cmd_vel" || true
pkill -f "drone_pose_to_odom_tf.py" || true
pkill -f "ros_gz_bridge.*parameter_bridge" || true
pkill -f "gz sim" || true
pkill -f "ruby.*gz sim" || true
pkill -f "virtual_lidar.py" || true
pkill -f "hit_monitor.py" || true
pkill -f "obstacle_mover.py" || true
pkill -f "nav2_minimal.launch.py" || true

echo "All simulation, bridge, and Nav2 processes stopped."