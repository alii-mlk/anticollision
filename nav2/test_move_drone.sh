#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash

echo "Enabling drone..."
gz topic -t /drone_1/enable -m gz.msgs.Boolean -p "data: true"

sleep 1

echo "Moving drone forward for 5 seconds..."
timeout 5s ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {z: 0.0}}" -r 10

echo "Sending stop command for 2 seconds..."
timeout 2s ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {z: 0.0}}" -r 10

echo "Movement test finished."