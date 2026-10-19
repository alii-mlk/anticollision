# Source this in every new terminal before doing any RL work:
#
#   source rl/env.sh
#
# ROS comes first and the venv second. Sourcing ROS puts rclpy on the Python
# path, and the venv keeps it visible because it was created with
# --system-site-packages. The other order loses rclpy.
#
# No "set -e" or "set -u" here: this file runs inside your own shell, so an
# error would close the terminal, and ROS's setup.bash breaks under set -u.

source /opt/ros/jazzy/setup.bash
source "$HOME/rlenv/bin/activate"
