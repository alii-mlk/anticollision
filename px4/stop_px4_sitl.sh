#!/usr/bin/env bash
# Stops PX4 SITL, Gazebo, and the GCS heartbeat.

echo "Stopping PX4 SITL processes..."

# Match the PX4 binary by exact process name. A substring match on "px4"
# would also match this script's own name and directory path.
pkill -x px4 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f "ruby.*gz sim" 2>/dev/null || true
pkill -f "gcs_heartbeat.py" 2>/dev/null || true
pkill -f "mavros_node" 2>/dev/null || true

echo "All PX4, Gazebo, and heartbeat processes stopped."
