#!/usr/bin/env bash
# Starts PX4 SITL with the gz_x500 quadcopter plus the GCS heartbeat that PX4
# needs before it will arm. Opens two terminals: the PX4 shell (pxh>) and the
# heartbeat sender.
#
# PX4 itself lives outside this repository (it is a large C++ build tree, and
# building it inside the VMware shared folder is slow and unreliable). Point
# PX4_DIR at it if it is not in the default location.
#
# Usage:
#   ./start_px4_sitl.sh
#   PX4_DIR=~/somewhere/PX4-Autopilot ./start_px4_sitl.sh
#   HEADLESS=0 ./start_px4_sitl.sh        # also open the Gazebo GUI

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/px4/PX4-Autopilot}"
HEADLESS="${HEADLESS:-1}"
LOGDIR="$WORKDIR/logs"

if [ ! -d "$PX4_DIR" ]; then
  echo "ERROR: PX4 not found at $PX4_DIR" >&2
  echo "Clone it with:" >&2
  echo "  mkdir -p ~/px4 && cd ~/px4" >&2
  echo "  git clone https://github.com/PX4/PX4-Autopilot.git --recursive" >&2
  echo "  cd PX4-Autopilot && git checkout v1.17.0" >&2
  echo "  git submodule update --init --recursive" >&2
  echo "  bash ./Tools/setup/ubuntu.sh --no-nuttx" >&2
  exit 1
fi

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

echo "Stopping any previous PX4 / Gazebo processes..."
# Match the PX4 binary by exact process name. A substring match on "px4"
# would also match this script's own name and directory path, killing it.
pkill -x px4 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f "ruby.*gz sim" 2>/dev/null || true
pkill -f "gcs_heartbeat.py" 2>/dev/null || true
sleep 2

echo "Starting PX4 SITL (gz_x500) from $PX4_DIR ..."

open_term "1 PX4 SITL (pxh shell)" "
  cd '$PX4_DIR' &&
  HEADLESS=$HEADLESS make px4_sitl gz_x500
"

# PX4 needs to boot and open its MAVLink port before the heartbeat is useful.
sleep 20

open_term "2 GCS Heartbeat" "
  cd '$WORKDIR' &&
  python3 gcs_heartbeat.py
"

echo ""
echo "Two terminals started."
echo "Wait for 'Ready for takeoff!' in the PX4 shell, then in that shell:"
echo "  commander takeoff        # arms and climbs to 2.5 m"
echo "  listener vehicle_local_position    # z should reach about -2.5 (NED)"
echo "  commander land"
echo ""
echo "One-time parameter setup (already saved if you ran it before; params live"
echo "in the PX4 build tree and survive restarts but not a clean rebuild):"
echo "  param set CBRK_SUPPLY_CHK 894281   # SITL has no real power monitoring"
echo "  param set COM_DISARM_PRFLT 0       # no auto-disarm while idle on ground"
echo "  param save"
echo ""
echo "To stop everything: ./stop_px4_sitl.sh"
