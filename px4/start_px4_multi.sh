#!/usr/bin/env bash
# Starts k PX4 SITL instances (one per drone in the scenario), each spawned at
# its own start point, all sharing one Gazebo world. Also starts one GCS
# heartbeat and one MAVROS bridge per drone.
#
# Port scheme, taken from PX4's own ROMFS/px4fmu_common/init.d-posix/px4-rc.mavlink:
#
#   udp_offboard_port_local  = 14580 + instance    PX4 listens here
#   udp_offboard_port_remote = 14540 + instance    MAVROS binds here
#   udp_gcs_port_local       = 18570 + instance    heartbeat target
#
# So drone i is reached at udp://:$((14540+i))@127.0.0.1:$((14580+i)), and each
# drone's MAVROS runs under its own /uavN namespace so their topics do not
# collide.
#
# The PX4 instances each get a terminal (their pxh> shells are useful), while
# the heartbeats and MAVROS run in the background with logs under logs/.
#
# Usage:
#   ./start_px4_multi.sh --scenario scenarios3d/s7_n15_k3
#   PX4_DIR=~/somewhere/PX4-Autopilot ./start_px4_multi.sh --scenario <dir>

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/px4/PX4-Autopilot}"
HEADLESS="${HEADLESS:-1}"
LOGDIR="$WORKDIR/logs"

# Which PX4 SITL build to run.
#
# The default build runs the simulation in lockstep: Gazebo only advances when
# the autopilot says so. That is fine for one vehicle, but with three PX4
# instances the physics has to stay in step with all of them, the estimators
# start missing sensor data ("Accel #0 fail: TIMEOUT", "Attitude failure"),
# and the drones fall out of the sky. PX4 ships a nolockstep configuration for
# exactly this case:
#
#   cd ~/px4/PX4-Autopilot && make px4_sitl_nolockstep
#   PX4_BUILD=px4_sitl_nolockstep ./start_px4_multi.sh --scenario <dir>
#
# Without lockstep the simulation runs in real time and runs are no longer
# bit-for-bit reproducible, which is the usual trade for multi-vehicle work.
PX4_BUILD="${PX4_BUILD:-px4_sitl_default}"

if [ "${1:-}" != "--scenario" ] || [ -z "${2:-}" ]; then
  echo "usage: $0 --scenario <scenario_dir>" >&2
  exit 1
fi
SCENARIO_DIR="$(cd "$2" && pwd)"
SCENARIO_FILE="$SCENARIO_DIR/scenario.yaml"

if [ ! -f "$SCENARIO_FILE" ]; then
  echo "ERROR: $SCENARIO_FILE not found" >&2
  exit 1
fi
if [ ! -d "$PX4_DIR" ]; then
  echo "ERROR: PX4 not found at $PX4_DIR" >&2
  exit 1
fi
if [ ! -x "$PX4_DIR/build/$PX4_BUILD/bin/px4" ]; then
  echo "ERROR: PX4 build '$PX4_BUILD' not found at" >&2
  echo "  $PX4_DIR/build/$PX4_BUILD/bin/px4" >&2
  echo "Build it with:  cd $PX4_DIR && make ${PX4_BUILD/px4_sitl_/px4_sitl_}" >&2
  exit 1
fi

mkdir -p "$LOGDIR"

# Old logs are removed before anything starts, so that waiting for a line like
# "Ready for takeoff" or "PARAMS DONE" cannot match one left over from the
# previous run.
rm -f "$LOGDIR"/PX4_instance_*.txt "$LOGDIR"/heartbeat_*.txt

# Written only once every drone is up and every SITL parameter has been read
# back. run_multi_mission.py waits for it. Bringing the stack up takes minutes
# and the last step is silent for most of it, which makes it very easy to
# start flying too early: drones that take off before SIM_BAT_MIN_PCT is
# applied drain their batteries and are landed by the failsafe mid-mission.
READY_MARKER="$LOGDIR/stack_ready"
rm -f "$READY_MARKER"

# One "x,y,z,0,0,0" spawn pose per drone, in scenario order.
mapfile -t POSES < <(python3 -c "
import re
text = open('$SCENARIO_FILE').read()
for m in re.finditer(r'start: \{x: ([-\d.]+), y: ([-\d.]+), z: ([-\d.]+)\}', text):
    print(f'{m.group(1)},{m.group(2)},{m.group(3)},0,0,0')
")
K=${#POSES[@]}

if [ "$K" -eq 0 ]; then
  echo "ERROR: no drones found in $SCENARIO_FILE" >&2
  exit 1
fi

echo "Scenario: $SCENARIO_DIR"
echo "Drones:   $K"

echo "Stopping any previous PX4 / Gazebo / MAVROS processes..."
pkill -x px4 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f "ruby.*gz sim" 2>/dev/null || true
pkill -f "gcs_heartbeat.py" 2>/dev/null || true
pkill -f "mavros_node" 2>/dev/null || true
pkill -f "parameter_bridge" 2>/dev/null || true
sleep 3

# With BATCH=1 nothing opens a window: everything runs in the background with
# its output going to a log file, which is what the unattended sweep needs.
#
# In either case PX4's output goes through a pipe rather than to a terminal,
# which makes its C library buffer whole blocks at a time. Waiting for a line
# to appear in the log would then wait for the next few kilobytes of output
# instead, so PX4 is run under stdbuf to keep it line by line.
open_term() {
  local title="$1"
  local cmd="$2"
  local logfile="$LOGDIR/$(echo "$title" | tr ' /' '__').txt"

  if [ "${BATCH:-0}" = "1" ]; then
    ( eval "$cmd" ) > "$logfile" 2>&1 &
  else
    gnome-terminal --title="$title" -- bash -lc "
      echo '===== $title ====='
      ( $cmd ) 2>&1 | tee '$logfile'
    "
  fi
}

# How long PX4 takes to come up depends on how fast the machine can run the
# simulation, so wait for it to say it is ready rather than sleeping a fixed
# number of seconds. On a machine simulating in slow motion any fixed guess is
# wrong, and starting the next instance too early is what turns a slow machine
# into a broken one.
wait_for_px4() {
  local i="$1"
  local limit="${PX4_READY_TIMEOUT:-240}"
  local log="$LOGDIR/PX4_instance_$i.txt"
  local waited=0
  while [ "$waited" -lt "$limit" ]; do
    if grep -q "Ready for takeoff" "$log" 2>/dev/null; then
      echo "    instance $i ready after ${waited}s"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
    # Without this the script looks frozen for minutes. Showing PX4's own
    # last complaint says whether it is still starting up or stuck on
    # something.
    if [ $((waited % 30)) -eq 0 ]; then
      local last
      last="$(grep -o "Preflight Fail:.*" "$log" 2>/dev/null | tail -n 1)"
      echo "    still waiting (${waited}s) ${last:+- $last}"
    fi
  done
  echo "    WARNING: instance $i never reported 'Ready for takeoff' (${limit}s)"
  echo "             see $log"
  return 1
}

for ((i = 0; i < K; i++)); do
  POSE="${POSES[$i]}"
  echo ""
  echo "--- drone $i: pose $POSE ---"

  # Every instance otherwise starts a uXRCE-DDS client on the same port 8888,
  # which floods the logs with "Exception sending a multicast message" and
  # wastes CPU. We reach PX4 over MAVLink/MAVROS, not uXRCE-DDS, so give each
  # instance its own port to keep them from colliding.
  DDS_PORT=$((8888 + i))

  open_term "PX4 instance $i" "
    cd '$PX4_DIR' &&
    PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500 PX4_GZ_MODEL_POSE='$POSE' \
    PX4_UXRCE_DDS_PORT=$DDS_PORT \
    HEADLESS=$HEADLESS stdbuf -oL -eL ./build/$PX4_BUILD/bin/px4 -i $i
  "

  # The heartbeat starts before we wait for the instance, not after. PX4 will
  # not report "Ready for takeoff" while its arming checks are still saying
  # "No connection to the GCS", so waiting first means waiting for something
  # only this heartbeat can bring about.
  # The heartbeat also applies the SITL parameters, over the same MAVLink
  # connection. MAVROS did this before, but it downloads the autopilot's whole
  # parameter list first, and with several instances that download does not
  # finish, so every attempt fails with "Unknown parameter to set".
  GCS_PORT=$((18570 + i))
  ( cd "$WORKDIR" && python3 gcs_heartbeat.py --port "$GCS_PORT" \
      --set CBRK_SUPPLY_CHK=894281 \
      --set COM_DISARM_PRFLT=0 \
      --set SIM_BAT_MIN_PCT=90 ) \
    > "$LOGDIR/heartbeat_$i.txt" 2>&1 &
  echo "    heartbeat -> udp $GCS_PORT"

  # Instance 0 also starts Gazebo, which is the slowest part of the startup.
  if [ "$i" -eq 0 ]; then
    echo "    waiting for Gazebo and PX4 to come up ..."
  else
    echo "    waiting for PX4 to come up ..."
  fi
  wait_for_px4 "$i" || true
done

# Waits for a MAVROS instance to reach its autopilot, and checks it is talking
# to the right one.
wait_for_mavros() {
  local i="$1"
  local tgt="$2"
  local log="$LOGDIR/mavros_$i.txt"
  local waited=0
  while [ "$waited" -lt "${MAVROS_READY_TIMEOUT:-90}" ]; do
    if grep -q "Got HEARTBEAT" "$log" 2>/dev/null; then
      local ids
      ids="$(grep -o "MY ID [0-9.]*, TARGET ID [0-9.]*" "$log" | tail -n 1)"
      echo "    /uav$i connected ($ids)"
      case "$ids" in
        *"TARGET ID $tgt.1") ;;
        *) echo "    WARNING: /uav$i is addressing the wrong autopilot;" \
                "expected TARGET ID $tgt.1" ;;
      esac
      return 0
    fi
    # A parameter MAVROS does not declare is a startup error, not a warning,
    # and the node never comes up at all. Say so instead of waiting it out.
    if grep -qiE "not declared|InvalidParameter|ParameterNotDeclared" "$log" 2>/dev/null; then
      echo "    ERROR: MAVROS rejected one of its parameters:"
      grep -iE "not declared|InvalidParameter|ParameterNotDeclared" "$log" | tail -n 2
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "    WARNING: /uav$i never reported a heartbeat; see $log"
  return 1
}

echo ""
echo "Starting one MAVROS per drone ..."
# PX4 gives instance i the MAVLink system ID i+1, while MAVROS addresses
# system 1 unless it is told otherwise. Left at the default, all k MAVROS
# nodes talk to system 1: instance 0 answers, and every other instance logs
# "Ignore command 400 from 1/191 to 1/1" and its drone can never arm. Worse,
# the unanswered commands get retried until two acknowledgements race and
# MAVROS aborts with "Promise already satisfied". So each MAVROS is given its
# own autopilot's system ID.
#
# The denied plugins are not used here and they are not idle either.
# guided_target logs "PositionTargetGlobal failed because no origin" many
# times per second, and waypoint, rallypoint and geofence keep asking the
# autopilot for lists. Disabling them cut each MAVROS from about 65% of a core
# to about 47%.
MAVROS_DENYLIST="${MAVROS_DENYLIST:-[guided_target,landing_target,waypoint,rallypoint,geofence]}"

rm -f "$LOGDIR"/mavros_*.txt
# One at a time, each waited for before the next is launched. Started together,
# all k nodes broadcast their autopilot-version request at once, the replies
# time out under the load, and a late reply then satisfies a promise that the
# retry has already satisfied, which aborts the node with
# "std::future_error: Promise already satisfied".
for ((i = 0; i < K; i++)); do
  BIND=$((14540 + i))
  REMOTE=$((14580 + i))
  TGT=$((i + 1))
  ( source /opt/ros/jazzy/setup.bash &&
    ros2 run mavros mavros_node --ros-args \
      -r __ns:=/uav$i \
      -p fcu_url:="udp://:$BIND@127.0.0.1:$REMOTE" \
      -p tgt_system:=$TGT \
      -p tgt_component:=1 \
      -p plugin_denylist:="$MAVROS_DENYLIST" ) \
    > "$LOGDIR/mavros_$i.txt" 2>&1 &
  echo "  /uav$i  <- udp://:$BIND@127.0.0.1:$REMOTE  (autopilot system $TGT)"
  wait_for_mavros "$i" "$TGT" || true
done

echo ""
echo "Bridging Gazebo's clock into ROS ..."
# The mission runner measures simulation time from /clock. Wall-clock time
# stops being a usable substitute as soon as the machine cannot simulate every
# drone at full speed, which is exactly the case this script exists for.
( source /opt/ros/jazzy/setup.bash &&
  ros2 run ros_gz_bridge parameter_bridge \
    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" ) \
  > "$LOGDIR/clock_bridge.txt" 2>&1 &
sleep 3
echo "  /clock"

echo ""
echo "Waiting for the SITL parameters to be applied ..."
# Each heartbeat process reports which parameters it set. Check that instead of
# assuming: a drone flying without SIM_BAT_MIN_PCT drains its battery and the
# failsafe lands it in the middle of the mission.
wait_for_params() {
  local i="$1"
  local log="$LOGDIR/heartbeat_$i.txt"
  local waited=0
  while [ "$waited" -lt "${PARAM_TIMEOUT:-240}" ]; do
    if grep -q "PARAMS DONE" "$log" 2>/dev/null; then
      grep -h "^PARAM OK" "$log" | sed "s/^/  uav$i /"
      return 0
    fi
    if grep -q "PARAMS INCOMPLETE" "$log" 2>/dev/null; then
      echo "  WARNING: uav$i did not receive every parameter:"
      grep -hE "^PARAM (OK|FAIL)|^PARAMS INCOMPLETE" "$log" | sed "s/^/    /"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "  WARNING: uav$i never finished applying parameters; see $log"
  return 1
}

for ((i = 0; i < K; i++)); do
  wait_for_params "$i" || true
done

echo ""
echo "SITL parameters applied."

printf 'scenario: %s\ndrones: %s\nready_at: %s\n' \
  "$SCENARIO_DIR" "$K" "$(date -Iseconds)" > "$READY_MARKER"

echo ""
echo "================================================================"
echo " READY. All $K drone(s) are up and their parameters are applied."
echo " Do not start the mission before this line appears."
echo "================================================================"
echo ""
# How fast the simulation is actually running decides whether this many drones
# are viable on this machine at all, so measure it rather than assume it.
"$WORKDIR/sim_speed.py" --window 8 || true
echo ""
echo "Check that each MAVROS connected:"
echo "  grep -l 'Got HEARTBEAT' $LOGDIR/mavros_*.txt"
echo ""
echo "Next:"
echo "  ./spawn_obstacles.py --scenario $SCENARIO_DIR"
echo "  ./run_multi_mission.py --scenario $SCENARIO_DIR"
echo ""
echo "To stop everything: ./stop_px4_sitl.sh"
