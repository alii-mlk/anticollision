#!/usr/bin/env bash
# Collects everything needed to work out why a multi-drone run misbehaved into
# one file, logs/diagnostics.txt.
#
# Run it while the simulation is still up if you can: some of what it collects
# (which processes are alive, whether each MAVROS is actually talking to its
# autopilot) only exists while things are running. The log files it reads
# survive either way.
#
# Usage:
#   ./collect_diagnostics.sh
#   ./collect_diagnostics.sh --scenario scenarios3d/s21_n10_k3

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$WORKDIR/logs"
OUT="$LOGDIR/diagnostics.txt"
SCENARIO=""

if [ "${1:-}" = "--scenario" ] && [ -n "${2:-}" ]; then
  SCENARIO="$(cd "$2" 2>/dev/null && pwd)"
fi

mkdir -p "$LOGDIR"

section() {
  echo ""
  echo "================================================================"
  echo "== $*"
  echo "================================================================"
}

{
  echo "diagnostics collected $(date -Iseconds)"
  echo "host: $(uname -a)"
  echo "cores: $(nproc)   memory: $(free -h | awk '/^Mem:/ {print $2" total, "$7" available"}')"

  section "processes"
  ps -eo pid,pcpu,pmem,etimes,comm,args --no-headers 2>/dev/null \
    | grep -E "px4|gz sim|ruby|mavros_node|parameter_bridge|run_mission|collision_monitor" \
    | grep -v grep

  section "simulation speed"
  "$WORKDIR/sim_speed.py" --window 8 2>&1

  section "crash reports"
  ls -la /var/crash/ 2>/dev/null | tail -n 20
  echo "--- recent apport entries ---"
  tail -n 20 /var/log/apport.log 2>/dev/null

  section "ROS topics"
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash 2>/dev/null
  timeout 15 ros2 topic list 2>&1 | grep -E "uav|clock|mission|collision|drone" | sort

  section "ROS nodes"
  timeout 15 ros2 node list 2>&1

  section "MAVROS parameter names"
  # MAVROS 2 renamed several parameters from MAVROS 1 (tgt_system and friends),
  # and passing a name the node does not declare is an error rather than a
  # no-op. This is here to read the real names off the running node.
  for node in $(timeout 15 ros2 node list 2>/dev/null | grep -i mavros | head -n 4); do
    echo "--- $node ---"
    timeout 15 ros2 param list "$node" 2>&1 \
      | grep -iE "system|component|target|tgt|plugin|url" | head -n 20
    for p in system_id component_id target_system_id target_component_id \
             tgt_system tgt_component; do
      value="$(timeout 8 ros2 param get "$node" "$p" 2>/dev/null)"
      [ -n "$value" ] && echo "    $p = $value"
    done
  done

  for log in "$LOGDIR"/mavros_*.txt; do
    [ -f "$log" ] || continue
    i="$(basename "$log" .txt | sed 's/mavros_//')"

    section "MAVROS $i: what it says about the connection"
    grep -iE "heartbeat|connected|sysid|system id|target|autopilot|version" "$log" \
      | head -n 20
    echo "--- errors and warnings ---"
    grep -iE "error|warn|fail|refus|timeout|exception|terminate|segmentation" "$log" \
      | head -n 25
    echo "--- last 40 lines ---"
    tail -n 40 "$log"

    section "MAVROS $i: is /uav$i/state live right now?"
    timeout 8 ros2 topic echo "/uav$i/state" --once 2>&1 | head -n 20
  done

  for log in "$LOGDIR"/PX4_instance_*.txt; do
    [ -f "$log" ] || continue
    i="$(basename "$log" .txt | sed 's/PX4_instance_//')"

    section "PX4 instance $i: identity and MAVLink links"
    grep -iE "MAV_SYS_ID|sys id|system id|partner|mode: (Normal|Onboard)|udp port" "$log" \
      | head -n 20

    section "PX4 instance $i: complaints"
    echo "--- ignored commands (count by kind) ---"
    grep -o "Ignore command [0-9]* from [0-9/]* to [0-9/]*" "$log" \
      | sort | uniq -c | sort -rn | head -n 10
    echo "--- estimator, sensor and failsafe trouble ---"
    grep -icE "TIMEOUT|Attitude failure|failsafe|critical|emergency" "$log" \
      | sed 's/^/matching lines: /'
    grep -iE "TIMEOUT|Attitude failure|failsafe|critical|emergency" "$log" \
      | head -n 20
    echo "--- preflight failures ---"
    grep -o "Preflight Fail:.*" "$log" | sort | uniq -c | sort -rn | head -n 10
    echo "--- last 25 lines ---"
    tail -n 25 "$log"
  done

  section "clock bridge"
  tail -n 20 "$LOGDIR/clock_bridge.txt" 2>/dev/null

  if [ -n "$SCENARIO" ]; then
    section "scenario"
    cat "$SCENARIO/scenario.yaml" 2>/dev/null

    for d in "$SCENARIO"/run/drone*; do
      [ -d "$d" ] || continue
      section "run results: $(basename "$d")"
      echo "--- mission.yaml ---"
      cat "$d/mission.yaml" 2>/dev/null
      echo "--- outcome.yaml ---"
      cat "$d/outcome.yaml" 2>/dev/null
      echo "--- mission.log (last 40) ---"
      tail -n 40 "$d/mission.log" 2>/dev/null
      echo "--- monitor.log (last 20) ---"
      tail -n 20 "$d/monitor.log" 2>/dev/null
    done

    section "summary"
    cat "$SCENARIO/run/summary.yaml" 2>/dev/null
  fi
} > "$OUT" 2>&1

echo "Written to $OUT"
echo "$(wc -l < "$OUT") lines, $(du -h "$OUT" | cut -f1)"
