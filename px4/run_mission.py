#!/usr/bin/env python3
"""Fly a drone to its scenario target and record the result.

This is the baseline mission: the drone flies straight at its target using
offboard position setpoints, with no avoidance. Obstacles are in its way, so
this run is expected to collide in a dense maze. It provides the "without
avoidance" numbers the avoidance component is later compared against.

The run ends when the drone reaches its target, is destroyed by
collision_monitor_3d.py (which disarms it), or the timeout expires.

Frames: scenario coordinates are Gazebo world coordinates, while MAVROS works
in PX4's local frame whose origin is the drone's spawn point, so the target is
converted with

    local = world - start

Time is measured on Gazebo's clock, not the wall clock. On a machine that
cannot run the simulation at full speed the two are not the same thing: with
three drones the simulator may only manage a third of real time, so a flight
that takes 40 simulated seconds occupies two wall-clock minutes. The flight
itself is unaffected, and simulated time is the honest measure of it, so
`--timeout` counts simulated seconds. `--wall-timeout` is only a safety net
for a simulation that has stopped advancing at all.

If nothing is publishing /clock the node falls back to counting wall-clock
time, which is what it always did and is correct when the simulation runs at
full speed.

Results are written to <out>/mission.yaml, and the flown path to
<out>/trajectory.csv, giving the travelled distance and the ratio against the
Euclidean distance.

Usage:
  ./run_mission.py --scenario scenarios3d/s1_n8_k1
  ./run_mission.py --scenario <dir> --out runs/r1 --timeout 180
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

SETPOINT_RATE_HZ = 20.0
PRESTREAM_SETPOINTS = 40      # ~2 s of setpoints before requesting OFFBOARD
TAKEOFF_ALTITUDE = 3.0        # climb here first, then head for the target
TAKEOFF_TOLERANCE = 0.8

# Travelled distance is measured on a smoothed position, not on raw samples.
# Pose arrives tens of times per second and each sample carries a little
# estimator jitter; summing raw steps turns that jitter into tens of metres of
# phantom travel (a hovering drone measured 16 m in one test). Low-pass
# filtering first, then ignoring sub-5 cm steps, brings a hover down to a few
# centimetres while leaving a real 20 m flight measured at 20 m.
POSITION_SMOOTHING = 0.3   # exponential moving average factor
MIN_STEP_M = 0.05

# When the drone disarms unexpectedly (a PX4 failsafe landing it, say), wait
# this long before declaring the run over. A collision also disarms the drone,
# via the monitor's force-disarm, and its "destroyed" message may arrive a
# moment later; without the grace period the run would be mislabelled.
DISARM_GRACE_S = 2.0

# How much wall-clock time a run is allowed per second of its simulated
# timeout, when no explicit wall-clock limit is given. Five covers a simulation
# running at a fifth of real time, which is well past anything usable; below
# that the run is not slow, it is stuck.
DEFAULT_WALL_TIMEOUT_FACTOR = 5.0

# How often to re-ask for offboard mode and for arming, in wall-clock seconds.
# These are asked for repeatedly rather than once: an attempt made before the
# MAVROS services have finished being discovered is simply lost, and PX4 may
# refuse the first request anyway.
REQUEST_INTERVAL_S = 1.0

# Give up if the drone has not armed after this many simulated seconds. This is
# reported as a separate outcome from a timeout, because the two are different
# failures: this one never left the ground, a timeout means it flew but did not
# arrive.
ARM_DEADLINE_S = 60.0


def parse_scenario(path):
    text = path.read_text()
    drones = []
    pattern = (r"start:\s*\{x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*z:\s*([-\d.]+)\}\s*"
               r"target:\s*\{x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*z:\s*([-\d.]+)\}")
    for m in re.finditer(pattern, text):
        v = [float(x) for x in m.groups()]
        drones.append({"start": tuple(v[0:3]), "target": tuple(v[3:6])})
    return drones


class MissionRunner(Node):
    def __init__(self, start, target, tolerance, timeout, wall_timeout,
                 out_dir, mavros_ns, drone_id):
        super().__init__("run_mission")

        self.start = start
        self.target_world = target
        # PX4's local frame originates at the spawn point.
        self.target_local = tuple(t - s for t, s in zip(target, start))
        self.tolerance = tolerance
        self.timeout = timeout
        self.wall_timeout = wall_timeout
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # MAVROS's own namespace, not a prefix added in front of one. Running
        # mavros_node with __ns:=/uav0 replaces its default /mavros namespace,
        # so the topics are /uav0/state rather than /uav0/mavros/state.
        ns = mavros_ns.rstrip("/")
        self.state = State()
        self.local_pos = None

        self.create_subscription(State, f"{ns}/state",
                                 self.state_cb, sensor_qos)
        self.create_subscription(PoseStamped,
                                 f"{ns}/local_position/pose",
                                 self.pose_cb, sensor_qos)
        # The collision monitor announces destruction here. Do not infer it
        # from the armed flag: a force-disarmed drone can still report armed
        # briefly, and PX4 may refuse an ordinary disarm outright.
        self.create_subscription(Bool,
                                 f"/collision_monitor/drone{drone_id}/destroyed",
                                 self.destroyed_cb, 10)
        # Gazebo's clock, bridged into ROS by start_px4_multi.sh. Subscribed to
        # as plain data rather than by setting use_sim_time: the setpoint
        # stream has to keep running at a real 20 Hz whatever the simulator is
        # doing, so the node's timer must stay on the wall clock.
        self.create_subscription(Clock, "/clock", self.clock_cb, sensor_qos)
        self.setpoint_pub = self.create_publisher(
            PoseStamped, f"{ns}/setpoint_position/local", 10)
        self.mission_done_pub = self.create_publisher(
            Bool, f"/mission/drone{drone_id}/done", 10)
        self.arming_client = self.create_client(CommandBool, f"{ns}/cmd/arming")
        self.mode_client = self.create_client(SetMode, f"{ns}/set_mode")

        self.sent = 0
        self.elapsed = 0.0        # simulated seconds since the run started
        self.sim_now = None       # latest /clock reading, seconds
        self.sim_start = None     # /clock at the first tick
        self.wall_start = time.monotonic()
        self.wall_elapsed = 0.0
        self.last_request = -REQUEST_INTERVAL_S
        self.climbed = False
        self.outcome = None
        self.was_armed = False
        self.destroyed = False
        self.disarmed_since = None

        self.trajectory = []       # (t, world x, y, z)
        self.travelled = 0.0
        self.last_world = None    # anchor for distance accumulation (smoothed)
        self.smoothed = None      # low-pass filtered position
        self.raw_world = None     # latest raw position, used for reporting

        self.euclidean = math.dist(start, target)
        self.timer = self.create_timer(1.0 / SETPOINT_RATE_HZ, self.tick)

        self.get_logger().info(
            f"Mission: world target ({target[0]:.2f}, {target[1]:.2f}, "
            f"{target[2]:.2f}), local ({self.target_local[0]:.2f}, "
            f"{self.target_local[1]:.2f}, {self.target_local[2]:.2f}), "
            f"Euclidean {self.euclidean:.2f} m"
        )

    def state_cb(self, msg):
        self.state = msg
        if msg.armed:
            self.was_armed = True

    def destroyed_cb(self, msg):
        if msg.data:
            self.destroyed = True

    def clock_cb(self, msg):
        self.sim_now = msg.clock.sec + msg.clock.nanosec * 1e-9

    def advance_time(self):
        """Update elapsed simulated time and elapsed wall-clock time."""
        self.wall_elapsed = time.monotonic() - self.wall_start

        if self.sim_now is None:
            # Nothing is publishing /clock, so simulated and wall-clock time
            # are assumed to be the same. True whenever the simulator keeps up.
            self.elapsed += 1.0 / SETPOINT_RATE_HZ
            return

        if self.sim_start is None:
            self.sim_start = self.sim_now
        self.elapsed = self.sim_now - self.sim_start

    @property
    def real_time_factor(self):
        if self.wall_elapsed <= 0.0:
            return 0.0
        return self.elapsed / self.wall_elapsed

    def pose_cb(self, msg):
        p = msg.pose.position
        self.local_pos = (p.x, p.y, p.z)
        world = (p.x + self.start[0], p.y + self.start[1], p.z + self.start[2])

        if self.smoothed is None:
            self.smoothed = list(world)
            self.last_world = tuple(world)
        else:
            a = POSITION_SMOOTHING
            self.smoothed = [s + a * (w - s)
                             for s, w in zip(self.smoothed, world)]
            step = math.dist(self.last_world, self.smoothed)
            if step >= MIN_STEP_M:
                self.travelled += step
                self.last_world = tuple(self.smoothed)

        # The reported position stays the raw one; only the distance
        # accumulator uses the smoothed track.
        self.raw_world = world
        self.trajectory.append((self.elapsed, *world))

    def make_setpoint(self, xyz):
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = "map"
        sp.pose.position.x, sp.pose.position.y, sp.pose.position.z = xyz
        sp.pose.orientation.w = 1.0
        return sp

    def tick(self):
        self.advance_time()

        # Climb straight up first, then fly to the target. Heading for the
        # target from the ground would drag the drone through obstacles at
        # ground level before it has any altitude.
        if not self.climbed:
            setpoint = (0.0, 0.0, TAKEOFF_ALTITUDE)
        else:
            setpoint = self.target_local

        # A destroyed drone is not flown any further, so stop the setpoint
        # stream before publishing anything else this tick.
        if self.destroyed and self.outcome is None:
            self.finish("destroyed")
            return

        self.setpoint_pub.publish(self.make_setpoint(setpoint))
        self.sent += 1

        if self.sent < PRESTREAM_SETPOINTS:
            return

        # Deadlines are checked before anything that can return early. A
        # mission whose autopilot never answers still has to end by itself.
        # The position check below used to come first, so a drone that never
        # armed never reached the timeout and had to be killed from outside.
        self.check_deadlines()

        if self.state.mode != "OFFBOARD" or not self.state.armed:
            if self.wall_elapsed - self.last_request >= REQUEST_INTERVAL_S:
                self.last_request = self.wall_elapsed
                if self.state.mode != "OFFBOARD":
                    self.request_mode("OFFBOARD")
                else:
                    self.request_arm()

        if self.local_pos is None:
            return

        # An unexpected disarm means PX4 gave up on the flight, typically a
        # failsafe landing it. Waiting out the timeout in that state tells us
        # nothing and costs minutes in a batch sweep, so end the run. The grace
        # period lets a collision's "destroyed" message win the race, since a
        # collision disarms the drone too.
        if self.was_armed and not self.state.armed:
            if self.disarmed_since is None:
                self.disarmed_since = self.elapsed
            elif self.elapsed - self.disarmed_since >= DISARM_GRACE_S:
                self.finish("disarmed")
                return
        else:
            self.disarmed_since = None

        if not self.climbed:
            if abs(self.local_pos[2] - TAKEOFF_ALTITUDE) <= TAKEOFF_TOLERANCE:
                self.climbed = True
                self.get_logger().info(
                    f"Reached {TAKEOFF_ALTITUDE:.1f} m, heading for the target.")
            return

        dist = math.dist(self.local_pos, self.target_local)
        if self.sent % 40 == 0:
            w = self.raw_world or (0, 0, 0)
            self.get_logger().info(
                f"mode={self.state.mode} armed={self.state.armed} "
                f"world=({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}) "
                f"dist={dist:.2f} m travelled={self.travelled:.1f} m "
                f"sim={self.elapsed:.0f} s rtf={self.real_time_factor:.2f}"
            )

        if dist <= self.tolerance:
            self.finish("reached")

    def check_deadlines(self):
        """End the run if any deadline has passed. finish() does not return."""
        if self.elapsed > self.timeout:
            self.finish("timeout")

        if not self.was_armed and self.elapsed > ARM_DEADLINE_S:
            self.get_logger().error(
                f"Never armed after {self.elapsed:.0f} simulated seconds "
                f"(connected={self.state.connected}, mode='{self.state.mode}'). "
                f"The autopilot is not answering this drone."
            )
            self.finish("not_armed")

        # A simulation that has stopped advancing would never reach the
        # simulated timeout above, so the run would hang forever. This is the
        # only thing the wall clock decides.
        if self.wall_elapsed > self.wall_timeout:
            self.get_logger().error(
                f"Wall-clock limit reached after {self.wall_elapsed:.0f} s "
                f"with only {self.elapsed:.0f} s simulated "
                f"(real time factor {self.real_time_factor:.2f}). "
                f"The simulator is not keeping up."
            )
            self.finish("stalled")

    def request_mode(self, mode):
        if self.mode_client.service_is_ready():
            req = SetMode.Request()
            req.base_mode = 0
            req.custom_mode = mode
            self.mode_client.call_async(req)
            self.get_logger().info(f"Requested mode {mode}")

    def request_arm(self):
        if self.arming_client.service_is_ready():
            req = CommandBool.Request()
            req.value = True
            self.arming_client.call_async(req)
            self.get_logger().info("Requested arm")

    def finish(self, outcome):
        self.outcome = outcome
        # Tell the monitor the flight is over. Without this it keeps watching
        # while PX4, no longer receiving setpoints, drops out of offboard and
        # descends, and it then reports a collision that has nothing to do
        # with the mission.
        self.mission_done_pub.publish(Bool(data=True))
        self.write_results()
        self.get_logger().info(
            f"Mission {outcome}: travelled {self.travelled:.2f} m, "
            f"Euclidean {self.euclidean:.2f} m, "
            f"ratio {self.travelled / self.euclidean:.3f}, "
            f"simulated time {self.elapsed:.1f} s in {self.wall_elapsed:.1f} s "
            f"of wall clock (real time factor {self.real_time_factor:.2f})"
        )
        raise SystemExit(0 if outcome == "reached" else 1)

    def write_results(self):
        ratio = self.travelled / self.euclidean if self.euclidean > 0 else 0.0
        w = self.raw_world or (0.0, 0.0, 0.0)
        (self.out_dir / "mission.yaml").write_text(
            f"outcome: {self.outcome}\n"
            f"euclidean_distance: {self.euclidean:.3f}\n"
            f"travelled_distance: {self.travelled:.3f}\n"
            f"ratio: {ratio:.3f}\n"
            f"simulation_time: {self.elapsed:.2f}\n"
            f"wall_time: {self.wall_elapsed:.2f}\n"
            f"real_time_factor: {self.real_time_factor:.3f}\n"
            f"clock_source: {'gazebo' if self.sim_now is not None else 'wall'}\n"
            f"final_world_position: {{x: {w[0]:.3f}, y: {w[1]:.3f}, z: {w[2]:.3f}}}\n"
            f"target_world_position: {{x: {self.target_world[0]:.3f}, "
            f"y: {self.target_world[1]:.3f}, z: {self.target_world[2]:.3f}}}\n"
        )
        with open(self.out_dir / "trajectory.csv", "w") as f:
            f.write("t,x,y,z\n")
            for t, x, y, z in self.trajectory:
                f.write(f"{t:.2f},{x:.3f},{y:.3f},{z:.3f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--drone-id", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.6)
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="mission timeout in SIMULATED seconds")
    ap.add_argument("--wall-timeout", type=float, default=0.0,
                    help="wall-clock safety limit in seconds "
                         f"(default: {DEFAULT_WALL_TIMEOUT_FACTOR:.0f} x "
                         "--timeout, which allows the simulation to run at a "
                         "fifth of real time)")
    ap.add_argument("--mavros-ns", default="/mavros",
                    help="MAVROS namespace: /mavros for a single drone, "
                         "/uav0, /uav1, ... for multi-drone runs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scenario_file = args.scenario / "scenario.yaml"
    if not scenario_file.exists():
        sys.exit(f"not found: {scenario_file}")

    drones = parse_scenario(scenario_file)
    if args.drone_id >= len(drones):
        sys.exit(f"drone {args.drone_id} not in scenario ({len(drones)} drones)")

    d = drones[args.drone_id]
    out_dir = args.out or args.scenario
    wall_timeout = (args.wall_timeout
                    or args.timeout * DEFAULT_WALL_TIMEOUT_FACTOR)

    rclpy.init()
    node = MissionRunner(d["start"], d["target"], args.tolerance,
                         args.timeout, wall_timeout, out_dir,
                         args.mavros_ns, args.drone_id)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node.outcome is None:
            node.outcome = "interrupted"
            node.write_results()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
