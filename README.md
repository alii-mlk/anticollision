# anticollision

Drone obstacle avoidance in simulation. The project has two stages: a 2D
baseline built on ROS 2 Nav2, and a 3D stage built on PX4 and MAVLink.

## Stage 1: Nav2 in 2D (complete)

A quadcopter navigates around obstacles using the ROS 2 Nav2 stack.

- *Static obstacles*: 48/48 randomized runs collision-free across 1–10
  obstacles; navigation time and detour overhead grow mildly with obstacle
  count (`results/2026-07-17_static_sweep/`).
- *Moving obstacles*: collision avoidance degrades sharply with obstacle
  speed. Total hits rise from 1 to 73 (8 runs per speed) between 0.2 and
  2.0 m/s, navigation time doubles, and beyond ~0.8 m/s obstacles outrun the
  drone entirely (`results/2026-07-20_speed_sweep/`). This quantifies where
  prediction-free Nav2 stops being sufficient.

## Stage 2: PX4 and MAVLink in 3D (in progress)

Nav2 only plans in 2D, and it is not how real drones are flown. This stage
replaces it with the standard autopilot stack: PX4 as the flight controller
and MAVLink as the protocol, so the software is independent of the drone.

Working so far: PX4 SITL with Gazebo, the MAVLink bridge (MAVROS), 3D maze
scenarios with obstacles standing on the ground, offboard flight to a 3D
target, and collision detection that destroys the drone on contact.

Note on PX4 3D navigation: PX4 has no maintained 3D path planner. The
PX4-Avoidance package is archived and ROS 1 only, the path planning interface
is withdrawn, and the built-in Collision Prevention is horizontal and works
only in Position mode. Planning therefore happens on the ROS 2 side and PX4
executes the setpoints, which is what PX4 itself now recommends. Details and
sources in `px4/FEASIBILITY_3D_NAVIGATION.md`.

## How the pieces fit together

Both stages use the same idea: the simulator provides physics, something
flies the drone, and our own code decides where it should go.

```
Stage 1:  our code -> Nav2 -> /cmd_vel -> bridge -> Gazebo
Stage 2:  our code -> MAVROS -> MAVLink -> PX4 -> Gazebo
```

In stage 2 the four programs are:

| Program | Role |
|---|---|
| Gazebo | the physics world: gravity, the drone's body, the obstacles |
| PX4 (SITL) | the flight controller, the software version of the board on a real drone. Keeps the drone stable and executes position commands |
| MAVROS | the translator between PX4's MAVLink protocol and ROS 2 |
| our Python nodes | the planner: chooses the target, watches for collisions, records metrics |

## Requirements

Stage 1 (Nav2):

- Ubuntu 24.04 with ROS 2 Jazzy (`/opt/ros/jazzy`)
- Gazebo (gz-sim, Harmonic) + `ros-jazzy-ros-gz-bridge`
- `ros-jazzy-navigation2`, `ros-jazzy-nav2-bringup`
- The X3 UAV model from Gazebo Fuel (auto-downloaded on first `gz sim` run with internet, cached under `~/.gz/fuel`)

Stage 2 (PX4) additionally:

- PX4-Autopilot v1.17.0 built for SITL, outside this repository (see
  `px4/start_px4_sitl.sh` for the clone and build commands). Do not build it
  inside a VM shared folder
- `ros-jazzy-mavros`, `ros-jazzy-mavros-msgs`, `ros-jazzy-mavros-extras`, plus
  the GeographicLib datasets (`sudo bash
  /opt/ros/jazzy/lib/mavros/install_geographiclib_datasets.sh`)

## Repository layout

```
Thesis Progress.docx          # running progress report
results/                      # curated, citable run evidence (one folder per experiment)
early_versions/               # superseded experiments, kept for project history

px4/                          # STAGE 2: PX4 + MAVLink, 3D
  start_px4_sitl.sh           # boots Gazebo + PX4 SITL (and the heartbeat below);
                              #   --scenario <dir> spawns the drone at the scenario start
  stop_px4_sitl.sh            # stops PX4, Gazebo, MAVROS and the heartbeat
  gcs_heartbeat.py            # PX4 refuses to arm with no ground station watching;
                              #   this sends the minimal MAVLink heartbeat to satisfy it
  gen_scenario_3d.py          # random 3D maze: n ground-standing boxes with random
                              #   footprint and height, k drones with 3D targets -> scenario.yaml
  spawn_obstacles.py          # inserts those obstacles into the running Gazebo world
  collision_monitor_3d.py     # 3D collision check; force-disarms ("destroys") the drone on contact
  run_mission.py              # flies a drone to its target via offboard setpoints,
                              #   records travelled/Euclidean distance, ratio, time
  offboard_goto.py            # minimal single-target offboard flight, used to verify the bridge
  FEASIBILITY_3D_NAVIGATION.md # why PX4 has no usable 3D planner, with sources
  scenarios3d/                # generated scenarios (gitignored; reproducible from seed)
  logs/                       # per-terminal logs (gitignored)

nav2/                         # STAGE 1: Nav2, 2D
  drone_nav2_world.sdf        # world: ground plane, obstacle_1 (1x4x2 wall at x=4), X3 drone
                              #   with multicopter motor/velocity-control + OdometryPublisher plugins
  start_drone_stack.sh        # launches Gazebo (headless) + all bridges in separate terminals
  stop_drone_stack.sh         # kills everything
  drone_pose_to_odom_tf.py    # republishes /model/drone_1/odometry as /odom + odom->base_link TF,
                              #   publishes static map->odom (identity)
  virtual_lidar.py            # analytic 2D lidar: raycasts drone pose against known obstacle
                              #   boxes, publishes sensor_msgs/LaserScan on /scan (see note below)
  nav2_params.yaml            # Nav2 config: mapless costmaps fed by /scan, holonomic DWB
  nav2_minimal.launch.py      # launches only controller/planner/behavior/bt_navigator + lifecycle mgr
  launch_nav2.sh              # runs the launch file (logs to logs/nav2.txt)
  send_nav2_goal.sh           # sends NavigateToPose goal, default (8,0); logs to logs/goal_<time>.txt
  test_move_drone.sh          # sanity check: drives the drone via /cmd_vel directly, bypassing Nav2
  view.sh + drone_view.rviz   # RViz2 view: odometry trail, lidar hits, costmap, planned path
  view_gazebo.sh              # attaches the Gazebo GUI to the running headless server
  gen_scenario.py             # random scenario generator: start/goal/N obstacles ->
                              #   scenarios/<name>/{world.sdf, scenario.yaml} (single source)
  hit_monitor.py              # counts obstacle hits (doesn't stop the run) + min clearance
  obstacle_mover.py           # obstacle ground truth: integrates motion (bouncing), publishes
                              #   /obstacles markers consumed by lidar + monitor + RViz
  run_batch.sh                # unattended sweep: N obstacles x seeds -> runs/batch_<ts>/
  compute_metrics.py          # batch dir -> metrics.csv + per-N summary (+ plots)
  scenarios/                  # generated scenarios (gitignored; reproducible from seed)
  runs/                       # batch outputs (gitignored; promote keepers to results/)
  logs/                       # all terminals and runs tee their output here (gitignored)
```

## Stage 2: running a 3D maze mission

Four terminals, in this order. All commands from the `px4/` directory, and
every ROS terminal needs `source /opt/ros/jazzy/setup.bash` first.

**0. Generate a scenario** (once per experiment; the seed makes it repeatable):

```bash
./gen_scenario_3d.py --n-obstacles 8 --seed 1     # -> scenarios3d/s1_n8_k1/
```

**1. Terminal A, the simulator.** Boots Gazebo and PX4, and spawns the drone
at the scenario's start point. Wait for `Ready for takeoff!` in the PX4 window
before continuing; it means PX4 has booted and passed its preflight checks.

```bash
./start_px4_sitl.sh --scenario scenarios3d/s1_n8_k1
```

Then put the obstacles into the world (the world starts empty):

```bash
./spawn_obstacles.py --scenario scenarios3d/s1_n8_k1
```

Expect `Acknowledged 8/8, 8 present in world 'default'`. The count is checked
against the world itself, because the spawn service acknowledges requests that
Gazebo may still fail to build.

**2. Terminal B, the MAVLink bridge.** Leave it running. Wait for
`CON: Got HEARTBEAT, connected. FCU: PX4 Autopilot`.

```bash
ros2 run mavros mavros_node --ros-args -p fcu_url:="udp://:14540@127.0.0.1:14580"
```

**3. Terminal C, the collision monitor.** Start it before the mission so it is
watching from the first moment.

```bash
./collision_monitor_3d.py --scenario scenarios3d/s1_n8_k1
```

**4. Terminal D, the mission.** The drone climbs to 3 m, then flies to its 3D
target.

```bash
./run_mission.py --scenario scenarios3d/s1_n8_k1
```

Results:

```bash
cat scenarios3d/s1_n8_k1/mission.yaml    # outcome, distances, ratio, time
cat scenarios3d/s1_n8_k1/outcome.yaml    # destroyed?, min clearance, collision point
```

Teardown: `./stop_px4_sitl.sh`, and Ctrl+C the MAVROS and monitor terminals.

### One-time PX4 parameters

These live in the PX4 build tree and survive restarts, but not a clean
rebuild. In the PX4 shell (`pxh>`):

```
param set CBRK_SUPPLY_CHK 894281   # SITL has no real power monitoring
param set COM_DISARM_PRFLT 0       # do not auto-disarm while idle on the ground
param set SIM_BAT_MIN_PCT 90       # keep the simulated battery full, so long runs
                                   #   are not cut short by a low-battery failsafe
param save
```

### Notes on the PX4 stage

- **Frames.** Scenario coordinates are Gazebo world coordinates, but MAVROS
  reports position in PX4's local frame, whose origin is wherever the drone
  spawned. The nodes convert with `world = local + start`. Getting this wrong
  sends the drone to a completely different place, so it is worth checking
  that a reached target's `final_world_position` is close to
  `target_world_position`.
- **Destroying a drone.** PX4 refuses an ordinary disarm command in flight
  ("Disarming denied: not landed"), so the monitor sends
  `MAV_CMD_COMPONENT_ARM_DISARM` with the force value 21196. Without this the
  drone survives its own collision and keeps flying, which produced a
  meaningless travelled distance of 350 m in one test.
- **Ratio for destroyed runs is not meaningful.** A drone that dies halfway
  has travelled less than the straight-line distance, giving a ratio below 1.
  Averages should be taken over successful runs only, with destroyed runs
  counted separately.

## Stage 1: running Nav2 (step by step)

All commands from the `nav2/` directory.

**1. Start the simulation + bridges:**

```bash
./start_drone_stack.sh
```

Opens 8 terminals: the Gazebo server (headless), the odometry/clock and cmd_vel
bridges, the odom/TF republisher, the virtual lidar, the hit monitor, the
obstacle mover, and an arming loop that enables the drone's motors (retries for
~20 s). Wait until the "3 Drone Odom TF Bridge" terminal starts printing
`Publishing /odom: ...` lines.

**2. Start Nav2 (new terminal):**

```bash
./launch_nav2.sh
```

Wait for `Managed nodes are active`, which means Nav2 is ready. Leave this running.

**3. Send a navigation goal (new terminal):**

```bash
./send_nav2_goal.sh           # default goal (8,0): straight line would cross the wall
./send_nav2_goal.sh 0 0       # custom goal: fly back, rounding the wall again
```

The terminal streams feedback (distance remaining, recoveries) and ends with
`Goal finished with status: SUCCEEDED`. Every run is also logged to
`logs/goal_<time>.txt` (the full feedback stream, which is the source for
trajectory metrics like path length and path/Euclidean ratio).

**4. Teardown:**

```bash
./stop_drone_stack.sh         # kills Gazebo + bridges; Ctrl+C the Nav2 terminal
```

## Randomized scenarios

Following the evaluation protocol (random start, random goal, N random fixed
obstacles, count hits instead of stopping).

**Generator parameters** (`./gen_scenario.py --help`):

| Parameter | Meaning |
|---|---|
| `--n-obstacles N` | required; number of random boxes to place |
| `--seed S` | required; same seed + same N = identical scenario (reproducible) |
| `--obstacle-speed V` | optional; m/s, random direction per obstacle (default 0 = static) |
| `--out DIR` | optional; output directory (default `scenarios/s<seed>_n<N>[_v<speed>]/`) |

Placement rules baked into the generator (constants at the top of
`gen_scenario.py`): workspace `[-9, 9]²`, start↔goal at least 8 m apart, 1.8 m
obstacle-free zone around start and goal, ≥ 0.4 m gap between obstacles,
footprints 0.5–2 m × 0.5–4 m, heights 2–3 m (always intersecting the 1 m
flight altitude). It writes the world SDF **and** `scenario.yaml` from one
source, so the simulation, the virtual lidar, and the hit monitor always agree
on where the obstacles are.

**Full command order for a scenario run** (each numbered step in its own
terminal, from `nav2/`):

```bash
./gen_scenario.py --n-obstacles 4 --seed 42        # 0. generate -> scenarios/s42_n4/

./start_drone_stack.sh scenarios/s42_n4            # 1. sim + bridges + hit monitor;
                                                   #    wait for "Publishing /odom" lines
./launch_nav2.sh                                   # 2. wait for "Managed nodes are active"

./view.sh                                          # 3. (optional) RViz: costmap/path/trail
./view_gazebo.sh                                   # 3b. (optional) Gazebo 3D window

./send_nav2_goal.sh --scenario scenarios/s42_n4    # 4. arms drone + sends scenario goal
                                                   #    (or: ./send_nav2_goal.sh X Y)

cat logs/hits_current.yaml                         # 5. hits / min clearance of the run
./stop_drone_stack.sh                              # 6. teardown (Ctrl+C Nav2 terminal)
```

`hit_monitor.py` counts each contact episode between the drone (modeled as a
0.3 m disc) and an obstacle; the run is not stopped, per the protocol. Live
totals go to `logs/hits_current.yaml`; hits are logged in its terminal.

Verified results: `s42_n4` (4 obstacles, 12 m lateral path) SUCCEEDED in 32 s,
0 recoveries, 0 hits, min clearance 2.09 m; `s7_n10` (10 obstacles, 8 m path)
SUCCEEDED in 18 s, 0 recoveries, 0 hits, min clearance 0.59 m.

## Batch evaluation

Unattended sweep over obstacle counts and seeds, per the evaluation protocol
(success rate, time, path/Euclidean ratio, and hit count as functions of N):

```bash
./run_batch.sh                                   # default: N in {1,2,4,6,8,10} x 5 seeds
N_LIST="1 4 8" SEEDS="1 2 3" ./run_batch.sh      # custom sweep
GOAL_TIMEOUT=300 ./run_batch.sh                  # longer per-goal watchdog (default 240 s)
```

The batch runner starts everything headless (no terminal windows). Per (N, seed)
it generates the scenario, brings up Gazebo + bridges + lidar + hit monitor +
Nav2, waits for readiness, arms the drone, sends the scenario goal with a
timeout, records all artifacts under `runs/batch_<timestamp>/n<N>_s<seed>/`
(feedback stream, hits, every component's log, final status), tears everything
down, and continues. Failures are recorded (`STACK_FAIL` / `NAV2_FAIL` /
`TIMEOUT`) without stopping the batch. Expect roughly 1.5–2.5 min per run on
the VM; a full default sweep (30 runs) is about an hour.

```bash
python3 compute_metrics.py runs/batch_<timestamp>
```

writes `metrics.csv` (one row per run), `summary.txt` (per-N aggregate table),
and `plots/*.png` (if matplotlib is installed: `sudo apt install
python3-matplotlib`).

## Moving obstacles (Phase C)

`gen_scenario.py --obstacle-speed V` gives every obstacle a random direction at
V m/s; obstacles bounce elastically off the workspace bounds. Motion is
integrated by `obstacle_mover.py` (launched by the stack and the batch runner),
which publishes the live footprints on `/obstacles`. The virtual lidar and the
hit monitor track that stream, so the drone senses the obstacles where they
*currently* are. Moving obstacles are not modeled in the Gazebo world (a
physical box frozen at its spawn pose would collide with the drone at a
position the obstacle has virtually left); RViz shows them via the marker
display.

Speed sweep for the degradation experiment (fixed N, rising speed):

```bash
for v in 0.2 0.4 0.8 1.2 1.6 2.0; do
  N_LIST="6" SEEDS="1 2 3 4 5 6 7 8" OBSTACLE_SPEED="$v" ./run_batch.sh
done
```

Each speed gets its own batch dir; `compute_metrics.py` reports and plots per
(N, speed). Merge batches by copying run dirs together before computing, or
compute per batch. Hits are counted from goal start to goal resolution (the
goal scripts reset the counter via `/hit_monitor/reset`, and a contact already
present at navigation start is not counted).

Measured result (N = 6, 8 seeds per speed, `results/2026-07-20_speed_sweep/`):
goal completion never fails, because runs are not stopped on contact, but hits
rise 1 → 73 across 0.2 → 2.0 m/s (~9 per flight at the top speed), navigation
time grows 32 → 71 s, and the path ratio approaches 2×. The drone's own top
speed is ~0.6–0.7 m/s, so beyond ~0.8 m/s obstacles outrun it and Nav2's
prediction-free replanning cannot compensate.

## Visualization

Two independent viewers; both attach to the running stack and can be opened or
closed at any time without affecting the simulation.

**RViz (recommended; shows what Nav2 "thinks"):**

```bash
./view.sh
```

Preconfigured view (`drone_view.rviz`): odometry arrow trail (the flown path),
red lidar points tracing the obstacles, the obstacle markers, the global
costmap with its inflation band, and the green planned path. The toolbar's
*2D Goal Pose* tool sends goals by clicking in the view (bypasses the logging
of `send_nav2_goal.sh`).

**Gazebo GUI (shows the simulated world itself):**

```bash
./view_gazebo.sh
```

Attaches Gazebo's 3D viewport to the headless server: you see the actual drone
model, the wall, and the ground plane. Works out of the box on real GPUs / WSL2.
In the VMware-on-Apple-Silicon VM it requires "Accelerate 3D Graphics" to be
**disabled** (see gotchas below); expect low framerates there (software
rendering).

## Architecture

```
Gazebo (gz sim -s, headless)
  ├─ /model/drone_1/odometry ──bridge──> drone_pose_to_odom_tf.py ──> /odom + TF (odom->base_link)
  ├─ /clock ──────────────────bridge──> ROS sim time (everything runs use_sim_time)
  └─ /drone_1/gazebo/command/twist <──bridge── /cmd_vel <── Nav2 controller
                                          /scan <── virtual_lidar.py <── /odom + /obstacles
```

Nav2 runs **mapless**: no static map, no AMCL. `map -> odom` is a static identity
transform; both costmaps are filled purely from `/scan` (obstacle layer + inflation).
DWB is configured holonomic (`min/max_vel_y` nonzero) so the planner can use the
drone's ability to strafe, and **fully heading-agnostic**. That took three
separate deviations from Nav2 defaults, each discovered through a failing run:

1. `yaw_goal_tolerance: 6.28`, no RotateToGoal critic. Requiring a final
   heading made the drone rotate in place at the goal, which the progress
   checker treats as being stuck.
2. No PathAlign/GoalAlign critics. Heading-alignment scoring forces
   rotate-before-translate behavior and stalled every lateral-dominant path.
3. `max_vel_theta: 0.0`. Empirically, any sustained yaw-rate command (e.g.
   wz=-1.0) stalls the X3's MulticopterVelocityControl completely: correct
   velocities arrive in Gazebo and the drone produces no thrust at all. Pure
   translation works. The drone therefore keeps its spawn heading forever.

### Why a "virtual" lidar?

The world originally used a `gpu_lidar` sensor, but rendering-based sensors need a
working GPU context (OGRE2) which this development VM (VMware on Apple Silicon)
cannot provide reliably: the sensor registers but never publishes. `virtual_lidar.py`
computes the identical `LaserScan` analytically (360 rays, ray/box intersection
against the obstacle footprints) with no rendering at all. From Nav2's
perspective the output is indistinguishable from a simulated lidar. The
obstacle footprints come from the scenario's `scenario.yaml` and, while
running, from the live `/obstacles` topic published by `obstacle_mover.py`, so
static and moving scenarios use the same code path. On a machine with a working
GPU, a real `gpu_lidar` + `gz-sim-sensors-system` + scan bridge can be swapped
back in for static scenarios.

### Notes / gotchas

- The stock `nav2_bringup navigation_launch.py` (Jazzy) launches extra servers
  (collision_monitor, route_server, docking...) that fail without their own
  config; hence the minimal launch file.
- ROS `setup.bash` breaks under `set -u`; scripts source it first.
- Scripts derive their working directory from their own location. Do not
  hardcode absolute paths (this repo is used through a VM shared folder).
- VMware on Apple Silicon: keep "Accelerate 3D Graphics" **disabled**. Counter-
  intuitively, enabling it makes the Gazebo GUI hang (buggy SVGA3D driver, and
  Mesa then refuses the software fallback); disabled, the GUI renders via
  llvmpipe, which is slow but stable. Rendering-based sensors (gpu_lidar) work
  in *neither* mode, hence the virtual lidar. On real GPUs / WSL2 none of this
  applies.
- RViz on the SVGA3D driver may log a GLSL link error for `indexed_8bit_image`
  (the costmap display shader). Harmless: everything else renders; at worst the
  costmap overlay is blank. Goes away with 3D acceleration disabled.
- Nav2's `bt_navigator` runs with raised IPC timeouts (`default_server_timeout:
  200`). With the defaults, Nav2 aborted goals spuriously whenever the
  RAM-limited VM froze briefly and delayed its internal service calls.

## Next steps

The Nav2 baseline is complete (static sweep and speed-degradation sweep, see
`results/`), and the PX4 stage now flies single drones through 3D mazes.

What remains:

1. **Multiple drones.** Several PX4 instances at once, each with its own
   target, each able to collide with obstacles and with the other drones. A
   run ends when every drone has either reached its target or been destroyed.
   Several PX4 instances plus Gazebo will probably not fit in an 8 GB VM, so
   this is where the experiments likely move to a stronger machine.
2. **Batch evaluation over (n, k).** For each combination of obstacle count
   and drone count: number of collisions, summed Euclidean distance, summed
   travelled distance, their ratio, simulation time and CPU time.
3. **The avoidance component.** PX4 provides no 3D avoidance, so the runs
   above are the "without avoidance" baseline. The thesis contribution is the
   component that closes that gap, evaluated with the same generator, batch
   runner and metrics for a with/without comparison.
