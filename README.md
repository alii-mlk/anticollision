# anticollision

Drone obstacle avoidance in simulation: a quadcopter (Gazebo X3) navigates around obstacles using the ROS 2 Nav2 stack.

**Status:** single static obstacle avoidance works end-to-end. The drone flies from (0,0) to a goal behind a wall and back, planning around the obstacle. Verified over repeated runs: ~18–19 s per 8 m leg, 0 recovery behaviors, path-length-to-Euclidean-distance ratio ≈ 1.3.

## Requirements

- Ubuntu 24.04 with ROS 2 Jazzy (`/opt/ros/jazzy`)
- Gazebo (gz-sim, Harmonic) + `ros-jazzy-ros-gz-bridge`
- `ros-jazzy-navigation2`, `ros-jazzy-nav2-bringup`
- The X3 UAV model from Gazebo Fuel (auto-downloaded on first `gz sim` run with internet, cached under `~/.gz/fuel`)

## Repository layout

```
Thesis Progress.docx          # running progress report
results/                      # curated, citable run evidence (one folder per experiment)
early_versions/               # superseded experiments, kept for project history
nav2/
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
  run_batch.sh                # unattended sweep: N obstacles x seeds -> runs/batch_<ts>/
  compute_metrics.py          # batch dir -> metrics.csv + per-N summary (+ plots)
  scenarios/                  # generated scenarios (gitignored; reproducible from seed)
  runs/                       # batch outputs (gitignored; promote keepers to results/)
  logs/                       # all terminals and runs tee their output here (gitignored)
```

## Running (step by step)

All commands from the `nav2/` directory.

**1. Start the simulation + bridges:**

```bash
./start_drone_stack.sh
```

Opens 6 terminals: the Gazebo server (headless), the odometry/clock and cmd_vel
bridges, the odom/TF republisher, the virtual lidar, and an arming loop that
enables the drone's motors (retries for ~20 s). Wait until the "3 Drone Odom TF
Bridge" terminal starts printing `Publishing /odom: ...` lines.

**2. Start Nav2 (new terminal):**

```bash
./launch_nav2.sh
```

Wait for `Managed nodes are active` — Nav2 is ready. Leave this running.

**3. Send a navigation goal (new terminal):**

```bash
./send_nav2_goal.sh           # default goal (8,0): straight line would cross the wall
./send_nav2_goal.sh 0 0       # custom goal: fly back, rounding the wall again
```

The terminal streams feedback (distance remaining, recoveries) and ends with
`Goal finished with status: SUCCEEDED`. Every run is also logged to
`logs/goal_<time>.txt` (full feedback stream — the source for trajectory metrics
like path length and path/Euclidean ratio).

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
| `--out DIR` | optional; output directory (default `scenarios/s<seed>_n<N>/`) |

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
0.3 m disc) and an obstacle — the run is not stopped, per the protocol. Live
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

The batch runner starts everything headless (no terminal windows): per (N, seed)
it generates the scenario, brings up Gazebo + bridges + lidar + hit monitor +
Nav2, waits for readiness, arms the drone, sends the scenario goal with a
timeout, records all artifacts under `runs/batch_<timestamp>/n<N>_s<seed>/`
(feedback stream, hits, every component's log, final status), tears everything
down, and continues — failures are recorded (`STACK_FAIL` / `NAV2_FAIL` /
`TIMEOUT`) without stopping the batch. Expect roughly 1.5–2.5 min per run on
the VM; a full default sweep (30 runs) is about an hour.

```bash
python3 compute_metrics.py runs/batch_<timestamp>
```

writes `metrics.csv` (one row per run), `summary.txt` (per-N aggregate table),
and `plots/*.png` (if matplotlib is installed: `sudo apt install
python3-matplotlib`).

## Visualization

Two independent viewers; both attach to the running stack and can be opened or
closed at any time without affecting the simulation.

**RViz (recommended — shows what Nav2 "thinks"):**

```bash
./view.sh
```

Preconfigured view (`drone_view.rviz`): odometry arrow trail (the flown path),
red lidar points tracing the wall, the global costmap with its inflation band,
and the green planned path. The toolbar's *2D Goal Pose* tool sends goals by
clicking in the view (bypasses the logging of `send_nav2_goal.sh`).

**Gazebo GUI (shows the simulated world itself):**

```bash
./view_gazebo.sh
```

Attaches Gazebo's 3D viewport to the headless server — you see the actual drone
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
                                          /scan <── virtual_lidar.py <── /odom + known obstacles
```

Nav2 runs **mapless**: no static map, no AMCL. `map -> odom` is a static identity
transform; both costmaps are filled purely from `/scan` (obstacle layer + inflation).
DWB is configured holonomic (`min/max_vel_y` nonzero) so the planner can use the
drone's ability to strafe — and **fully heading-agnostic**, which took three
separate deviations from Nav2 defaults, each discovered through a failing run:

1. `yaw_goal_tolerance: 6.28`, no RotateToGoal critic — requiring a final
   heading made the drone rotate in place at the goal, which the progress
   checker treats as being stuck.
2. No PathAlign/GoalAlign critics — heading-alignment scoring forces
   rotate-before-translate behavior and stalled every lateral-dominant path.
3. `max_vel_theta: 0.0` — empirically, any sustained yaw-rate command (e.g.
   wz=-1.0) stalls the X3's MulticopterVelocityControl completely: correct
   velocities arrive in Gazebo and the drone produces no thrust at all. Pure
   translation works. The drone therefore keeps its spawn heading forever.

### Why a "virtual" lidar?

The world originally used a `gpu_lidar` sensor, but rendering-based sensors need a
working GPU context (OGRE2) which this development VM (VMware on Apple Silicon)
cannot provide reliably — the sensor registers but never publishes. `virtual_lidar.py`
computes the identical `LaserScan` analytically (360 rays, ray/box intersection
against the obstacle footprints listed in `OBSTACLES`) with no rendering at all.
From Nav2's perspective the output is indistinguishable from a simulated lidar.
**The `OBSTACLES` list must be kept in sync with the world SDF.** On a machine with
a working GPU, a real `gpu_lidar` + `gz-sim-sensors-system` + scan bridge can be
swapped back in.

### Notes / gotchas

- The stock `nav2_bringup navigation_launch.py` (Jazzy) launches extra servers
  (collision_monitor, route_server, docking...) that fail without their own config —
  hence the minimal launch file.
- ROS `setup.bash` breaks under `set -u`; scripts source it first.
- Scripts derive their working directory from their own location — do not hardcode
  absolute paths (this repo is used through a VM shared folder).
- VMware on Apple Silicon: keep "Accelerate 3D Graphics" **disabled**. Counter-
  intuitively, enabling it makes the Gazebo GUI hang (buggy SVGA3D driver, and Mesa
  then refuses the software fallback); disabled, the GUI renders via llvmpipe —
  slow but stable. Rendering-based sensors (gpu_lidar) work in *neither* mode,
  hence the virtual lidar. On real GPUs / WSL2 none of this applies.
- RViz on the SVGA3D driver may log a GLSL link error for `indexed_8bit_image`
  (the costmap display shader). Harmless: everything else renders; at worst the
  costmap overlay is blank. Goes away with 3D acceleration disabled.

## Next steps

1. Randomized obstacle scenarios: generate world SDF + `OBSTACLES` list from a single
   source, batch-run goals, aggregate metrics from `logs/`.
2. Metrics script (success rate, navigation time, path/Euclidean ratio, min obstacle
   clearance) parsing `logs/goal_*.txt`.
3. Moving obstacles: feed `virtual_lidar.py` live obstacle poses from Gazebo instead
   of a static list.
