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
collision.sdf                 # earlier experiment: drones colliding
3 drones move manually.sdf    # earlier experiment: manual multi-drone control
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
  logs/                       # all terminals and runs tee their output here (gitignored candidate)
  old/                        # superseded prototype (dead-reckoning bridge)
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
drone's ability to strafe. The final yaw tolerance is effectively disabled
(`yaw_goal_tolerance: 6.28`, no RotateToGoal critic): a multicopter's heading is
irrelevant, and requiring one made the drone rotate in place at the goal, which
the progress checker treats as being stuck.

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
