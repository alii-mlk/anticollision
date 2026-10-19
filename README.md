# anticollision

Drone obstacle avoidance in simulation. The project has three stages: a 2D
baseline built on ROS 2 Nav2, a 3D stage built on PX4 and MAVLink that
measures k drones flying through mazes without avoidance, and an avoidance
policy trained with reinforcement learning and compared against that baseline
on the same scenarios.

## Stage 1: Nav2 in 2D (complete)

A quadcopter navigates around obstacles using the ROS 2 Nav2 stack.

- *Static obstacles*: 48/48 randomized runs collision-free across 1-10
  obstacles; navigation time and detour overhead grow mildly with obstacle
  count (`results/2026-07-17_static_sweep/`).
- *Moving obstacles*: collision avoidance degrades sharply with obstacle
  speed. Total hits rise from 1 to 73 (8 runs per speed) between 0.2 and
  2.0 m/s, navigation time doubles, and beyond ~0.8 m/s obstacles outrun the
  drone entirely (`results/2026-07-20_speed_sweep/`). This quantifies where
  prediction-free Nav2 stops being sufficient.

## Stage 2: PX4 and MAVLink in 3D (baseline complete)

Nav2 only plans in 2D, and it is not how real drones are flown. This stage
replaces it with the standard autopilot stack: PX4 as the flight controller
and MAVLink as the protocol, so the software is independent of the drone.

Working: PX4 SITL with Gazebo, the MAVLink bridge (MAVROS), 3D maze scenarios
with obstacles standing on the ground, k drones flying at once with a target
each, collision detection against obstacles and against other drones that
destroys the drone on contact, and an unattended sweep over (n, k).

Note on PX4 3D navigation: PX4 has no maintained 3D path planner. The
PX4-Avoidance package is archived and ROS 1 only, the path planning interface
is withdrawn, and the built-in Collision Prevention is horizontal and works
only in Position mode. Planning therefore happens on the ROS 2 side and PX4
executes the setpoints, which is what PX4 itself now recommends. Details and
sources in `px4/FEASIBILITY_3D_NAVIGATION.md`.

Because there was no avoidance system to install, the drones in this stage fly
straight at their targets. That makes the sweep below a "without avoidance"
baseline, which is the measurement the avoidance component has to be compared
against.

**Sweep results** (n in {4, 8, 12, 16, 20}, k in {1, 2, 3}, 5 seeds each, 75
runs, `results/2026-09-06_3d_sweep/`):

- Collisions per run rise with obstacle count at every drone count. For k = 3
  they go from 1.2 at n = 4 to 2.0 at n = 20.
- The share of drones reaching their target falls the same way. For k = 1 it
  goes from 60% at n = 4 to 20% at n = 20.
- Ratio over the drones that arrive stays between 1.09 and 1.11 in every
  configuration, because without avoidance they fly almost straight. All of
  the degradation is in how many drones survive, not in how long their paths
  are.
- CPU time depends on the number of drones and not on the number of
  obstacles: about 14 s, 32 s and 50 s per run for k = 1, 2 and 3.

Drone-to-drone collisions do occur, and more often when there are few
obstacles (0.4 per run at n = 4, k = 3), because the drones then fly straighter
and cross each other's paths more.

The same seed produces the same drone start and target points at every n, so
the summed Euclidean distance is identical down each k column (21.5, 37.6 and
53.3 m for k = 1, 2, 3). Obstacle density is varied against a fixed set of
flights.

## Stage 3: avoidance with reinforcement learning (complete)

A neural network decides which way each drone flies. Its input is a 3D lidar
scan plus the target, and its output is a direction. It is trained with PPO
(Stable-Baselines3, an actor-critic method) in MuJoCo through the
MuJoCo-Drones-Gym package, because training inside Gazebo and PX4 would take
far too long, and it is then flown in Gazebo with PX4 without any further
training.

The lidar is computed by our own code (`rl/lidar.py`, analytic rays against
the obstacle boxes, the ground and the other drones), so the network sees the
same scan in both simulators. The output is a direction rather than motor
commands, and each simulator's own controller turns it into flight. Flying
straight, the MuJoCo environment
gives the same outcome as Gazebo on all 25 single-drone scenarios of the
stage 2 sweep, and the observation built during a Gazebo flight is identical to
the one in training.

**Sweep results** (the same 75 scenarios as stage 2, both arms flown on
2026-10-19, `results/2026-10-19_rl_sweep/`):

- Drones reaching their target: 74/150 (49%) flying straight, 135/150 (90%)
  with the policy. Destroyed drones: 76 and 15.
- Same drone, same scenario: the policy saved 61 drones and lost none.
- `ratio+` rises from 1.09-1.11 to 1.13-1.34 and grows with obstacle count,
  because the drones now fly around the obstacles.
- No collisions between drones with the policy (3 flying straight), and no two
  drones closer than 2.66 m.
- Costs: missions take about 30 s instead of 12 s (the policy flies at the
  1.04 m/s it was trained at), and the network adds about 3.5 s of CPU time to
  a one-drone mission.
- Weak points: the crash rate rises with drone count (4%, 6% and 15% of drones
  for k = 1, 2, 3; none of these crashes happened near another drone), and the
  drones it saves often pass close to an obstacle (median 0.45 m).

The straight arm was re-run rather than reused, because a system update had
changed MAVROS, Gazebo and the bridge after September. It reproduces the
September sweep: 72 of 75 runs identical and the same totals.

## How the pieces fit together

All three stages use the same idea: the simulator provides physics, something
flies the drone, and our own code decides where it should go.

```
Stage 1:  our code -> Nav2 -> /cmd_vel -> bridge -> Gazebo
Stage 2:  our code -> MAVROS -> MAVLink -> PX4 -> Gazebo
Stage 3:  lidar + policy network -> velocity setpoint -> MAVROS -> MAVLink -> PX4 -> Gazebo
          (trained beforehand in MuJoCo, with the same lidar code)
```

In stage 2 the four programs are:

| Program | Role |
|---|---|
| Gazebo | the physics world: gravity, the drone's body, the obstacles |
| PX4 (SITL) | the flight controller, the software version of the board on a real drone. Keeps the drone stable and executes position commands |
| MAVROS | the translator between PX4's MAVLink protocol and ROS 2 |
| our Python nodes | the planner: chooses the target, watches for collisions, records metrics |

Stage 3 changes only the planner. `run_mission.py --policy` replaces the
straight-line setpoint with the network's direction. Everything else, from the
scenarios to the collision monitor and the metrics, is the same code.

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
  /opt/ros/jazzy/lib/mavros/install_geographiclib_datasets.sh`). The September
  sweep ran on MAVROS 2.14 and the October sweeps on 2.16. 2.16 moved its
  topics from `/uavN/...` to `/uavN/mavros/...`, so the nodes look the prefix
  up (`px4/mavros_ns.py`) instead of assuming it

Stage 3 (RL) additionally:

- a Python venv at `~/rlenv` with MuJoCo, Gymnasium, Stable-Baselines3, PyTorch
  (CPU build) and MuJoCo-Drones-Gym, created by `rl/setup_env.sh`. Every
  terminal that trains or flies the policy needs `source rl/env.sh` first

## Repository layout

```
Thesis Progress.docx          # running progress report
results/                      # curated, citable run evidence (one folder per experiment)
early_versions/               # superseded experiments, kept for project history

rl/                           # STAGE 3: reinforcement learning avoidance
  setup_env.sh                # one-time setup of the venv ~/rlenv (CPU PyTorch, MuJoCo,
                              #   Gymnasium, Stable-Baselines3, MuJoCo-Drones-Gym)
  env.sh                      # source it in every RL terminal: ROS first, then the venv
  smoke_test.py               # checks each piece of the toolchain on its own
  verify_venv.py              # checks every installed file against its package record;
                              #   --fix reinstalls damaged packages (after a full disk)
  lidar.py                    # the 3D lidar: 82 analytic rays against boxes, ground and
                              #   other drones, shared by training and the Gazebo flights
  test_lidar.py               # hand-worked distances plus a random-maze geometric check
  env.py                      # Gymnasium environment: random maze from gen_scenario_3d.py,
                              #   MuJoCo or point-mass flight, reward, geometric collisions
  check_env.py                # interface check, calibration, and straight flight compared
                              #   with the Gazebo sweep scenario by scenario
  diagnose_flight.py          # shows the MuJoCo-Drones-Gym controller bug (see stage 3 notes)
  train.py                    # PPO training with periodic evaluation -> runs/<name>/
  policy.py                   # loads a trained run for the Gazebo side: observation in,
                              #   velocity setpoint out
  evaluate.py                 # flies the 25 single-drone scenarios in MuJoCo through
                              #   policy.py and checks its observation against training
  runs/                       # training output (gitignored; the evaluated model is
                              #   copied into results/2026-10-19_rl_sweep/model/)
  logs/                       # (gitignored)

px4/                          # STAGE 2: PX4 + MAVLink, 3D
  start_px4_sitl.sh           # boots Gazebo + PX4 SITL for ONE drone (and the heartbeat below);
                              #   --scenario <dir> spawns the drone at the scenario start
  start_px4_multi.sh          # the same for k drones: k PX4 instances sharing one Gazebo world,
                              #   a heartbeat and a MAVROS each, then the SITL parameters
  stop_px4_sitl.sh            # stops PX4, Gazebo, MAVROS, the heartbeats and the clock bridge
  gcs_heartbeat.py            # PX4 refuses to arm with no ground station watching;
                              #   this sends the minimal MAVLink heartbeat to satisfy it,
                              #   and applies the SITL parameters with --set NAME=VALUE
  gen_scenario_3d.py          # random 3D maze: n ground-standing boxes with random
                              #   footprint and height, k drones with 3D targets -> scenario.yaml
  spawn_obstacles.py          # inserts those obstacles into the running Gazebo world
  collision_monitor_3d.py     # 3D collision check against obstacles and other drones;
                              #   force-disarms ("destroys") the drone on contact
  run_mission.py              # flies one drone to its target via offboard setpoints,
                              #   records travelled/Euclidean distance, ratio, time;
                              #   --policy <run dir> steers it with the trained network
  run_multi_mission.py        # runs a whole scenario: k missions + k monitors, then the summary
  run_batch_3d.py             # unattended sweep over (n, k, seed) -> runs3d/batch_<ts>/
  compute_metrics_3d.py       # batch dir -> metrics.csv + the per-(n,k) table
  compare_sweeps_3d.py        # a straight batch against a policy batch: per configuration,
                              #   per drone (saved / lost), clearances and drone separation
  mavros_ns.py                # finds the namespace MAVROS publishes under (it changed in 2.16)
  sim_speed.py                # real time factor of the running simulation, and which
                              #   processes are using the CPU
  set_sim_speed.py            # slows Gazebo down deliberately, for a machine that cannot
                              #   keep up (see the note below)
  collect_diagnostics.sh      # dumps everything needed to debug a bad run into logs/
  offboard_goto.py            # minimal single-target offboard flight, used to verify the bridge
  FEASIBILITY_3D_NAVIGATION.md # why PX4 has no usable 3D planner, with sources
  scenarios3d/                # generated scenarios (gitignored; reproducible from seed)
  runs3d/                     # batch outputs (gitignored; promote keepers to results/)
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

## Stage 2: running k drones

`start_px4_multi.sh` does the whole bring-up, so this needs two terminals
rather than four. Generate a scenario with more than one drone:

```bash
./gen_scenario_3d.py --n-obstacles 10 --k-drones 3 --seed 21
```

```bash
./start_px4_multi.sh --scenario scenarios3d/s21_n10_k3
```

This starts the PX4 instances one at a time, waiting for each to report
`Ready for takeoff!` before starting the next, then one heartbeat and one
MAVROS per drone, then the Gazebo clock bridge, then the SITL parameters. It
takes a few minutes. Two things in its output are worth reading:

```
/uav0 connected (MY ID 1.191, TARGET ID 1.1)
/uav1 connected (MY ID 2.191, TARGET ID 2.1)
/uav2 connected (MY ID 3.191, TARGET ID 3.1)
```

The target IDs must be 1.1, 2.1, 3.1 and not three times 1.1 (see the note on
system IDs below), and every drone must report `PARAMS DONE`. It finishes with
a `READY` banner and writes `logs/stack_ready`.

Then, in the second terminal:

```bash
./spawn_obstacles.py --scenario scenarios3d/s21_n10_k3
./run_multi_mission.py --scenario scenarios3d/s21_n10_k3 --timeout 180
```

`run_multi_mission.py` waits for `logs/stack_ready` before it starts, so it
cannot begin flying while the parameters are still being applied. It prints a
per-drone table and writes `run/summary.yaml`.

## Stage 2: the (n, k) sweep

Fully unattended. It generates each scenario, brings the stack up, spawns the
obstacles, flies, records, and tears everything down before the next run.

```bash
./run_batch_3d.py --n 4 8 12 16 20 --k 1 2 3 --seeds 1 2 3 4 5 --timeout 180
./compute_metrics_3d.py runs3d/batch_<timestamp>
```

That is 75 runs and takes about 100 minutes. `--timeout` is in **simulated**
seconds. Keep the machine on mains power and stop it sleeping, or it throttles
and the later runs get slower than the early ones. Sleeping does not corrupt
the results, because simulated time comes from Gazebo's clock and CPU time
from the process accounting, but it does distort the `wall(s)` and `rtf`
columns.

The metrics table reports two ratios. `ratio` is over every drone, which is
the metric as specified. `ratio+` is over only the drones that reached their
target, and is the one that says how efficient a completed path was. They
differ because a destroyed drone contributes the distance it managed against
the full distance it was asked to cover, which pulls `ratio` below 1.

### SITL parameters

These are applied automatically, by the heartbeat process, over MAVLink:

```
CBRK_SUPPLY_CHK 894281   # SITL has no real power monitoring
COM_DISARM_PRFLT 0       # do not auto-disarm while idle on the ground
SIM_BAT_MIN_PCT 90       # keep the simulated battery full, so long runs
                         #   are not cut short by a low-battery failsafe
SDLOG_MODE -1            # no PX4 flight logs (.ulg); nothing here reads them,
                         #   and over a sweep they would fill the disk
```

They used to be set through MAVROS, which does not work with more than one
drone: MAVROS downloads the autopilot's whole parameter list before it will
set anything, and with three instances that download does not finish, so every
attempt fails with "Unknown parameter to set". `ros2 param set` reports
success anyway, so the failure was silent.

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
- **MAVLink system IDs.** PX4 gives instance *i* the system ID *i+1*, while
  MAVROS addresses system 1 unless it is told otherwise. Left at the default,
  every MAVROS talks to instance 0: the others log
  `Ignore command 400 from 1/191 to 1/1` and their drones can never arm. The
  unanswered commands are then retried until two acknowledgements race and
  MAVROS aborts with `std::future_error: Promise already satisfied`. Each
  MAVROS therefore gets `-p tgt_system:=<i+1>`. This was the reason three
  drones appeared not to work at all.
- **Start the MAVROS nodes one at a time.** Started together they all
  broadcast their version request at once, the replies time out, and the same
  promise race follows.
- **One flight per stack start.** After a mission the drones are left in the
  air or in return-to-launch, so a second mission on the same stack would start
  airborne and count the descent as travelled distance. `run_mission.py`
  refuses to start (outcome `not_at_start`) unless the drone is on the ground,
  disarmed and at its start point. Restart the stack between flights; the
  batch runner already does.
- **Ratio for destroyed runs is not meaningful on its own.** A drone that dies
  halfway has travelled less than the straight-line distance, giving a ratio
  below 1. This is why the metrics table reports `ratio+` alongside `ratio`.
- **`Accel #0 fail: TIMEOUT` at startup is harmless.** It appears when a later
  instance spawns its model into the shared Gazebo world, which stalls the
  sensor stream for a moment. The last instance to start never sees it. It
  scales with drone count, which makes it look like a performance limit, but
  it is not: with three drones flying, Gazebo uses about 40% of one core and
  the simulation runs at 0.99 of real time.
- **If a machine cannot keep up**, `sim_speed.py` measures the real time
  factor and `set_sim_speed.py` slows Gazebo down to a speed it can hold. An
  even slowdown is harmless, since simulated time, distances and collisions
  are unaffected and the run simply takes longer. What breaks a run is the
  simulator falling behind unevenly, because the autopilot then decides its
  sensors are dead.

## Stage 3: training and flying the policy

Everything runs in the VM, from the repository root unless noted. Once:

```bash
bash rl/setup_env.sh
```

Then in every terminal that uses the policy:

```bash
source rl/env.sh
```

**Check the pieces** (both write their reports to `rl/logs/`):

```bash
python rl/test_lidar.py
python rl/check_env.py
```

`check_env.py` ends with the straight-line comparison against the Gazebo
sweep, which should agree on 25 of 25 scenarios for both flight models.

**Train.** About 31 minutes for 2 million decisions on the VM with 4
environments:

```bash
python rl/train.py --dynamics mujoco --steps 2000000 --name mujoco_1
```

The run directory `rl/runs/mujoco_1/` gets `best.zip` (the version with the
most arrivals on 100 held-out mazes), the lidar and environment settings it was
trained with, and the evaluation history in `eval.csv`.

**Evaluate in MuJoCo through the deployment code:**

```bash
python rl/evaluate.py rl/runs/mujoco_1
```

It flies the 25 single-drone Gazebo scenarios straight and with the policy, and
checks that the observation `policy.py` builds is the one the network was
trained on (it should report a largest difference of 0).

**Fly it in Gazebo.** The same as stage 2 with `--policy` added, from `px4/`
(where the venv is `source ../rl/env.sh`):

```bash
./run_multi_mission.py --scenario scenarios3d/s21_n10_k3 --timeout 180 --policy ../rl/runs/mujoco_1
```

**The comparison sweep**, both arms, about 4 hours together:

```bash
./run_batch_3d.py --n 4 8 12 16 20 --k 1 2 3 --seeds 1 2 3 4 5 --timeout 180
./run_batch_3d.py --n 4 8 12 16 20 --k 1 2 3 --seeds 1 2 3 4 5 --timeout 180 --policy ../rl/runs/mujoco_1
python3 compare_sweeps_3d.py runs3d/<straight batch> runs3d/<policy batch>
```

The policy arm refuses to start if the venv is not active. To repeat the
reported sweep with the same model, use
`--policy ../results/2026-10-19_rl_sweep/model`, the copy of the model that was
evaluated.

### Notes on the RL stage

- **Why train in MuJoCo.** One Gazebo flight takes about a minute of real
  time, and the policy needs hundreds of thousands of decisions. The MuJoCo
  environment runs about 900 decisions per second per environment, without
  rendering, so the VM's missing GPU does not matter.
- **Bug in MuJoCo-Drones-Gym (commit ca170dc).** Its waypoint action mode
  creates a new position controller at every control step, so the controller
  state is lost, the attitude oscillates with growing amplitude and the drone
  flips. Hovering and climbing still work, which makes it easy to miss.
  `rl/env.py` uses the package's RPM mode and calls its controller itself, one
  instance per episode. `rl/diagnose_flight.py` reproduces the failure.
- **Same speed in both simulators.** The direction is flown at 1.04 m/s level
  and 1.17 m/s climbing, the speeds measured in MuJoCo for the waypoint
  distance used in training, and `run_mission.py` sends the same speeds to PX4
  as velocity setpoints.
- **Training stability.** With a constant learning rate the policy peaked and
  then got worse as its action noise shrank. A learning rate decaying to zero
  and `target_kl` 0.03 keep it stable.
- **PyTorch on ARM.** On ARM64 Linux a plain `pip install torch` picks the CUDA
  build, about 9 GB of NVIDIA libraries that cannot be used in the VM.
  `setup_env.sh` installs from the CPU-only index instead.
- **Known limitation.** The policy was trained with one drone and static
  obstacles. Other drones appear in its lidar only in Gazebo, which is one of
  two possible reasons the crash rate rises at k = 3 (the other is that the
  k = 3 mazes are a different set; see the results README).

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
footprints 0.5-2 m × 0.5-4 m, heights 2-3 m (always intersecting the 1 m
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
`TIMEOUT`) without stopping the batch. Expect roughly 1.5-2.5 min per run on
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
speed is ~0.6-0.7 m/s, so beyond ~0.8 m/s obstacles outrun it and Nav2's
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

## Status

All three stages are complete: the Nav2 baseline (static and moving obstacle
sweeps), the PX4 baseline over (n, k) without avoidance
(`results/2026-09-06_3d_sweep/`), and the RL policy compared against it on the
same scenarios (`results/2026-10-19_rl_sweep/`).

Possible improvements to the policy, none of them done: training with several
drones in the episode, a larger penalty for flying close to obstacles (to widen
the margins of the drones it saves), and training at a higher speed (to shorten
the missions).
