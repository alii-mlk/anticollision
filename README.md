# anticollision

Collision avoidance for drones, in simulation. The work went through four
stages, each building on the one before:

1. Nav2 in 2D: a quadcopter flies around obstacles with the ROS 2 navigation
   stack.
2. PX4 and MAVLink in 3D: the drones are flown by the PX4 autopilot, the
   standard software on real drones, through 3D mazes. There is no 3D
   avoidance in PX4, so the drones fly straight at their targets. This is the
   baseline.
3. Avoidance with reinforcement learning: a neural network, trained in
   MuJoCo, chooses the direction of each drone from a 3D lidar. It is flown in
   Gazebo with PX4 on the same scenarios as the baseline.
4. One policy for many drones: the same kind of network, trained with many
   drones in the maze and told about its nearest neighbours, so the drones
   also avoid each other.

## How the repository is organised

```
nav2/                  stage 1: Nav2 in 2D
px4/                   stages 2 to 4 in Gazebo: starting PX4 and the drones,
                       scenarios, missions, collision monitor, sweeps, metrics
rl/                    stages 3 and 4: lidar, training environments, training
                       and evaluation of the policy
results/               one folder per experiment, plus the tables and figures
                       for the thesis (results/README.md lists them)
early_versions/        the first experiments, before stage 1, kept for history
Thesis Progress.docx   my notes on the work, in the order it was done
```

Every script starts with a comment or docstring that says what it does and
how to run it. The list of all files with one line each is in
[Files](#files) below.

How a Gazebo experiment goes through the scripts (stages 2 to 4, all in
`px4/`):

```
gen_scenario_3d.py        n obstacles and k drones from a seed -> scenario.yaml
run_batch_3d.py           for every (n, k, seed):
  start_px4_multi.sh        Gazebo, one PX4 and one MAVROS per drone
  spawn_obstacles.py        puts the boxes into the world
  run_multi_mission.py      per drone: run_mission.py (flies it, straight or
                            with --policy) and collision_monitor_3d.py
                            (destroys it on contact)
                          -> runs3d/batch_<time>/
compute_metrics_3d.py     -> metrics.csv, drones.csv, summary.txt
compare_sweeps_3d.py      a straight batch against a policy batch
                          -> copied to results/<date>/
results/make_tables.py    the thesis tables, results/make_figures.py the figures
```

How the policy is made (stages 3 and 4, in `rl/`):

```
lidar.py                  the 3D lidar, the same code in MuJoCo and in Gazebo
env.py, swarm.py          the training mazes, one drone or many
train.py                  PPO -> rl/runs/<name>/ (model, settings, evaluations)
evaluate.py,              checks in MuJoCo, and the large evaluation on
check_swarm.py,             hundreds of new mazes
evaluate_large.py
policy.py                 loads a trained model for px4/run_mission.py --policy
```

## Results

### Stage 1: Nav2 in 2D

- Static obstacles: 48 of 48 random runs reached the goal without touching an
  obstacle, with 1 to 10 obstacles. Navigation time and detour grow a little
  with the number of obstacles (`results/2026-07-17_static_sweep/`).
- Moving obstacles: the collisions grow quickly with the obstacle speed. The
  total number of hits goes from 1 to 73 (8 runs per speed) between 0.2 and
  2.0 m/s, and the navigation time doubles. From about 0.8 m/s on the
  obstacles are faster than the drone (`results/2026-07-20_speed_sweep/`).
  Nav2 only reacts to where the obstacles are now, so it cannot keep up.

### Stage 2: PX4 and MAVLink in 3D, without avoidance

Nav2 only plans in 2D, and real drones are not flown with it. This stage uses
the standard autopilot stack instead: PX4 as the flight controller and
MAVLink as the protocol, so the software does not depend on one drone.

It has PX4 SITL with Gazebo, the MAVLink bridge (MAVROS), 3D maze scenarios
with obstacles standing on the ground, k drones flying at once with a target
each, collision detection against obstacles and other drones that destroys
the drone on contact, and an unattended sweep over (n, k).

PX4 has no maintained 3D path planner. The PX4-Avoidance package is archived
and only works with ROS 1, the path planning interface was withdrawn, and the
built-in Collision Prevention is horizontal and only works in Position mode.
So the planning runs on the ROS 2 side and PX4 executes the setpoints, which
is also what PX4 recommends now. Details and sources are in
`px4/FEASIBILITY_3D_NAVIGATION.md`. Because there was nothing to install, the
drones in this stage fly straight at their targets, and the sweep is the
baseline without avoidance.

Sweep (n in {4, 8, 12, 16, 20}, k in {1, 2, 3}, 5 seeds each, 75 runs,
`results/2026-09-06_3d_sweep/`):

- Collisions per run grow with the number of obstacles for every number of
  drones. For k = 3 from 1.2 at n = 4 to 2.0 at n = 20.
- The share of drones that reach their target falls in the same way. For
  k = 1 from 60% at n = 4 to 20% at n = 20.
- The ratio of travelled to straight-line distance over the drones that
  arrive stays between 1.09 and 1.11 everywhere, because without avoidance
  they fly almost straight. More obstacles mean fewer drones arrive, while
  the paths of the ones that arrive stay the same.
- CPU time depends on the number of drones and not on the number of
  obstacles: about 14 s, 32 s and 51 s per run for k = 1, 2 and 3.

Drones also collide with each other, more often with few obstacles (0.4 per
run at n = 4, k = 3), because then they fly straighter and cross each other's
paths more.

The same seed gives the same start and target points at every n, so the
summed Euclidean distance is the same down each k column (21.5, 37.6 and
53.3 m for k = 1, 2, 3), and only the number of obstacles changes.

### Stage 3: avoidance with reinforcement learning

A neural network decides which way each drone flies. Its input is a 3D lidar
scan plus the target, and its output is a direction. It is trained with PPO
(Stable-Baselines3, an actor-critic method) in MuJoCo through the
MuJoCo-Drones-Gym package, because training inside Gazebo and PX4 would take
far too long. Then it is flown in Gazebo with PX4 without more training.

The lidar is computed by my own code (`rl/lidar.py`, rays against the
obstacle boxes, the ground and the other drones), so the network sees the
same scan in both simulators. The output is a direction instead of motor
commands, and each simulator's own controller turns it into flight. Flying
straight, the MuJoCo environment gives the same outcome as Gazebo on all 25
single-drone scenarios of the stage 2 sweep, and the input built during a
Gazebo flight is the same as in training.

Sweep (the same 75 scenarios as stage 2, both arms flown on 2026-10-19,
`results/2026-10-19_rl_sweep/`):

- Drones that reach their target: 74 of 150 (49%) flying straight, 135 of 150
  (90%) with the policy. Destroyed drones: 76 and 15.
- Same drone, same scenario: the policy saved 61 drones and lost none.
- `ratio+` goes from 1.09 to 1.11 up to 1.13 to 1.34 and grows with the
  number of obstacles, because the drones now fly around them.
- No collisions between drones with the policy (3 flying straight), and no
  two drones closer than 1.97 m. This is a lower bound, see the results
  README.
- Costs: missions take about 30 s instead of 12 s (the policy flies at the
  1.04 m/s it was trained at). With one drone the mission node uses about
  3.6 s of CPU time per mission with the network, against 0.9 s flying
  straight.
- Limitations: more drones crash with more drones in the run (4%, 6% and 15%
  for k = 1, 2, 3, none of them near another drone), and the drones it saves
  often pass close to an obstacle (median 0.45 m).

The straight arm was flown again instead of reusing September's, because a
system update had changed MAVROS, Gazebo and the bridge. It gives the same
result as in September: 72 of 75 runs identical and the same totals.

### Stage 4: one policy for many drones

The stage 3 policy was trained alone, and in MuJoCo with ten drones it hit
another drone as often as flying straight (8% of the flights). The new policy
is trained with 4 to 8 drones in each maze, all flown by the same network.
Each drone is told the position, velocity and predicted closest approach of
its three nearest neighbours within 6 m (the drones are assumed to broadcast
their position and velocity). Training also costs more for passing close to
obstacles and to other drones. Details are in
`results/2026-11-15_swarm_sweep/README.md`.

- MuJoCo, 400 new mazes per drone count for 1 to 10 drones: 95.2% of the
  13,600 flights arrived (stage 3 policy 78.9%, straight 43.9%), and no drone
  hit another one (740 and 804 for the other two). The share that arrives no
  longer falls as drones are added.
- Gazebo, the same 75 scenarios: 147 of 150 drones arrived (98%), against 135
  with the stage 3 policy and 74 flying straight. It saved 73 drones against
  straight and lost none. The 3 destroyed drones all hit an obstacle.
- Gazebo with 4 drones, 25 more scenarios: 93 of 100 arrived, against 49
  flying straight. No contact between drones with the policy in any Gazebo
  run (5 drones destroyed by contact flying straight).
- Costs: memory 963 MB with one drone and 3,511 MB with four (621 and 2,482
  flying straight), most of the difference in the mission nodes, which each
  load PyTorch. The simulator side uses about the same processor time as
  flying straight. Four drones is the most the VM can simulate at full speed.

`results/make_tables.py` and `results/make_figures.py` make the tables and
figures for the thesis from the result folders, into `results/thesis/`.

## How the pieces fit together

In every stage the simulator provides the physics, something flies the
drone, and my own code decides where it should go.

```
Stage 1:  my code -> Nav2 -> /cmd_vel -> bridge -> Gazebo
Stage 2:  my code -> MAVROS -> MAVLink -> PX4 -> Gazebo
Stage 3:  lidar + policy network -> velocity setpoint -> MAVROS -> MAVLink -> PX4 -> Gazebo
          (trained beforehand in MuJoCo, with the same lidar code)
Stage 4:  the same, with the 3 nearest drones' position and velocity as extra
          inputs (published by the collision monitors), trained with many drones
```

In stage 2 the four programs are:

| Program | Role |
|---|---|
| Gazebo | the physics world: gravity, the drone's body, the obstacles |
| PX4 (SITL) | the flight controller, the software version of the board on a real drone. Keeps the drone stable and executes position commands |
| MAVROS | the translator between PX4's MAVLink protocol and ROS 2 |
| my Python nodes | the planner: chooses the target, watches for collisions, records the metrics |

Stages 3 and 4 only change the planner. `run_mission.py --policy` replaces the
straight-line setpoint with the network's direction. Everything else, from the
scenarios to the collision monitor and the metrics, is the same code.

## Requirements

Everything runs in an Ubuntu VM (VMware on a MacBook Air M2). The repository
is used through the VM's shared folder, so no script contains an absolute
path.

Stage 1 (Nav2):

- Ubuntu 24.04 with ROS 2 Jazzy (`/opt/ros/jazzy`)
- Gazebo (gz-sim, Harmonic) and `ros-jazzy-ros-gz-bridge`
- `ros-jazzy-navigation2`, `ros-jazzy-nav2-bringup`
- the X3 UAV model from Gazebo Fuel (downloaded on the first `gz sim` run with
  internet, then cached under `~/.gz/fuel`)

Stage 2 (PX4) also needs:

- PX4-Autopilot v1.17.0 built for SITL, outside this repository (see
  `px4/start_px4_sitl.sh` for the clone and build commands). Do not build it
  inside a VM shared folder.
- `ros-jazzy-mavros`, `ros-jazzy-mavros-msgs`, `ros-jazzy-mavros-extras`, and
  the GeographicLib datasets (`sudo bash
  /opt/ros/jazzy/lib/mavros/install_geographiclib_datasets.sh`). The September
  sweep ran on MAVROS 2.14 and the later ones on 2.16. 2.16 moved its topics
  from `/uavN/...` to `/uavN/mavros/...`, so the nodes look the prefix up
  (`px4/mavros_ns.py`).

Stages 3 and 4 (RL) also need:

- a Python venv at `~/rlenv` with MuJoCo, Gymnasium, Stable-Baselines3,
  PyTorch (CPU build) and MuJoCo-Drones-Gym, made by `rl/setup_env.sh`. Every
  terminal that trains or flies the policy needs `source rl/env.sh` first.
  Stable-Baselines3 also brings matplotlib, which `results/make_figures.py`
  uses.

## Files

```
README.md
Thesis Progress.docx          my notes on the work, in order
early_versions/               the first experiments, before stage 1 (see its README)

nav2/                         STAGE 1: Nav2 in 2D
  drone_nav2_world.sdf        world: ground plane, one 1x4x2 m wall at x = 4, the X3 drone
                              with motor, velocity control and odometry plugins
  start_drone_stack.sh        starts Gazebo (headless) and all bridges, each in its terminal
  stop_drone_stack.sh         stops all of it
  drone_pose_to_odom_tf.py    turns the drone's Gazebo odometry into /odom and the
                              odom -> base_link transform, plus a fixed map -> odom
  virtual_lidar.py            2D lidar computed from the drone pose and the known obstacle
                              boxes, published as a LaserScan on /scan
  nav2_params.yaml            Nav2 settings: no map, costmaps from /scan, sideways motion
  nav2_minimal.launch.py      starts only the Nav2 servers this project needs
  launch_nav2.sh              runs that launch file (log in logs/nav2.txt)
  send_nav2_goal.sh           sends a navigation goal, (8, 0) by default
  test_move_drone.sh          moves the drone with /cmd_vel directly, without Nav2
  view.sh, drone_view.rviz    RViz view: flown path, lidar points, costmap, planned path
  view_gazebo.sh              opens the Gazebo window on the running simulation
  gen_scenario.py             random scenario: start, goal, N obstacles ->
                              scenarios/<name>/world.sdf and scenario.yaml
  hit_monitor.py              counts contacts with obstacles (the run goes on) and the
                              smallest clearance
  obstacle_mover.py           moves the obstacles for the moving-obstacle runs and
                              publishes where they are on /obstacles
  run_batch.sh                unattended sweep over N obstacles and seeds -> runs/
  compute_metrics.py          a batch folder -> metrics.csv, summary and plots

px4/                          STAGES 2 TO 4: PX4 and MAVLink in 3D
  FEASIBILITY_3D_NAVIGATION.md  why PX4 has no usable 3D planner, with sources
  start_px4_sitl.sh           starts Gazebo and PX4 for one drone, with the heartbeat
  start_px4_multi.sh          the same for k drones: k PX4 instances in one Gazebo world,
                              a heartbeat and a MAVROS for each, then the SITL parameters
  stop_px4_sitl.sh            stops PX4, Gazebo, MAVROS, the heartbeats and the clock bridge
  gcs_heartbeat.py            PX4 does not arm without a ground station, so this sends the
                              MAVLink heartbeat one would send, and sets the SITL parameters
  gen_scenario_3d.py          random 3D maze: n boxes standing on the ground, k drones
                              with 3D targets -> scenario.yaml
  spawn_obstacles.py          puts those boxes into the running Gazebo world
  run_mission.py              flies one drone to its target with offboard setpoints and
                              records distances, ratio and time; --policy flies it with
                              the trained network
  collision_monitor_3d.py     checks one drone against the obstacles and the other drones,
                              destroys it on contact, publishes its position and velocity
                              for the others, records its closest distance to another drone
  run_multi_mission.py        a whole scenario: k missions and k monitors, then the summary
  run_batch_3d.py             unattended sweep over (n, k, seed) -> runs3d/batch_<time>/,
                              with CPU time and memory measured during every flight
  compute_metrics_3d.py       a batch folder -> metrics.csv (per run), drones.csv (per
                              drone) and the tables per (n, k)
  compare_sweeps_3d.py        a straight batch against a policy batch: per configuration,
                              per drone (saved or lost), clearances, distance between drones
  mavros_ns.py                finds where MAVROS puts its topics (this changed in 2.16)
  sim_speed.py                real time factor of the running simulation, and which
                              processes use the CPU
  set_sim_speed.py            slows Gazebo down on purpose, for a machine that cannot keep up
  collect_diagnostics.sh      collects everything needed to debug a bad run into logs/
  offboard_goto.py            the first offboard flight to one 3D point, to test the bridge

rl/                           STAGES 3 AND 4: the policy
  setup_env.sh                one-time setup of the venv ~/rlenv
  env.sh                      source it in every RL terminal: ROS first, then the venv
  smoke_test.py               checks every part of the toolchain on its own
  verify_venv.py              checks the installed files against their package records,
                              --fix reinstalls damaged packages (needed after a full disk)
  lidar.py                    the 3D lidar: 82 rays against boxes, ground and other drones,
                              the same code in training and in Gazebo
  test_lidar.py               distances worked out by hand, and a check on random mazes
  env.py                      the training environment with one drone: random maze, MuJoCo
                              or point-mass flight, reward, collisions
  check_env.py                checks env.py, measures the flight speeds, and compares
                              flying straight with the Gazebo sweep scenario by scenario
  diagnose_flight.py          shows the bug in MuJoCo-Drones-Gym's controller (see the notes)
  swarm.py                    many drones in one maze flown by one policy, and the vector
                              environment that trains on them (stage 4)
  check_swarm.py              checks swarm.py against env.py and Gazebo, and the neighbour
                              inputs
  train.py                    PPO training with regular evaluations -> runs/<name>/
  policy.py                   loads a trained model for the Gazebo side: observation in,
                              velocity setpoint out
  evaluate.py                 flies the 25 single-drone scenarios in MuJoCo through
                              policy.py and checks its input against training
  evaluate_large.py           flies policies on hundreds of new mazes per drone count,
                              with 95% intervals
  diagnose_crashes.py         where a policy hits obstacles and what its lidar showed

results/                      one folder per experiment (see results/README.md)
  make_tables.py              the thesis tables from the result folders -> results/thesis/
  make_figures.py             the thesis figures (needs matplotlib, run it in the venv)
  thesis/                     the tables and figures
```

The folders `nav2/logs`, `nav2/scenarios`, `nav2/runs`, `px4/logs`,
`px4/scenarios3d`, `px4/runs3d`, `rl/logs` and `rl/runs` are made by the
scripts and are gitignored. The runs used in the thesis are copied to
`results/`.

## Running stage 1: Nav2

All commands from the `nav2/` folder.

### One flight in the fixed world

Start the simulation and the bridges:

```bash
./start_drone_stack.sh
```

This opens 8 terminals: the Gazebo server (headless), the odometry/clock and
cmd_vel bridges, the odom/TF node, the virtual lidar, the hit monitor, the
obstacle mover, and a loop that enables the drone's motors (it retries for
about 20 s). Wait until the "3 Drone Odom TF Bridge" terminal prints
`Publishing /odom: ...` lines.

Start Nav2 in a new terminal, and wait for `Managed nodes are active`:

```bash
./launch_nav2.sh
```

Send a goal in a new terminal:

```bash
./send_nav2_goal.sh           # default goal (8,0): the straight line would cross the wall
./send_nav2_goal.sh 0 0       # any goal: fly back, around the wall again
```

The terminal shows the feedback (distance left, recoveries) and ends with
`Goal finished with status: SUCCEEDED`. Every goal is also logged to
`logs/goal_<time>.txt`, the full feedback stream, which the metrics (path
length, path/Euclidean ratio) are computed from.

Stop everything with `./stop_drone_stack.sh`, and Ctrl+C the Nav2 terminal.

### Random scenarios

As in the evaluation protocol: random start, random goal, N random fixed
obstacles, and hits are counted instead of stopping the run.

| Parameter of `gen_scenario.py` | Meaning |
|---|---|
| `--n-obstacles N` | number of random boxes, required |
| `--seed S` | required, the same seed and N give the same scenario |
| `--obstacle-speed V` | m/s in a random direction for each obstacle, default 0 (static) |
| `--out DIR` | output folder, default `scenarios/s<seed>_n<N>[_v<speed>]/` |

The placement rules are constants at the top of `gen_scenario.py`: workspace
from -9 to 9 m in x and y, start and goal at least 8 m apart, 1.8 m without
obstacles around start and goal, at least 0.4 m between obstacles, footprints
of 0.5 to 2 m by 0.5 to 4 m, heights of 2 to 3 m (always higher than the 1 m
flight altitude). It writes the world SDF and `scenario.yaml` from the same
data, so the simulation, the virtual lidar and the hit monitor always agree
on where the obstacles are.

All commands for one scenario, each numbered step in its own terminal:

```bash
./gen_scenario.py --n-obstacles 4 --seed 42        # 0. generate -> scenarios/s42_n4/

./start_drone_stack.sh scenarios/s42_n4            # 1. simulation, bridges, hit monitor;
                                                   #    wait for "Publishing /odom" lines
./launch_nav2.sh                                   # 2. wait for "Managed nodes are active"

./view.sh                                          # 3. (optional) RViz: costmap, path, flown path
./view_gazebo.sh                                   # 3b. (optional) Gazebo 3D window

./send_nav2_goal.sh --scenario scenarios/s42_n4    # 4. arms the drone and sends the goal
                                                   #    (or: ./send_nav2_goal.sh X Y)

cat logs/hits_current.yaml                         # 5. hits and min clearance of the run
./stop_drone_stack.sh                              # 6. stop (Ctrl+C the Nav2 terminal)
```

`hit_monitor.py` counts each contact between the drone (a disc of 0.3 m) and
an obstacle. The run is not stopped, as the protocol says. The running totals
are in `logs/hits_current.yaml`.

Two checks: `s42_n4` (4 obstacles, 12 m sideways path) reached the goal in
32 s with no recoveries, no hits and 2.09 m minimum clearance. `s7_n10` (10
obstacles, 8 m path) reached it in 18 s with no recoveries, no hits and
0.59 m minimum clearance.

### Batch evaluation

An unattended sweep over obstacle counts and seeds, with success rate, time,
path/Euclidean ratio and hit count for each N:

```bash
./run_batch.sh                                   # default: N in {1,2,4,6,8,10}, 5 seeds
N_LIST="1 4 8" SEEDS="1 2 3" ./run_batch.sh      # own choice of N and seeds
GOAL_TIMEOUT=300 ./run_batch.sh                  # longer time limit per goal (default 240 s)
```

The batch runner starts everything in the background, without terminal
windows. For every (N, seed) it generates the scenario, starts Gazebo, the
bridges, the lidar, the hit monitor and Nav2, waits until they are ready,
arms the drone, sends the goal with a time limit, saves everything under
`runs/batch_<time>/n<N>_s<seed>/`, stops it all and goes on. A failed run is
recorded (`STACK_FAIL`, `NAV2_FAIL` or `TIMEOUT`) and the batch continues. A
run takes about 1.5 to 2.5 minutes on the VM, so the default sweep (30 runs)
takes about an hour.

```bash
python3 compute_metrics.py runs/batch_<time>
```

writes `metrics.csv` (one row per run), `summary.txt` (a table per N) and
`plots/*.png` (if matplotlib is installed: `sudo apt install
python3-matplotlib`).

### Moving obstacles

`gen_scenario.py --obstacle-speed V` gives every obstacle a random direction
at V m/s, and the obstacles bounce off the edges of the workspace. The motion
is computed by `obstacle_mover.py` (started by the stack and by the batch
runner), which publishes the current footprints on `/obstacles`. The virtual
lidar and the hit monitor follow that topic, so the drone senses the
obstacles where they are at that moment. The moving obstacles are not put into
the Gazebo world, because a box frozen at its spawn pose would hit the drone
at a place the obstacle has already left. RViz shows them as markers.

The speed sweep (fixed N, increasing speed):

```bash
for v in 0.2 0.4 0.8 1.2 1.6 2.0; do
  N_LIST="6" SEEDS="1 2 3 4 5 6 7 8" OBSTACLE_SPEED="$v" ./run_batch.sh
done
```

Each speed gets its own batch folder, and `compute_metrics.py` reports per
(N, speed). To combine batches, copy their run folders together before
computing the metrics. Hits are counted from the start of the goal until it
ends (the goal scripts reset the counter through `/hit_monitor/reset`), and a
contact that already exists when navigation starts is not counted.

Result (N = 6, 8 seeds per speed, `results/2026-07-20_speed_sweep/`): the goal
is always reached in the end, because runs are not stopped on contact, but
the hits go from 1 to 73 between 0.2 and 2.0 m/s (about 9 per flight at the
highest speed), the navigation time from 32 to 71 s, and the path becomes
almost twice the straight line. The drone flies at about 0.6 to 0.7 m/s, so
from about 0.8 m/s on the obstacles are faster than the drone and Nav2's
replanning cannot keep up.

### Watching a run

Both viewers attach to the running stack and can be opened or closed at any
time without changing the simulation.

RViz shows what Nav2 sees:

```bash
./view.sh
```

The view (`drone_view.rviz`) shows the flown path as arrows, the lidar points
on the obstacles in red, the obstacle markers, the global costmap with its
inflation band, and the planned path in green. The *2D Goal Pose* tool sends
goals by clicking in the view (without the logging of `send_nav2_goal.sh`).

The Gazebo window shows the simulated world itself:

```bash
./view_gazebo.sh
```

It attaches Gazebo's 3D view to the headless server, with the drone model,
the wall and the ground. It works normally with a real GPU or WSL2. In the
VMware VM on Apple Silicon "Accelerate 3D Graphics" has to be turned off (see
the notes below), and the frame rate is low because of software rendering.

### How the Nav2 setup works

```
Gazebo (gz sim -s, headless)
  ├─ /model/drone_1/odometry ──bridge──> drone_pose_to_odom_tf.py ──> /odom + TF (odom->base_link)
  ├─ /clock ──────────────────bridge──> ROS sim time (every node uses use_sim_time)
  └─ /drone_1/gazebo/command/twist <──bridge── /cmd_vel <── Nav2 controller
                                          /scan <── virtual_lidar.py <── /odom + /obstacles
```

Nav2 runs without a map and without AMCL. `map -> odom` is a fixed identity
transform, and both costmaps are filled only from `/scan` (obstacle layer
plus inflation). The DWB controller is set up as holonomic (`min/max_vel_y`
not zero), so the planner can use the drone's ability to fly sideways, and
the heading of the drone plays no role. That needed three changes to the
Nav2 defaults, each found through a failing run:

1. `yaw_goal_tolerance: 6.28` and no RotateToGoal critic. Asking for a final
   heading made the drone turn in place at the goal, which the progress
   checker took as being stuck.
2. No PathAlign/GoalAlign critics. Scoring the heading forces the drone to
   turn before moving, and every mostly sideways path stalled.
3. `max_vel_theta: 0.0`. Any lasting yaw-rate command (for example
   wz = -1.0) stalls the X3's MulticopterVelocityControl completely: the
   velocities arrive in Gazebo and the drone produces no thrust at all. Pure
   translation works. So the drone keeps its spawn heading all the time.

### Why a virtual lidar

The world first used a `gpu_lidar` sensor, but sensors that render need a
working GPU context (OGRE2), which this VM (VMware on Apple Silicon) cannot
provide reliably: the sensor registers but never publishes anything.
`virtual_lidar.py` computes the same kind of `LaserScan` from geometry (360
rays against the obstacle footprints) without any rendering. For Nav2 it is a
normal lidar. The obstacle footprints come from the scenario's
`scenario.yaml` and, while running, from the `/obstacles` topic of
`obstacle_mover.py`, so static and moving scenarios use the same code. On a
machine with a working GPU, a real `gpu_lidar` with `gz-sim-sensors-system`
and a scan bridge could be used again for static scenarios.

### Notes on the Nav2 stage

- The standard `nav2_bringup navigation_launch.py` (Jazzy) starts extra
  servers (collision_monitor, route_server, docking and others) that fail
  without their own configuration, so this project has its own minimal launch
  file.
- ROS's `setup.bash` breaks under `set -u`, so the scripts source it first.
- The scripts find their working folder from their own location. Do not put
  absolute paths in them (the repository is used through a VM shared folder).
- VMware on Apple Silicon: keep "Accelerate 3D Graphics" turned off. With it
  on, the Gazebo window hangs (a faulty SVGA3D driver, and Mesa then refuses
  the software fallback). With it off the window renders with llvmpipe, slowly
  but stably. Rendering sensors (gpu_lidar) work in neither mode, hence the
  virtual lidar. None of this applies to a real GPU or WSL2.
- RViz on the SVGA3D driver can log a GLSL link error for
  `indexed_8bit_image` (the costmap shader). The rest still renders, at worst
  the costmap overlay stays empty. It does not happen with 3D acceleration
  off.
- Nav2's `bt_navigator` runs with longer IPC timeouts
  (`default_server_timeout: 200`). With the defaults Nav2 aborted goals for no
  reason whenever the VM, which has little RAM, froze for a moment and
  delayed its internal service calls.

## Running stage 2: PX4 in 3D

### One drone

Four terminals, in this order. All commands from the `px4/` folder, and every
ROS terminal needs `source /opt/ros/jazzy/setup.bash` first.

0\. Generate a scenario (once per experiment, the seed makes it repeatable):

```bash
./gen_scenario_3d.py --n-obstacles 8 --seed 1     # -> scenarios3d/s1_n8_k1/
```

1\. Terminal A, the simulator. Starts Gazebo and PX4 and puts the drone at the
scenario's start point. Wait for `Ready for takeoff!` in the PX4 window, which
means PX4 has started and passed its preflight checks.

```bash
./start_px4_sitl.sh --scenario scenarios3d/s1_n8_k1
```

Then put the obstacles into the world (it starts empty):

```bash
./spawn_obstacles.py --scenario scenarios3d/s1_n8_k1
```

It should print `Acknowledged 8/8, 8 present in world 'default'`. The count is
checked against the world itself, because the spawn service acknowledges
requests that Gazebo can still fail to build.

2\. Terminal B, the MAVLink bridge. Leave it running and wait for
`CON: Got HEARTBEAT, connected. FCU: PX4 Autopilot`.

```bash
ros2 run mavros mavros_node --ros-args -p fcu_url:="udp://:14540@127.0.0.1:14580"
```

3\. Terminal C, the collision monitor. Start it before the mission, so it is
watching from the first moment.

```bash
./collision_monitor_3d.py --scenario scenarios3d/s1_n8_k1
```

4\. Terminal D, the mission. The drone climbs to 3 m, then flies to its 3D
target.

```bash
./run_mission.py --scenario scenarios3d/s1_n8_k1
```

The results:

```bash
cat scenarios3d/s1_n8_k1/mission.yaml    # outcome, distances, ratio, time
cat scenarios3d/s1_n8_k1/outcome.yaml    # destroyed or not, min clearance, collision point
```

To stop: `./stop_px4_sitl.sh`, and Ctrl+C the MAVROS and monitor terminals.

### k drones

`start_px4_multi.sh` starts everything, so this needs two terminals instead of
four. Generate a scenario with more than one drone:

```bash
./gen_scenario_3d.py --n-obstacles 10 --k-drones 3 --seed 21
```

```bash
./start_px4_multi.sh --scenario scenarios3d/s21_n10_k3
```

It starts the PX4 instances one at a time, waiting for each to report
`Ready for takeoff!` before the next, then one heartbeat and one MAVROS per
drone, then the Gazebo clock bridge, then the SITL parameters. This takes a
few minutes. Two things in its output should be checked:

```
/uav0 connected (MY ID 1.191, TARGET ID 1.1)
/uav1 connected (MY ID 2.191, TARGET ID 2.1)
/uav2 connected (MY ID 3.191, TARGET ID 3.1)
```

The target IDs must be 1.1, 2.1, 3.1 and not three times 1.1 (see the note on
system IDs below), and every drone must report `PARAMS DONE`. At the end it
prints a `READY` banner and writes `logs/stack_ready`.

Then, in the second terminal:

```bash
./spawn_obstacles.py --scenario scenarios3d/s21_n10_k3
./run_multi_mission.py --scenario scenarios3d/s21_n10_k3 --timeout 180
```

`run_multi_mission.py` waits for `logs/stack_ready`, so it cannot start flying
while the parameters are still being set. It prints a table per drone and
writes `run/summary.yaml`.

### The (n, k) sweep

This runs on its own. It generates each scenario, starts the stack, spawns
the obstacles, flies, records and stops everything before the next run.

```bash
./run_batch_3d.py --n 4 8 12 16 20 --k 1 2 3 --seeds 1 2 3 4 5 --timeout 180
./compute_metrics_3d.py runs3d/batch_<time>
```

That is 75 runs and takes about 100 minutes. `--timeout` is in simulated
seconds. Keep the machine on power and stop it from sleeping, or it slows
down and the later runs get slower than the first ones. Sleep does not
change the results, because simulated time comes from Gazebo's clock and CPU
time from the process accounting, but it does change the `wall(s)` and `rtf`
columns.

The metrics table has two ratios. `ratio` is over every drone, as in the
evaluation plan. `ratio+` is over the drones that reached their target only,
and shows how long a completed path was. They differ because a destroyed
drone counts the distance it managed against the full distance it was asked
to fly, which pulls `ratio` below 1.

### SITL parameters

These are set automatically by the heartbeat process, over MAVLink:

```
CBRK_SUPPLY_CHK 894281   # SITL has no real power monitoring
COM_DISARM_PRFLT 0       # do not disarm by itself while waiting on the ground
SIM_BAT_MIN_PCT 90       # keep the simulated battery full, so long runs are
                         #   not stopped by a low-battery failsafe
SDLOG_MODE -1            # no PX4 flight logs (.ulg); nothing here reads them,
                         #   and over a sweep they would fill the disk
```

At first they were set through MAVROS, which does not work with more than one
drone: MAVROS downloads the autopilot's whole parameter list before it sets
anything, and with three instances that download does not finish, so every
attempt fails with "Unknown parameter to set". `ros2 param set` still reports
success, so nothing showed the failure.

### Notes on the PX4 stage

- Frames: scenario coordinates are Gazebo world coordinates, but MAVROS
  reports the position in PX4's local frame, whose origin is where the drone
  spawned. The nodes convert with `world = local + start`. If this is wrong
  the drone flies somewhere else entirely, so for a drone that arrived,
  `final_world_position` should be close to `target_world_position`.
- Destroying a drone: PX4 refuses a normal disarm command in flight
  ("Disarming denied: not landed"), so the monitor sends
  `MAV_CMD_COMPONENT_ARM_DISARM` with the force value 21196. Without this the
  drone survived its own collision and kept flying, and one test recorded a
  travelled distance of 350 m.
- MAVLink system IDs: PX4 gives instance *i* the system ID *i+1*, while MAVROS
  talks to system 1 unless told otherwise. With the default every MAVROS
  talks to instance 0: the others log `Ignore command 400 from 1/191 to 1/1`
  and their drones never arm. The unanswered commands are then sent again
  until two acknowledgements arrive at once and MAVROS crashes with
  `std::future_error: Promise already satisfied`. So each MAVROS gets
  `-p tgt_system:=<i+1>`. This was why three drones seemed not to work at all.
- The MAVROS nodes are started one at a time. Started together they all ask
  for the autopilot version at once, the replies time out, and the same crash
  follows.
- One flight per stack start: after a mission the drones are still in the air
  or returning home, so a second mission on the same stack would start in the
  air and count the descent as travelled distance. `run_mission.py` refuses
  to start (outcome `not_at_start`) unless the drone is on the ground,
  disarmed and at its start point. Restart the stack between flights; the
  batch runner does this.
- The ratio of a destroyed drone means little on its own: a drone destroyed
  halfway has flown less than the straight-line distance, so its ratio is
  below 1. This is why the metrics table has `ratio+` next to `ratio`.
- `Accel #0 fail: TIMEOUT` at startup can be ignored. It appears when a later
  instance adds its model to the shared Gazebo world, which stops the sensor
  data for a moment. The last instance to start never shows it. It grows with
  the number of drones, so it looks like a performance limit, but it is not:
  with three drones flying, Gazebo uses about 40% of one core and the
  simulation runs at 0.99 of real time.
- If a machine cannot keep up, `sim_speed.py` measures the real time factor
  and `set_sim_speed.py` slows Gazebo down to a speed the machine can hold. An
  even slowdown does no harm, since simulated time, distances and collisions
  stay the same and the run only takes longer. What breaks a run is a
  simulator that falls behind unevenly, because the autopilot then thinks its
  sensors have failed.

## Running stages 3 and 4: the policy

Everything runs in the VM, from the repository root unless noted. Once:

```bash
bash rl/setup_env.sh
```

Then in every terminal that uses the policy:

```bash
source rl/env.sh
```

### Stage 3: one drone

Check the parts first (both write their reports to `rl/logs/`):

```bash
python rl/test_lidar.py
python rl/check_env.py
```

`check_env.py` ends with flying straight compared against the Gazebo sweep,
which should agree on 25 of 25 scenarios for both flight models.

Train. About 31 minutes for 2 million decisions on the VM with 4
environments:

```bash
python rl/train.py --dynamics mujoco --steps 2000000 --name mujoco_1
```

The run folder `rl/runs/mujoco_1/` gets `best.zip` (the version with the most
arrivals on 100 held-out mazes), the lidar and environment settings it was
trained with, and the evaluations in `eval.csv`.

Evaluate in MuJoCo through the code that flies it in Gazebo:

```bash
python rl/evaluate.py rl/runs/mujoco_1
```

It flies the 25 single-drone Gazebo scenarios straight and with the policy,
and checks that the input `policy.py` builds is the one the network was
trained on (the largest difference should be 0).

Fly it in Gazebo. The same as stage 2 with `--policy` added, from `px4/`
(where the venv is `source ../rl/env.sh`):

```bash
./run_multi_mission.py --scenario scenarios3d/s21_n10_k3 --timeout 180 --policy ../rl/runs/mujoco_1
```

The comparison sweep, both arms, about 4 hours in total:

```bash
./run_batch_3d.py --n 4 8 12 16 20 --k 1 2 3 --seeds 1 2 3 4 5 --timeout 180
./run_batch_3d.py --n 4 8 12 16 20 --k 1 2 3 --seeds 1 2 3 4 5 --timeout 180 --policy ../rl/runs/mujoco_1
python3 compare_sweeps_3d.py runs3d/<straight batch> runs3d/<policy batch>
```

The policy arm does not start if the venv is not active. To repeat the
reported sweep with the same model, use
`--policy ../results/2026-10-19_rl_sweep/model`, the copy of the model that
was evaluated. The stage 4 policy is in `results/2026-11-15_swarm_sweep/model/`.

### Stage 4: many drones

```bash
python rl/train.py --dynamics mujoco --drones 4 6 8 8 --neighbours 3 --near-cost 0.5 --n-steps 256 --steps 4000000 --eval-every 400000 --heldout-per-n 8 --name swarm_4
```

About an hour on the VM. `--drones` gives the number of drones in each of the
training processes, `--neighbours 3` adds the inputs about the three nearest
drones. Then the checks and the large evaluation (under an hour):

```bash
python rl/check_swarm.py --policy rl/runs/swarm_4/final.zip
python rl/evaluate.py rl/runs/swarm_4/final.zip
python rl/evaluate_large.py rl/runs/swarm_4/final.zip --drones 1 2 3 4 6 8 10 --mazes 400 --first-seed 300000 --name swarm_scaling
```

In Gazebo nothing changes except the model passed to `--policy`. The
collision monitors publish every drone's velocity next to its position, and
each mission node gives the flying neighbours to the policy. The stage 4
sweeps, from `px4/` after `source ../rl/env.sh`:

```bash
./run_batch_3d.py --n 4 8 12 16 20 --k 1 2 3 --seeds 1 2 3 4 5 --timeout 180 --policy ../rl/runs/swarm_4/final.zip
./run_batch_3d.py --n 4 8 12 16 20 --k 4 --seeds 1 2 3 4 5 --timeout 180 --policy ../rl/runs/swarm_4/final.zip
./run_batch_3d.py --n 4 8 12 16 20 --k 4 --seeds 1 2 3 4 5 --timeout 180
```

Then the tables and figures for the thesis, from the repository root:

```bash
python3 results/make_tables.py
python results/make_figures.py
```

### Notes on the RL stages

- Why train in MuJoCo: one Gazebo flight takes about a minute of real time,
  and training needs hundreds of thousands of decisions. The MuJoCo
  environment makes about 900 decisions per second per environment without
  rendering, so the missing GPU of the VM does not matter.
- Bug in MuJoCo-Drones-Gym (commit ca170dc): its waypoint action mode makes a
  new position controller at every control step, so the controller state is
  lost, the attitude oscillates more and more and the drone flips. Hovering
  and climbing still work, which makes it easy to miss. `rl/env.py` uses the
  package's RPM mode and calls its controller itself, one controller per
  episode. `rl/diagnose_flight.py` shows the failure.
- Same speed in both simulators: the direction is flown at 1.04 m/s level and
  1.17 m/s climbing, the speeds measured in MuJoCo for the waypoint distance
  used in training, and `run_mission.py` sends the same speeds to PX4 as
  velocity setpoints.
- Training stability: with a constant learning rate the policy got better and
  then worse again as its action noise got smaller. A learning rate that
  decreases to zero and `target_kl` 0.03 keep it stable.
- PyTorch on ARM: on ARM64 Linux a plain `pip install torch` installs the
  CUDA build, about 9 GB of NVIDIA libraries that the VM cannot use.
  `setup_env.sh` installs from the CPU-only index instead.
- The stage 3 policy was trained with one drone and static obstacles, so it
  only met other drones in Gazebo. That was one of two possible reasons why
  more drones crash at k = 3 (the other is that the k = 3 mazes are a
  different set, see its results README). Flying the same scenarios in MuJoCo
  later showed it was the mazes: the policy reaches the same drones there
  whether they fly together or each alone.

## Status

Done: the Nav2 baseline (static and moving obstacles), the PX4 baseline over
(n, k) without avoidance (`results/2026-09-06_3d_sweep/`), the RL policy
compared with it on the same scenarios (`results/2026-10-19_rl_sweep/`), and
the policy trained with many drones, evaluated in MuJoCo with up to 10 drones
and in Gazebo with up to 4 (`results/2026-11-15_swarm_sweep/`).

Not done: training at a higher speed (to make the missions shorter), and
Gazebo runs with more than 4 drones, which need a faster machine than the VM.
