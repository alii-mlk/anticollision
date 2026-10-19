# 3D (n, k) sweep with and without the RL policy (2026-10-19)

The same 75 runs as the September baseline (n in {4, 8, 12, 16, 20} obstacles,
k in {1, 2, 3} drones, seeds 1 to 5), flown twice on the same day with the same
software: once straight at the target as before, and once with every drone
steered by the trained policy. Both use the same scenarios, the same mission
code and the same collision monitor, so they can be compared run by run and
drone by drone.

The policy is `model/best.zip`, the trained network as Stable-Baselines3 saves
it, next to the lidar (`lidar.json`) and the settings (`config.json`) it was
trained with. These three files are copied from `rl/runs/mujoco_1`, which is
gitignored. The network was trained with PPO in MuJoCo only, one drone per
episode, and had never been flown in Gazebo before this. It sees the 82-ray lidar
(`rl/lidar.py`), the direction and distance to the target, its velocity and
its altitude, and outputs a direction. `run_mission.py --policy` sends that
direction to PX4 as an offboard velocity setpoint (1.04 m/s level, 1.17 m/s
climb, the speeds it was trained at), with a new decision every 0.104 s of
simulated time. Other drones are put into the lidar as cubes at their
published positions.

Source batches (gitignored, reproducible from the seeds):
`px4/runs3d/batch_20261019_105440_straight` (97 minutes) and
`px4/runs3d/batch_20261019_132454_policy` (128 minutes), both on the VM.

```
python3 compute_metrics_3d.py runs3d/<batch>
python3 compare_sweeps_3d.py runs3d/<straight batch> runs3d/<policy batch>
./run_batch_3d.py --n 4 8 12 16 20 --k 1 2 3 --seeds 1 2 3 4 5 --timeout 180 \
    --policy ../results/2026-10-19_rl_sweep/model
```

`straight/` and `policy/` hold each sweep's metrics.csv and summary.txt in the
same format as the September folder. `comparison.txt` is the full output of
`compare_sweeps_3d.py`.

## The baseline was re-run first

An `apt upgrade` on 2026-10-11 moved MAVROS from 2.14 to 2.16, gz-sim from 8.14
to 8.15 and ros-gz-bridge from 1.0.22 to 1.0.24, after the September sweep. So
the straight arm was flown again on the current software rather than reused.

It reproduces September: 72 of 75 runs have the same arrivals and the same
collisions, the obstacle collisions are identical in all 15 configurations, and
the totals are the same (74 of 150 drones reached). The three runs that differ
are all seed 5 with two or three drones, and they differ only in drone to drone
collisions, which depend on small timing differences in when each drone
starts. The upgrade did not change the baseline.

## Result

| n | k | reached, straight | reached, policy | collisions/run | ratio+ |
|--:|--:|--:|--:|:--|:--|
| 4 | 1 | 3/5 | 4/5 | 0.4 -> 0.2 | 1.105 -> 1.128 |
| 4 | 2 | 10/10 | 10/10 | 0.0 -> 0.0 | 1.097 -> 1.127 |
| 4 | 3 | 11/15 | 14/15 | 0.8 -> 0.2 | 1.097 -> 1.166 |
| 8 | 1 | 3/5 | 5/5 | 0.4 -> 0.0 | 1.099 -> 1.137 |
| 8 | 2 | 5/10 | 9/10 | 1.0 -> 0.2 | 1.102 -> 1.150 |
| 8 | 3 | 8/15 | 15/15 | 1.4 -> 0.0 | 1.095 -> 1.205 |
| 12 | 1 | 2/5 | 5/5 | 0.6 -> 0.0 | 1.092 -> 1.155 |
| 12 | 2 | 4/10 | 9/10 | 1.2 -> 0.2 | 1.103 -> 1.171 |
| 12 | 3 | 8/15 | 13/15 | 1.4 -> 0.4 | 1.102 -> 1.253 |
| 16 | 1 | 1/5 | 5/5 | 0.8 -> 0.0 | 1.091 -> 1.172 |
| 16 | 2 | 4/10 | 10/10 | 1.2 -> 0.0 | 1.104 -> 1.180 |
| 16 | 3 | 6/15 | 11/15 | 1.8 -> 0.8 | 1.104 -> 1.294 |
| 20 | 1 | 1/5 | 5/5 | 0.8 -> 0.0 | 1.090 -> 1.273 |
| 20 | 2 | 3/10 | 9/10 | 1.4 -> 0.2 | 1.102 -> 1.209 |
| 20 | 3 | 5/15 | 11/15 | 2.0 -> 0.8 | 1.109 -> 1.335 |
| all | | 74/150 (49%) | 135/150 (90%) | 76 -> 15 drones destroyed | |

Drones reaching their target go from 49% to 90%, and destroyed drones from 76
to 15. Looking at the same drone in the same scenario, the policy saved 61
drones and lost none. All 15 drones destroyed with the policy were also
destroyed flying straight.

Without avoidance `ratio+` was between 1.09 and 1.11 in every configuration.
With the policy it is between 1.13 and 1.34 and grows with the number of
obstacles, because the drones now fly around the obstacles instead of straight
at the target.

Drone to drone collisions go from 3 to 0. Flying straight, three runs had two
drones pass within 1 m of each other in the air. With the policy no two drones
came closer than 2.66 m in any of the 50 multi-drone runs.

The result in Gazebo matches what MuJoCo predicted for the 25 single-drone
scenarios (`rl/evaluate.py`): MuJoCo said 23 of 25, Gazebo gave 24 of 25. The
one Gazebo crash (n = 4, seed 2) is one of the two MuJoCo crashes. The other
one (n = 20, seed 4) touched the obstacle in MuJoCo and missed it by less than
a millimetre in Gazebo.

## Costs

The missions take longer: about 30 s of simulated time instead of 11 to 12 s.
The detours add a little, but most of the difference is the speed. The policy
flies at the 1.04 m/s it was trained at, while PX4's position controller
averaged 1.70 m/s on the same drones flying straight. The training speed was
set by calibrating the MuJoCo controller (`rl/check_env.py`). A faster policy
should be possible, but it would have to be trained again.

The controller's own CPU time (lidar plus network, every 0.104 s) is 3.1 to
3.9 s per mission with one drone and 13 to 20 s per run with three, against
about 1 s per drone for the straight-line mission node. The total CPU time per
run is 2 to 3 times the baseline, mostly because the runs last longer. The
real time factor stayed between 0.94 and 1.00 in every run, so the VM kept up
with three drones and their policies.

## Limitations

Destroyed drones by number of drones in the run:

| k | straight | policy |
|--:|--:|--:|
| 1 | 15/25 (60%) | 1/25 (4%) |
| 2 | 24/50 (48%) | 3/50 (6%) |
| 3 | 37/75 (49%) | 11/75 (15%) |

Flying straight, the crash rate does not depend on k. With the policy it rises
at k = 3. All 15 crashes are obstacle hits, at 4 to 7 m altitude, and none
happened with another drone close by. Of the 14 crashes in runs with more than
one drone, the nearest other drone in the 5 s before the crash was 2.8 m away
once, 3.8 m once, and 5.5 m or more (or not in the air) in the other twelve.

There are two possible reasons, and this data does not separate them. The
first is that the k = 3 mazes are a different set of mazes: the generator
places the obstacles after all k start and target points, so the same seed
gives other obstacles for a different k. The second is that the policy was
trained alone. In the k = 3 runs another drone was within the 10 m lidar range
for 72% of the time in the air, which is an input the policy never saw in
training, even when the other drone is too far away to matter. Flying the
k = 3 scenarios in MuJoCo with each drone alone would tell the two apart.

Clearance to obstacles, over the drones that arrived:

| | drones | median closest approach | under 0.5 m |
|:--|--:|--:|--:|
| straight, drones that arrive both ways | 74 | 1.65 m | 19 |
| policy, the same 74 drones | 74 | 1.96 m | 6 |
| policy, the 61 drones it saved | 61 | 0.45 m | 35 |

On the flights that also succeed without avoidance, the policy keeps more
distance from the obstacles than flying straight. The drones it saves pass much
closer: more than half come within 0.5 m of an obstacle, and two within a
centimetre. In training the penalty for being within 1 m of an obstacle is
small: it grows quadratically to at most 0.1 per decision at zero clearance,
and is 0.025 at 0.5 m, against 10 for a crash. So the policy learned to get
through gaps rather than to keep away from walls. Raising that penalty is the
first thing to try if larger margins are wanted.
