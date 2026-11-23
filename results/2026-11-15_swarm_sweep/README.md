# Many drones with one shared policy (2026-11-13 to 2026-11-16)

Stage 4. The policy of October was trained with one drone and flew as if it were alone:
other drones were only boxes in its lidar. In MuJoCo with ten drones it hit
another drone in 8% of the flights, as often as flying straight. This folder
holds the policy that replaced it and everything measured with it.

The new policy is trained with several drones in the same maze, all flown by
the same network. Each drone is also told about its three nearest flying
neighbours within 6 m (ten numbers each: direction and closeness, relative
velocity, the predicted relative position at the closest point of approach,
and how soon that is), which assumes the drones broadcast their position and
velocity. In Gazebo the collision monitors publish both, and every mission
node passes them to its policy. Training also gives a small cost for being
within 2 m of another drone.

The October policy is called "trained alone" below, the new one "final
policy".

## The policy

`model/final.zip`, with the lidar (`lidar.json`), the settings (`config.json`)
and the evaluations during training (`eval.csv`), copied from
`rl/runs/swarm_4` (gitignored). PPO in MuJoCo from scratch, 4 million
decisions in 59 minutes on the VM, four scenes of 4, 6, 8 and 8 drones with
0 to 20 obstacles. A drone that arrives or crashes gets a new mission, so the
scenes stay crowded. Input: 82 lidar rays, 8 values for target, velocity and
altitude, 30 for the neighbours.

```
python rl/train.py --dynamics mujoco --drones 4 6 8 8 --neighbours 3 \
    --near-cost 0.5 --n-steps 256 --steps 4000000 --eval-every 400000 \
    --heldout-per-n 8 --name swarm_4
```

It was chosen on 200 new mazes per drone count (`mujoco/large_compare_3.txt`,
seeds from 200000), against the earlier many-drone policy without neighbour
inputs and a second run that started from that policy. The second run did
equally well there. This one was kept because it is one training run from
scratch and needs no checkpoint picking. The results below come from other
mazes than the ones it was chosen on.

## Results in MuJoCo

400 new mazes per drone count (seeds from 300000), with 4, 8, 12, 16 or 20
obstacles as in the Gazebo sweep.
From `mujoco/large_swarm_scaling.txt` and `large_swarm_scaling_2.txt`:

| drones | straight | trained alone | final policy |
|--:|--:|--:|--:|
| 1 | 45.5% | 78.5% | 93.8% [90.9, 95.7] |
| 3 | 44.8% | 81.9% | 95.7% [94.4, 96.7] |
| 10 | 43.5% | 77.3% | 95.3% [94.7, 96.0] |
| all, 13,600 flights | 43.9% | 78.9% | 95.2% [94.8, 95.5] |
| hit another drone | 804 | 740 | 0 |

Share of drones that arrived, with the 95% interval. With the final policy no
drone hit another one in 13,600 flights, so the rate is at most about 0.03%
per flight (95% upper bound). The share that arrives no longer falls as drones
are added. What is left are crashes, 3 to 6% of the flights, and under 1%
timeouts.

The October policy does worse here than in its Gazebo sweep (78.9% against
90%). That sweep had 75 scenarios and the policy had been picked on small
evaluation sets, which together flattered it.

`mujoco/check_swarm_swarm_4.txt`: the many-drone simulator flies one drone the
same as the one-drone environment, the drones in one process do not affect
each other, and the neighbour inputs agree with hand-worked cases.
`mujoco/evaluate_swarm_4.txt`: what the policy sees in Gazebo is the same as
in training, to the last digit, with one drone and with four.
`mujoco/check_swarm_mujoco_1.txt`: the same checks with the October policy,
which reaches 23, 47 and 64 drones for 1, 2 and 3 drones in MuJoCo whether
the drones fly together or each alone, against 24, 47 and 64 in Gazebo.

## Results in Gazebo

The same 75 scenarios as in October (4 to 20 obstacles, 1 to 3 drones, seeds
1 to 5), and 25 more with 4 drones. The straight arm for 1 to 3 drones is the
October batch: the 15 straight runs flown again on 2026-11-15 to measure
memory gave the same result in 14 of 15.

| drones | straight | trained alone | final policy |
|--:|--:|--:|--:|
| 1 to 3 | 74/150 (49%) | 135/150 (90%) | 147/150 (98%) [94.3, 99.3] |
| 4 | 49/100 (49%) | not flown | 93/100 (93%) [86.3, 96.6] |

All 10 drones destroyed with the final policy hit an obstacle. Comparing the
same drone in the same scenario, it saved 73 drones and lost none against
flying straight with 1 to 3 drones, and 44 with none lost with 4 drones.
Against the October policy it was better in 11 scenarios, equal in 64 and
worse in none.

Per scenario, the share of drones that arrived went up by 51 percentage points
on average (standard deviation 40, 75 scenarios) for 1 to 3 drones, and by 44
(standard deviation 22, 25 scenarios) for 4 drones. The deviation is large
because a scenario has only 1 to 4 drones, so its share moves in steps of 25
to 100 points.

MuJoCo predicted Gazebo closely: on the 25 one-drone scenarios 25 arrivals in
both, on the 50 two-drone flights 46 in MuJoCo and 50 in Gazebo, on the 75
three-drone flights 70 and 72. Stage 5 compares the two simulators drone by
drone (`results/2026-11-21_safety_layer/`): the same outcome for 240 of the
250 drones.

### Path, time and clearance

The detours grow with the number of obstacles: the distance flown over the
straight line is 1.15 with 4 obstacles and 1.27 with 20 (standard deviation
0.06 and 0.13). On the drones that arrive anyway, flying straight or with the
policy (1 to 3 drones), the policy flies 6.7% more distance (standard
deviation 6.2) and takes 15 s longer (standard deviation 3.4). Most of that is speed: the policy
flies at the 1.04 m/s it was trained at, while PX4's own position control
averaged 1.7 m/s.

On the 74 drones that arrived with all three controllers, the median closest
approach to an obstacle was 1.65 m flying straight, 1.96 m with the October
policy and 2.28 m with the final one, and 19, 6 and 1 of them came within
0.5 m. The final policy was trained with a five times larger cost for passing
close to obstacles than the October one.

### Distance between drones

The collision monitors compare live positions and saw 5 drones destroyed by
touching another drone flying straight (2 to 4 drones), and none with either
policy.

The closest distance between two drones can only be given as a lower bound
for these runs. Each drone's track counts time from the start of its own
mission node, the nodes of one run start up to about a second apart, and these
runs did not record when. On the four runs where two drones touched, the true
offsets were 0.2 to 0.94 s. Allowing up to 2 s, no two drones with the final
policy came closer than 1.74 m with 2 or 3 drones and 1.25 m with 4. Since
2026-11-17 the monitors record this distance themselves and each node saves
when its clock started (`sim_start`), so later runs give the exact value.
`separation.csv` and `separation_k4.csv` have the value per run.

## Costs

Processor time of the simulator, PX4 and MAVROS, in cores busy: 0.83, 1.73,
2.61 and 3.43 for 1 to 4 drones with the final policy, about the same as
flying straight. The mission node with the policy (lidar and network) uses 12%
of a core with one drone and 25% per drone with four, against 8 to 13% flying
straight. In these runs the policy decided every 0.15 s and not every 0.104 s
as in training: the node only checked on its 20 Hz timer, so a decision
waited for every third tick. This was found and fixed in stage 5, where the
policy decides 44% more often and its node uses about a quarter more
processor time.

Memory, measured on 2026-11-16 in `ram_straight/` and `ram_policy/` (one run
per obstacle count, averaged), in MB:

| drones | straight | final policy |
|--:|--:|--:|
| 1 | 621 | 963 |
| 2 | 1,109 | 1,678 |
| 3 | 1,717 | 2,524 |
| 4 | 2,482 | 3,511 |

Pages shared between processes are divided among them, so the parts add up.
The largest part is MAVROS (1,443 MB of the 3,511 with 4 drones). The policy
adds about 300 MB per drone, almost all of it in the mission nodes, which each
load PyTorch and the network.
The obstacles add about 10 MB per 4 obstacles, in Gazebo.
`ram_policy/memory_example_n4_k4.txt` lists every counted process of one run.

## Limitations

- Four drones is the limit of the VM. With the policy the real time factor
  was 0.88 to 0.94, flying straight 0.72 to 0.97. With five drones the
  simulation ran at 0.5 to 0.6 of real time, so that result is not valid and
  is not here.
- In Gazebo there are 25 scenarios per drone count, so the intervals are
  wide. The large MuJoCo evaluation gives the tighter numbers, and MuJoCo
  predicted the Gazebo results within a few drones.
- There is no guarantee of avoidance, only measured rates: no drone to drone
  contact in 13,600 MuJoCo flights and 225 Gazebo flights with more than one
  drone, and crashes in 3 to 6% of the MuJoCo flights. Stage 5 adds a safety
  layer and a recovery around this policy, see
  `results/2026-11-21_safety_layer/`.
- One training run. Another seed could give a somewhat different policy.
- An earlier memory batch (`px4/runs3d/batch_20261116_121949_policy`) is not
  used: from its 7th run on, 241 MB that were not part of the stack were
  counted. Process matching was fixed and the memory flown again. The times
  in `ram_straight/` and `ram_policy/` are not used either, the VM ran slow
  during the policy part.

## Files

| | |
|:--|:--|
| `policy/` | the 75-run sweep with the final policy, `px4/runs3d/batch_20261115_153421_policy` |
| `k4_straight/`, `k4_policy/` | 4 drones, 25 runs each, `batch_20261116_054900_straight` and `batch_20261116_111911_policy` |
| `ram_straight/`, `ram_policy/` | memory per part of the stack, 1 to 4 drones, `batch_20261116_200448_straight` and `batch_20261116_203353_policy` |
| `comparison.txt`, `comparison_k4.txt` | `compare_sweeps_3d.py`, against the October straight batch and for 4 drones |
| `separation*.csv` | closest distance between drones per run (lower bounds) |
| `examples/` | one scenario (n = 12, k = 3, seed 2) and the paths flown in it, straight and with the policy, for figure 8 |
| `model/` | the policy |
| `mujoco/` | the MuJoCo evaluations and checks named above, and `crashes_swarm_1.txt`, which showed that the obstacle crashes happen with the obstacle in view |

Every batch folder has `metrics.csv` (one row per run), `drones.csv` (one row
per drone) and `summary.txt` from `compute_metrics_3d.py`. The memory columns
(`ram`, `ctrl ram`) in `policy/`, `k4_straight/` and `k4_policy/` come from the
first version of the memory measurement and are not used: in `policy/` the
simulator column includes about 310 MB of the launcher script, and none of
the three counts the helper processes. All memory numbers come from
`ram_straight/` and `ram_policy/`.

The tables and figures for the thesis are made from these files and the
October folder:

```
python3 results/make_tables.py
```

```
python results/make_figures.py
```

The first runs anywhere and writes `results/tables/`. The second needs
matplotlib, so it runs in the VM after `source rl/env.sh`, and writes
`results/figures/`.
