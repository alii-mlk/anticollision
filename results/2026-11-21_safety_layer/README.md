# A safety layer and a recovery for the policy (2026-11-19 to 2026-11-21)

Stage 5. After stage 4 the professor asked for no crashes, and for an
explanation of why the results in Gazebo and in MuJoCo are not the same. This
folder holds the work on both questions: a fix of the decision timing in
Gazebo, a drone by drone comparison of the two simulators, a safety layer that
checks every command of the policy (`rl/shield.py`), a recovery for drones that
get stuck (`rl/recovery.py`), and the final tests. The policy itself is the
one of stage 4 (`results/2026-11-15_swarm_sweep/model/`) and was not trained
again.

## The decision timing in Gazebo

The policy was trained to decide every 0.104 s. In Gazebo the mission node
only checked whether a decision was due on its 20 Hz setpoint timer, and asked
for "at least 0.104 s since the last one", so each decision waited for the
third tick: in all the policy runs of stages 3 and 4 the policy decided about
every 0.15 s (149 drones of the November sweep, 0.148 to 0.154 s; the October
sweep the same; 0.13 to 0.15 s with 4 drones, where the simulation ran slower
than real time and a tick held less simulated time). Now `px4/run_mission.py` checks 100 times a second on a fixed grid
of 0.104 s of simulated time and sends the new command at once. Every run since
has a decision period of 0.1042 s (`gazebo/timing_test/` is the first one; the
period is also in every `drones.csv`).

The timing was not what made the two simulators differ. Flown in MuJoCo with a
decision every 0.146 s, the closest it can get to 0.15 s, the policy crashed
5.2, 3.6, 4.3 and 3.1% of the flights with 1 to 4 drones, against 5.8, 3.5,
3.9 and 3.1% at 0.104 s on the same mazes (`mujoco/large_swarm_scaling_0146.txt`
and `../2026-11-15_swarm_sweep/mujoco/large_swarm_scaling_2.txt`).

Deciding 44% more often costs processor time: the mission node with the policy
went from about 20 to 25% of a core per drone with 1 to 3 drones and from 25 to
32% with 4. With 4 drones the simulation then ran at 0.85 to 0.92 of real
time instead of about 0.92. PX4 runs in lockstep with Gazebo (the default
`px4_sitl_default` build), so the flights are the same, they only take longer
on the clock.

## Why Gazebo and MuJoCo differ

The final policy flew the 100 Gazebo scenarios (1 to 4 drones, 250 drones) in
both simulators, with the same starts, targets and obstacles, and every drone
was compared with itself (`rl/compare_simulators.py`):

| Gazebo timing | same outcome | arrived only in Gazebo | only in MuJoCo | McNemar p |
|---|--:|--:|--:|--:|
| decision every 0.15 s (`mujoco/compare_final_0104.txt`) | 240 of 250 | 8 | 2 | 0.11 |
| decision every 0.104 s (`mujoco/compare_fixed_timing.txt`) | 241 of 250 | 7 | 2 | 0.18 |

So on the same scenarios the two simulators agree on 96% of the drones, and the
split of the others between the two columns is what chance would give. Of the
7 drones that arrived only in Gazebo, 6 passed a box with 1 to 17 cm to spare
there: near misses in Gazebo that MuJoCo's slightly different drone turns into
touches. Both drones follow the commanded velocity about equally well: one
second ahead, 95% of the positions are within 0.46 m of where the commands
alone would put them in MuJoCo and within 0.44 to 0.46 m in Gazebo.

The rates that looked different in stage 4 (2% crashes in Gazebo with 1 to 3
drones against 4% in MuJoCo, 7% against 3% with 4 drones) come from different
mazes and few Gazebo flights: Gazebo is better with 1 to 3 drones and worse
with 4, and neither difference is significant (Fisher test, p = 0.28 and
0.08). The explanation is therefore: small differences in how the two drones
move only matter for drones that already pass within centimetres of a box, and
the Gazebo sets are too small to show rates of a few percent precisely.

## The safety layer

`rl/shield.py` sits between the policy and the drone, in both simulators.
Before a command is sent it predicts where the drone goes if it flies the
command until the next decision and then brakes, and if it flies it for a
look-ahead of 0.5 s and then brakes. The prediction uses a first-order model
of how the drone follows a velocity command (the drone's velocity moves
towards the command with a time constant, the lag, 0.4 s). The command is
sent unchanged if along these paths the drone keeps 0.30 m from every box, the
ground and the edges of the area, and 0.40 m from every other flying drone
(assumed to keep its velocity). Otherwise the safe command with the velocity
closest to the policy's is sent, chosen from 266 directions at four speeds,
the policy's own direction, and hovering. If none is safe, which only happens
when the drone is already closer than the margin, the one that keeps it
furthest from everything is sent. The settings are in `shield_settings.json`.

Why it does not touch a box, in the model: hovering is braking now, and every
command sent was checked so that braking at the next decision keeps the
margin. So braking is always possible, and by induction the drone keeps the
margin for the whole flight. The real drone follows the model only
approximately, so the guarantee needs the model's error to stay below the
margin. Other drones move and can turn, so for them the distance stays a
measured result.

`checks/check_shield_final.txt` (`rl/check_shield.py`):

1. 12 situations with answers worked out by hand: a wall at two distances, a
   box beside the path, the ground, the edge of the area, a drone coming head
   on, a drone already too close, a drone boxed in. All as expected.
2. The guarantee in the model: 120 flights with three made-up pilots, one of
   them always aiming at the nearest box, with and without the decision period
   varying by 12%. The closest any drone came to a box, the ground or the edge
   was 0.300 m, and a safe command was always found.
3. How well the model predicts real flights, half a second ahead: in MuJoCo 95%
   of the predictions are within 0.07 m (worst 0.25 m); in the final Gazebo
   sweep within 0.08 m with 1 to 3 drones (worst 0.31 m) and 0.12 m with 4
   drones (worst 0.68 m, see the limitations).
4. The 100 Gazebo scenarios in MuJoCo: the policy alone crashed 16 drones, with
   the layer none, and with the layer and the recovery all 250 arrived. One
   check of the layer takes 0.06 to 0.16 ms.

## Tuning the settings with an evolution strategy

The four settings (margin, drone margin, look-ahead, lag) were tuned with the
evolution strategy of Salimans et al. (2017) by `rl/tune_shield.py`: six
mirrored pairs of settings around the current ones per iteration, ranked, and
a step towards the better ones. Each setting flew 336 flights of a separate
tuning block (seeds from 400000), scored 1 per arrival, -10 per crash and
-0.01 per second. After 5 iterations (`es/`, 65 settings) every setting
tried had 0 crashes and 1 to 3 timeouts, and all scores were within 0.01 of
each other, so
there was nothing for the method to climb and the run was stopped. The
defaults were kept. On ordinary mazes they are about as good as these four
numbers can be, and the remaining failures were of a kind the settings cannot
fix (next section).

## The recovery

To see where the layer still failed, the policy flew 120 mazes each with 2, 4
and 8 drones of the tuning block, with and without the layer
(`mujoco/large_mine_400000.txt`): 56 crashes without it, none with it, but 9
timeouts. For 5 of those 9 drones the layer had not changed a single command:
they were the policy's own dead ends, a drone going back and forth behind a box
until its time ran out. The layer adds a few of its own where a gap is
narrower than it allows (1.30 m with the defaults).

`rl/recovery.py` gets such a drone out. A drone that has not come 0.5 m closer
to its target for 8 s climbs straight up, just high enough to pass above
every box near the straight line to its target, flies level to the point above
the target, and descends onto it. Boxes stand on the ground, so there is
nothing above a drone, and the scenario generator keeps 2.5 m around every
target free, so the way down is clear. The safety layer still checks every
command of the recovery. If the drone gets stuck again during it, the cruise
altitude goes up by a metre.

Before the full tests it was tried on the hard cases:

- the tuning block again (`mujoco/large_mine_400000_recovery.txt`): 1,680 of
  1,680 flights arrived, 29 needed the recovery;
- every maze of the final test block where any drone of any earlier
  evaluation did not arrive (`mujoco/large_hard_300000_recovery.txt`, 819
  mazes, 5,334 flights): all arrived, 253 needed the recovery;
- in Gazebo, the 19 runs that ever had a crash, a timeout or a difference
  between the simulators (`gazebo/hard_runs/`): 63 of 63 drones arrived, 6
  needed the recovery.

## Results

Gazebo, the 100 scenarios of stage 4, every drone deciding every 0.104 s:

| drones | controller | arrived | crashed | timed out |
|---|---|--:|--:|--:|
| 1 to 3 | final policy | 147 of 150 | 3 | 0 |
| 1 to 3 | with the layer | 149 of 150 | 0 | 1 |
| 1 to 3 | with the layer and the recovery | 150 of 150 | 0 | 0 |
| 4 | final policy | 92 of 100 | 8 | 0 |
| 4 | with the layer | 97 of 100 | 0 | 3 |
| 4 | with the layer and the recovery | 100 of 100 | 0 | 0 |

Flying straight, 74 of 150 and 49 of 100 arrived on the same scenarios. Drone
by drone, 239 arrived with all three controllers, 7 crashed with the policy
alone and arrived with the layer, and 4 crashed with the policy alone, timed
out with the layer and arrived with the recovery. No drone that arrived with
the policy alone failed with the layer or the recovery.

MuJoCo, the 13,600 flights of the final evaluation (400 mazes for each of 1,
2, 3, 4, 6, 8 and 10 drones, seeds from 300000):

| controller | arrived | crashed | timed out |
|---|--:|--:|--:|
| final policy | 95.2% | 589 | 69 |
| with the layer | 99.2% | 0 | 109 |
| with the layer and the recovery | 100% | 0 | 0 |

The last row holds for every number of drones. With no failure in 13,600
flights the rate of crashes, and of timeouts, is at most 0.028% per flight (95%
upper bound). Flight by flight, every flight that crashed or timed out with
the policy alone or with the layer alone arrives with the layer and the
recovery. The same flights in this run and in the run on the hard mazes are
identical to the last decision: the simulation is deterministic, so the
results can be repeated exactly.

What it costs a drone that arrives with all three controllers (Table 20): in
MuJoCo the mean flight time is 19.43 s with the policy alone, 19.48 s with the
layer and 19.50 s with the layer and the recovery, and the path ratio 1.151,
1.153 and 1.155; in Gazebo the mean times differ by less than 0.3 s. The layer
replaces about 1% of the commands. The recovery was needed in 2.3% of the
MuJoCo flights and by 6 of the 250 Gazebo drones, which arrived after 35 to
50 s; the longest MuJoCo flight took 56 s. Memory is the same with and without the layer (960,
1,682, 2,528 and 3,515 MB for 1 to 4 drones), and the mission node's
processor time varies more between days than between the controllers.

No drone touched another one in any of these runs. The monitors now measure
the distance between drones directly: the closest two drones came was 1.43 m,
centre to centre, in the final sweep.

## More drones: 15 and 20 in MuJoCo (2026-11-22)

Gazebo stops at four drones in the VM, MuJoCo does not. So the three
controllers were flown again with 15 and 20 drones in the same 24 m square,
twice as crowded as the 10 drones of the final evaluation. The policy was
trained with 4 to 8 drones and sees only its 3 nearest neighbours.

The scenario generator keeps 2.5 m free around every start and target, and
with 20 drones that is 40 points in the square, so it often cannot place 16 or
20 boxes. Leaving those mazes out would have kept mostly mazes with few boxes.
`evaluate_large.py --redraw` gives such a maze a new seed with the same number
of boxes (36 of the 400 mazes with 15 drones, 191 of 400 with 20). The redrawn
mazes are the ones with many boxes, so they are harder than the others, not
easier: the policy alone arrives in 90.7% and 92.6% of their flights against
95.6% and 96.9% in the mazes that were not redrawn.

| drones | controller | flights | arrived | crashed | hit drone | timed out | recovery used |
|---|---|---:|---:|---:|---:|---:|---:|
| 15 | final policy | 6,000 | 95.2% | 270 | 10 | 19 | |
| 15 | with layer | 6,000 | 99.3% | 0 | 0 | 41 | |
| 15 | with layer and recovery | 6,000 | 100% | 0 | 0 | 0 | 114 |
| 20 | final policy | 8,000 | 94.8% | 391 | 20 | 22 | |
| 20 | with layer | 8,000 | 99.5% | 0 | 0 | 41 | |
| 20 | with layer and recovery | 8,000 | 100% | 0 | 0 | 0 | 162 |

(Table 22; all three on the same mazes, drone for drone.)

- The policy alone arrives about as often as with 1 to 10 drones (95.2% and
  94.8% against 95.2%); crashes into boxes do not grow with the crowd.
- But for the first time the final policy touched other drones: 5 contacts
  with 15 drones and 10 with 20 (each contact is two drones, so 10 and 20
  flights), after none in 13,600 flights with up to 10. Two drones meeting is
  much more likely when the area is twice as full, and three neighbours may be
  too few to see all of them.
- With the layer there was no contact and no crash, and with the recovery
  every one of the 14,000 flights arrived. Together with the final evaluation
  that is 27,600 MuJoCo flights without a crash, so the crash rate is at most
  0.014% per flight (95% upper bound; 0.027% for these 14,000 alone).
- The layer works harder: it changed 1.9% and 2.2% of the commands (with the
  recovery 1.7% and 2.0%) against 1.2 to 1.5% with up to 10 drones, and it
  found no safe command 2,813 and 5,918 times, where it keeps the command with
  the most room. The closest any drone came to anything was 0.13 m with 15
  drones and 0.17 m with 20 (0.21 m with up to 10), less than the margin but
  never a touch.

The two runs took 44 and 34 minutes in the VM.

## Limitations

- The guarantee holds for the model. The real drones do not follow the model
  exactly, and the worst prediction error measured (0.68 m half a second ahead,
  with 4 drones in Gazebo) is larger than the 0.30 m margin. So in Gazebo and
  in MuJoCo the absence of crashes is a measured result, not a proof.
- One near miss: in the final sweep one drone (scenario n16, 4 drones, seed 4,
  drone 3) came within 0.078 m of a box; the next lowest was 0.19 m. It was in
  a corridor 1.65 m wide, where the layer has only a 35 cm band in which it
  keeps the margin on both sides, and the layer had to reverse it several
  times. In one reversal PX4 kept the drone moving the old way for 0.4 s after
  the new command, much longer than the model expects (probably its own
  smoothing of sharp setpoint changes), and that moment is the 0.68 m error
  above. Then the recovery climbed it out and it arrived
  (`examples/n16_k4_s4/`).
- Between drones nothing is guaranteed: the other drones are assumed to keep
  their velocity, and they broadcast position and velocity to each other.
- The recovery relies on two properties of these mazes: boxes stand on the
  ground, and every target has free space around it. It flies over the boxes,
  which is fine in an empty sky but would not be in a building.
- A MuJoCo flight ends after 62.5 s and a Gazebo flight after 180 s, so a slow
  recovery counts as a timeout sooner in MuJoCo. The longest flight with the
  recovery was 56 s.
- Gazebo with at most 4 drones, the limit of the VM; larger swarms only in
  MuJoCo.
- The settings of the layer were chosen by hand from the measured model error;
  the evolution strategy found no better ones on ordinary mazes. One policy,
  from one training run.

## Files

| | |
|:--|:--|
| `gazebo/policy_timing/`, `policy_timing_k4/` | the final policy alone with the fixed timing, 75 and 25 runs (`px4/runs3d/batch_20261120_033828_policy`, `_k4_policy`) |
| `gazebo/shield/`, `shield_k4/` | with the layer (`batch_20261120_033828_shield`, `_k4_shield`); run n4_k1_s3 was flown again on 2026-11-20 after the network of the VM changed during its first try |
| `gazebo/shield_recovery/`, `shield_recovery_k4/` | with the layer and the recovery, the final sweep (`batch_20261121_101748_shield_recovery`, `_k4_shield_recovery`) |
| `gazebo/hard_runs/` | the 19 hardest runs with the layer and the recovery (`batch_20261120_144821_shield_recovery`) |
| `gazebo/timing_test/` | the first run with the fixed timing: `decision_period` in `mission.yaml`, and the per-decision log used to fit the lag |
| `mujoco/large_shield_default.*` | the 13,600 flights with the layer |
| `mujoco/large_final_recovery.*` | the 13,600 flights with the layer and the recovery |
| `mujoco/large_swarm_scaling_0146.*` | the policy alone deciding every 0.146 s |
| `mujoco/large_mine_400000*.*` | the tuning block, without and with the recovery |
| `mujoco/large_hard_300000_recovery.*` | the hard mazes of the final test block with the recovery |
| `mujoco/compare_*` | the Gazebo scenarios flown again in MuJoCo, drone by drone |
| `checks/` | `rl/check_shield.py` before the first night and at the end, and the output of the steps after the two long Gazebo runs |
| `es/` | the evolution strategy run: one row per iteration, every setting tried, the settings of the run |
| `examples/n8_k4_s4/` | the stuck drone (drone 3) flown three ways: crashed, held at the gap, climbed over; for figure 10 |
| `examples/n16_k4_s4/` | the near miss |
| `shield_settings.json` | the settings of the layer used in all runs |

The per-flight CSV files are gzipped. Every Gazebo folder has `metrics.csv`,
`drones.csv` and `summary.txt` from `px4/compute_metrics_3d.py`; since stage 5
`drones.csv` also has the decision period, how many commands the layer
changed, and how long the recovery flew. `mujoco/large_crowd_recovery` (the
policy alone and with the layer and the recovery) and `mujoco/large_crowd_shield`
(the layer alone) are the runs with 15 and 20 drones. Tables 15 to 22 and
figures 9 and 10 of `results/tables/` and `results/figures/` are made from this
folder.

## How it was run

In the VM, from the repository root, after `source rl/env.sh`:

```bash
python rl/check_shield.py --gazebo-batch px4/runs3d/<batch>
python rl/evaluate_large.py results/2026-11-15_swarm_sweep/model/final.zip --shield default --shield-only --recovery --no-straight --drones 1 2 3 4 6 8 10 --mazes 400 --first-seed 300000 --name final_recovery
python rl/evaluate_large.py results/2026-11-15_swarm_sweep/model/final.zip --shield default --recovery --no-straight --drones 15 20 --mazes 400 --first-seed 300000 --redraw --name crowd_recovery
python rl/evaluate_large.py results/2026-11-15_swarm_sweep/model/final.zip --shield default --shield-only --no-straight --drones 15 20 --mazes 400 --first-seed 300000 --redraw --name crowd_shield
```

and from `px4/`:

```bash
./run_batch_3d.py --n 4 8 12 16 20 --k 1 2 3 --seeds 1 2 3 4 5 --timeout 180 --policy ../rl/runs/swarm_4/final.zip --shield default --recovery
./run_batch_3d.py --n 4 8 12 16 20 --k 4 --seeds 1 2 3 4 5 --timeout 180 --policy ../rl/runs/swarm_4/final.zip --shield default --recovery
```

Without `--recovery` it is the layer alone, without `--shield` the policy
alone. `--runs n20_k3_s5 n8_k4_s4 ...` flies a list of runs instead of the
whole grid.
