# 3D (n, k) sweep without avoidance (2026-09-06)

Stage 2, PX4 and MAVLink in 3D. 75 runs: n in {4, 8, 12, 16, 20} obstacles
standing on the ground, k in {1, 2, 3} drones, 5 seeds each. PX4 SITL with
Gazebo, one PX4 instance and one MAVROS per drone, all in one world. Each
drone flies straight at its own 3D target with offboard position setpoints,
and is destroyed (force-disarmed) when it touches an obstacle or another
drone. A run ends when every drone has reached its target or been destroyed.

Source batch: px4/runs3d/batch_20260906_195643 (gitignored, every scenario can
be made again from its seed with gen_scenario_3d.py). The sweep took 99
minutes on the VM.

There is no avoidance in these runs. PX4 has no maintained 3D path planner
that could be installed (see px4/FEASIBILITY_3D_NAVIGATION.md), so this is the
baseline without avoidance that the avoidance component is compared with.

## Result

Collisions per run grow with the number of obstacles at every drone count,
from 1.2 to 2.0 for k = 3, and the share of drones that reach their target
falls, from 60% to 20% for k = 1.

There are two ratio columns. `ratio` is over every drone, as in the
evaluation plan, and it falls from 0.93 to 0.60 as n grows. This happens
because a destroyed drone counts the distance it managed against the full
distance it was asked to fly. `ratio+` is over the drones that arrive only,
and it stays between 1.086 and 1.110 in all 15 configurations, because
without avoidance the drones fly almost straight. So more obstacles mean more
destroyed drones, while the paths of the drones that arrive stay the same
length. With avoidance, `ratio+` should go up (the drones fly around
obstacles), together with more arrivals and fewer collisions.

CPU time depends on the number of drones and not on the number of obstacles:
about 14, 32 and 51 seconds per run for k = 1, 2 and 3.

Collisions between drones happen at n = 4, k = 3 (0.4 per run) and n = 8,
k = 3 (0.2 per run) and nowhere else. With few obstacles the drones fly
straighter and cross each other's paths more often. With many obstacles they
are usually destroyed before they can meet.

The generator draws the start and target points from the seed before the
obstacles, so the same seed gives the same missions at every n. The summed
Euclidean distance is therefore the same down each k column (21.5, 37.6 and
53.3 m), and only the number of obstacles changes between the rows.

The real time factor was 0.88 to 0.99 on average per configuration, and 0.76
to 1.00 for single runs (7 of the 75 runs were below 0.88). Time is measured
on Gazebo's clock, so a slower run gives the same result and only takes
longer.
