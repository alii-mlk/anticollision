# 3D (n, k) sweep without avoidance (2026-09-06)

75 runs: n in {4, 8, 12, 16, 20} ground-standing obstacles, k in {1, 2, 3}
drones, 5 seeds each. PX4 SITL with Gazebo, one PX4 instance and one MAVROS
per drone, all sharing one world. Each drone flies straight at its own 3D
target with offboard position setpoints and is destroyed (force-disarmed) when
it touches an obstacle or another drone. A run ends when every drone has
reached its target or been destroyed.

Source batch: px4/runs3d/batch_20260906_195643 (gitignored; every scenario is
reproducible from its seed with gen_scenario_3d.py). Sweep took 99 minutes on
the VM.

There is no avoidance in these runs. PX4 has no maintained 3D path planner to
install (see px4/FEASIBILITY_3D_NAVIGATION.md), so this is the "without
avoidance" baseline that the avoidance component has to be compared against.

## Result

Collisions per run rise with obstacle count at every drone count, from 1.2 to
2.0 for k = 3, and the share of drones reaching their target falls, from 60%
to 20% for k = 1. So the baseline degrades with obstacle density exactly as
expected.

The two ratio columns are the useful part. `ratio` is over every drone and is
the metric as specified; it falls from 0.93 to 0.60 as n grows. That is not a
path getting shorter, it is destroyed drones contributing the distance they
managed against the full distance they were asked to cover. `ratio+`, over
only the drones that arrive, stays between 1.086 and 1.110 in every one of the
15 configurations. Without avoidance the drones fly almost straight, so all of
the degradation is in survival and none of it in path length. This is the
number the avoidance component should change: it should raise `ratio+` (real
detours) while raising the reached count and lowering the collisions.

CPU time depends on the number of drones and not on the number of obstacles:
about 14, 32 and 50 seconds per run for k = 1, 2 and 3. Simulation cost here is
per vehicle.

Drone-to-drone collisions appear at n = 4, k = 3 (0.4 per run) and n = 8,
k = 3 (0.2 per run) and nowhere else. With few obstacles the drones fly
straighter and cross each other's paths more often; with many obstacles they
are usually destroyed before they can meet.

The generator draws the drone start and target points from the seed before the
obstacles, so the same seed gives the same missions at every n. The summed
Euclidean distance is therefore identical down each k column (21.5, 37.6 and
53.3 m), which means obstacle density is varied against a fixed set of
flights.

The real time factor stayed between 0.88 and 0.99 throughout, so the machine
kept up and the simulation was not slowed at any point.
