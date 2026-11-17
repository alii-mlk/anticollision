# Moving-obstacle speed sweep (2026-07-20)

Stage 1, Nav2 in 2D. 48 runs: 6 obstacles, speeds {0.2, 0.4, 0.8, 1.2, 1.6,
2.0} m/s, 8 seeds each, random start and goal. The obstacles move only in the
virtual sensing (obstacle_mover.py), bouncing off the edges of the workspace.
Hits are counted from the start of navigation until the goal ends. A contact
that already exists when navigation starts is not counted (see
hit_monitor.py).

Source batches: nav2/runs/batch_20260720_{013154,014019,014921,020004,021039,022350},
merged into nav2/runs/sweep_speed_n6 (gitignored, the scenarios can be made
again from the seeds).

Result: the drone always reaches the goal in the end (48 of 48), because the
protocol does not stop a run on contact. The collisions grow quickly with the
obstacle speed: from 1 hit in total at 0.2 m/s to 73 at 2.0 m/s (about 9 per
flight). The mean navigation time doubles (32 to 71 s) and the path becomes
about twice as long as the straight line. The drone itself flies at about 0.6
to 0.7 m/s, so from about 0.8 m/s on the obstacles are faster than the drone,
and Nav2, which only reacts to where the obstacles are now, cannot keep up.
