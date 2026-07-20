# Moving-obstacle speed sweep — 2026-07-20

48 runs: 6 obstacles, speeds {0.2, 0.4, 0.8, 1.2, 1.6, 2.0} m/s x 8 seeds,
randomized start/goal. Obstacles move virtually (obstacle_mover.py bouncing
within workspace bounds); hits are counted from navigation start to goal
resolution, with pre-existing overlaps at navigation start excluded
(hit_monitor.py adoption logic).

Source batches: nav2/runs/batch_20260720_{013154,014019,014921,020004,021039,022350},
merged into nav2/runs/sweep_speed_n6 (gitignored; scenarios reproducible from seeds).

Result: goal completion never fails (48/48 — runs are not stopped on contact,
per protocol), but collision avoidance degrades sharply with obstacle speed:
total hits rise 1 -> 73 across the sweep (~9 per flight at 2.0 m/s), mean
navigation time doubles (32 -> 71 s), and the path/Euclidean ratio grows to
~2x. The drone's own speed limit is ~0.6-0.7 m/s: beyond ~0.8 m/s obstacles
outrun the drone and Nav2's prediction-free replanning cannot compensate.
This quantifies where plain Nav2 stops being sufficient and motivates the
predictive avoidance component to be added on top of it.
