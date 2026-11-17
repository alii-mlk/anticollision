# Static-obstacle sweep (2026-07-17)

Stage 1, Nav2 in 2D. 48 runs: N obstacles in {1, 2, 4, 6, 8, 10}, 8 seeds
each, random start and goal (gen_scenario.py defaults), Nav2 configured so the
final heading does not matter (see nav2/nav2_params.yaml).

Source batch: nav2/runs/batch_20260717_184042 (gitignored, the scenarios can be
made again from the seeds). 7 of the 48 runs failed the first time with
timeout errors inside Nav2. The cause was the small amount of RAM in the
virtual machine: when it ran out of memory, the simulation froze for a moment
and Nav2's internal service calls timed out. I raised a Nav2 timeout parameter
(default_server_timeout) and ran those runs again with the same seeds and
settings. No run failed because of navigation.

Result: 48 of 48 reached the goal, with 0 collisions. Navigation time and the
path/Euclidean ratio grow with N, and the minimum clearance to an obstacle
falls to about 0.37 m (the costmap inflation radius of 0.8 m is what keeps
this margin). The numbers are in summary.txt and metrics.csv, the plots in
plots/.
