# Static-obstacle sweep (2026-07-17)

48 runs: N obstacles in {1, 2, 4, 6, 8, 10} x 8 seeds each, randomized start/goal
(gen_scenario.py defaults), Nav2 heading-agnostic config (see nav2/nav2_params.yaml).

Source batch: nav2/runs/batch_20260717_184042 (gitignored; scenarios reproducible
from seeds). 7 of the 48 runs failed on their first attempt with timeout errors
inside Nav2. The cause was the limited RAM of the virtual machine: when the VM
ran out of memory, the simulation froze briefly and Nav2's internal service
calls timed out. A Nav2 timeout parameter (default_server_timeout) was raised
and the affected runs were repeated with identical seeds and configuration.
No run ever failed for navigation reasons.

Result: 48/48 SUCCEEDED, 0 collisions. Navigation time and path/Euclidean ratio
rise with N; minimum obstacle clearance falls to ~0.37 m (inflation radius 0.8 m
acting as the effective margin). See summary.txt, metrics.csv, plots/.
