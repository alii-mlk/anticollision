# Results

One folder per experiment, named by the day it was run. Every folder has its
own README with the setup, the numbers and the limitations. The raw run
folders are large and gitignored (`nav2/runs/`, `px4/runs3d/`, `rl/runs/`), so
only what the thesis uses is copied here.

| Folder | Stage | What it holds |
|---|---|---|
| `2026-07-14_first_success/` | 1 | the logs of the first Nav2 flight that reached its goal (8, 0) behind the wall |
| `2026-07-17_static_sweep/` | 1 | Nav2, 48 runs with 1 to 10 static obstacles |
| `2026-07-20_speed_sweep/` | 1 | Nav2, 48 runs with obstacles moving at 0.2 to 2.0 m/s |
| `2026-09-06_3d_sweep/` | 2 | PX4, 75 runs with 4 to 20 obstacles and 1 to 3 drones, flying straight (the baseline) |
| `2026-10-19_rl_sweep/` | 3 | the same 75 runs flown straight again and with the first policy, trained with one drone |
| `2026-11-15_swarm_sweep/` | 4 | the policy trained with many drones: MuJoCo with 1 to 10 drones, Gazebo with 1 to 4 drones, CPU and memory |
| `2026-11-21_safety_layer/` | 5 | the same policy with the safety layer and the recovery: the decision timing fix, Gazebo and MuJoCo compared drone by drone, the evolution strategy run, the final tests |
| `tables/` | | every table of the thesis as CSV, and all of them as text in `all_tables.txt` |
| `figures/` | | every figure of the thesis as PDF and PNG |

The two scripts:

- `make_tables.py` reads the folders above and writes `tables/all_tables.txt`
  (all tables as text) and `tables/*.csv`. It only needs the Python
  standard library: `python3 results/make_tables.py`.
- `make_figures.py` draws `figures/*.pdf` and `*.png` from the same
  data. It needs matplotlib, which is in the RL environment:
  `source rl/env.sh`, then `python results/make_figures.py`.

The Gazebo batch folders from October on (for example
`2026-11-15_swarm_sweep/policy/`) have the same three files, written by
`px4/compute_metrics_3d.py`. The September folder has no `drones.csv`.

| File | Content |
|---|---|
| `metrics.csv` | one row per run: drones that arrived, collisions, distances, ratio, times, CPU and memory |
| `drones.csv` | one row per drone: outcome, distances, flight time, closest approach to an obstacle |
| `summary.txt` | the table per number of obstacles and drones, averaged over the seeds |
