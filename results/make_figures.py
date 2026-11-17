#!/usr/bin/env python3
"""The figures of the thesis, drawn from the curated result folders.

Needs matplotlib, which the project's Python environment has (it comes with
Stable-Baselines3). In the VM, from the repository root:

  source rl/env.sh
  python results/make_figures.py

Writes every figure as PDF (for the thesis) and PNG (to look at) into
results/thesis/figures/. The data is read with the functions of
make_tables.py, and every figure has a table with the same numbers in
results/thesis/tables.txt:

  01 learning curve                  table 1
  02 Gazebo arrivals by obstacles    tables 2 and 3
  03 MuJoCo, 1 to 10 drones          table 13
  04 path and time by obstacles      table 7
  05 clearance to obstacles          table 9
  06 processor time                  table 11
  07 memory                          table 12
  08 one scenario seen from above    (the paths are in examples/)

The three controllers have the same colour and marker in every figure. The
colours were checked for colour-blind readers, and the markers tell the
controllers apart in grey print.
"""

import csv
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_tables as data  # noqa: E402

OUT = data.OUT / "figures"
EXAMPLE = data.NOVEMBER / "examples" / "n12_k3_s2"
EXAMPLE_RUN = "n12_k3_s2"

CONTROLLERS = ("straight", "trained alone", "final policy")
COLOR = {"straight": "#2a78d6", "trained alone": "#eb6834",
         "final policy": "#1baf7a"}
MARKER = {"straight": "o", "trained alone": "s", "final policy": "^"}
# Small sideways shifts so the points of the three controllers do not hide
# each other, as a fraction of the distance between two x values.
SHIFT = {"straight": -0.12, "trained alone": 0.0, "final policy": 0.12}
PART_COLOR = dict(zip(data.MEMORY_PARTS, ("#2a78d6", "#eb6834", "#1baf7a",
                                          "#eda100", "#e87ba4", "#008300")))
PART_NAME = {"gazebo": "Gazebo", "px4": "PX4 autopilots", "mavros": "MAVROS",
             "helpers": "helpers", "mission": "mission nodes",
             "monitor": "collision monitors"}
DRONE_COLOR = ("#2a78d6", "#eb6834", "#1baf7a")

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OBSTACLE = "#e1e0d9"

WIDTH = 6.3   # inches: the text width of an A4 page with 2.5 cm margins


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.titlecolor": INK,
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.facecolor": "white",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "figure.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {OUT / name}.pdf / .png")


def controller_legend(fig, arms=CONTROLLERS, y=1.0):
    handles = [Line2D([], [], color=COLOR[a], marker=MARKER[a], label=a)
               for a in arms]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, y), handlelength=2.2)


def percent(successes, n):
    """Share in percent and its 95% interval, as (value, low, high)."""
    low, high = data.wilson(successes, n)
    return 100 * successes / n, 100 * low, 100 * high


def error_bars(values):
    """[[distance down], [distance up]] for errorbar from (value, low, high)."""
    return [[v - lo for v, lo, _ in values], [hi - v for v, _, hi in values]]


# ---------------------------------------------------------------- figures

def fig_learning_curve():
    checkpoints, final, straight = data.training_curve()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), sharey=True,
                             squeeze=False)
    color = COLOR["final policy"]
    for ax, (name, title) in zip(axes[0], data.TRAINING_SETS):
        points = checkpoints[name]
        x = [d / 1e6 for d, _, _ in points]
        values = [percent(r, n) for _, r, n in points]
        ax.fill_between(x, [lo for _, lo, _ in values],
                        [hi for _, _, hi in values], color=color, alpha=0.15,
                        linewidth=0)
        ax.plot(x, [v for v, _, _ in values], color=color,
                marker=MARKER["final policy"])
        d, r, n = final[name]
        ax.plot([d / 1e6], [100 * r / n], linestyle="none", marker="o",
                markersize=7, markerfacecolor="none", markeredgecolor=INK,
                markeredgewidth=1)
        s = 100 * straight[name][0] / straight[name][1]
        ax.axhline(s, color=MUTED, linewidth=1)
        ax.text(0.1, s - 2.5, "flying straight", ha="left", va="top",
                color=INK2, fontsize=7.5)
        ax.set_title(f"{title}, {n} flights")
        ax.set_xlabel("training decisions (millions)")
        ax.set_xlim(0, 4.3)
    axes[0][0].set_ylim(0, 102)
    axes[0][0].set_ylabel("arrived (%)")
    handles = [Line2D([], [], color=color, marker=MARKER["final policy"],
                      label="checkpoints, with 95% interval"),
               Line2D([], [], linestyle="none", marker="o", markersize=7,
                      markerfacecolor="none", markeredgecolor=INK,
                      label="final.zip, the policy used")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, "01_learning_curve")


def fig_gazebo_arrivals(drones):
    obstacles = (4, 8, 12, 16, 20)
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.4), sharey=True,
                             squeeze=False)
    for ax, k in zip(axes[0], (1, 2, 3, 4)):
        for arm in CONTROLLERS:
            points = []
            for n in obstacles:
                ds = [d for d in drones[arm].values()
                      if d["k"] == k and d["n"] == n]
                if ds:
                    points.append((n, percent(sum(d["arrived"] for d in ds),
                                              len(ds))))
            if not points:
                continue
            values = [v for _, v in points]
            ax.errorbar([n + 4 * SHIFT[arm] for n, _ in points],
                        [v for v, _, _ in values], yerr=error_bars(values),
                        color=COLOR[arm], marker=MARKER[arm], elinewidth=0.8,
                        capsize=0)
        ax.set_title(f"{k} drone{'s' if k > 1 else ''}, "
                     f"{5 * k} per point", fontsize=8.5)
        ax.set_xticks(obstacles)
        ax.set_xlabel("obstacles")
    axes[0][0].set_ylim(0, 105)
    axes[0][0].set_ylabel("drones that arrived (%)")
    controller_legend(fig)
    fig.tight_layout()
    save(fig, "02_gazebo_arrivals")


def fig_mujoco_scaling():
    flights = data.mujoco_flights()
    counts = (1, 2, 3, 4, 6, 8, 10)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), squeeze=False)
    jobs = ((axes[0][0], lambda f: f["outcome"] == "reached", "arrived"),
            (axes[0][1], lambda f: f["outcome"] == "hit_drone",
             "destroyed by another drone"))
    for ax, hit, title in jobs:
        for arm in CONTROLLERS:
            values = []
            for k in counts:
                fs = [f for f in flights[arm] if f["k"] == k]
                values.append(percent(sum(hit(f) for f in fs), len(fs)))
            ax.fill_between(counts, [lo for _, lo, _ in values],
                            [hi for _, _, hi in values], color=COLOR[arm],
                            alpha=0.15, linewidth=0)
            # Not clipped, so the points that lie on the axis at 0 are
            # drawn whole.
            ax.plot(counts, [v for v, _, _ in values], color=COLOR[arm],
                    marker=MARKER[arm], clip_on=False, zorder=3)
        ax.set_title(title)
        ax.set_xticks(counts)
        ax.set_xlabel("drones in the maze")
        ax.set_ylabel("flights (%)")
    axes[0][0].set_ylim(0, 102)
    axes[0][1].set_ylim(0, 10)
    controller_legend(fig)
    fig.tight_layout()
    save(fig, "03_mujoco_scaling")


def fig_path_and_time(drones):
    obstacles = (4, 8, 12, 16, 20)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), squeeze=False)
    jobs = ((axes[0][0], lambda d: d["travelled"] / d["euclidean"],
             "distance flown / straight line"),
            (axes[0][1], lambda d: d["time"], "flight time (s)"))
    for ax, measure, label in jobs:
        for arm in CONTROLLERS:
            means, stds = [], []
            for n in obstacles:
                m, s = data.mean_std(measure(d) for d in drones[arm].values()
                                     if d["arrived"] and d["n"] == n
                                     and d["k"] in (1, 2, 3))
                means.append(m)
                stds.append(s)
            ax.errorbar([n + 4 * SHIFT[arm] for n in obstacles], means,
                        yerr=stds, color=COLOR[arm], marker=MARKER[arm],
                        elinewidth=0.8, capsize=0)
        ax.set_xticks(obstacles)
        ax.set_xlabel("obstacles")
        ax.set_ylabel(label)
    axes[0][0].set_title("path, mean and standard deviation")
    axes[0][1].set_title("time, mean and standard deviation")
    axes[0][1].set_ylim(bottom=0)
    controller_legend(fig)
    fig.tight_layout()
    save(fig, "04_path_and_time")


def fig_clearance(drones):
    keys = data.arrived_with_all(drones)
    fig, axes = plt.subplots(1, 1, figsize=(WIDTH * 0.62, 2.6), squeeze=False)
    ax = axes[0][0]
    for arm in CONTROLLERS:
        values = sorted(drones[arm][key]["min_clearance"] for key in keys)
        n = len(values)
        ax.step([0.0] + values, [0.0] + [100 * (i + 1) / n for i in range(n)],
                where="post", color=COLOR[arm])
        median = statistics.median(values)
        ax.plot([median], [50], linestyle="none", marker=MARKER[arm],
                color=COLOR[arm])
    ax.axvline(0.5, color=MUTED, linewidth=1)
    ax.text(0.53, 97, "0.5 m", color=INK2, fontsize=7.5, va="top")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 102)
    ax.set_xlabel("closest approach to an obstacle (m)")
    ax.set_ylabel("drones, at most this close (%)")
    ax.set_title(f"the {len(keys)} drones that arrived with all three, "
                 "markers at the median", fontsize=8.5)
    controller_legend(fig)
    fig.tight_layout()
    save(fig, "05_clearance")


def fig_cpu(runs, drones):
    counts = (1, 2, 3, 4)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), squeeze=False)
    for arm in CONTROLLERS:
        sim, ctrl = [], []
        for k in counts:
            rs = [r for r in runs[arm].values() if int(r["k_drones"]) == k]
            ds = [d for d in drones[arm].values() if d["k"] == k
                  and d["controller_cpu_time"] and d["time"] > 0]
            if not rs:
                continue
            sim.append((k, data.mean_std(float(r["cpu_time"])
                                         / float(r["wall_time_measured"])
                                         for r in rs)))
            ctrl.append((k, data.mean_std(100 * float(d["controller_cpu_time"])
                                          / d["time"] for d in ds)))
        for ax, points in ((axes[0][0], sim), (axes[0][1], ctrl)):
            ax.errorbar([k + SHIFT[arm] for k, _ in points],
                        [m for _, (m, _) in points],
                        yerr=[s for _, (_, s) in points], color=COLOR[arm],
                        marker=MARKER[arm], elinewidth=0.8, capsize=0)
    axes[0][0].set_title("simulator, autopilots and MAVROS")
    axes[0][0].set_ylabel("processor cores busy")
    axes[0][1].set_title("mission node, per drone")
    axes[0][1].set_ylabel("one core (%)")
    for ax in axes[0]:
        ax.set_xticks(counts)
        ax.set_xlabel("drones")
        ax.set_ylim(bottom=0)
    controller_legend(fig)
    fig.tight_layout()
    save(fig, "06_cpu")


def fig_memory():
    counts = (1, 2, 3, 4)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True,
                             squeeze=False)
    for ax, (arm, folder) in zip(axes[0], data.MEMORY.items()):
        runs = [r for r in data.read(folder / "metrics.csv")
                if r["status"] == "OK"]
        bottoms = [0.0] * len(counts)
        for part in data.MEMORY_PARTS:
            heights = [statistics.fmean(float(r[f"memory_pss_{part}_mb"])
                                        for r in runs if int(r["k_drones"]) == k)
                       for k in counts]
            # A thin white edge leaves a gap between the parts of a bar.
            ax.bar(counts, heights, bottom=bottoms, width=0.6,
                   color=PART_COLOR[part], edgecolor="white", linewidth=0.8)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        for k, total in zip(counts, bottoms):
            ax.text(k, total + 40, f"{total:.0f}", ha="center", va="bottom",
                    color=INK2, fontsize=7.5)
        ax.set_title(arm)
        ax.set_xticks(counts)
        ax.set_xlabel("drones")
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("memory (MB)")
    axes[0][0].set_ylim(0, 4000)
    handles = [Patch(color=PART_COLOR[p], label=PART_NAME[p])
               for p in reversed(data.MEMORY_PARTS)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save(fig, "07_memory")


def read_scenario(path):
    text = path.read_text()
    number = r"([-\d.]+)"
    starts = [tuple(map(float, m)) for m in re.findall(
        rf"start: {{x: {number}, y: {number}, z: {number}}}", text)]
    targets = [tuple(map(float, m)) for m in re.findall(
        rf"target: {{x: {number}, y: {number}, z: {number}}}", text)]
    boxes = [tuple(map(float, m)) for m in re.findall(
        rf"x_min: {number}, x_max: {number}, y_min: {number}, "
        rf"y_max: {number}, z_min: {number}, z_max: {number}", text)]
    return starts, targets, boxes


def read_track(path):
    """(t, x, y, z) rows, without the part before the clock restarted."""
    with open(path) as f:
        rows = [tuple(map(float, r)) for r in list(csv.reader(f))[1:]]
    restart = max((i for i in range(1, len(rows)) if rows[i][0] < rows[i - 1][0]),
                  default=0)
    return rows[restart:]


def fig_example(drones):
    starts, targets, boxes = read_scenario(EXAMPLE / "scenario.yaml")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.2), squeeze=False,
                             gridspec_kw={"height_ratios": [3.0, 1.3]})
    longest = 0.0
    for col, (arm, prefix) in enumerate((("straight", "straight"),
                                         ("final policy", "policy"))):
        top, side = axes[0][col], axes[1][col]
        for x0, x1, y0, y1, _, z1 in boxes:
            top.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    facecolor=OBSTACLE, edgecolor=MUTED,
                                    linewidth=0.6))
            top.text((x0 + x1) / 2, (y0 + y1) / 2, f"{z1:.1f}", ha="center",
                     va="center", fontsize=6, color=INK2)
        arrived = 0
        for i in range(len(starts)):
            track = read_track(EXAMPLE / f"{prefix}_drone{i}.csv")
            color = DRONE_COLOR[i]
            top.plot([r[1] for r in track], [r[2] for r in track],
                     color=color, linewidth=1.3)
            top.plot([starts[i][0]], [starts[i][1]], marker="o",
                     color=color, linestyle="none")
            top.text(starts[i][0] + 0.5, starts[i][1] + 0.5, str(i + 1),
                     color=INK, fontsize=7.5)
            top.plot([targets[i][0]], [targets[i][1]], marker="*",
                     markersize=9, color=color, linestyle="none")
            side.plot([r[0] for r in track], [r[3] for r in track],
                      color=color, linewidth=1.3)
            longest = max(longest, track[-1][0])
            if drones[arm][(EXAMPLE_RUN, i)]["arrived"]:
                arrived += 1
            else:
                end = track[-1]
                top.plot([end[1]], [end[2]], marker="X", markersize=7,
                         color=INK, linestyle="none")
                side.plot([end[0]], [end[3]], marker="X", markersize=7,
                          color=INK, linestyle="none")
        top.set_title(f"{arm}: {arrived} of {len(starts)} arrived")
        top.set_aspect("equal")
        top.set_xlim(-12.5, 12.5)
        top.set_ylim(-12.5, 12.5)
        top.set_xlabel("x (m)")
        side.set_xlabel("time (s)")
    axes[0][0].set_ylabel("y (m)")
    axes[1][0].set_ylabel("altitude (m)")
    for side in axes[1]:
        side.set_xlim(0, longest + 1)
        side.set_ylim(0, 12)
    handles = ([Line2D([], [], color=DRONE_COLOR[i], label=f"drone {i + 1}")
                for i in range(len(starts))]
               + [Line2D([], [], linestyle="none", marker="o", color=MUTED,
                         label="start"),
                  Line2D([], [], linestyle="none", marker="*", markersize=9,
                         color=MUTED, label="target"),
                  Line2D([], [], linestyle="none", marker="X", markersize=7,
                         color=INK, label="destroyed"),
                  Patch(facecolor=OBSTACLE, edgecolor=MUTED,
                        label="obstacle, height in m")])
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, "08_example_flight")


def main():
    style()
    runs = {arm: data.gazebo_runs(arm) for arm in data.ARMS}
    drones = {arm: data.gazebo_drones(arm) for arm in data.ARMS}
    print(f"writing to {OUT}")
    fig_learning_curve()
    fig_gazebo_arrivals(drones)
    fig_mujoco_scaling()
    fig_path_and_time(drones)
    fig_clearance(drones)
    fig_cpu(runs, drones)
    fig_memory()
    fig_example(drones)


if __name__ == "__main__":
    main()
