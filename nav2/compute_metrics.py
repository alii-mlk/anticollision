#!/usr/bin/env python3
"""Aggregate metrics from a run_batch.sh output directory.

Reads every run's artifacts (scenario/scenario.yaml, goal.txt, hits.yaml,
status.txt) and produces:

  <batch>/metrics.csv    one row per run
  <batch>/summary.txt    per-N aggregate table (also printed to stdout)
  <batch>/plots/*.png    success rate / time / ratio / hits vs N
                         (only if matplotlib is available)

Metrics per run:
  status          SUCCEEDED / ABORTED / TIMEOUT / STACK_FAIL / NAV2_FAIL
  nav_time_s      last navigation_time in the feedback stream (sim seconds)
  path_len_m      integrated length of the flown trajectory (from feedback)
  euclid_m        straight-line start->goal distance (from scenario.yaml)
  ratio           path_len / euclid  -- the professor's overhead metric
  hits            contact episodes counted by hit_monitor
  min_clearance_m closest approach to any obstacle (negative = penetration)

Usage:  python3 compute_metrics.py runs/batch_20260718_101500
No dependencies beyond the standard library (matplotlib optional).
"""

import csv
import math
import re
import sys
from pathlib import Path


def parse_scenario(path):
    text = path.read_text()
    def grab(key):
        m = re.search(rf"{key}:\s*{{x:\s*([-\d.]+),\s*y:\s*([-\d.]+)", text)
        return (float(m.group(1)), float(m.group(2))) if m else None
    n = re.search(r"n_obstacles:\s*(\d+)", text)
    seed = re.search(r"seed:\s*(\d+)", text)
    return {
        "seed": int(seed.group(1)) if seed else None,
        "n_obstacles": int(n.group(1)) if n else None,
        "start": grab("start"),
        "goal": grab("goal"),
    }


def parse_goal_feedback(path):
    """Positions + last navigation_time from the action feedback stream."""
    if not path.exists():
        return [], None
    text = path.read_text(errors="replace")
    # First position match is the goal echo in the "Sending goal" block; skip it.
    pts = [(float(x), float(y)) for x, y in
           re.findall(r"position:\s+x: ([-\d.e+]+)\s+y: ([-\d.e+]+)", text)][1:]
    times = re.findall(r"navigation_time:\s+sec: (\d+)\s+nanosec: (\d+)", text)
    nav_time = None
    if times:
        s, ns = times[-1]
        nav_time = int(s) + int(ns) / 1e9
    return pts, nav_time


def parse_hits(path):
    if not path.exists():
        return None, None
    text = path.read_text()
    hits = re.search(r"hits:\s*(\d+)", text)
    clearance = re.search(r"min_clearance:\s*([-\d.]+|null)", text)
    c = clearance.group(1) if clearance else None
    return (int(hits.group(1)) if hits else None,
            None if c in (None, "null") else float(c))


def path_length(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def collect(batch_dir):
    rows = []
    for run_dir in sorted(batch_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("n"):
            continue
        scen_file = run_dir / "scenario" / "scenario.yaml"
        if not scen_file.exists():
            continue
        scen = parse_scenario(scen_file)
        status_file = run_dir / "status.txt"
        status = status_file.read_text().strip() if status_file.exists() else "MISSING"
        pts, nav_time = parse_goal_feedback(run_dir / "goal.txt")
        hits, min_clear = parse_hits(run_dir / "hits.yaml")

        euclid = (math.dist(scen["start"], scen["goal"])
                  if scen["start"] and scen["goal"] else None)
        plen = path_length(pts) if len(pts) > 1 else None
        ratio = (plen / euclid) if (plen and euclid) else None

        rows.append({
            "run": run_dir.name,
            "n_obstacles": scen["n_obstacles"],
            "seed": scen["seed"],
            "status": status,
            "nav_time_s": round(nav_time, 1) if nav_time else None,
            "path_len_m": round(plen, 2) if plen else None,
            "euclid_m": round(euclid, 2) if euclid else None,
            "ratio": round(ratio, 3) if ratio else None,
            "hits": hits,
            "min_clearance_m": min_clear,
        })
    return rows


def summarize(rows):
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_obstacles"], []).append(r)

    lines = []
    header = (f"{'N':>3} {'runs':>5} {'success':>8} {'mean_t(s)':>10} "
              f"{'mean_ratio':>11} {'hits':>5} {'min_clear(m)':>13}")
    lines.append(header)
    lines.append("-" * len(header))
    for n in sorted(k for k in by_n if k is not None):
        rs = by_n[n]
        ok = [r for r in rs if r["status"] == "SUCCEEDED"]
        rate = f"{len(ok)}/{len(rs)}"
        mean = lambda vals: (sum(vals) / len(vals)) if vals else None
        mt = mean([r["nav_time_s"] for r in ok if r["nav_time_s"]])
        mr = mean([r["ratio"] for r in ok if r["ratio"]])
        hits = sum(r["hits"] or 0 for r in rs)
        clears = [r["min_clearance_m"] for r in rs if r["min_clearance_m"] is not None]
        worst = min(clears) if clears else None
        lines.append(f"{n:>3} {len(rs):>5} {rate:>8} "
                     f"{(f'{mt:.1f}' if mt is not None else '-'):>10} "
                     f"{(f'{mr:.3f}' if mr is not None else '-'):>11} "
                     f"{hits:>5} "
                     f"{(f'{worst:.2f}' if worst is not None else '-'):>13}")
    return "\n".join(lines)


def try_plots(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available -- skipping plots)")
        return

    by_n = {}
    for r in rows:
        if r["n_obstacles"] is not None:
            by_n.setdefault(r["n_obstacles"], []).append(r)
    ns = sorted(by_n)
    if not ns:
        return

    out_dir.mkdir(exist_ok=True)

    def per_n(fn):
        return [fn(by_n[n]) for n in ns]

    plots = [
        ("success_rate", "Success rate",
         per_n(lambda rs: sum(r["status"] == "SUCCEEDED" for r in rs) / len(rs))),
        ("mean_nav_time", "Mean navigation time (s, successes)",
         per_n(lambda rs: _mean([r["nav_time_s"] for r in rs
                                 if r["status"] == "SUCCEEDED" and r["nav_time_s"]]))),
        ("mean_ratio", "Mean path/Euclidean ratio (successes)",
         per_n(lambda rs: _mean([r["ratio"] for r in rs
                                 if r["status"] == "SUCCEEDED" and r["ratio"]]))),
        ("total_hits", "Total hits",
         per_n(lambda rs: sum(r["hits"] or 0 for r in rs))),
    ]
    for name, title, values in plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, values, marker="o")
        ax.set_xlabel("Number of obstacles")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if name == "success_rate":
            ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=120)
        plt.close(fig)
    print(f"plots written to {out_dir}/")


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <batch_dir>")
    batch_dir = Path(sys.argv[1])
    if not batch_dir.is_dir():
        sys.exit(f"not a directory: {batch_dir}")

    rows = collect(batch_dir)
    if not rows:
        sys.exit("no runs found (expected subdirs like n4_s2/ with scenario/)")

    csv_path = batch_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    (batch_dir / "summary.txt").write_text(summary + "\n")

    print(f"{len(rows)} runs -> {csv_path}")
    print()
    print(summary)
    print()
    try_plots(rows, batch_dir / "plots")


if __name__ == "__main__":
    main()
