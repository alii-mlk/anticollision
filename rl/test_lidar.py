#!/usr/bin/env python3
"""Tests for rl/lidar.py.

Three kinds of check:

  hand-worked cases   distances worked out on paper for simple scenes, with
                      the reasoning in a comment next to each one
  random scenes       a geometric check that does not reuse the lidar's own
                      algorithm: every reported hit point must lie on a surface,
                      the point just before it must be in free space, and a
                      ray that reports no hit must pass through free space all
                      the way to the maximum range
  a real scenario     a scenario.yaml from the PX4 stage, if one is on disk

Runs on its own or under pytest. The report also goes to rl/logs/test_lidar.txt.

Usage:
  python rl/test_lidar.py
"""

import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lidar import (Lidar, boxes_from_scenario, cast, drone_boxes,  # noqa: E402
                   make_directions, starts_from_scenario)

LOG = HERE / "logs" / "test_lidar.txt"
R = 10.0                      # max range used in the hand-worked cases
S2 = math.sqrt(2.0)

LINES = []


def say(text):
    print(text, flush=True)
    LINES.append(text)


def ray(origin, direction, boxes, ground=True, max_range=R):
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return float(cast(origin, d[None, :], boxes, max_range, ground)[0])


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"got {a!r}, expected {b!r}"


# A 1 x 2 x 2 box: x from 3 to 4, y from -1 to 1, z from 0 to 2.
WALL = [[3, 4, -1, 1, 0, 2]]


# --- hand-worked cases ---------------------------------------------------

def test_straight_at_a_face():
    # From x = 0 towards +x, the near face is at x = 3.
    close(ray((0, 0, 1), (1, 0, 0), WALL), 3.0)


def test_miss_sideways():
    # Towards +y the ray never enters x in [3, 4]. Horizontal, so no ground.
    close(ray((0, 0, 1), (0, 1, 0), WALL), R)


def test_pointing_away():
    # The box is behind the ray.
    close(ray((0, 0, 1), (-1, 0, 0), WALL), R)


def test_grazing_an_edge_counts():
    # At y = 1 the ray runs exactly along the box's y = 1 face. Touching
    # counts as a hit, as in collision_monitor_3d.py, so it still reports 3.
    close(ray((0, 1, 1), (1, 0, 0), WALL), 3.0)


def test_just_past_the_edge_misses():
    # A hair outside y = 1 it misses.
    close(ray((0, 1.0001, 1), (1, 0, 0), WALL), R)


def test_inside_a_box_reads_zero():
    # (3.5, 0, 1) is inside the wall: 0 in every direction.
    out = cast((3.5, 0, 1), make_directions(), WALL, R)
    assert np.all(out == 0.0), out


def test_diagonal_onto_a_corner():
    # Box x in [3, 4], y in [3, 4]. Along (1, 1, 0) the ray reaches x = 3 and
    # y = 3 at the same moment, after 3 * sqrt(2).
    close(ray((0, 0, 1), (1, 1, 0), [[3, 4, 3, 4, 0, 2]]), 3 * S2)


def test_flies_over_a_short_box():
    # At z = 3 a box 2 m tall is underneath the ray, so no hit...
    close(ray((0, 0, 3), (1, 0, 0), [[3, 4, -1, 1, 0, 2]]), R)
    # ...while a box 4 m tall is hit at its face, 3 m away.
    close(ray((0, 0, 3), (1, 0, 0), [[3, 4, -1, 1, 0, 4]]), 3.0)


def test_downward_ray_onto_a_box_top():
    # From (0, 0, 5) along (1, 0, -1): after t the point is
    # (t / sqrt 2, 0, 5 - t / sqrt 2). It meets the top face z = 2 when
    # t / sqrt 2 = 3, at x = 3, inside x in [2, 4]. So t = 3 * sqrt(2).
    # The ground would only come at t = 5 * sqrt(2), further away.
    close(ray((0, 0, 5), (1, 0, -1), [[2, 4, -1, 1, 0, 2]]), 3 * S2)


def test_ground_straight_down():
    close(ray((0, 0, 2), (0, 0, -1), []), 2.0)


def test_ground_at_45_degrees():
    # From 2 m up at 45 degrees down, the ground is 2 * sqrt(2) away.
    close(ray((0, 0, 2), (1, 0, -1), []), 2 * S2)


def test_ground_can_be_switched_off():
    close(ray((0, 0, 2), (0, 0, -1), [], ground=False), R)


def test_drone_on_the_ground_sees_ground_only_below():
    # At z = 0 the downward ray reads 0, but a horizontal ray lies in the
    # ground plane and must not.
    close(ray((0, 0, 0), (0, 0, -1), []), 0.0)
    close(ray((0, 0, 0), (1, 0, 0), []), R)


def test_beyond_max_range_is_capped():
    close(ray((0, 0, 1), (1, 0, 0), [[15, 16, -1, 1, 0, 2]]), R)


def test_nearest_of_two_boxes_wins():
    boxes = [[6, 7, -1, 1, 0, 2], [3, 4, -1, 1, 0, 2]]
    close(ray((0, 0, 1), (1, 0, 0), boxes), 3.0)


def test_other_drone_is_seen():
    # A drone at (5, 0, 1) with radius 0.35 is a cube from x = 4.65 to 5.35.
    boxes = drone_boxes([(5, 0, 1)], 0.35)
    close(ray((0, 0, 1), (1, 0, 0), boxes), 4.65)


def test_no_obstacles_at_all():
    out = cast((0, 0, 3), make_directions(), [], R, ground=False)
    assert np.all(out == R), out


# --- configuration -------------------------------------------------------

def test_directions_layout():
    d = make_directions()
    assert d.shape == (82, 3), d.shape            # 16 x 5 rings + 2 poles
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.allclose(d[-2], (0, 0, 1)) and np.allclose(d[-1], (0, 0, -1))


def test_observation_is_scaled_and_float32():
    lidar = Lidar()
    obs = lidar.observe((0, 0, 3), WALL)
    assert obs.dtype == np.float32 and obs.shape == (lidar.n_rays,)
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_config_round_trip():
    lidar = Lidar(n_azimuth=8, max_range=7.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lidar.json"
        lidar.save(path)
        again = Lidar.load(path)
    assert again == lidar and np.allclose(again.directions, lidar.directions)
    assert Lidar() != lidar


# --- random scenes, checked geometrically ----------------------------------

def distance_to_boxes(points, boxes):
    """Distance from each point to the nearest box, 0 inside.

    Closest-point distance, the same formula as collision_monitor_3d.py. It is
    a different method from the lidar's slab intersection, which is what makes
    it a fair check on it.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 1, 3)
    b = np.asarray(boxes, dtype=float).reshape(1, -1, 6)
    lo, hi = b[..., [0, 2, 4]], b[..., [1, 3, 5]]
    gap = np.maximum(np.maximum(lo - p, 0.0), p - hi)
    return np.sqrt((gap ** 2).sum(axis=2)).min(axis=1)


def in_free_space(points, boxes):
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    return (p[:, 2] > 0.0) & (distance_to_boxes(p, boxes) > 0.0)


def random_scene(rng, n_boxes=12, bounds=12.0):
    boxes = []
    for _ in range(n_boxes):
        w, d = rng.uniform(1.0, 3.0, size=2)
        cx, cy = rng.uniform(-bounds, bounds, size=2)
        h = rng.uniform(3.0, 10.0)
        boxes.append([cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2, 0.0, h])
    return np.array(boxes)


def test_random_scenes_geometrically():
    rng = np.random.default_rng(0)
    lidar = Lidar()
    eps, step = 1e-6, 0.05
    hits = misses = 0
    for _ in range(200):
        boxes = random_scene(rng)
        # An origin in free space, at a flying altitude.
        for _ in range(100):
            origin = np.array([*rng.uniform(-12, 12, size=2), rng.uniform(0.5, 9.0)])
            if in_free_space(origin, boxes)[0]:
                break
        else:
            continue
        samples = np.arange(step, lidar.max_range, step)
        for d, t in zip(lidar.directions, lidar.ranges(origin, boxes)):
            if t < lidar.max_range:
                hits += 1
                p = origin + t * d
                on_box = distance_to_boxes(p, boxes)[0] < 1e-7
                on_ground = abs(p[2]) < 1e-7
                assert on_box or on_ground, f"hit at {p} is on no surface"
                before = origin + (t - eps) * d
                assert in_free_space(before, boxes)[0], \
                    f"the ray passed through something before t = {t}"
            else:
                misses += 1
                points = origin + samples[:, None] * d
                free = in_free_space(points, boxes)
                assert free.all(), (f"reported no hit, but the point at "
                                    f"{samples[~free][0]:.2f} m is inside something")
    say(f"    {hits} hits and {misses} misses checked across 200 random scenes")
    assert hits > 1000 and misses > 1000, (hits, misses)


# --- a real scenario -----------------------------------------------------

def find_scenario():
    root = HERE.parent / "px4"
    for pattern in ("scenarios3d/*/scenario.yaml",
                    "runs3d/*/*/scenario/scenario.yaml"):
        found = sorted(root.glob(pattern))
        if found:
            return found[0]
    return None


def test_real_scenario():
    path = find_scenario()
    if path is None:
        say("    (no scenario.yaml on disk, skipped)")
        return
    boxes = boxes_from_scenario(path)
    starts = starts_from_scenario(path)
    assert len(boxes) and len(starts), path
    lidar = Lidar()
    for s in starts:
        # Take-off altitude, where the policy starts steering.
        out = lidar.ranges((s[0], s[1], 3.0), boxes)
        assert np.all(np.isfinite(out))
        assert np.all((out >= 0) & (out <= lidar.max_range))
    say(f"    {path.relative_to(HERE.parent)}: {len(boxes)} boxes, "
        f"{len(starts)} drones")


def test_speed():
    rng = np.random.default_rng(1)
    lidar = Lidar()
    boxes = np.vstack([random_scene(rng, 20), drone_boxes(rng.uniform(-5, 5, (2, 3)), 0.35)])
    n = 2000
    start = time.perf_counter()
    for _ in range(n):
        lidar.ranges((0.0, 0.0, 3.0), boxes)
    per_scan = (time.perf_counter() - start) / n
    say(f"    {lidar.n_rays} rays against {len(boxes)} boxes: "
        f"{per_scan * 1e6:.0f} us per scan")
    # The environment steps in roughly 170 us; the lidar must not dominate it.
    assert per_scan < 2e-3, per_scan


# --- runner --------------------------------------------------------------

def main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = []

    say(f"lidar tests {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for name, fn in tests:
        try:
            fn()
            say(f"PASS  {name}")
        except Exception:
            failed.append(name)
            say(f"FAIL  {name}")
            for line in traceback.format_exc().strip().splitlines()[-4:]:
                say(f"    {line}")
    say("")
    say(f"{len(tests) - len(failed)}/{len(tests)} passed"
        + (f"; failed: {', '.join(failed)}" if failed else ""))

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(LINES) + "\n")
    print(f"report written to {LOG}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
