"""Getting a stuck drone out: climb over the boxes (stage 5).

The policy only reacts to what is around it, and now and then it gets lost
behind a box, going back and forth until its time runs out. On 120 mazes of
the tuning block (seeds from 400000, 2, 4 and 8 drones) nine drones with the
safety layer timed out, and for five of them the layer had not changed a
single command: they were the policy's own dead ends, not the layer's. The
layer adds a few of its own, where a gap is narrower than it allows (two
drone radii plus two margins, 1.30 m with the defaults); in Gazebo one drone
met a 1.18 m gap in three mazes.

A drone that has not come PROGRESS metres closer to its target for
STUCK_AFTER seconds goes over the obstacles instead, the way the recoveries of
Nav2's behaviour tree take over in stage 1:

  1. climb straight up to the cruise altitude, high enough to pass above every
     box near the straight line to the target with room to spare. Boxes stand
     on the ground, so there is never anything above a drone.
  2. fly level towards the point above the target
  3. within NEAR_TARGET of it, fly straight at the target. The scenario
     generator keeps 2.5 m around every target free of boxes, so the way down
     is clear.

The safety layer still checks every one of these commands, so the recovery
cannot fly a drone into a box or another drone. If the drone gets stuck again
during the recovery (another drone in the way), the cruise altitude goes up by
a metre, up to just below the ceiling. The boxes are at most 10 m high and the
ceiling is 12 m, so there is always an altitude that clears them.

Usage, per drone and mission:
  recovery = Recovery(target, boxes, margin=shield.settings.margin)
  action, mode = recovery.action(t, pos, policy_action)   # mode: policy or recovery
"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402

STUCK_AFTER = 8.0      # s without coming PROGRESS m closer to the target
PROGRESS = 0.5         # m
CLEARANCE = 0.3        # m above the box tops, on top of the drone's radius and the margin
NEAR_TARGET = 1.0      # m, horizontally, from the point above the target
LEVEL_GAIN = 0.5       # vertical command per metre off the cruise altitude, while flying level


def footprint_gap(p, q, box):
    """Horizontal distance between the segment p-q and a box's footprint.

    Checked at points 10 cm apart, which is plenty next to CLEARANCE.
    """
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    steps = max(2, int(math.ceil(float(np.linalg.norm(q - p)) / 0.1)) + 1)
    points = p[None, :] + np.linspace(0.0, 1.0, steps)[:, None] * (q - p)[None, :]
    lo, hi = np.array([box[0], box[2]]), np.array([box[1], box[3]])
    gap = np.maximum(np.maximum(lo - points, 0.0), points - hi)
    return float(np.sqrt((gap ** 2).sum(axis=1)).min())


class Recovery:
    """Watches one drone's progress and flies it over the boxes when it is stuck."""

    def __init__(self, target, boxes, margin, radius=maze.DRONE_RADIUS, ceiling=maze.CEILING):
        self.target = np.asarray(target, dtype=np.float64).reshape(3)
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
        self.keep = radius + margin + CLEARANCE      # from the box tops to the drone's centre
        self.highest = ceiling - margin - 0.2
        self.mark = None          # distance to the target when progress was last made
        self.mark_time = None
        self.cruise = None        # cruise altitude; None while the policy flies
        self.started = 0          # how often the recovery started or raised its altitude

    def cruise_altitude(self, pos):
        """Lowest altitude that clears every box near the straight line to the target."""
        tops = [b[5] for b in self.boxes
                if footprint_gap(pos[:2], self.target[:2], b) < self.keep]
        z = max(tops, default=0.0) + self.keep
        return min(max(z, pos[2], self.target[2]), self.highest)

    def action(self, t, pos, policy_action):
        """The action to fly at time t (seconds of this mission), and who chose it."""
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        to_target = self.target - pos
        dist = float(np.linalg.norm(to_target))
        level = to_target[:2]
        across = float(np.linalg.norm(level))
        climbing = (self.cruise is not None and across > NEAR_TARGET
                    and pos[2] < self.cruise - 0.2)

        # Progress. Climbing takes the drone away from a lower target on
        # purpose, so the clock only runs while it is not climbing.
        if self.mark is None or climbing or dist < self.mark - PROGRESS:
            self.mark, self.mark_time = dist, t
        if t - self.mark_time > STUCK_AFTER:
            self.cruise = (self.cruise_altitude(pos) if self.cruise is None
                           else min(self.cruise + 1.0, self.highest))
            self.started += 1
            self.mark, self.mark_time = dist, t
            climbing = across > NEAR_TARGET and pos[2] < self.cruise - 0.2

        if self.cruise is None:
            return np.asarray(policy_action, dtype=np.float64).reshape(3), "policy"
        if across <= NEAR_TARGET:
            action = to_target / max(dist, 1e-9)
        elif climbing:
            action = np.array([0.0, 0.0, 1.0])
        else:
            up = float(np.clip(LEVEL_GAIN * (self.cruise - pos[2]), -0.5, 0.5))
            action = np.array([level[0] / across, level[1] / across, up])
            action /= np.linalg.norm(action)
        return action, "recovery"
