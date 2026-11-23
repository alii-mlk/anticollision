"""Safety layer between the avoidance policy and the drone (stage 5).

The policy is a neural network, so nothing guarantees that a command it gives
is safe, and in the large MuJoCo evaluation about 4% of the flights still end
against a box. The safety layer checks every command before it is sent and
replaces it when it is not safe. It is plain geometry, and like the lidar and
the observation the same code runs in MuJoCo and in Gazebo.

How a command is checked

  1. Predict where the drone goes if it flies the command until the next
     decision and then brakes, and also if it flies the command for a longer
     look-ahead time and then brakes. The prediction uses a first-order model
     of how the drone follows a velocity command: its velocity moves towards
     the commanded one with a time constant `lag`. This is the point model of
     env.py. On the first Gazebo test run with three drones it predicted the
     position half a second ahead to within 19 cm with a lag of 0.4 s, where
     assuming an instant change of velocity was off by up to 44 cm. Braking
     means the command zero. Because the next decision in Gazebo can come a little
     early or late, the first path is checked with the hold 20% shorter and
     20% longer too.
  2. The command is safe if along all these paths the drone keeps at least
     `margin` from every box, the ground and the edges of the area, and at
     least `drone_margin` from every other flying drone. Other drones are
     assumed to keep their velocity. Distances are from the drone's sphere
     (radius 0.35 m, as in collision_monitor_3d.py), so a margin of 0 would
     mean touching.
  3. A safe command is sent unchanged. Otherwise the safe command whose
     velocity is closest to the policy's is sent. The commands tried are 266
     directions (24 headings at 11 elevations, plus straight up and down) at
     four speeds, the policy's own direction at those speeds, and hovering.
  4. If no command is safe, which only happens when the drone is already
     closer than the margin, the command that keeps it furthest from
     everything is sent, and among equally good ones the one closest to the
     policy's.

Why the drone does not touch a box, the ground or the edge of the area

Call the drone's state safe if braking from it keeps the margin. A drone
hovering at its start is in a safe state, since the scenario generator keeps
2.5 m free around every start. If the drone is in a safe state at a decision,
then hovering, which is braking now, passes the check, so a safe command is
always found. The command sent passed the check with the hold equal to the
time until the next decision, so up to the next decision the drone keeps the
margin, and the state it reaches there is safe again (braking from it is the
second half of a path that was checked). So by induction every state is safe
and the drone keeps the margin for the whole flight. This holds for the
model. The real drone does not follow the model exactly: if its position never
differs from the model's by more than e, it keeps at least margin - e, and it
never touches anything as long as e < margin. check_shield.py measures e on
MuJoCo and Gazebo flights. Boxes, the ground and the edges do not move. Other
drones do, and they can turn, so for them the argument does not hold and the
distance between drones stays a measured result.

The path is checked at points at most about 5 cm apart, so a check can be
off by up to about 2.5 cm.

Usage:
  shield = Shield(ShieldSettings.load("rl/shield_settings.json"),
                  decision_dt=policy.decision_dt,
                  speed_xy=policy.speed_xy, speed_z=policy.speed_z)
  action, what = shield.filter(action, pos, vel, boxes, others_pos, others_vel)
  # what: "unchanged", "changed", "hover" or "no_safe"
"""

import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402
from lidar import drone_boxes, make_directions  # noqa: E402

# The area the drones may use: leaving it ends a flight in env.py.
AREA_LIMIT = maze.scenario_gen.BOUNDS + maze.OUT_MARGIN   # |x| and |y| (m)
CEILING = maze.CEILING                                     # z (m)

SAMPLE_STEP = 0.04        # s between checked points while the command is held
BRAKE_POINTS = 10         # checked points along the braking part, evenly spaced
BRAKE_END = 5.0           # the end of braking is taken at 5 lags (for the other drones)
EARLY_LATE = 0.2          # the next decision may come this much earlier or later
SPEEDS = (1.0, 0.75, 0.5, 0.25)   # action lengths tried, 1 = full speed
DIRECTIONS = make_directions(24, tuple(range(-75, 76, 15)), poles=True)
CHUNK = 64                # candidates checked at a time
OUTCOMES = ("unchanged", "changed", "hover", "no_safe")


@dataclass
class ShieldSettings:
    """The four numbers of the safety layer. ES tunes them in phase W."""

    margin: float = 0.3         # m kept from boxes, the ground and the edges
    drone_margin: float = 0.4   # m kept from other drones
    lookahead: float = 0.5      # s the command is held in the longer check
    lag: float = 0.4            # s, how fast the drone follows a new velocity

    @classmethod
    def load(cls, path=None):
        """Defaults, or the values in a JSON file (missing keys keep the default)."""
        if path is None or str(path) == "default":
            return cls()
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown shield settings in {path}: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in data.items()})

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def describe(self):
        return (f"margin {self.margin:.2f} m, drone margin {self.drone_margin:.2f} m, "
                f"look-ahead {self.lookahead:.2f} s, lag {self.lag:.2f} s")


def lag_path(lag, holds, step=SAMPLE_STEP, brake_points=BRAKE_POINTS):
    """Where the checked points are, as coefficients of the first-order model.

    With velocity v0 at position p0, holding the command velocity c for h
    seconds and then braking gives, at every checked point n,

        position_n = p0 + g_n * v0 + b_n * c,   at time t_n

    While the command is held (time t):
        g = lag (1 - e),  b = t - lag (1 - e),  with e = exp(-t / lag)
    Braking after the hold h (fraction u of the braking distance, which is
    reached at time h - lag ln(1 - u)):
        g = lag (1 - e_h) + e_h lag u,  b = h - lag (1 - e_h) + (1 - e_h) lag u

    Returns the arrays t, g and b, one value per checked point, all holds
    together.
    """
    if lag <= 0:
        raise ValueError("the lag must be positive")
    t_all, g_all, b_all = [0.0], [0.0], [0.0]
    for h in holds:
        k = max(1, math.ceil(h / step - 1e-9))
        for t in np.linspace(0.0, h, k + 1)[1:]:
            e = math.exp(-t / lag)
            t_all.append(t)
            g_all.append(lag * (1.0 - e))
            b_all.append(t - lag * (1.0 - e))
        e_h = math.exp(-h / lag)
        for j in range(1, brake_points + 1):
            u = j / brake_points
            s = -lag * math.log(1.0 - u) if u < 1.0 else BRAKE_END * lag
            t_all.append(h + s)
            g_all.append(lag * (1.0 - e_h) + e_h * lag * u)
            b_all.append(h - lag * (1.0 - e_h) + (1.0 - e_h) * lag * u)
    return np.array(t_all), np.array(g_all), np.array(b_all)


class Shield:
    """Checks the policy's commands and replaces the unsafe ones."""

    def __init__(self, settings=None, decision_dt=maze.DECISION_EVERY / maze.CTRL_FREQ,
                 speed_xy=maze.FLOWN_SPEED_XY, speed_z=maze.FLOWN_SPEED_Z,
                 radius=maze.DRONE_RADIUS):
        self.settings = settings or ShieldSettings()
        s = self.settings
        self.decision_dt = float(decision_dt)
        self.radius = float(radius)
        # An action is a direction of length at most 1; this turns it into the
        # velocity the drone is asked to fly, as rl/policy.py does.
        self.scale = np.array([speed_xy, speed_xy, speed_z], dtype=np.float64)
        dt = self.decision_dt
        holds = sorted({round(dt * (1.0 - EARLY_LATE), 6), round(dt, 6),
                        round(dt * (1.0 + EARLY_LATE), 6),
                        round(max(s.lookahead, dt), 6)})
        self.holds = holds
        self.t, self.g, self.b = lag_path(s.lag, holds)

        grid = [d * speed for d in DIRECTIONS for speed in SPEEDS]
        self.grid = np.vstack([np.array(grid), np.zeros((1, 3))])
        self.counts = Counter()

    def describe(self):
        return (f"{self.settings.describe()}, holds "
                f"{', '.join(f'{h:.3f}' for h in self.holds)} s")

    # --- the check ------------------------------------------------------

    def paths(self, pos, vel, velocities):
        """Checked points for each commanded velocity: (commands, points, 3)."""
        base = np.asarray(pos, dtype=np.float64) + self.g[:, None] * np.asarray(vel, dtype=np.float64)
        return base[None] + self.b[None, :, None] * np.asarray(velocities)[:, None, :]

    def slack(self, pos, vel, velocities, boxes, others_pos, others_vel):
        """For each commanded velocity, the smallest spare distance along its paths.

        Spare distance is the clearance minus the margin it has to keep. A
        command is safe when this is >= 0.
        """
        P = self.paths(pos, vel, velocities)                    # (M, N, 3)
        r = self.radius
        clear = np.minimum.reduce([
            P[..., 2] - r,                                     # the ground
            CEILING - P[..., 2],                               # the ceiling
            AREA_LIMIT - np.abs(P[..., 0]),                    # the edges
            AREA_LIMIT - np.abs(P[..., 1]),
        ])
        if len(boxes):
            lo, hi = boxes[:, [0, 2, 4]], boxes[:, [1, 3, 5]]
            Pb = P[..., None, :]                               # (M, N, 1, 3)
            gap = np.maximum(np.maximum(lo - Pb, 0.0), Pb - hi)  # (M, N, B, 3)
            clear = np.minimum(clear, np.sqrt((gap ** 2).sum(axis=-1)).min(axis=-1) - r)
        spare = clear - self.settings.margin
        if len(others_pos):
            # The other drones keep their velocity: (N, J, 3).
            Q = others_pos[None, :, :] + self.t[:, None, None] * others_vel[None, :, :]
            apart = np.linalg.norm(P[:, :, None, :] - Q[None], axis=-1).min(axis=-1)
            spare = np.minimum(spare, apart - 2.0 * r - self.settings.drone_margin)
        return spare.min(axis=1)

    # --- choosing the command -------------------------------------------

    def filter(self, action, pos, vel, boxes=(), others_pos=None, others_vel=None,
               wrecks=None):
        """The command to send instead of `action`, and what was done.

        action: the policy's action (a direction, length at most 1).
        pos, vel: the drone's position and velocity in the world frame.
        boxes: (B, 6) obstacles as [x_min, x_max, y_min, y_max, z_min, z_max].
        others_pos, others_vel: the other drones that are flying.
        wrecks: positions of destroyed drones. In Gazebo they stay where they
            fell, so they are kept away from like boxes.
        """
        a0 = maze.clip_action(action)
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        vel = np.asarray(vel, dtype=np.float64).reshape(3)
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
        if wrecks is not None and len(wrecks):
            boxes = np.vstack([boxes, drone_boxes(wrecks, self.radius)])
        others_pos = np.asarray(others_pos if others_pos is not None else [],
                                dtype=np.float64).reshape(-1, 3)
        if others_vel is None:
            others_vel = np.zeros_like(others_pos)
        others_vel = np.asarray(others_vel, dtype=np.float64).reshape(-1, 3)
        if len(others_vel) != len(others_pos):
            raise ValueError(f"{len(others_pos)} other drones but {len(others_vel)} velocities")
        scene = (boxes, others_pos, others_vel)

        wanted = a0 * self.scale
        spare0 = float(self.slack(pos, vel, wanted[None], *scene)[0])
        if spare0 >= 0.0:
            return self._done(a0, "unchanged", spare0)

        # Every command tried, the policy's own first, closest velocity first.
        norm = float(np.linalg.norm(a0))
        own = [a0[None]]
        if norm > 1e-9:
            own.append((a0 / norm)[None] * np.array(SPEEDS)[:, None])
        actions = np.vstack(own + [self.grid])
        velocities = actions * self.scale
        order = np.argsort(np.linalg.norm(velocities - wanted, axis=1), kind="stable")

        spare = np.full(len(actions), -np.inf)
        for start in range(0, len(order), CHUNK):
            idx = order[start:start + CHUNK]
            spare[idx] = self.slack(pos, vel, velocities[idx], *scene)
            safe = idx[spare[idx] >= 0.0]
            if safe.size:
                choice = safe[0]
                what = "hover" if not actions[choice].any() else "changed"
                return self._done(actions[choice], what, float(spare[choice]))

        # Nothing keeps the margin: keep as far from everything as possible.
        best = spare.max()
        ties = order[spare[order] >= best - 1e-9]
        choice = ties[0]
        return self._done(actions[choice], "no_safe", float(best))

    def _done(self, action, what, spare):
        self.counts[what] += 1
        self.last_spare = spare
        return np.asarray(action, dtype=np.float64), what


def shield_scene(shield, scene, actions, counts=None, recoveries=None):
    """The shield applied to every flying drone of a swarm.Scene (MuJoCo).

    actions has one row per flying drone, in drone order, as
    swarm.fly_scenarios passes them. In MuJoCo a drone that has crashed or
    arrived leaves the scene, so the other drones are the ones still flying.
    counts, if given, collects what the shield did per drone:
    counts[(obstacles, scene seed, drone)][what] += 1, and
    counts[...]["recovery"] the decisions flown by the recovery.
    recoveries, if given (a dict, empty at the start), turns on the recovery of
    rl/recovery.py: a stuck drone climbs over the boxes. Its commands go
    through the shield like the policy's.
    """
    flying = np.flatnonzero(scene.present)
    out = np.array(actions, dtype=np.float64, copy=True).reshape(len(flying), 3)
    for row, i in enumerate(flying):
        key = (scene.n_obstacles, scene.scene_seed, int(i))
        others = flying[flying != i] if scene.interact else flying[:0]
        action = out[row]
        if recoveries is not None:
            recovery = recoveries.get(key)
            if recovery is None:
                from recovery import Recovery
                recovery = recoveries[key] = Recovery(scene.target[i], scene.boxes,
                                                      shield.settings.margin, shield.radius)
            action, mode = recovery.action(scene.decisions[i] * shield.decision_dt,
                                           scene.pos[i], action)
            if counts is not None and mode == "recovery":
                counts[key]["recovery"] += 1
        out[row], what = shield.filter(action, scene.pos[i], scene.vel[i], scene.boxes,
                                       scene.pos[others], scene.vel[others])
        if counts is not None:
            counts[key][what] += 1
    return out
