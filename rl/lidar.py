"""3D lidar computed analytically from the scenario geometry.

The policy is trained in MuJoCo and then flown in Gazebo with PX4, so it has to
get the same kind of input in both. For that reason neither simulator's own
sensor is used. Instead every ray is intersected with the
obstacle boxes from scenario.yaml in plain numpy, and the same code runs during
training and during the Gazebo flight. It is the 3D version of what
nav2/virtual_lidar.py does in 2D, and it needs no GPU.

What a ray can hit:

  boxes         axis-aligned, as [x_min, x_max, y_min, y_max, z_min, z_max],
                the same order as scenario.yaml
  the ground    the plane z = 0, seen only by rays pointing downwards
  other drones  passed in as small boxes, see drone_boxes()

Rays are fixed in the world frame rather than rotated with the drone. Our
drones never change heading (the offboard setpoints hold it), so world and body
frames are the same, and fixing the rays to the world removes any chance of a
frame mismatch between the two simulators.

Touching counts as a hit: a ray that grazes an edge or a face reports it, which
matches collision_monitor_3d.py, where touching is a collision. A point inside a
box reads 0 in every direction.

Usage:
  lidar = Lidar()
  ranges = lidar.ranges(position, boxes)     # metres, capped at max_range
  obs = lidar.observe(position, boxes)       # the same, scaled to [0, 1]
"""

import json
import math
import re
from pathlib import Path

import numpy as np

DEFAULT_MAX_RANGE = 10.0
DEFAULT_AZIMUTHS = 16
DEFAULT_ELEVATIONS_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)

# Below this a direction component counts as zero, i.e. the ray is parallel to
# that pair of box faces.
PARALLEL_EPS = 1e-12

_BOX_PATTERN = re.compile(
    r"x_min:\s*([-\d.]+),\s*x_max:\s*([-\d.]+),\s*"
    r"y_min:\s*([-\d.]+),\s*y_max:\s*([-\d.]+),\s*"
    r"z_min:\s*([-\d.]+),\s*z_max:\s*([-\d.]+)")
_START_PATTERN = re.compile(
    r"start:\s*\{x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*z:\s*([-\d.]+)\}")


def make_directions(n_azimuth=DEFAULT_AZIMUTHS,
                    elevations_deg=DEFAULT_ELEVATIONS_DEG, poles=True):
    """Unit ray directions: n_azimuth headings at each elevation.

    With poles=True a ray straight up and one straight down are added, since
    the elevation rings never reach them.
    """
    dirs = []
    for elevation in elevations_deg:
        e = math.radians(elevation)
        for i in range(n_azimuth):
            a = 2.0 * math.pi * i / n_azimuth
            dirs.append((math.cos(e) * math.cos(a),
                         math.cos(e) * math.sin(a),
                         math.sin(e)))
    if poles:
        dirs.append((0.0, 0.0, 1.0))
        dirs.append((0.0, 0.0, -1.0))
    d = np.array(dirs, dtype=np.float64)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def cast(origin, directions, boxes, max_range=DEFAULT_MAX_RANGE, ground=True):
    """Distance along each ray to the nearest surface, capped at max_range.

    Slab method: for each box, the part of the ray inside each pair of
    parallel faces is an interval of distances, and the ray hits the box where
    the three intervals overlap. Everything is done for all rays and all boxes
    at once.
    """
    o = np.asarray(origin, dtype=np.float64).reshape(1, 1, 3)
    d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    n = d.shape[0]
    dist = np.full(n, float(max_range))

    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
    if b.shape[0]:
        lo = b[:, [0, 2, 4]][None, :, :]          # (1, M, 3)
        hi = b[:, [1, 3, 5]][None, :, :]
        dd = d[:, None, :]                         # (N, 1, 3)

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - o) / dd
            t2 = (hi - o) / dd
        t_lo = np.minimum(t1, t2)                  # (N, M, 3)
        t_hi = np.maximum(t1, t2)

        # A ray parallel to a pair of faces never crosses them. It is either
        # between them for its whole length or never. Dividing by zero would
        # give infinities, or NaN when the origin sits exactly on a face, so
        # this case is set explicitly instead.
        parallel = np.abs(dd) < PARALLEL_EPS
        between = (o >= lo) & (o <= hi)
        t_lo = np.where(parallel, np.where(between, -np.inf, np.inf), t_lo)
        t_hi = np.where(parallel, np.where(between, np.inf, -np.inf), t_hi)

        t_near = t_lo.max(axis=2)                  # (N, M)
        t_far = t_hi.min(axis=2)
        hit = (t_near <= t_far) & (t_far >= 0.0)
        # A negative t_near means the origin is inside the box: distance 0.
        t = np.where(hit, np.maximum(t_near, 0.0), np.inf)
        dist = np.minimum(dist, t.min(axis=1))

    if ground:
        oz = float(o[0, 0, 2])
        dz = d[:, 2]
        down = dz < -PARALLEL_EPS
        t_ground = np.full(n, np.inf)
        t_ground[down] = max(oz, 0.0) / -dz[down]
        dist = np.minimum(dist, t_ground)

    return np.minimum(dist, float(max_range))


def drone_boxes(positions, radius):
    """Other drones as cubes of side 2 * radius, so the lidar sees them too."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if not p.size:
        return np.zeros((0, 6))
    r = float(radius)
    return np.column_stack([p[:, 0] - r, p[:, 0] + r,
                            p[:, 1] - r, p[:, 1] + r,
                            p[:, 2] - r, p[:, 2] + r])


def boxes_from_scenario(path):
    """Obstacle boxes from a scenario.yaml written by gen_scenario_3d.py."""
    text = Path(path).read_text()
    found = [[float(v) for v in m.groups()] for m in _BOX_PATTERN.finditer(text)]
    return np.array(found, dtype=np.float64).reshape(-1, 6)


def starts_from_scenario(path):
    """Drone start points from a scenario.yaml."""
    text = Path(path).read_text()
    found = [[float(v) for v in m.groups()] for m in _START_PATTERN.finditer(text)]
    return np.array(found, dtype=np.float64).reshape(-1, 3)


class Lidar:
    """One fixed lidar configuration, used unchanged for training and flying.

    The configuration is saved next to a trained policy, so the code that
    flies it in Gazebo can check it is using the same lidar the policy was
    trained with.
    """

    def __init__(self, n_azimuth=DEFAULT_AZIMUTHS,
                 elevations_deg=DEFAULT_ELEVATIONS_DEG, poles=True,
                 max_range=DEFAULT_MAX_RANGE, ground=True):
        self.n_azimuth = int(n_azimuth)
        self.elevations_deg = tuple(float(e) for e in elevations_deg)
        self.poles = bool(poles)
        self.max_range = float(max_range)
        self.ground = bool(ground)
        self.directions = make_directions(self.n_azimuth, self.elevations_deg,
                                          self.poles)

    @property
    def n_rays(self):
        return self.directions.shape[0]

    def ranges(self, position, boxes):
        """Distances in metres, capped at max_range."""
        return cast(position, self.directions, boxes, self.max_range,
                    self.ground)

    def observe(self, position, boxes):
        """Distances scaled to [0, 1], which is what the network sees."""
        return (self.ranges(position, boxes) / self.max_range).astype(np.float32)

    def config(self):
        return {"n_azimuth": self.n_azimuth,
                "elevations_deg": list(self.elevations_deg),
                "poles": self.poles,
                "max_range": self.max_range,
                "ground": self.ground}

    def save(self, path):
        Path(path).write_text(json.dumps(self.config(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def __eq__(self, other):
        return isinstance(other, Lidar) and self.config() == other.config()
