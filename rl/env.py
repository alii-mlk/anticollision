"""Gymnasium environment for training the avoidance policy.

One drone in a random 3D maze, starting at take-off altitude and asked to reach
a target. Every episode draws a new maze, so the policy has to learn to avoid
obstacles in general rather than memorise a layout.

How it matches the PX4 stage, where the trained policy is evaluated:

  mazes        drawn by px4/gen_scenario_3d.py itself, imported rather than
               copied, so training and evaluation use the same generator.
               Given the same (n, seed) this builds the same scenario the
               Gazebo sweep flew.
  sensing      rl/lidar.py, the same code that will run in Gazebo.
  collisions   judged geometrically, with the closest-point distance that
               collision_monitor_3d.py uses: the drone is a sphere of radius
               0.35 m and touching a box ends the episode. The boxes therefore
               never need to exist inside MuJoCo.
  the action   a 3D direction, turned into a waypoint LOOKAHEAD metres along it.
               In Gazebo PX4 gets the same direction as an offboard velocity
               setpoint, at the speeds this waypoint gives here (FLOWN_SPEED_XY
               and FLOWN_SPEED_Z).
  the start    take-off altitude above the scenario start point, which is
               where run_mission.py hands over after its vertical climb.

The flight itself comes from one of two models:

  "mujoco"  MuJoCo-Drones-Gym: a Crazyflie in MuJoCo, flown by the package's
            own PID controller towards each waypoint. Used for training. The
            controller is called directly rather than through the
            package's waypoint mode, which has a bug; see MujocoDynamics.
  "point"   a point mass that turns towards the commanded velocity with a lag.
            Much simpler, but fast and independent of the package, so it helps
            tell a problem with the environment from a problem with the drone
            model.

MuJoCo works in its own local frame, with the drone starting at the origin;
the maze is placed around it with  world = local + offset, in the same way as
world = local + start in the PX4 stage.

Observation (90 values, all in [-1, 1]):
  0-81   lidar ranges, 0 = touching, 1 = nothing within 10 m
  82-84  unit vector towards the target
  85     distance to the target / 30, capped at 1
  86-88  velocity / 3 m/s, clipped
  89     altitude / ceiling

Training uses one drone. Other drones only appear in Gazebo, where rl/policy.py
adds them to the lidar.
"""

import math
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "px4"))

import gen_scenario_3d as scenario_gen  # noqa: E402
from lidar import Lidar  # noqa: E402

# Shared with the PX4 stage. Keep these equal to the values there.
TAKEOFF_ALTITUDE = 3.0    # run_mission.py climbs here first
DRONE_RADIUS = 0.35       # collision_monitor_3d.py
TOLERANCE = 0.6           # run_mission.py arrival tolerance

# Workspace. The maze is within +-12 m (gen_scenario_3d.BOUNDS); leaving it by
# more than the margin, or climbing past the ceiling, ends the episode.
CEILING = 12.0
OUT_MARGIN = 3.0

# Timing. MuJoCo-Drones-Gym defaults: physics at 240 Hz, control at 48 Hz. The
# policy decides every DECISION_EVERY control steps, about 10 times a second;
# collisions are checked at every control step in between.
SIM_FREQ = 240
CTRL_FREQ = 48
DECISION_EVERY = 5
SETTLE_STEPS = 48         # one second of hovering after reset
MAX_DECISIONS = 600       # about a minute of flight

# Waypoint distance along the chosen direction. The package's position gains
# turn a waypoint d metres ahead into a steady speed of about 0.42 d m/s
# (measured at 1 m, see rl/diagnose_flight.py), so 2.5 m gives about 1 m/s at a
# full-length action. Shorter actions give slower flight, so the policy can
# also choose its speed.
LOOKAHEAD = 2.5
POINT_SPEED = 1.05        # point model speed for a full-length action (m/s)

# What a full-length action actually does in MuJoCo with that waypoint, measured
# by check_env.py's calibration: 1.04 m/s level, 1.17 m/s climbing. The policy
# learned under these speeds, so they are what PX4 is asked to fly in Gazebo,
# as velocity setpoints, one axis at a time.
FLOWN_SPEED_XY = 1.04
FLOWN_SPEED_Z = 1.17
POINT_LAG = 0.3           # point model velocity time constant (s)

V_SCALE = 3.0
DIST_SCALE = 30.0

# Reward. The progress term is the main signal; the rest shape it.
PROGRESS = 1.0            # per metre closer to the target
TIME_COST = 0.01          # per decision, so dawdling is not free
SAFE_DISTANCE = 1.0       # clearance below which the proximity cost starts (m)
NEAR_COST = 0.1           # proximity cost at zero clearance, per decision
CRASH_COST = 10.0         # hitting a box or the ground, or leaving the workspace
ARRIVAL_BONUS = 10.0


def build_observation(lidar, pos, vel, target, boxes, v_scale=V_SCALE,
                      dist_scale=DIST_SCALE, ceiling=CEILING):
    """The 90 numbers the policy sees, from the drone's state and the scene.

    This is the only place the observation is put together. The training
    environment calls it, and so does the code that flies the policy in
    Gazebo (rl/policy.py), so the two can never drift apart.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    scan = lidar.observe(pos, boxes)
    to_target = np.asarray(target, dtype=np.float64) - pos
    dist = float(np.linalg.norm(to_target))
    unit = to_target / dist if dist > 1e-9 else np.zeros(3)
    extra = np.array([
        *unit,
        min(dist / dist_scale, 1.0),
        *np.clip(vel / v_scale, -1.0, 1.0),
        min(max(pos[2] / ceiling, 0.0), 1.0),
    ], dtype=np.float32)
    return np.concatenate([scan, extra]).astype(np.float32)


def clip_action(action):
    """Clip each component to [-1, 1], then the whole vector to length 1."""
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
    norm = float(np.linalg.norm(a))
    return a / norm if norm > 1.0 else a


def clearance_to_boxes(pos, boxes, radius=DRONE_RADIUS):
    """Gap between the drone's sphere and the nearest box; <= 0 means contact."""
    if not len(boxes):
        return math.inf
    lo, hi = boxes[:, [0, 2, 4]], boxes[:, [1, 3, 5]]
    gap = np.maximum(np.maximum(lo - pos, 0.0), pos - hi)
    return float(np.sqrt((gap ** 2).sum(axis=1)).min()) - radius


class PointDynamics:
    """A point mass whose velocity follows the commanded one with a lag."""

    name = "point"

    def __init__(self):
        self.dt = 1.0 / CTRL_FREQ

    def reset(self):
        self.pos = np.array([0.0, 0.0, TAKEOFF_ALTITUDE])
        self.vel = np.zeros(3)
        return self.pos.copy(), self.vel.copy()

    def step(self, direction):
        wanted = direction * POINT_SPEED
        self.vel = self.vel + (wanted - self.vel) * (self.dt / POINT_LAG)
        self.pos = self.pos + self.vel * self.dt
        return self.pos.copy(), self.vel.copy()


class MujocoDynamics:
    """A Crazyflie in MuJoCo-Drones-Gym, flown towards waypoints by its PID.

    The package's own waypoint mode (ActionType.PID) is not used, because it
    does not fly: it creates a new PIDControl on every control step, which
    wipes the controller's internal state, and the attitude then oscillates
    with growing amplitude until the drone flips over. Hover and climb still
    work, since they need no tilt, which is why it is easy to miss.
    rl/diagnose_flight.py shows this. Here the same controller is called
    directly, one instance kept for the whole episode, with the drone's state
    read from MuJoCo, and its motor speeds are sent through ActionType.RPM.
    Drone model, physics and controller are still all the package's own.
    """

    name = "mujoco"

    def __init__(self):
        from multi_drone_mujoco.control.pid_control import PIDControl
        from multi_drone_mujoco.envs.base_aviary import BaseAviary
        from multi_drone_mujoco.utils.enums import (ActionType, DroneModel,
                                                    ObservationType, Physics)
        import mujoco

        out = HERE / "runs" / "aviary"
        out.mkdir(parents=True, exist_ok=True)
        self.sim = BaseAviary(
            drone_model=DroneModel.CF2X, num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, TAKEOFF_ALTITUDE]]),
            physics=Physics.MJC, sim_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
            gui=False, record=False, obstacles=False,
            obs_type=ObservationType.KIN, act_type=ActionType.RPM,
            output_folder=str(out))
        self._controller_class = PIDControl
        self.ctrl = None

        model = self.sim.model
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.qadr = int(model.jnt_qposadr[j])
                self.vadr = int(model.jnt_dofadr[j])
                break
        else:
            raise RuntimeError("the drone model has no free joint")

    def _state(self):
        """Position, quaternion (w, x, y, z), velocity, angular velocity."""
        q, v = self.sim.data.qpos, self.sim.data.qvel
        return (q[self.qadr:self.qadr + 3].copy(), q[self.qadr + 3:self.qadr + 7].copy(),
                v[self.vadr:self.vadr + 3].copy(), v[self.vadr + 3:self.vadr + 6].copy())

    def _fly_to(self, waypoint):
        pos, quat, vel, ang_vel = self._state()
        rpm, _, _ = self.ctrl.computeControl(
            control_timestep=1.0 / CTRL_FREQ, cur_pos=pos, cur_quat=quat,
            cur_vel=vel, cur_ang_vel=ang_vel,
            target_pos=np.asarray(waypoint, dtype=np.float64), target_rpy=np.zeros(3))
        self.sim.step(np.asarray(rpm, dtype=np.float64).reshape(4))
        pos, _, vel, _ = self._state()
        return pos, vel

    def reset(self):
        # A fresh controller per episode, so nothing carries over from the
        # last one, kept for the whole episode. The drone appears in mid-air
        # with its motors stopped, so it is held at the start for a second
        # before the policy takes over.
        self.sim.reset()
        self.ctrl = self._controller_class(self.sim)
        hold = np.array([0.0, 0.0, TAKEOFF_ALTITUDE])
        for _ in range(SETTLE_STEPS):
            pos, vel = self._fly_to(hold)
        return pos, vel

    def step(self, direction):
        # The waypoint is recomputed from the current position at every
        # control step, so the drone keeps moving along the direction for as
        # long as it is held, rather than stopping at a fixed point.
        pos = self._state()[0]
        return self._fly_to(pos + direction * LOOKAHEAD)


def make_dynamics(name):
    if name == "mujoco":
        return MujocoDynamics()
    if name == "point":
        return PointDynamics()
    raise ValueError(f"unknown dynamics '{name}', expected 'mujoco' or 'point'")


class MazeEnv(gym.Env):
    """Fly from the start to the target without touching anything."""

    metadata = {"render_modes": []}

    def __init__(self, dynamics="mujoco", n_obstacles=(0, 20), lidar=None,
                 max_decisions=MAX_DECISIONS):
        super().__init__()
        self.dyn = make_dynamics(dynamics)
        self.n_range = (int(n_obstacles[0]), int(n_obstacles[1]))
        self.lidar = lidar or Lidar()
        self.max_decisions = int(max_decisions)

        n_obs = self.lidar.n_rays + 8
        self.observation_space = spaces.Box(-1.0, 1.0, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

    # --- scenes ---------------------------------------------------------

    def _make_scene(self, n, seed):
        """One drone and n boxes, drawn the same way gen_scenario_3d.py draws them."""
        for _ in range(50):
            s = seed if seed is not None else int(self.np_random.integers(2 ** 31 - 1))
            rng = random.Random(s)
            try:
                drones = scenario_gen.generate_drones(1, rng)
                obstacles = scenario_gen.generate_obstacles(n, drones, rng)
            except SystemExit:
                # The generator gives up on a crowded draw; try another seed,
                # unless a specific scenario was asked for.
                if seed is not None:
                    raise RuntimeError(f"gen_scenario_3d could not build n={n} seed={seed}")
                continue
            boxes = np.array([[*o["rect"], 0.0, o["height"]] for o in obstacles],
                             dtype=np.float64).reshape(-1, 6)
            return drones[0], boxes, s
        raise RuntimeError(f"could not draw a scene with {n} obstacles")

    # --- gym interface --------------------------------------------------

    def reset(self, seed=None, options=None):
        """options: {"n": obstacles, "seed": scenario seed}, both optional.

        With both given the scene is the one gen_scenario_3d.py writes for
        --n-obstacles n --k-drones 1 --seed seed.
        """
        super().reset(seed=seed)
        options = options or {}
        n = options.get("n")
        if n is None:
            n = int(self.np_random.integers(self.n_range[0], self.n_range[1] + 1))
        drone, self.boxes, self.scene_seed = self._make_scene(int(n), options.get("seed"))
        self.n_obstacles = int(n)

        sx, sy, _ = drone["start"]
        self.start = np.array([sx, sy, TAKEOFF_ALTITUDE])
        self.target = np.array(drone["target"], dtype=np.float64)
        self.offset = np.array([sx, sy, 0.0])

        local, vel = self.dyn.reset()
        self.pos = local + self.offset
        self.vel = vel
        self.last_pos = self.pos.copy()
        self.travelled = 0.0
        self.decisions = 0
        self.min_clearance = math.inf
        self.max_altitude = float(self.pos[2])
        self.prev_dist = float(np.linalg.norm(self.target - self.pos))
        self.euclidean = float(np.linalg.norm(self.target - self.start))
        return self._obs(), self._info(None)

    def step(self, action):
        a = clip_action(action)

        outcome = None
        clearance = math.inf
        for _ in range(DECISION_EVERY):
            local, self.vel = self.dyn.step(a)
            self.pos = local + self.offset
            self.travelled += float(np.linalg.norm(self.pos - self.last_pos))
            self.last_pos = self.pos.copy()
            self.max_altitude = max(self.max_altitude, float(self.pos[2]))

            box_gap = clearance_to_boxes(self.pos, self.boxes)
            ground_gap = self.pos[2] - DRONE_RADIUS
            clearance = min(box_gap, ground_gap)
            self.min_clearance = min(self.min_clearance, clearance)

            if box_gap <= 0.0:
                outcome = "collided"
            elif ground_gap <= 0.0:
                outcome = "ground"
            elif (abs(self.pos[0]) > scenario_gen.BOUNDS + OUT_MARGIN
                  or abs(self.pos[1]) > scenario_gen.BOUNDS + OUT_MARGIN
                  or self.pos[2] > CEILING):
                outcome = "out_of_bounds"
            elif np.linalg.norm(self.target - self.pos) <= TOLERANCE:
                outcome = "reached"
            if outcome:
                break

        self.decisions += 1
        dist = float(np.linalg.norm(self.target - self.pos))
        reward = PROGRESS * (self.prev_dist - dist) - TIME_COST
        closeness = max(0.0, 1.0 - max(clearance, 0.0) / SAFE_DISTANCE)
        reward -= NEAR_COST * closeness ** 2
        self.prev_dist = dist

        if outcome == "reached":
            reward += ARRIVAL_BONUS
        elif outcome is not None:
            reward -= CRASH_COST

        terminated = outcome is not None
        truncated = not terminated and self.decisions >= self.max_decisions
        if truncated:
            outcome = "timeout"
        return self._obs(), float(reward), terminated, truncated, self._info(outcome)

    # --- helpers --------------------------------------------------------

    def _obs(self):
        return build_observation(self.lidar, self.pos, self.vel, self.target, self.boxes)

    def _info(self, outcome):
        return {
            "outcome": outcome,
            "n_obstacles": self.n_obstacles,
            "scene_seed": self.scene_seed,
            "decisions": self.decisions,
            "euclidean": self.euclidean,
            "travelled": self.travelled,
            "distance_to_target": float(np.linalg.norm(self.target - self.pos)),
            "min_clearance": self.min_clearance,
            # Highest point of the flight. Boxes reach at most 10 m, so a
            # policy that just climbs over everything shows up here.
            "max_altitude": self.max_altitude,
        }
