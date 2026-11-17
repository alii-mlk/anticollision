"""Several drones in one maze, all flown by the same policy.

rl/env.py trains one drone alone. Here k drones share a maze, each with its own
start and target, and every one of them is flown by the same network. Touching
another drone ends the flight for both, as in collision_monitor_3d.py.

A drone learns about the others in two ways. They are in its lidar as cubes of
side 2 * DRONE_RADIUS, which is what rl/policy.py does in Gazebo. And, with
neighbours=K, it is given the relative position and velocity of its K nearest
neighbours as extra inputs (env.neighbour_features), as if the drones broadcast
them. The lidar alone was not enough: a drone is small between
the rays and a range says nothing about where it is going, and policies trained
with only the lidar hit other drones as often as flying straight does.

Everything else is taken from rl/env.py: the flight models, the observation,
the reward and its constants, the collision rule. The scenes come from
px4/gen_scenario_3d.py, so (n, seed) with k drones is the scenario the Gazebo
sweep flew with --k-drones k.

Two ways to use a scene:

  respawn=False  every drone flies its mission once and is then taken out.
                 For evaluation.
  respawn=True   for training. PPO needs every drone to be flying all the
                 time, so a drone that finishes gets a new mission in the same
                 maze: after arriving it flies back the way it came, after a
                 crash it is put back at its start, or above its target, or
                 somewhere free if other drones are near both. The maze is replaced
                 when every drone has finished its first mission; later
                 missions still under way are cut off there and counted as
                 truncated, not as a success or a crash.

SwarmVecEnv runs several scenes in separate processes and presents every drone
as one environment to Stable-Baselines3.

Each drone has its own MuJoCo instance. The drones do not push air at each
other in this model, and contact is judged geometrically, so nothing is lost by
keeping them apart, and the flight model that was checked against Gazebo is
used as it is.
"""

import math
import multiprocessing
import random
import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "px4"))

import env as maze  # noqa: E402
import gen_scenario_3d as scenario_gen  # noqa: E402
from lidar import Lidar, drone_boxes  # noqa: E402


# A drone put back into a maze somewhere other than the ends of its route
# keeps at least this gap to every box (m).
RESPAWN_CLEARANCE = 1.0


class Scene:
    """k drones in one maze. Arrays are indexed by drone."""

    def __init__(self, k, dynamics="mujoco", n_obstacles=(0, 20), lidar=None,
                 max_decisions=None, respawn=True, interact=True, seed=None,
                 neighbours=0):
        self.k = int(k)
        self.dyn = [maze.make_dynamics(dynamics) for _ in range(self.k)]
        self.n_range = (int(n_obstacles[0]), int(n_obstacles[1]))
        self.lidar = lidar or Lidar()
        # How many nearest drones each drone is told about; 0 for the policies
        # trained before those inputs existed.
        self.neighbours = int(neighbours)
        self.n_obs = self.lidar.n_rays + 8 + self.neighbours * maze.NEIGHBOUR_FEATURES
        self.max_decisions = int(max_decisions or maze.MAX_DECISIONS)
        self.respawn = bool(respawn)
        # With interact=False the drones neither see nor touch each other:
        # k separate flights that happen to share a maze.
        self.interact = bool(interact)
        self.rng = np.random.default_rng(seed)

    # --- scenes ---------------------------------------------------------

    def draw(self, n, seed):
        """k drones and n boxes, drawn the way gen_scenario_3d.py draws them."""
        for _ in range(50):
            s = seed if seed is not None else int(self.rng.integers(2 ** 31 - 1))
            rng = random.Random(s)
            try:
                drones = scenario_gen.generate_drones(self.k, rng)
                obstacles = scenario_gen.generate_obstacles(n, drones, rng)
            except SystemExit:
                if seed is not None:
                    raise RuntimeError(
                        f"gen_scenario_3d could not build n={n} k={self.k} seed={seed}")
                continue
            boxes = np.array([[*o["rect"], 0.0, o["height"]] for o in obstacles],
                             dtype=np.float64).reshape(-1, 6)
            return drones, boxes, s
        raise RuntimeError(f"could not draw a scene with {n} obstacles and {self.k} drones")

    def reset(self, n=None, seed=None):
        """A new maze. With n and seed given it is the Gazebo scenario."""
        if n is None:
            n = int(self.rng.integers(self.n_range[0], self.n_range[1] + 1))
        drones, boxes, used = self.draw(int(n), seed)
        return self.begin(drones, boxes, seed=used)

    def begin(self, drones, boxes, seed=None):
        """Start all drones on the given missions in the given maze."""
        if len(drones) != self.k:
            raise ValueError(f"{len(drones)} missions for {self.k} drones")
        k = self.k
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
        self.n_obstacles = len(self.boxes)
        self.scene_seed = seed

        # The two ends of each drone's route. Both are free of obstacles at
        # take-off altitude, because the generator keeps a clear radius around
        # every start and every target.
        self.home = np.array([[d["start"][0], d["start"][1], maze.TAKEOFF_ALTITUDE]
                              for d in drones], dtype=np.float64)
        self.goal = np.array([d["target"] for d in drones], dtype=np.float64)
        self.above_goal = self.goal.copy()
        self.above_goal[:, 2] = maze.TAKEOFF_ALTITUDE

        self.pos = np.zeros((k, 3))
        self.vel = np.zeros((k, 3))
        self.offset = np.zeros((k, 3))
        self.target = self.goal.copy()
        self.start = self.home.copy()
        self.outbound = np.ones(k, dtype=bool)
        self.present = np.ones(k, dtype=bool)
        self.first_done = np.zeros(k, dtype=bool)
        self.mission = np.zeros(k, dtype=int)
        self.travelled = np.zeros(k)
        self.decisions = np.zeros(k, dtype=int)
        self.min_clearance = np.full(k, math.inf)
        self.max_altitude = np.zeros(k)
        self.prev_dist = np.zeros(k)
        self.euclidean = np.zeros(k)

        for i in range(k):
            self._place(i, self.home[i])
            self._begin_mission(i, self.home[i])
        return self._observe()

    def _place(self, i, point):
        """Put drone i at a point at take-off altitude, hovering.

        The flight models always start at their own origin at take-off
        altitude, so only the horizontal offset moves; a vertical offset would
        put MuJoCo's ground plane at the wrong height.
        """
        self.offset[i] = (point[0], point[1], 0.0)
        local, vel = self.dyn[i].reset()
        self.pos[i] = local + self.offset[i]
        self.vel[i] = vel

    def _begin_mission(self, i, start):
        self.start[i] = start
        self.travelled[i] = 0.0
        self.decisions[i] = 0
        self.min_clearance[i] = math.inf
        self.max_altitude[i] = self.pos[i, 2]
        self.prev_dist[i] = np.linalg.norm(self.target[i] - self.pos[i])
        self.euclidean[i] = np.linalg.norm(self.target[i] - self.start[i])

    def _next_mission(self, i, outcome):
        """Training only: keep drone i flying after its mission has ended."""
        if outcome == "reached":
            # It is where it is; send it back along its route.
            self.outbound[i] = not self.outbound[i]
        else:
            # Put it back at one end of its route, the one with more room. If
            # another drone is too close to both, anywhere free in the maze: a
            # drone must not appear on top of another one, or both would be
            # punished for a crash neither could avoid.
            others = [j for j in range(self.k) if j != i and self.present[j]]
            spots = (self.home[i], self.above_goal[i])
            room = [min((float(np.linalg.norm(p - self.pos[j])) for j in others),
                        default=math.inf) for p in spots]
            best = 0 if room[0] >= room[1] else 1
            spot, self.outbound[i] = spots[best], best == 0
            if room[best] < scenario_gen.DRONE_GAP:
                free = self._free_spot(others)
                if free is not None:
                    spot = free
                    # From there, fly to whichever end of its route is further.
                    self.outbound[i] = (np.linalg.norm(self.goal[i] - free)
                                        >= np.linalg.norm(self.home[i] - free))
            self._place(i, spot)
        self.target[i] = self.goal[i] if self.outbound[i] else self.home[i]
        self.mission[i] += 1
        self._begin_mission(i, self.pos[i].copy())

    def _free_spot(self, others):
        """A random point at take-off altitude clear of boxes and drones."""
        for _ in range(30):
            x, y = self.rng.uniform(-scenario_gen.BOUNDS, scenario_gen.BOUNDS, size=2)
            p = np.array([x, y, maze.TAKEOFF_ALTITUDE])
            if maze.clearance_to_boxes(p, self.boxes) < RESPAWN_CLEARANCE:
                continue
            if any(np.linalg.norm(p - self.pos[j]) < scenario_gen.DRONE_GAP for j in others):
                continue
            return p
        return None

    @property
    def finished(self):
        """True once no drone is flying (only happens with respawn=False)."""
        return not self.present.any()

    # --- one decision for every drone -----------------------------------

    def step(self, actions):
        """actions: (k, 3). Returns observations, rewards, dones, infos."""
        k = self.k
        actions = np.asarray(actions, dtype=np.float64).reshape(k, 3)
        a = [maze.clip_action(actions[i]) for i in range(k)]
        radius = maze.DRONE_RADIUS
        limit = scenario_gen.BOUNDS + maze.OUT_MARGIN

        flying = [i for i in range(k) if self.present[i]]
        outcome = [None] * k
        obstacle_gap = np.full(k, math.inf)    # to the nearest box or the ground
        others_gap = np.full(k, math.inf)      # to the nearest other drone

        for _ in range(maze.DECISION_EVERY):
            live = [i for i in flying if outcome[i] is None]
            if not live:
                break
            for i in live:
                local, self.vel[i] = self.dyn[i].step(a[i])
                new = local + self.offset[i]
                self.travelled[i] += float(np.linalg.norm(new - self.pos[i]))
                self.pos[i] = new
                self.max_altitude[i] = max(self.max_altitude[i], new[2])

            # All drones have moved; now judge each one. Two drones that touch
            # are both judged before either is taken out.
            for i in live:
                p = self.pos[i]
                box_gap = maze.clearance_to_boxes(p, self.boxes)
                ground_gap = p[2] - radius
                drone_gap = math.inf
                if self.interact:
                    for j in live:
                        if j != i:
                            gap = float(np.linalg.norm(p - self.pos[j])) - 2 * radius
                            drone_gap = min(drone_gap, gap)
                obstacle_gap[i] = min(box_gap, ground_gap)
                others_gap[i] = drone_gap
                self.min_clearance[i] = min(self.min_clearance[i], box_gap,
                                            ground_gap, drone_gap)

                if box_gap <= 0.0:
                    outcome[i] = "collided"
                elif drone_gap <= 0.0:
                    outcome[i] = "hit_drone"
                elif ground_gap <= 0.0:
                    outcome[i] = "ground"
                elif abs(p[0]) > limit or abs(p[1]) > limit or p[2] > maze.CEILING:
                    outcome[i] = "out_of_bounds"
                elif np.linalg.norm(self.target[i] - p) <= maze.TOLERANCE:
                    outcome[i] = "reached"

        # The same reward as rl/env.py, plus the same kind of proximity cost
        # for other drones, which starts further out.
        rewards = np.zeros(k, dtype=np.float32)
        for i in flying:
            self.decisions[i] += 1
            dist = float(np.linalg.norm(self.target[i] - self.pos[i]))
            reward = maze.PROGRESS * (self.prev_dist[i] - dist) - maze.TIME_COST
            closeness = max(0.0, 1.0 - max(obstacle_gap[i], 0.0) / maze.SAFE_DISTANCE)
            reward -= maze.NEAR_COST * closeness ** 2
            closeness = max(0.0, 1.0 - max(others_gap[i], 0.0) / maze.DRONE_SAFE_DISTANCE)
            reward -= maze.DRONE_NEAR_COST * closeness ** 2
            self.prev_dist[i] = dist
            if outcome[i] == "reached":
                reward += maze.ARRIVAL_BONUS
            elif outcome[i] is not None:
                reward -= maze.CRASH_COST
            elif self.decisions[i] >= self.max_decisions:
                outcome[i] = "timeout"
            rewards[i] = reward

        dones = np.array([o is not None for o in outcome])
        infos = [self._info(i, outcome[i]) for i in range(k)]
        ended = [i for i in flying if outcome[i] is not None]
        for i in ended:
            infos[i]["TimeLimit.truncated"] = outcome[i] == "timeout"
            if self.mission[i] == 0:
                self.first_done[i] = True

        if not self.respawn:
            for i in ended:
                self.present[i] = False
            return self._observe(), rewards, dones, infos

        for i in ended:
            infos[i]["terminal_observation"] = self._obs(i)

        if self.first_done.all():
            # Every drone has flown its first mission, so the maze is replaced.
            # The later missions still under way stop here.
            for i in flying:
                if outcome[i] is None:
                    dones[i] = True
                    infos[i]["outcome"] = "cut"
                    infos[i]["TimeLimit.truncated"] = True
                    infos[i]["terminal_observation"] = self._obs(i)
            return self.reset(), rewards, dones, infos

        for i in ended:
            self._next_mission(i, outcome[i])
        return self._observe(), rewards, dones, infos

    # --- helpers --------------------------------------------------------

    def _obs(self, i):
        boxes = self.boxes
        others = []
        if self.interact:
            others = [j for j in range(self.k) if j != i and self.present[j]]
            if others:
                boxes = np.vstack([boxes, drone_boxes(self.pos[others], maze.DRONE_RADIUS)])
        obs = maze.build_observation(self.lidar, self.pos[i], self.vel[i],
                                     self.target[i], boxes)
        if not self.neighbours:
            return obs
        return np.concatenate([obs, maze.neighbour_features(
            self.pos[i], self.vel[i], self.pos[others], self.vel[others],
            self.neighbours)])

    def _observe(self):
        obs = np.zeros((self.k, self.n_obs), dtype=np.float32)
        for i in range(self.k):
            if self.present[i]:
                obs[i] = self._obs(i)
        return obs

    def _info(self, i, outcome):
        return {
            "outcome": outcome,
            "n_obstacles": self.n_obstacles,
            "k_drones": self.k,
            "scene_seed": self.scene_seed,
            "drone": i,
            "mission": int(self.mission[i]),
            "decisions": int(self.decisions[i]),
            "euclidean": float(self.euclidean[i]),
            "travelled": float(self.travelled[i]),
            "distance_to_target": float(np.linalg.norm(self.target[i] - self.pos[i])),
            "min_clearance": float(self.min_clearance[i]),
            "max_altitude": float(self.max_altitude[i]),
        }


def fly_scenarios(scene, act, scenarios):
    """Fly each (n, seed) scenario once; returns one info per drone.

    scene must have respawn=False. act(scene, observations) takes the
    observations of the drones still flying, one row each, and returns one
    action per row.
    """
    if scene.respawn:
        raise ValueError("fly_scenarios needs a scene made with respawn=False")
    infos = []
    for n, seed in scenarios:
        try:
            obs = scene.reset(n=n, seed=seed)
        except RuntimeError:
            continue
        while not scene.finished:
            flying = scene.present.copy()
            actions = np.zeros((scene.k, 3))
            actions[flying] = act(scene, obs[flying])
            obs, _, dones, step_infos = scene.step(actions)
            infos.extend(info for info, done in zip(step_infos, dones) if done)
    return infos


def _scene_worker(conn, k, scene_kwargs):
    try:
        scene = Scene(k, **scene_kwargs)
        while True:
            command, data = conn.recv()
            if command == "step":
                conn.send(scene.step(data))
            elif command == "reset":
                if data is not None:
                    scene.rng = np.random.default_rng(data)
                conn.send(scene.reset())
            else:
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        conn.close()


class SwarmVecEnv(VecEnv):
    """Scenes in separate processes, every drone one environment for PPO.

    sizes lists the number of drones in each scene, one process per scene:
    [1, 2, 3, 4] gives four processes and ten environments. Finished drones
    are restarted inside the scene, so, as with any Stable-Baselines3 vector
    environment, the observation returned with done=True already belongs to
    the next episode and the last one of the old episode is in
    info["terminal_observation"].
    """

    def __init__(self, sizes, **scene_kwargs):
        self.sizes = [int(k) for k in sizes]
        if not self.sizes or min(self.sizes) < 1:
            raise ValueError("sizes must list at least one scene with at least one drone")
        scene_kwargs["respawn"] = True
        self.first = np.cumsum([0] + self.sizes[:-1])
        self.conns, self.procs = [], []
        self.closed = False

        # fork, so the workers share the parent's memory until they write to
        # it. Each worker builds its own MuJoCo instances after the fork.
        ctx = multiprocessing.get_context("fork")
        for k in self.sizes:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_scene_worker, args=(child, k, scene_kwargs),
                               daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

        lidar = scene_kwargs.get("lidar") or Lidar()
        n_obs = (lidar.n_rays + 8
                 + int(scene_kwargs.get("neighbours", 0)) * maze.NEIGHBOUR_FEATURES)
        super().__init__(sum(self.sizes),
                         spaces.Box(-1.0, 1.0, (n_obs,), np.float32),
                         spaces.Box(-1.0, 1.0, (3,), np.float32))

    def seed(self, seed=None):
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        self._seeds = [seed + i for i in range(self.num_envs)]
        return self._seeds

    def reset(self):
        for j, conn in enumerate(self.conns):
            conn.send(("reset", self._seeds[self.first[j]]))
        obs = np.concatenate([conn.recv() for conn in self.conns])
        self._seeds = [None for _ in range(self.num_envs)]
        return obs

    def step_async(self, actions):
        actions = np.asarray(actions).reshape(self.num_envs, 3)
        for j, conn in enumerate(self.conns):
            start = self.first[j]
            conn.send(("step", actions[start:start + self.sizes[j]]))

    def step_wait(self):
        results = [conn.recv() for conn in self.conns]
        obs = np.concatenate([r[0] for r in results])
        rewards = np.concatenate([r[1] for r in results])
        dones = np.concatenate([r[2] for r in results])
        infos = [info for r in results for info in r[3]]
        return obs, rewards, dones, infos

    def close(self):
        if self.closed:
            return
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
        self.closed = True

    # Stable-Baselines3 calls these, but the scenes have nothing to give.

    def _count(self, indices):
        return self.num_envs if indices is None else len(np.atleast_1d(indices))

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None] * (sum(self.sizes) if indices is None
                             else len(np.atleast_1d(indices)))
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self._count(indices)
