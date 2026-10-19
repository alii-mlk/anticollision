"""A trained avoidance policy, packaged for flying it outside training.

Loads a run directory written by rl/train.py and gives the code that flies the
drone in Gazebo (px4/run_mission.py) three things:

  observe()   the observation, built by the same function the training
              environment uses (env.build_observation), from the scenario's
              boxes, the other drones' positions, and the drone's own state
  action()    the network's choice, without exploration noise, clipped the
              same way the training environment clips it
  velocity()  that choice as a velocity to send PX4, in m/s

Everything comes from the run's own files rather than from the current
defaults: the lidar from lidar.json, the scales, timing and speeds from
config.json. Changing env.py later therefore cannot silently change what a
trained policy sees. The network's input size is checked against the lidar
before anything flies.

Why velocities: in training the waypoint is recomputed LOOKAHEAD metres ahead
at every control step, so what the drone actually follows is a commanded
velocity, measured at FLOWN_SPEED_XY level and FLOWN_SPEED_Z climbing for a
full-length action. Sending PX4 the same speeds reproduces the flight the
policy trained on, without depending on how PX4's position controller is
tuned.

Other drones are added to the lidar as cubes of side 2 * DRONE_RADIUS, so the
drone keeps the same clearance from them that it learned to keep from boxes.
Note that the policy was trained with one drone, so it has only ever seen
obstacles that stand still.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402
from lidar import Lidar, drone_boxes  # noqa: E402


class AvoidancePolicy:
    def __init__(self, path, threads=1):
        path = Path(path).expanduser().resolve()
        run_dir = path if path.is_dir() else path.parent
        model_file = run_dir / "best.zip" if path.is_dir() else path
        for needed in (model_file, run_dir / "lidar.json", run_dir / "config.json"):
            if not needed.exists():
                raise FileNotFoundError(f"missing {needed}")

        import torch
        # Several drones each run their own copy, next to PX4 and Gazebo. One
        # thread each is plenty for a network this small and keeps them from
        # fighting the simulator for cores.
        torch.set_num_threads(threads)
        from stable_baselines3 import PPO

        # The training schedules are not needed to fly, and replacing them
        # avoids unpickling the learning-rate function stored in the file.
        self.model = PPO.load(str(model_file), device="cpu", custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.2,
        })
        self.model_file = model_file
        self.lidar = Lidar.load(run_dir / "lidar.json")

        config = json.loads((run_dir / "config.json").read_text())
        settings = config.get("env", {})
        self.dynamics = config.get("dynamics", "unknown")

        def setting(name):
            return float(settings.get(name, getattr(maze, name)))

        self.v_scale = setting("V_SCALE")
        self.dist_scale = setting("DIST_SCALE")
        self.ceiling = setting("CEILING")
        self.radius = setting("DRONE_RADIUS")
        self.tolerance = setting("TOLERANCE")
        self.decision_dt = setting("DECISION_EVERY") / setting("CTRL_FREQ")
        # Runs trained before these were recorded fall back to env.py's values.
        self.speed_xy = setting("FLOWN_SPEED_XY")
        self.speed_z = setting("FLOWN_SPEED_Z")

        expected = int(self.model.observation_space.shape[0])
        if expected != self.lidar.n_rays + 8:
            raise ValueError(
                f"the network takes {expected} inputs, but lidar.json gives "
                f"{self.lidar.n_rays} rays + 8 = {self.lidar.n_rays + 8}")

    def observe(self, pos, vel, target, boxes, others=()):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
        others = np.asarray(others, dtype=np.float64).reshape(-1, 3)
        if len(others):
            boxes = np.vstack([boxes, drone_boxes(others, self.radius)])
        return maze.build_observation(self.lidar, pos, vel, target, boxes,
                                      self.v_scale, self.dist_scale, self.ceiling)

    def action(self, obs):
        a, _ = self.model.predict(obs, deterministic=True)
        return maze.clip_action(a)

    def velocity(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(3)
        return np.array([a[0] * self.speed_xy, a[1] * self.speed_xy, a[2] * self.speed_z])

    def describe(self):
        return (f"{self.model_file} ({self.dynamics}), {self.lidar.n_rays} rays, "
                f"decision every {self.decision_dt:.3f} s, "
                f"{self.speed_xy:.2f} m/s level, {self.speed_z:.2f} m/s climb")
