#!/usr/bin/env python3
"""Train the avoidance policy with PPO.

Several copies of rl/env.py run in parallel and PPO learns from all of them.
At regular intervals the current policy is flown, without exploration noise,
on two fixed sets of mazes and scored on how many drones arrive and how many
crash:

  gazebo   the 25 single-drone scenarios of the Gazebo sweep, the ones the
           policy will finally be compared on
  heldout  mazes the policy never trains on, five obstacle counts with
           several seeds each

With --drones the training mazes hold several drones that all fly with the
policy being trained (rl/swarm.py), and more sets are scored: the same two with
three drones in each maze (gazebo_k3 and heldout_k3) and a crowded one with
eight (heldout_k8). With --neighbours K every drone is also told about its K
nearest neighbours. These sets are small and only show how training is going;
the version to use is chosen afterwards with rl/evaluate_large.py.

Flying straight at the target is scored on the same two sets once at the
start, so every evaluation can be read against the baseline.

Everything goes into rl/runs/<name>/:

  progress.csv     PPO's own statistics, one row per update
  eval.csv         the evaluations, plus the straight-line baseline
  monitor/         every training episode with its outcome
  best.zip         the policy with the most arrivals on the held-out sets
  step_<n>.zip     the policy at every evaluation, after n decisions
  latest.zip       the most recent policy, saved at every evaluation
  final.zip        the policy at the end of training
  lidar.json       the lidar it was trained with
  config.json      environment, reward and PPO settings

Usage (in the VM, after  source rl/env.sh):
  python rl/train.py --dynamics point --steps 20000 --eval-every 10000 \
      --report-every 4096 --heldout-per-n 2 --name smoke
  python rl/train.py --dynamics point --steps 2000000 --name point_1
  python rl/train.py --dynamics mujoco --steps 2000000 --name mujoco_1
  python rl/train.py --dynamics mujoco --drones 1 2 3 4 --n-steps 512 \
      --steps 3000000 --name swarm_1
  python rl/train.py --dynamics mujoco --drones 4 6 8 8 --neighbours 3 \
      --init-from rl/runs/swarm_2/step_1600000.zip --lr 1e-4 --near-cost 0.5 \
      --n-steps 256 --steps 3000000 --eval-every 300000 --heldout-per-n 8 \
      --name swarm_3
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env as maze  # noqa: E402

GAZEBO_SCENARIOS = [(n, s) for n in (4, 8, 12, 16, 20) for s in (1, 2, 3, 4, 5)]
HELDOUT_N = (4, 8, 12, 16, 20)
HELDOUT_FIRST_SEED = 101     # far from the 1 to 5 used by the Gazebo sweep

OUTCOMES = ("reached", "collided", "hit_drone", "ground", "out_of_bounds", "timeout")


def mem_available_mb():
    """Free memory in the VM, read from /proc so no extra package is needed."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def straight_policy(env, obs):
    # Observation entries 82 to 84 are the unit vector towards the target.
    # Works for one observation and for one row per drone.
    return obs[..., env.lidar.n_rays:env.lidar.n_rays + 3]


def evaluate(env, act, scenarios):
    """Fly each scenario once and summarise the outcomes, one per drone.

    env is a single-drone MazeEnv or a several-drone swarm.Scene.
    """
    if isinstance(env, maze.MazeEnv):
        infos = []
        for n, seed in scenarios:
            try:
                obs, _ = env.reset(options={"n": n, "seed": seed})
            except RuntimeError:
                continue
            while True:
                obs, _, terminated, truncated, info = env.step(act(env, obs))
                if terminated or truncated:
                    infos.append(info)
                    break
    else:
        from swarm import fly_scenarios
        infos = fly_scenarios(env, act, scenarios)
    counts = Counter(i["outcome"] for i in infos)
    arrived = [i for i in infos if i["outcome"] == "reached"]
    return {
        "episodes": len(infos),
        **{k: counts.get(k, 0) for k in OUTCOMES},
        "ratio_reached": (float(np.mean([i["travelled"] / i["euclidean"] for i in arrived]))
                          if arrived else 0.0),
        "mean_min_clearance": float(np.mean([max(i["min_clearance"], -1.0) for i in infos]))
                              if infos else 0.0,
        "mean_max_altitude": float(np.mean([i["max_altitude"] for i in infos]))
                             if infos else 0.0,
    }


def describe(result):
    n = result["episodes"]
    parts = [f"reached {result['reached']}/{n}", f"collided {result['collided']}"]
    for k in ("hit_drone", "ground", "out_of_bounds", "timeout"):
        if result[k]:
            parts.append(f"{k} {result[k]}")
    if result["reached"]:
        parts.append(f"ratio {result['ratio_reached']:.3f}")
    parts.append(f"highest {result['mean_max_altitude']:.1f} m")
    return ", ".join(parts)


def make_callback(eval_sets, eval_every, report_every, out, started):
    from stable_baselines3.common.callbacks import BaseCallback

    eval_file = out / "eval.csv"

    class EvalCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = eval_every
            self.next_report = report_every
            self.best = None
            self.last_report = (0, time.monotonic())

        def heartbeat(self):
            """One line from the last 100 training episodes, with exploration."""
            episodes = list(self.model.ep_info_buffer)
            done, then = self.last_report
            now = time.monotonic()
            rate = (self.num_timesteps - done) / max(now - then, 1e-9)
            self.last_report = (self.num_timesteps, now)
            if not episodes:
                print(f"{self.num_timesteps:>11,} decisions  {rate:,.0f}/s  "
                      f"(no finished episodes yet)", flush=True)
                return
            counts = Counter(e.get("outcome") for e in episodes)
            # "cut": a later mission stopped because its maze was replaced.
            mix = ", ".join(f"{k} {counts[k]}" for k in (*OUTCOMES, "cut")
                            if counts.get(k))
            print(f"{self.num_timesteps:>11,} decisions  {rate:,.0f}/s  "
                  f"last {len(episodes)} episodes: {mix}  "
                  f"mean return {np.mean([e['r'] for e in episodes]):.1f}", flush=True)

        def _policy(self, env, obs):
            return self.model.predict(obs, deterministic=True)[0]

        def run_eval(self, label):
            rows = []
            for set_name, (eval_env, scenarios) in eval_sets.items():
                r = evaluate(eval_env, self._policy, scenarios)
                rows.append((set_name, r))
                print(f"    {label} {set_name:<10} {describe(r)}", flush=True)
            write_eval_rows(eval_file, self.num_timesteps, started, label, rows)

            self.model.save(out / "latest.zip")
            if label == "policy":
                # Every evaluated version is kept, so the choice between them
                # can be made later on a larger set (rl/evaluate_large.py).
                self.model.save(out / f"step_{self.num_timesteps}.zip")
            held = [r for name, r in rows if name.startswith("heldout")]
            arrived = sum(r["reached"] for r in held)
            crashed = sum(r["collided"] + r["hit_drone"] for r in held)
            score = (arrived, -crashed)
            if self.best is None or score > self.best:
                self.best = score
                self.model.save(out / "best.zip")
                print(f"    new best: {arrived} of {sum(r['episodes'] for r in held)} "
                      f"arrivals on the held-out mazes", flush=True)

        def _on_step(self):
            if self.num_timesteps >= self.next_report:
                self.next_report += report_every
                self.heartbeat()
            if self.num_timesteps >= self.next_eval:
                self.next_eval += eval_every
                minutes = (time.monotonic() - started) / 60
                print(f"--- evaluation at {self.num_timesteps:,} decisions, "
                      f"{minutes:.1f} min, {mem_available_mb()} MB free ---", flush=True)
                self.run_eval("policy")
            return True

    return EvalCallback()


def write_eval_rows(path, decisions, started, label, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["decisions", "minutes", "policy", "set", "episodes", *OUTCOMES,
                        "ratio_reached", "mean_min_clearance", "mean_max_altitude",
                        "mem_free_mb"])
        for set_name, r in rows:
            w.writerow([decisions, f"{(time.monotonic() - started) / 60:.1f}", label, set_name,
                        r["episodes"], *[r[k] for k in OUTCOMES],
                        f"{r['ratio_reached']:.3f}", f"{r['mean_min_clearance']:.3f}",
                        f"{r['mean_max_altitude']:.2f}", mem_available_mb()])


def env_settings():
    names = ["TAKEOFF_ALTITUDE", "DRONE_RADIUS", "TOLERANCE", "CEILING", "OUT_MARGIN",
             "SIM_FREQ", "CTRL_FREQ", "DECISION_EVERY", "SETTLE_STEPS", "MAX_DECISIONS",
             "LOOKAHEAD", "POINT_SPEED", "POINT_LAG", "FLOWN_SPEED_XY",
             "FLOWN_SPEED_Z", "V_SCALE", "DIST_SCALE",
             "PROGRESS", "TIME_COST", "SAFE_DISTANCE", "NEAR_COST", "CRASH_COST",
             "ARRIVAL_BONUS", "DRONE_SAFE_DISTANCE", "DRONE_NEAR_COST",
             "NEIGHBOUR_FEATURES", "NEIGHBOUR_RANGE", "NEIGHBOUR_MISS_RANGE",
             "NEIGHBOUR_HORIZON"]
    return {n: getattr(maze, n) for n in names}


def warm_start(model, path):
    """Start from an already trained policy, which may have fewer inputs.

    Every weight is copied. Where the new network has more inputs than the old
    one (the neighbour inputs, which come after the old ones), the weights of
    the extra inputs start at zero, so at first the new policy flies like the
    old one and then learns what to do with them.
    """
    import torch
    from stable_baselines3 import PPO

    old = PPO.load(str(path), device="cpu", custom_objects={
        "learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.2})
    state = model.policy.state_dict()
    widened = []
    for name, value in old.policy.state_dict().items():
        if name not in state:
            raise SystemExit(f"--init-from: {path} has a weight '{name}' the new network lacks")
        target = state[name]
        if value.shape == target.shape:
            state[name] = value.clone()
        elif (value.dim() == 2 and value.shape[0] == target.shape[0]
              and value.shape[1] < target.shape[1]):
            wider = torch.zeros_like(target)
            wider[:, :value.shape[1]] = value
            state[name] = wider
            widened.append(f"{name} {value.shape[1]} -> {target.shape[1]} inputs")
        else:
            raise SystemExit(f"--init-from: '{name}' is {tuple(value.shape)} in {path} "
                             f"but {tuple(target.shape)} in the new network")
    model.policy.load_state_dict(state)
    print(f"started from {path}" + (f"; widened {', '.join(widened)}" if widened else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dynamics", choices=["point", "mujoco"], default="point")
    ap.add_argument("--steps", type=int, default=2_000_000,
                    help="total policy decisions, summed over all environments")
    ap.add_argument("--envs", type=int, default=4, help="environments in parallel")
    ap.add_argument("--drones", type=int, nargs="+", default=None,
                    help="train with several drones per maze: one number per "
                         "process, the drones in its maze (e.g. 1 2 3 4). "
                         "Replaces --envs")
    ap.add_argument("--neighbours", type=int, default=0,
                    help="tell every drone the relative position and velocity "
                         "of this many nearest drones (needs --drones)")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from this trained model .zip instead of from "
                         "random weights; it may have fewer inputs")
    ap.add_argument("--n-min", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=200_000)
    ap.add_argument("--report-every", type=int, default=20_000,
                    help="decisions between one-line progress reports")
    ap.add_argument("--heldout-per-n", type=int, default=20,
                    help="held-out mazes per obstacle count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="run directory name under rl/runs/")
    # PPO
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=1024, help="rollout length per env")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    # Both added after the first point-model run (point_1): the held-out score
    # peaked at 94/100 around 800k decisions and then fell to about 80 while
    # training went on. As the policy's noise shrank (std 0.79 to 0.26), the
    # same learning rate moved the actual behaviour further each update
    # (approx_kl 0.015 to 0.045, clip fraction 0.24 to 0.36) and the updates
    # overshot. Decaying the learning rate and capping each update's change
    # are the usual fixes for this.
    # Reward. The defaults are the constants in rl/env.py.
    ap.add_argument("--near-cost", type=float, default=None,
                    help=f"cost per decision at zero clearance (default {maze.NEAR_COST})")
    ap.add_argument("--safe-distance", type=float, default=None,
                    help="clearance in metres below which that cost starts "
                         f"(default {maze.SAFE_DISTANCE})")
    ap.add_argument("--drone-near-cost", type=float, default=None,
                    help="the same cost for the gap to another drone "
                         f"(default {maze.DRONE_NEAR_COST})")
    ap.add_argument("--drone-safe-distance", type=float, default=None,
                    help=f"gap below which it starts (default {maze.DRONE_SAFE_DISTANCE})")
    ap.add_argument("--no-lr-decay", action="store_true",
                    help="keep the learning rate constant instead of decaying it to 0")
    ap.add_argument("--target-kl", type=float, default=0.03,
                    help="stop an update's epochs early once the policy has moved "
                         "this far (0 disables)")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    # Set before the environments are made, so the worker processes and
    # config.json both get the changed values.
    if args.near_cost is not None:
        maze.NEAR_COST = args.near_cost
    if args.safe_distance is not None:
        maze.SAFE_DISTANCE = args.safe_distance
    if args.drone_near_cost is not None:
        maze.DRONE_NEAR_COST = args.drone_near_cost
    if args.drone_safe_distance is not None:
        maze.DRONE_SAFE_DISTANCE = args.drone_safe_distance
    if args.neighbours and not args.drones:
        ap.error("--neighbours needs --drones: one drone alone has no neighbours")

    name = args.name or f"{args.dynamics}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("eval.csv",):
        (out / stale).unlink(missing_ok=True)

    n_range = (args.n_min, args.n_max)
    lidar = maze.Lidar()
    lidar.save(out / "lidar.json")

    ppo_kwargs = dict(n_steps=args.n_steps, batch_size=args.batch_size,
                      n_epochs=args.epochs, gamma=args.gamma, gae_lambda=0.95,
                      clip_range=0.2, ent_coef=args.ent_coef,
                      target_kl=args.target_kl or None)
    lr0 = args.lr
    # SB3 calls this with the fraction of training still left, 1 down to 0.
    learning_rate = lr0 if args.no_lr_decay else (lambda remaining: lr0 * remaining)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    (out / "config.json").write_text(json.dumps({
        "dynamics": args.dynamics, "steps": args.steps, "envs": args.envs,
        "drones": args.drones, "neighbours": args.neighbours,
        "init_from": str(args.init_from) if args.init_from else None,
        "n_obstacles": list(n_range), "seed": args.seed,
        "ppo": {**ppo_kwargs, "learning_rate": lr0,
                "lr_schedule": "constant" if args.no_lr_decay else "linear to 0"},
        "policy": {"net_arch": [256, 256]},
        "env": env_settings(), "lidar": lidar.config(),
    }, indent=2) + "\n")

    def make():
        return maze.MazeEnv(dynamics=args.dynamics, n_obstacles=n_range,
                            neighbours=args.neighbours)

    # fork shares the parent's memory with the workers until they write to
    # it, which matters on a VM with under 5 GB free.
    if args.drones:
        from stable_baselines3.common.vec_env import VecMonitor
        from swarm import Scene, SwarmVecEnv
        (out / "monitor").mkdir(exist_ok=True)
        venv = VecMonitor(
            SwarmVecEnv(args.drones, dynamics=args.dynamics, n_obstacles=n_range,
                        neighbours=args.neighbours),
            filename=str(out / "monitor" / "swarm"),
            info_keywords=("outcome", "n_obstacles", "k_drones"))
        layout = (f"{len(args.drones)} mazes with {', '.join(map(str, args.drones))} "
                  f"drones ({venv.num_envs} environments)")
    else:
        vec_cls = SubprocVecEnv if args.envs > 1 else DummyVecEnv
        vec_kwargs = {"start_method": "fork"} if args.envs > 1 else None
        venv = make_vec_env(make, n_envs=args.envs, seed=args.seed, vec_env_cls=vec_cls,
                            vec_env_kwargs=vec_kwargs, monitor_dir=str(out / "monitor"),
                            monitor_kwargs={"info_keywords": ("outcome", "n_obstacles")})
        layout = f"{args.envs} environments"

    # The evaluation environments are made after the workers were forked, so
    # the workers do not carry copies of them.
    eval_env = make()
    heldout = [(n, HELDOUT_FIRST_SEED + i) for n in HELDOUT_N
               for i in range(args.heldout_per_n)]
    eval_sets = {"gazebo": (eval_env, GAZEBO_SCENARIOS), "heldout": (eval_env, heldout)}
    if args.drones:
        # The same two sets with three drones per maze. Half as many held-out
        # mazes, which still makes more flights than the one-drone set.
        eval_scene = Scene(3, dynamics=args.dynamics, respawn=False,
                           neighbours=args.neighbours)
        eval_sets["gazebo_k3"] = (eval_scene, GAZEBO_SCENARIOS)
        eval_sets["heldout_k3"] = (eval_scene, [
            (n, HELDOUT_FIRST_SEED + i) for n in HELDOUT_N
            for i in range(max(1, args.heldout_per_n // 2))])
        # A crowded set, where drones meet often.
        crowd = Scene(8, dynamics=args.dynamics, respawn=False,
                      neighbours=args.neighbours)
        eval_sets["heldout_k8"] = (crowd, [
            (n, HELDOUT_FIRST_SEED + i) for n in HELDOUT_N
            for i in range(max(1, args.heldout_per_n // 4))])

    started = time.monotonic()
    print(f"run {out}")
    print(f"{args.dynamics} flight model, {layout}, "
          f"{args.steps:,} decisions, {mem_available_mb()} MB free")
    print("--- baseline: flying straight at the target ---", flush=True)
    rows = []
    for set_name, (set_env, scenarios) in eval_sets.items():
        r = evaluate(set_env, straight_policy, scenarios)
        rows.append((set_name, r))
        print(f"    straight {set_name:<10} {describe(r)}", flush=True)
    write_eval_rows(out / "eval.csv", 0, started, "straight", rows)

    model = PPO("MlpPolicy", venv, learning_rate=learning_rate,
                policy_kwargs=policy_kwargs, seed=args.seed,
                device="cpu", verbose=0, **ppo_kwargs)
    model.set_logger(configure(str(out), ["csv"]))
    if args.init_from:
        warm_start(model, args.init_from)

    callback = make_callback(eval_sets, args.eval_every, args.report_every, out, started)
    try:
        model.learn(total_timesteps=args.steps, callback=callback)
    except KeyboardInterrupt:
        print("interrupted; saving what there is")
    model.save(out / "final.zip")

    print(f"--- final evaluation, {(time.monotonic() - started) / 60:.1f} min ---", flush=True)
    callback.run_eval("final")
    venv.close()
    print(f"done; results in {out}")


if __name__ == "__main__":
    main()
