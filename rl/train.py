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

Flying straight at the target is scored on the same two sets once at the
start, so every evaluation can be read against the baseline.

Everything goes into rl/runs/<name>/:

  progress.csv     PPO's own statistics, one row per update
  eval.csv         the evaluations, plus the straight-line baseline
  monitor/         every training episode with its outcome
  best.zip         the policy with the most arrivals on the held-out set
  latest.zip       the most recent policy, saved at every evaluation
  final.zip        the policy at the end of training
  lidar.json       the lidar it was trained with
  config.json      environment, reward and PPO settings

Usage (in the VM, after  source rl/env.sh):
  python rl/train.py --dynamics point --steps 20000 --eval-every 10000 \
      --report-every 4096 --heldout-per-n 2 --name smoke
  python rl/train.py --dynamics point --steps 2000000 --name point_1
  python rl/train.py --dynamics mujoco --steps 2000000 --name mujoco_1
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

OUTCOMES = ("reached", "collided", "ground", "out_of_bounds", "timeout")


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
    return obs[env.lidar.n_rays:env.lidar.n_rays + 3]


def evaluate(env, act, scenarios):
    """Fly each scenario once and summarise the outcomes."""
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
    for k in ("ground", "out_of_bounds", "timeout"):
        if result[k]:
            parts.append(f"{k} {result[k]}")
    if result["reached"]:
        parts.append(f"ratio {result['ratio_reached']:.3f}")
    parts.append(f"highest {result['mean_max_altitude']:.1f} m")
    return ", ".join(parts)


def make_callback(eval_env, eval_sets, eval_every, report_every, out, started):
    from stable_baselines3.common.callbacks import BaseCallback

    eval_file = out / "eval.csv"

    class EvalCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = eval_every
            self.next_report = report_every
            self.best = None
            self.last_report = (0, time.time())

        def heartbeat(self):
            """One line from the last 100 training episodes, with exploration."""
            episodes = list(self.model.ep_info_buffer)
            done, then = self.last_report
            now = time.time()
            rate = (self.num_timesteps - done) / max(now - then, 1e-9)
            self.last_report = (self.num_timesteps, now)
            if not episodes:
                print(f"{self.num_timesteps:>11,} decisions  {rate:,.0f}/s  "
                      f"(no finished episodes yet)", flush=True)
                return
            counts = Counter(e.get("outcome") for e in episodes)
            mix = ", ".join(f"{k} {counts[k]}" for k in OUTCOMES if counts.get(k))
            print(f"{self.num_timesteps:>11,} decisions  {rate:,.0f}/s  "
                  f"last {len(episodes)} episodes: {mix}  "
                  f"mean return {np.mean([e['r'] for e in episodes]):.1f}", flush=True)

        def _policy(self, env, obs):
            return self.model.predict(obs, deterministic=True)[0]

        def run_eval(self, label):
            rows = []
            for set_name, scenarios in eval_sets.items():
                r = evaluate(eval_env, self._policy, scenarios)
                rows.append((set_name, r))
                print(f"    {label} {set_name:<8} {describe(r)}", flush=True)
            write_eval_rows(eval_file, self.num_timesteps, started, label, rows)

            self.model.save(out / "latest.zip")
            held = dict(rows)["heldout"]
            score = (held["reached"], -held["collided"])
            if self.best is None or score > self.best:
                self.best = score
                self.model.save(out / "best.zip")
                print(f"    new best: {held['reached']} arrivals on the held-out set",
                      flush=True)

        def _on_step(self):
            if self.num_timesteps >= self.next_report:
                self.next_report += report_every
                self.heartbeat()
            if self.num_timesteps >= self.next_eval:
                self.next_eval += eval_every
                minutes = (time.time() - started) / 60
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
            w.writerow([decisions, f"{(time.time() - started) / 60:.1f}", label, set_name,
                        r["episodes"], *[r[k] for k in OUTCOMES],
                        f"{r['ratio_reached']:.3f}", f"{r['mean_min_clearance']:.3f}",
                        f"{r['mean_max_altitude']:.2f}", mem_available_mb()])


def env_settings():
    names = ["TAKEOFF_ALTITUDE", "DRONE_RADIUS", "TOLERANCE", "CEILING", "OUT_MARGIN",
             "SIM_FREQ", "CTRL_FREQ", "DECISION_EVERY", "SETTLE_STEPS", "MAX_DECISIONS",
             "LOOKAHEAD", "POINT_SPEED", "POINT_LAG", "FLOWN_SPEED_XY",
             "FLOWN_SPEED_Z", "V_SCALE", "DIST_SCALE",
             "PROGRESS", "TIME_COST", "SAFE_DISTANCE", "NEAR_COST", "CRASH_COST",
             "ARRIVAL_BONUS"]
    return {n: getattr(maze, n) for n in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dynamics", choices=["point", "mujoco"], default="point")
    ap.add_argument("--steps", type=int, default=2_000_000,
                    help="total policy decisions, summed over all environments")
    ap.add_argument("--envs", type=int, default=4, help="environments in parallel")
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
        "n_obstacles": list(n_range), "seed": args.seed,
        "ppo": {**ppo_kwargs, "learning_rate": lr0,
                "lr_schedule": "constant" if args.no_lr_decay else "linear to 0"},
        "policy": {"net_arch": [256, 256]},
        "env": env_settings(), "lidar": lidar.config(),
    }, indent=2) + "\n")

    def make():
        return maze.MazeEnv(dynamics=args.dynamics, n_obstacles=n_range)

    # fork shares the parent's memory with the workers until they write to
    # it, which matters on a VM with under 5 GB free.
    vec_cls = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    vec_kwargs = {"start_method": "fork"} if args.envs > 1 else None
    venv = make_vec_env(make, n_envs=args.envs, seed=args.seed, vec_env_cls=vec_cls,
                        vec_env_kwargs=vec_kwargs, monitor_dir=str(out / "monitor"),
                        monitor_kwargs={"info_keywords": ("outcome", "n_obstacles")})

    eval_env = make()
    eval_sets = {
        "gazebo": GAZEBO_SCENARIOS,
        "heldout": [(n, HELDOUT_FIRST_SEED + i) for n in HELDOUT_N
                    for i in range(args.heldout_per_n)],
    }

    started = time.time()
    print(f"run {out}")
    print(f"{args.dynamics} flight model, {args.envs} environments, "
          f"{args.steps:,} decisions, {mem_available_mb()} MB free")
    print("--- baseline: flying straight at the target ---", flush=True)
    rows = []
    for set_name, scenarios in eval_sets.items():
        r = evaluate(eval_env, straight_policy, scenarios)
        rows.append((set_name, r))
        print(f"    straight {set_name:<8} {describe(r)}", flush=True)
    write_eval_rows(out / "eval.csv", 0, started, "straight", rows)

    model = PPO("MlpPolicy", venv, learning_rate=learning_rate,
                policy_kwargs=policy_kwargs, seed=args.seed,
                device="cpu", verbose=0, **ppo_kwargs)
    model.set_logger(configure(str(out), ["csv"]))

    callback = make_callback(eval_env, eval_sets, args.eval_every, args.report_every,
                             out, started)
    try:
        model.learn(total_timesteps=args.steps, callback=callback)
    except KeyboardInterrupt:
        print("interrupted; saving what there is")
    model.save(out / "final.zip")

    print(f"--- final evaluation, {(time.time() - started) / 60:.1f} min ---", flush=True)
    callback.run_eval("final")
    venv.close()
    print(f"done; results in {out}")


if __name__ == "__main__":
    main()
