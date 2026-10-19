#!/usr/bin/env python3
"""Smoke test: check that every piece the RL stage needs works in the VM.

Each layer is checked on its own, so one failure does not hide the others:

  1. imports      numpy, mujoco, gymnasium, torch, stable_baselines3
  2. ros          rclpy is still importable from this venv, which the policy
                  needs later when it runs inside run_mission.py
  3. mujoco       a tiny model steps headlessly and falls under gravity
  4. package      MuJoCo-Drones-Gym can build a drone environment and step it
  5. sb3          PPO trains for a few steps, on the package environment if it
                  worked and on CartPole otherwise, so torch and SB3 are tested
                  even when the package is broken
  6. speed        steps per second, which decides whether training takes hours
                  or days

Nothing here renders anything. The VM has no working GPU, and the RL stage
does not need one, because the lidar is computed analytically.

The report goes to the terminal and to rl/logs/smoke_test.txt.

Usage (inside the VM, with the venv active):
  python rl/smoke_test.py
  python rl/smoke_test.py --package-dir ~/mujoco-drones-gym
"""

import argparse
import importlib
import inspect
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "logs" / "smoke_test.txt"

results = []
lines = []


def say(text=""):
    print(text, flush=True)
    lines.append(text)


def check(name, fn):
    """Run one check, record PASS or FAIL, and keep going either way."""
    say(f"--- {name} ---")
    start = time.time()
    try:
        detail = fn()
        status = "PASS"
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc().strip().splitlines()
        # The last few frames are the useful ones; the rest is library noise.
        for line in tb[-8:]:
            say(f"    {line}")
        status = "FAIL"
    took = time.time() - start
    say(f"{status}  {name}  ({took:.1f} s)  {detail or ''}")
    say()
    results.append((name, status, detail))
    return status == "PASS"


def check_imports():
    found = []
    for mod in ("numpy", "mujoco", "gymnasium", "torch", "stable_baselines3"):
        m = importlib.import_module(mod)
        found.append(f"{mod} {getattr(m, '__version__', '?')}")
    import torch
    found.append(f"torch threads {torch.get_num_threads()}")
    return ", ".join(found)


def check_ros():
    import rclpy  # noqa: F401
    return f"rclpy from {Path(rclpy.__file__).parent}"


def check_mujoco():
    import mujoco
    import numpy as np

    xml = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <geom type="plane" size="20 20 0.1"/>
        <body name="probe" pos="0 0 5">
          <freejoint/>
          <geom type="box" size="0.1 0.1 0.05" mass="0.5"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    z0 = float(data.qpos[2])
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    z1 = float(data.qpos[2])
    # Free fall for half a second drops it by about g t^2 / 2 = 1.23 m.
    drop = z0 - z1
    if not 1.0 < drop < 1.5:
        raise RuntimeError(f"fell {drop:.2f} m in 0.5 s, expected about 1.23 m")

    n = 20000
    start = time.time()
    for _ in range(n):
        mujoco.mj_step(model, data)
    rate = n / (time.time() - start)
    np.asarray(data.qpos)
    return f"fell {drop:.2f} m in 0.5 s as expected; {rate:,.0f} raw physics steps/s"


def make_package_env():
    """Build the package's hover environment headlessly.

    The constructor's keyword arguments are read from its signature rather than
    assumed, because the package is new and its interface may still change.
    """
    from multi_drone_mujoco.envs.hover_aviary import HoverAviary

    params = inspect.signature(HoverAviary.__init__).parameters
    kwargs = {}
    for key, value in (("gui", False), ("record", False),
                       ("render_mode", None)):
        if key in params:
            kwargs[key] = value
    env = HoverAviary(**kwargs)
    return env, kwargs, list(params)


def check_package():
    env, kwargs, params = make_package_env()
    obs, _ = env.reset(seed=0)
    n = 300
    start = time.time()
    steps = 0
    for _ in range(n):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            obs, _ = env.reset()
    rate = steps / (time.time() - start)
    env.close()
    shape = getattr(obs, "shape", type(obs).__name__)
    say(f"    constructor arguments: {', '.join(p for p in params if p != 'self')}")
    say(f"    used: {kwargs}")
    say(f"    observation space: {env.observation_space}")
    say(f"    action space:      {env.action_space}")
    return f"stepped {steps} times, obs {shape}, {rate:,.0f} env steps/s"


def check_package_enums():
    enums = importlib.import_module("multi_drone_mujoco.utils.enums")
    found = []
    for name in dir(enums):
        obj = getattr(enums, name)
        members = getattr(obj, "__members__", None)
        if members and not name.startswith("_"):
            found.append(f"{name}[{', '.join(members)}]")
    for entry in found:
        say(f"    {entry}")
    return f"{len(found)} enum classes"


def check_sb3(use_package):
    from stable_baselines3 import PPO

    if use_package:
        env, _, _ = make_package_env()
        label = "HoverAviary"
    else:
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        label = "CartPole-v1 (package environment unavailable)"

    model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, n_epochs=2,
                verbose=0, device="cpu")
    start = time.time()
    model.learn(total_timesteps=1024)
    took = time.time() - start
    env.close()
    return f"PPO trained 1024 steps on {label} in {took:.1f} s ({1024 / took:,.0f} steps/s)"


def package_version(package_dir):
    if not package_dir or not (package_dir / ".git").exists():
        return "unknown (pass --package-dir to record the commit)"
    try:
        out = subprocess.run(
            ["git", "-C", str(package_dir), "log", "-1", "--format=%h %ad",
             "--date=short"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package-dir", type=Path,
                    default=Path.home() / "mujoco-drones-gym",
                    help="where MuJoCo-Drones-Gym was cloned, to record its commit")
    args = ap.parse_args()

    say(f"smoke test {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"python {sys.version.split()[0]} at {sys.executable}")
    say(f"platform {platform.platform()}, {os.cpu_count()} cpus")
    say(f"in a venv: {sys.prefix != sys.base_prefix}")
    say(f"MuJoCo-Drones-Gym commit: {package_version(args.package_dir.expanduser())}")
    say()

    check("1 imports", check_imports)
    check("2 ros visible from this venv", check_ros)
    check("3 mujoco steps headlessly", check_mujoco)
    package_ok = check("4 package hover environment", check_package)
    if package_ok:
        check("4b package enums", check_package_enums)
    check("5 sb3 trains", lambda: check_sb3(package_ok))

    say("=== summary ===")
    for name, status, _ in results:
        say(f"  {status}  {name}")
    failed = [n for n, s, _ in results if s == "FAIL"]
    say()
    say("all passed" if not failed else f"{len(failed)} failed: {', '.join(failed)}")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines) + "\n")
    print(f"\nreport written to {LOG}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
