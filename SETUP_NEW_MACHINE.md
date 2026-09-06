# Setting the project up on another machine

Written while moving the simulation from a VMware Ubuntu VM on a MacBook Air
to a Windows laptop with an RTX GPU, running Ubuntu under WSL2. The reason for
the move is at the end.

Everything below assumes Ubuntu 24.04. On Windows that means WSL2 with an
Ubuntu 24.04 distribution.

## 1. Ubuntu 24.04 under WSL2 (skip on native Ubuntu)

In Windows PowerShell:

```powershell
wsl --list --verbose
```

If Ubuntu 24.04 is not listed:

```powershell
wsl --install Ubuntu-24.04
```

Then update everything and check the GPU is visible from inside Ubuntu:

```bash
sudo apt update && sudo apt upgrade -y
nproc          # how many cores WSL got
free -h        # how much memory WSL got
glxinfo -B     # after: sudo apt install -y mesa-utils
```

`glxinfo -B` should name the real GPU rather than a software renderer. This is
the whole point of the move, so it is worth confirming before installing
anything else.

## 2. ROS 2 Jazzy

```bash
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-jazzy-desktop
```

## 3. Gazebo Harmonic and the ROS bridge

```bash
sudo apt install -y ros-jazzy-ros-gz
gz sim --version        # expect Gazebo Sim 8.x
```

## 4. MAVROS

```bash
sudo apt install -y ros-jazzy-mavros ros-jazzy-mavros-msgs ros-jazzy-mavros-extras
sudo bash /opt/ros/jazzy/lib/mavros/install_geographiclib_datasets.sh
```

That last script is required. Without the datasets MAVROS complains and some
conversions fail.

## 5. PX4 v1.17.0

Build it outside any Windows-mounted folder (`/mnt/c/...`). Those are slow and
unreliable for large builds; use the Linux home directory.

```bash
mkdir -p ~/px4 && cd ~/px4
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
git checkout v1.17.0
git submodule update --init --recursive
bash ./Tools/setup/ubuntu.sh --no-nuttx
```

Then build both configurations. The second one is for multi-drone runs:

```bash
make px4_sitl_default
make px4_sitl_nolockstep
```

Each build takes tens of minutes the first time.

## 6. This project

Unzip the project somewhere in the Linux filesystem, again not under
`/mnt/c/`, then:

```bash
cd <project>/px4
chmod +x *.sh *.py
```

Sanity check with one drone:

```bash
./start_px4_sitl.sh
```

Wait for `Ready for takeoff!`, then in another terminal:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run mavros mavros_node --ros-args -p fcu_url:="udp://:14540@127.0.0.1:14580"
```

and in a third:

```bash
source /opt/ros/jazzy/setup.bash
cd <project>/px4
./offboard_goto.py --x 5 --y -4 --z 3
```

The drone should take off and fly to that point. If it does, the stack works.

Multi-drone, which is what the new machine is for:

```bash
./gen_scenario_3d.py --n-obstacles 10 --k-drones 3 --seed 21
PX4_BUILD=px4_sitl_nolockstep ./start_px4_multi.sh --scenario scenarios3d/s21_n10_k3
./spawn_obstacles.py --scenario scenarios3d/s21_n10_k3
./run_multi_mission.py --scenario scenarios3d/s21_n10_k3
```

## Notes on running several drones

For a long time three drones failed on the laptop VM (VMware on a MacBook Air,
6 cores, 6 GB): the PX4 instances reported `Accel #0 fail: TIMEOUT`, the
estimates went unstable, and drones fell out of the sky without touching
anything. That looked like the machine being too slow, and it was not. Three
drones now fly correctly on that same VM, and while they do, Gazebo uses about
40% of one core out of six and the simulation runs at 0.99 of real time.

The real causes were all in how the drones were addressed and configured, and
they are described in the comments in `px4/start_px4_multi.sh`. The short
version: PX4 gives instance *i* the MAVLink system ID *i+1*, so every MAVROS
has to be told which autopilot it is talking to; the MAVROS nodes have to be
started one at a time; and the SITL parameters cannot be set through MAVROS,
because it insists on downloading the autopilot's whole parameter list first
and that download stalls with several instances running.

The lesson worth keeping is to measure before theorising about speed:

```bash
cd <project>/px4
./sim_speed.py --window 20
```

This prints the real time factor (simulated seconds per wall-clock second) and
which processes are using the processor. `start_px4_multi.sh` runs it
automatically once the drones are up.

If a machine genuinely cannot keep up, running slower than real time is not by
itself a problem: the physics is the same and simulated time is still the
honest measure of a flight. What breaks a run is the simulator falling behind
*unevenly*, because the autopilot then sees its sensor data arrive late and
decides the sensors are dead. Telling Gazebo to aim at a speed it can actually
hold turns that into an even slowdown every drone shares:

```bash
./set_sim_speed.py 0.3        # or ./set_sim_speed.py show / reset
```

Runs then take about three times longer in wall-clock time, while their
simulated duration, distances, collisions and ratio stay the same. The mission
runner measures simulated time from Gazebo's clock, so `--timeout` still means
simulated seconds, and `--wall-timeout` is only there to catch a simulation
that has stopped entirely.

Two other things that free up real capacity on a laptop:

- Boot the VM to a text console instead of the GNOME desktop. Everything here
  runs headless anyway, and the desktop costs roughly a core and a gigabyte:
  `sudo systemctl set-default multi-user.target`, then reboot. Put it back
  with `graphical.target`.
- Keep the machine on mains power and give it air. A fanless laptop throttles
  under a sustained load like this, so a batch that starts fine can slow down
  after a few minutes.

## State of the work when this was written

Working and verified:

- one, two and three drones: simultaneous 3D maze missions, per-drone metrics,
  summary of collisions, summed euclidean and travelled distance, ratio,
  simulated time and the real time factor
- collision detection against obstacles, with destroy-on-collision
- `start_px4_multi.sh` for k instances with correct ports, namespaces and
  MAVLink system IDs, and SITL parameters applied over plain MAVLink

Not yet verified:

- drone-to-drone collision detection: the code and its geometry are tested,
  but no run has actually made two drones hit each other, because the
  generator keeps their start points apart
- `run_batch_3d.py` and `compute_metrics_3d.py`: the unattended (n, k) sweep
  and its metrics table have never been run end to end

Those two are the next things to do, in that order.
