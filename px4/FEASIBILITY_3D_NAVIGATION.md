# Feasibility note: PX4 3D navigation and collision avoidance

I wrote this after the PX4 SITL and the MAVLink bridge were working, and
before building the 3D evaluation on top of them. The plan was to use "the
PX4 3D navigation system", so I first checked what that is today.

Checked in July 2026 against the current PX4 documentation and repositories.

## What I found

### 1. PX4-Avoidance is archived and cannot run on this stack

The repository (github.com/PX4/PX4-Avoidance) has the 3D avoidance software
that people usually mean by PX4 obstacle avoidance: a local planner based on
the 3DVFH+ algorithm, a global planner on an octomap occupancy grid, and a
safe landing planner. It was archived on 1 August 2024 and carries the notice
"This project is currently not maintained."

It also targets Ubuntu 20.04, ROS 1 Noetic and Gazebo 11. This project uses
Ubuntu 24.04, ROS 2 Jazzy and Gazebo Harmonic. Using it would mean porting
archived ROS 1 code to ROS 2, which would be a project of its own and is not
the subject of the thesis.

### 2. The Path Planning Interface has been withdrawn

This was the MAVLink interface through which PX4 handed a desired path to an
external planner and received a trajectory back. The current documentation
says that it "along with the features Obstacle avoidance in Missions and Safe
Landing are no longer supported or maintained, and should not be used in any
PX4 version", and that the code "was abandoned due to architectural
constraints of the implementation making it hard to maintain, extend, and
adopt".

So the official way to connect a 3D planner to PX4 missions does not exist
anymore either.

### 3. PX4 still has Collision Prevention, but it is 2D

Collision Prevention is part of current PX4. It keeps a map of obstacle
distances in sectors around the vehicle (36 or 72 depending on the version)
and slows or stops the vehicle before it reaches an obstacle. The obstacle
data comes from supported rangefinders or from a companion computer through
the MAVLink OBSTACLE_DISTANCE message.

It has two limits for this project. It only works in the horizontal plane, so
it does not do 3D avoidance, and it only works in Position mode, not in
autonomous or offboard flight. It brakes before an obstacle but does not plan
a path around it.

## What this means for the plan

PX4 has no maintained 3D path planner to switch to. What PX4 recommends now
is to plan on the companion computer in ROS 2 and let PX4 execute the
setpoints in offboard mode, which is the setup I already had working.

That setup already works from start to end. Through MAVROS the drone arms,
takes off and flies to any 3D target with position setpoints sent at 20 Hz
(`offboard_goto.py`). It reached a target at (5, -4, 6) in 4.7 s and held it
within a few centimetres.

Options I considered:

| Option | Assessment |
|---|---|
| Port PX4-Avoidance to ROS 2 | Archived ROS 1 code, a lot of work, not the thesis topic |
| Use PX4 Collision Prevention as the avoidance system | Horizontal only and Position mode only, so it cannot be used for a 3D evaluation |
| Plan in ROS 2, execute through PX4 offboard setpoints | Works now, is what PX4 recommends, and is already tested |

So I kept PX4 and MAVLink, as the professor asked, because they give a
realistic drone model, the standard protocol and support for several
vehicles, and the 3D planning and avoidance runs on the ROS 2 side through
offboard setpoints. Collision Prevention could be added later as an extra
safety layer by sending it OBSTACLE_DISTANCE messages, but it cannot be the
main mechanism.

The original plan was to show that an existing system gets worse in harder
scenarios and then add my component. Since PX4 has no maintained 3D avoidance
at all, the comparison is instead between offboard flight without any
avoidance and the same flight with my 3D avoidance component, measured with
the same metrics.

## Sources

- PX4-Avoidance repository: https://github.com/PX4/PX4-Avoidance
- Path Planning Interface (current docs, deprecation notice):
  https://docs.px4.io/main/en/computer_vision/path_planning_interface.html
- Collision Prevention (current docs):
  https://docs.px4.io/main/en/computer_vision/collision_prevention.html
