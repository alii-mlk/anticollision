# Feasibility note: PX4 3D navigation and collision avoidance

Written after Phase D (PX4 SITL + MAVLink bridge working) and before building the
3D evaluation on top of it. The question is what "the PX4 3D navigation system"
actually is today, because the rest of the plan depends on the answer.

Checked in July 2026 against the current PX4 documentation and repositories.

## What I found

**1. PX4-Avoidance is archived and cannot run on this stack.**

The repository (github.com/PX4/PX4-Avoidance) contains the 3D avoidance software
usually meant when people refer to PX4 obstacle avoidance: a local planner based
on the 3DVFH+ algorithm, a global planner on an octomap occupancy grid, and a
safe landing planner. It was archived on 1 August 2024 and carries the notice
"This project is currently not maintained."

It also targets Ubuntu 20.04, ROS 1 Noetic and Gazebo 11. Our stack is Ubuntu
24.04, ROS 2 Jazzy and Gazebo Harmonic. Using it would mean porting archived ROS 1
code to ROS 2, which is a project in itself and not the subject of the thesis.

**2. The Path Planning Interface has been withdrawn.**

This was the MAVLink interface through which PX4 handed a desired path to an
external planner and received a trajectory back. The current documentation states
that it "along with the features Obstacle avoidance in Missions and Safe Landing
are no longer supported or maintained, and should not be used in any PX4 version",
and explains that the code "was abandoned due to architectural constraints of the
implementation making it hard to maintain, extend, and adopt".

So the official channel for plugging a 3D planner into PX4 missions no longer
exists either.

**3. What PX4 does still provide is Collision Prevention, and it is 2D.**

Collision Prevention is built into current PX4. It keeps an obstacle distance map
divided into sectors around the vehicle (36 or 72 depending on version) and slows
or stops the vehicle before it reaches an obstacle. Obstacle data comes either
from supported rangefinders or from a companion computer through the MAVLink
OBSTACLE_DISTANCE message.

Two limits matter for us. It works in the horizontal plane, so it does not solve
3D avoidance, and it applies in Position mode rather than in autonomous or
offboard flight. It is a safety braking layer, not a path planner.

## What this means for the plan

There is no maintained 3D path planner inside PX4 to switch to. What PX4 now
recommends instead is exactly the architecture we already have working after
Phase D: the planning runs on the companion computer side in ROS 2, and PX4
executes the resulting setpoints in offboard mode.

Phase D already demonstrates this path end to end. Through MAVROS we arm the
drone, take off, and fly to an arbitrary 3D target with position setpoints
streamed at 20 Hz (`offboard_goto.py`). The drone reached a target at
(5, -4, 6) in 4.7 s and held it within a few centimetres.

Options considered:

| Option | Assessment |
|---|---|
| Port PX4-Avoidance to ROS 2 | Archived ROS 1 code, large effort, not the thesis topic |
| Use PX4 Collision Prevention as the avoidance system | Horizontal only and Position mode only, so it cannot satisfy a 3D evaluation |
| Plan in ROS 2, execute through PX4 offboard setpoints | Works today, is what PX4 recommends, and is what Phase D already proved |

Recommendation: keep PX4 and MAVLink as the professor asked, since they give us
the realistic drone model, the standard protocol, and multi-vehicle support, and
do the 3D planning and avoidance on the ROS 2 side through offboard setpoints.
Optionally enable Collision Prevention later as an extra reactive safety layer,
feeding it OBSTACLE_DISTANCE messages, but it cannot be the main mechanism.

One consequence is worth pointing out. The original plan was to show that a
baseline system degrades and then add our component. If PX4 has no maintained 3D
avoidance at all, the gap the thesis fills is larger than expected, not smaller.
The comparison would then be between simple offboard flight without avoidance and
the same flight with our 3D avoidance component, measured with the metrics we
already collect.

## Sources

- PX4-Avoidance repository: https://github.com/PX4/PX4-Avoidance
- Path Planning Interface (current docs, deprecation notice):
  https://docs.px4.io/main/en/computer_vision/path_planning_interface.html
- Collision Prevention (current docs):
  https://docs.px4.io/main/en/computer_vision/collision_prevention.html
