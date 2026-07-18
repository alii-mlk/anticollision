# Early versions

Superseded experiments kept for the project's history. Nothing here is part of
the current pipeline (see `../nav2/`).

- `3 drones move manually.sdf` — first Gazebo world: three X3 drones driven by
  hand-published gz topics, before any ROS 2 integration.
- `collision.sdf` — follow-up world where drones were deliberately flown into
  each other to observe collisions.
- `swarm_collision_test.py` — script driving the collision experiments via
  `gz topic` subprocess calls.
- `drone_nav2_bridge.py` — first ROS↔Gazebo bridge prototype. Replaced because
  it used subprocess-per-message publishing, the wrong command topic, and
  dead-reckoned odometry instead of reading the simulated pose.
