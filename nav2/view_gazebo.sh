#!/usr/bin/env bash

# Attaches the Gazebo GUI to the already-running simulation server
# (start_drone_stack.sh runs the server headless with `gz sim -s`).
# Close the window anytime; the simulation keeps running.
#
# VM note (VMware on Apple Silicon): the GUI only works with
# "Accelerate 3D Graphics" turned off in the VM's display settings.
# Mesa then uses llvmpipe software rendering, which is slow but stable.
# With acceleration on, the SVGA3D driver makes the GUI hang. On a
# machine with a real GPU (or WSL2) it works normally.

gz sim -g "$@"
