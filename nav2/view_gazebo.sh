#!/usr/bin/env bash

# Attaches the Gazebo GUI to the already-running simulation server
# (start_drone_stack.sh runs the server headless with `gz sim -s`).
# Close the window anytime; the simulation keeps running.
#
# VM note (VMware on Apple Silicon): the GUI only works with
# "Accelerate 3D Graphics" DISABLED in the VM's display settings --
# Mesa then falls back to llvmpipe software rendering, which is slow
# but stable. With acceleration enabled, the buggy SVGA3D driver makes
# the GUI hang. On real hardware / WSL2 with a proper GPU, it just works.

gz sim -g "$@"
