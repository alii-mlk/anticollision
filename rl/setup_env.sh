#!/usr/bin/env bash
# One-time setup of the RL environment in the VM, then the smoke test.
#
# Creates the venv at ~/rlenv, installs MuJoCo, Gymnasium and Stable-Baselines3
# (which brings PyTorch), clones MuJoCo-Drones-Gym into ~/mujoco-drones-gym and
# installs it, then runs rl/smoke_test.py. Safe to run again: steps that are
# already done are skipped.
#
# The package and the venv live in the home directory, not in the shared
# folder, for the same reason PX4 does: the shared folder is slow for large
# trees of small files.
#
# Usage:
#   bash rl/setup_env.sh

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/rlenv"
PKG_DIR="$HOME/mujoco-drones-gym"
PKG_URL="https://github.com/tau-intelligence/MuJoCo-drones-gym.git"

# ROS has to be sourced before set -u, because its setup.bash reads variables
# that may not exist yet.
source /opt/ros/jazzy/setup.bash
set -e

echo "=== 1. venv at $VENV ==="
if [ -d "$VENV" ]; then
  echo "  already exists, keeping it"
else
  python3 -m venv --system-site-packages "$VENV"
  echo "  created"
fi
source "$VENV/bin/activate"
echo "  using $(python --version) at $(command -v python)"

echo ""
echo "=== 2. PyTorch, CPU build only ==="
# On ARM64 Linux a plain "pip install torch" now picks the CUDA build meant
# for NVIDIA ARM servers. That pulls about 9 GB of GPU libraries into a VM
# that has no GPU, and the first install here was left with a truncated
# libcusparse ("file too short"). So any CUDA build is removed first, and
# torch is installed from PyTorch's CPU index before anything else can pull
# the CUDA build in as a dependency.
pip install --upgrade pip
cuda_pkgs="$(pip list --format=freeze 2>/dev/null \
  | grep -iE '^(nvidia-|triton|cuda-)' | cut -d= -f1 | tr '\n' ' ')"
if [ -n "$cuda_pkgs" ]; then
  echo "  removing the CUDA build: torch $cuda_pkgs"
  pip uninstall -y torch $cuda_pkgs
fi
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "=== 2b. MuJoCo, Gymnasium, Stable-Baselines3 ==="
pip install mujoco gymnasium stable-baselines3

# Stable-Baselines3 depends on torch; make sure that did not swap the CPU
# build back for the CUDA one. This is the check that matters: torch itself
# must be the CPU build and must import.
if ! python -c "import torch, sys; sys.exit(0 if torch.version.cuda is None else 1)"; then
  echo "  ERROR: torch is the CUDA build, or does not import"
  pip list --format=freeze | grep -iE '^(torch|nvidia-)'
  exit 1
fi
python -c "import torch; print('  torch', torch.__version__, '| cuda build:', torch.version.cuda)"

# A stray NVIDIA library is a waste of disk rather than a fault, since the CPU
# build of torch never loads one. Report what pulled each one in, so it can be
# removed if nothing actually needs it.
stray="$(pip list --format=freeze 2>/dev/null | grep -iE '^nvidia-' | cut -d= -f1)"
if [ -n "$stray" ]; then
  echo "  note: NVIDIA libraries still installed (harmless, not loaded):"
  for p in $stray; do
    echo "    $p  required by: $(pip show "$p" 2>/dev/null | sed -n 's/^Required-by: //p')"
  done
fi

# The downloaded CUDA wheels stay in pip's cache otherwise, taking the space
# straight back.
pip cache purge >/dev/null 2>&1 || true

echo ""
echo "=== 3. MuJoCo-Drones-Gym at $PKG_DIR ==="
if [ -d "$PKG_DIR/.git" ]; then
  echo "  already cloned, keeping it"
else
  git clone "$PKG_URL" "$PKG_DIR"
fi
# A failure here is not fatal: the smoke test still checks everything else,
# and it reports whether the package works at all.
if ( cd "$PKG_DIR" && pip install -e . ); then
  echo "  installed"
else
  echo "  WARNING: the package did not install; the smoke test will report it"
fi

echo ""
echo "=== 4. smoke test ==="
set +e
python "$WORKDIR/smoke_test.py" --package-dir "$PKG_DIR"
status=$?

echo ""
echo "From now on, start every RL terminal with:  source rl/env.sh"
exit $status
