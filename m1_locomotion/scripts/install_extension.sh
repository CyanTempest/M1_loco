#!/usr/bin/env bash
# Run inside an already installed Isaac Lab v2.3.2 Python environment on Linux.
set -euo pipefail
M1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ROBOT_LAB_ROOT:?Set ROBOT_LAB_ROOT to a new path or the pinned robot_lab checkout}"
if [[ ! -e "$ROBOT_LAB_ROOT" ]]; then
    git clone --depth 1 --branch v2.3.2 https://github.com/fan-ziqi/robot_lab.git "$ROBOT_LAB_ROOT"
fi
M1_ROBOT_LAB_REV="$(git -C "$ROBOT_LAB_ROOT" rev-parse HEAD)"
if [[ "$M1_ROBOT_LAB_REV" != "09f6a9dfdf48f32f38bb851dfa3a7d44db32b270" ]]; then
    echo "robot_lab version mismatch: $M1_ROBOT_LAB_REV" >&2
    exit 1
fi
# RSL locomotion does not need the upstream package's optional AMP/CusRL dependency stack.
python -m pip install toml psutil colorama xacrodoc
python -m pip install --no-deps -e "$ROBOT_LAB_ROOT/source/robot_lab"
python -m pip install --no-deps -e "$M1_ROOT"
python "$M1_ROOT/scripts/prepare_assets.py"
python "$M1_ROOT/scripts/preflight.py"
