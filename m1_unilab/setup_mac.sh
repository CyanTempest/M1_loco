#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd "$(dirname "$0")" && pwd)"
env_dir="${M1_UNILAB_ENV:-$package_dir/.venv}"
export UV_CACHE_DIR="${M1_UV_CACHE:-$package_dir/.uv-cache}"
uv venv --python 3.12 --allow-existing "$env_dir"
uv pip install --python "$env_dir/bin/python" \
  'unilab[mujoco,unirl] @ git+https://github.com/Motphys/UniLab.git@f7f72afead492c4cbfcc64333bee39509c9a2204' \
  'torch==2.14.1' 'rsl-rl-lib==5.5.2' 'unisim-core==2.7.0' \
  'mujoco==3.11.0' 'mjbatch-uni==0.2.5' \
  -e "$package_dir/../m1_locomotion" -e "$package_dir"
uv run --no-project --python "$env_dir/bin/python" python -m m1_unilab.validate --terrain flat
uv run --no-project --python "$env_dir/bin/python" python -m m1_unilab.validate --terrain mixed --num-envs 12 --steps 1100
uv run --no-project --python "$env_dir/bin/python" python -m m1_unilab.validate_full --num-envs 8 --steps 100
echo "Ready: bash $package_dir/train_mac_full.sh"

