#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd "$(dirname "$0")" && pwd)"
env_dir="${M1_UNILAB_ENV:-$package_dir/.venv}"
if [[ ! -x "$env_dir/bin/python" && -z "${M1_UNILAB_ENV:-}" ]]; then
  env_dir="$package_dir/../../work/unilab-venv"
fi
if [[ ! -x "$env_dir/bin/python" ]]; then
  echo "Python environment missing. Run: bash \"$package_dir/setup_mac.sh\"" >&2
  exit 1
fi
export UV_CACHE_DIR="${M1_UV_CACHE:-$package_dir/../../work/uv-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
cd "$package_dir"
launch=(uv run --no-project --python "$env_dir/bin/python" python -m m1_unilab.train
  --profile b2w-full --terrain mixed --device mps
  --num-envs 512 --iterations 30000 --torch-threads 2 --run-name full_512_30000)
if [[ "$(uname -s)" == Darwin ]]; then
  exec caffeinate -i "${launch[@]}" "$@"
fi
exec "${launch[@]}" "$@"
