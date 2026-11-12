#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd "$(dirname "$0")" && pwd)"
env_dir="${M1_UNILAB_ENV:-$package_dir/.venv}"
if [[ ! -x "$env_dir/bin/python" && -z "${M1_UNILAB_ENV:-}" ]]; then
  env_dir="$package_dir/../../work/unilab-venv"
fi
export UV_CACHE_DIR="${M1_UV_CACHE:-$package_dir/../../work/uv-cache}"
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONFAULTHANDLER=1
cd "$package_dir"
exec uv run --no-project --python "$env_dir/bin/python" python -m m1_unilab.interactive "$@"
