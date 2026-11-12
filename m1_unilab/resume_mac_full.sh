#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd "$(dirname "$0")" && pwd)"
checkpoint="$package_dir/checkpoints/recovery_1000.pt"
if [[ ! -f "$checkpoint" ]]; then
  echo "Missing preserved checkpoint: $checkpoint" >&2
  exit 1
fi
# The preserved checkpoint contains 12,312,576 / (512 * 24) = 1,002 updates.
# Resume counters repeat a label; use actual sample counts for the 30k target.
exec bash "$package_dir/train_mac_full.sh" \
  --checkpoint "$checkpoint" --iterations 28998 \
  --run-name full_512_30000_libccd "$@"
