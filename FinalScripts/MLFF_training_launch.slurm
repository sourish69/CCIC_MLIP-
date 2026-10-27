#!/bin/bash
# Usage: ./submit.sh [config.json] [--restart]
#   Checks the config, then submits the 4-model SLURM job from the current
#   directory. --restart resumes each model from its latest checkpoint.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(realpath "${1:-config.json}")"
RESTART=0
[ "${2:-}" = "--restart" ] && RESTART=1

python "$HERE/train_mlff.py" --config "$CONFIG" --stage check

sbatch --chdir "$PWD" \
       --export=ALL,CONFIG="$CONFIG",TRAIN_SCRIPT="$HERE/train_mlff.py",RESTART="$RESTART" \
       "$HERE/run_mlff.sbatch"
