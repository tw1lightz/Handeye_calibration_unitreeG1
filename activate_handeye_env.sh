#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/home/unitree/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="hand_eye_calib"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run this with:" >&2
    echo "  source $ROOT_DIR/activate_handeye_env.sh" >&2
    exit 1
fi

if [[ ! -f "$CONDA_SH" ]]; then
    echo "conda activation script not found: $CONDA_SH" >&2
    return 1
fi

if [[ ! -x "/home/unitree/miniforge3/envs/$CONDA_ENV_NAME/bin/python" ]]; then
    echo "Project env not found: $CONDA_ENV_NAME" >&2
    return 1
fi

mkdir -p "$ROOT_DIR/.cache/matplotlib"
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
