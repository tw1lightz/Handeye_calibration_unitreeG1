#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="/home/unitree/miniforge3/bin/conda"
MPL_CACHE_DIR="$ROOT_DIR/.cache/matplotlib"
CONDA_ENV_NAME="hand_eye_calib"

if [[ ! -x "$CONDA_BIN" ]]; then
    echo "conda not found: $CONDA_BIN" >&2
    exit 1
fi

if [[ ! -x "/home/unitree/miniforge3/envs/$CONDA_ENV_NAME/bin/python" ]]; then
    echo "Conda env not found: $CONDA_ENV_NAME" >&2
    exit 1
fi

mkdir -p "$MPL_CACHE_DIR"
export MPLCONFIGDIR="$MPL_CACHE_DIR"

exec "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python "$@"
