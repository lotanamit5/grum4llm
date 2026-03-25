#!/usr/bin/env bash
# SBATCH worker script for GRUMs fit_grum.py

set -e
set -o pipefail

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <config_path> <output_json_path>"
    exit 1
fi

CONFIG_PATH="$1"
OUTPUT_JSON="$2"

# Deactivate threads for numpy/torch to avoid contention on large nodes
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

# Run experiment
python scripts/fit_grum.py --config "$CONFIG_PATH" --output_json "$OUTPUT_JSON"
