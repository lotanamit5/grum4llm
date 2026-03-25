#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# SBATCH worker script for GRUMs fit_grum.py

set -e
set -o pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_path>"
    exit 1
fi

CONFIG_PATH="$1"

# Optimization: Match threads to allocated CPUs (16)
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

# Run experiment
# output_json is resolved internally by fit_grum.py using trial_id and exp_dir
python experiments/fit_grum.py --config "$CONFIG_PATH"
