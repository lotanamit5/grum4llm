#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# SBATCH worker script for GRUMs fit_grum.py

set -e
set -o pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <exp_dir> <job_id> <config_path>"
    exit 1
fi

EXP_DIR="$1"
JOB_ID="$2"
CONFIG_PATH="$3"

OUTPUT_JSON="${EXP_DIR}/outputs/${JOB_ID}.json"

# Optimization: Match threads to allocated CPUs (16)
# This prevents over-subscription while utilizing the full allocation.
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

# Run experiment
python scripts/fit_grum.py --config "$CONFIG_PATH" --output_json "$OUTPUT_JSON"
