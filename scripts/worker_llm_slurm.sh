#!/bin/bash
#SBATCH -A bml
#SBATCH -p bml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -e
set -o pipefail

# Environment setup
# Assuming conda environment 'env' exists and has transformers, torch, etc.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

CONFIG_PATH=$1

if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: sbatch $0 <config_path>"
    exit 1
fi

echo "Starting GRUM LLM Worker with config: $CONFIG_PATH"
python experiments/fit_grum_llm.py --config "$CONFIG_PATH"

echo "Worker finished successfully."
