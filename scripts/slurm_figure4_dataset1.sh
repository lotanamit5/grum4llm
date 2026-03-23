#!/usr/bin/env bash
#SBATCH --job-name=grums-fig4-d1
#SBATCH --output=out/fig4-d1_%j.out
#SBATCH --error=err/fig4-d1_%j.err
#SBATCH --cpus-per-task=32

set -e
set -o pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 4 (Dataset 1)..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure4_dataset1_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 4 (Dataset 1) complete."
