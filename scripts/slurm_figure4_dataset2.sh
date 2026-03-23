#!/usr/bin/env bash
#SBATCH --job-name=grums-fig4-d2
#SBATCH --output=out/fig4-d2_%j.out
#SBATCH --error=err/fig4-d2_%j.err
#SBATCH --cpus-per-task=32

set -e
set -o pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 4 (Dataset 2)..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure4_dataset2_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 4 (Dataset 2) complete."
