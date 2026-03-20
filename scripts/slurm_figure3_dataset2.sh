#!/usr/bin/env bash
#SBATCH --job-name=grums-fig3-d2
#SBATCH --output=out/grums-figures-d2_%j.out
#SBATCH --error=err/grums-figures-d2_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=64

set -e
set -o pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 3 (Dataset 2)..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure3_dataset2_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 3 (Dataset 2) complete."
