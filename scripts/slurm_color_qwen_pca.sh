#!/usr/bin/env bash
#SBATCH --job-name=grums-color-qwen-pca
#SBATCH --output=logs/slurm/color_qwen_pca_%j.out
#SBATCH --error=logs/slurm/color_qwen_pca_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

set -e
set -o pipefail

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 1: Generating Qwen PCA dataset..."
python scripts/generate_color_dataset.py --config configs/color_sweep.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2: Running orchestrator active learning sweeps (Qwen PCA)..."
python scripts/run_experiment_orchestration.py configs/color_sweep.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Qwen PCA color ranking experiment complete."
