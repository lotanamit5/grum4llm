#!/usr/bin/env bash
#SBATCH --job-name=grums-fig5-sushi
#SBATCH --output=logs/slurm/fig5_sushi_%j.out
#SBATCH --error=logs/slurm/fig5_sushi_%j.err
#SBATCH --cpus-per-task=64

set -e
set -o pipefail

# ── resolve repo root ─────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── activate conda environment ────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sushi experiment running via orchestrator..."
python scripts/run_experiment_orchestration.py --config configs/repro/figure5_sushi_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 5 (Sushi) complete."
