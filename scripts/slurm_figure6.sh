#!/usr/bin/env bash
# Match configs/repro/figure6_orchestration.yml max_parallel_subprocesses to Slurm CPUs.
#SBATCH --job-name=grums-fig6
#SBATCH --output=logs/slurm/fig6_%j.out
#SBATCH --error=logs/slurm/fig6_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=128

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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 6..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure6_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 6 complete."
