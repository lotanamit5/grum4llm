#!/usr/bin/env bash
# Parallelism: match configs/repro/figure4_* max_parallel_subprocesses to --cpus-per-task.
# Multi-node sharding: same pattern as scripts/slurm_figure3.sh (seed-range split or job array).
#SBATCH --job-name=grums-fig4
#SBATCH --output=logs/slurm/fig4_%j.out
#SBATCH --error=logs/slurm/fig4_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

set -e
set -o pipefail

# ── resolve repo root ─────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── activate conda environment ────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

mkdir -p logs/slurm

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 4 (Dataset 1)..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure4_dataset1_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 4 (Dataset 2)..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure4_dataset2_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 4 complete."
