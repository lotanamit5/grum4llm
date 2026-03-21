#!/usr/bin/env bash
# Fig. 3 orchestration: keep configs/repro/figure3_* max_parallel_subprocesses <= --cpus-per-task.
# Two nodes: submit two jobs with different sweep.seeds ranges (duplicate YAML, adjust start/stop),
# or use sbatch --array and a small wrapper that picks a seed slice from SLURM_ARRAY_TASK_ID.
#SBATCH --job-name=grums-fig3
#SBATCH --output=logs/slurm/fig3_%j.out
#SBATCH --error=logs/slurm/fig3_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

set -e
set -o pipefail

# ── resolve repo root (same directory as this script's parent) ────────────────
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 3 (Dataset 1)..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure3_dataset1_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 3 (Dataset 2)..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure3_dataset2_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 3 complete."
