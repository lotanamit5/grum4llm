#!/usr/bin/env bash
# Fig. 2: asymptotic social choice (synthetic). Match max_parallel_subprocesses in
# configs/repro/figure2_orchestration.yml to --cpus-per-task (128 for a full standard node).
# Reduce both if you hit memory limits (each subprocess runs MC-EM).
#SBATCH --job-name=grums-fig2
#SBATCH --output=logs/slurm/fig2_%j.out
#SBATCH --error=logs/slurm/fig2_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=128

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

mkdir -p logs/slurm

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 2 (asymptotic social choice)..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure2_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 2 complete."
