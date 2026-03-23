#!/usr/bin/env bash
# Fig. 5: Sushi (not YAML-orchestrated). Each criterion run uses --n-jobs parallel repeats
# inside one Python process; keep --cpus-per-task >= --n-jobs in the loop below.
#SBATCH --job-name=grums-fig5-sushi
#SBATCH --output=logs/slurm/fig5_sushi_%j.out
#SBATCH --error=logs/slurm/fig5_sushi_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

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
