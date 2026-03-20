#!/usr/bin/env bash
#SBATCH --job-name=grums-fig3
#SBATCH --output=out/grums-figures_%j.out
#SBATCH --error=err/grums-figures_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=128

set -e
set -o pipefail

# ── resolve repo root (same directory as this script's parent) ────────────────
REPO_ROOT="$SLURM_SUBMIT_DIR"
echo "Root dir: $REPO_ROOT"
cd "$REPO_ROOT"

# ── activate conda environment ────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 3 (Dataset 1)..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure3_dataset1_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Figure 3 (Dataset 2)..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure3_dataset2_orchestration.yml

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 3 complete."
