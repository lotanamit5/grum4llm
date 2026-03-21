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
mkdir -p results/repro/sushi

CRITERIA=(random d_opt e_opt social personalized)

for criterion in "${CRITERIA[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sushi experiment: criterion=${criterion}..."
    python scripts/run_sushi_experiment.py \
        --criterion  "${criterion}" \
        --metric     social \
        --repeats    20 \
        --rounds     100 \
        --n-jobs     16 \
        --output-json "results/repro/sushi/sushi_${criterion}.json"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 5 (Sushi) complete."
