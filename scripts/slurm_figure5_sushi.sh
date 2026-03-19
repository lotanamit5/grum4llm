#!/usr/bin/env bash
#SBATCH --job-name=grums-fig5-sushi
#SBATCH --output=logs/slurm/fig5_sushi_%j.out
#SBATCH --error=logs/slurm/fig5_sushi_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -e
set -o pipefail

# ── resolve repo root ─────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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
        --n-jobs     4 \
        --output-json "results/repro/sushi/sushi_${criterion}.json"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 5 (Sushi) complete."
