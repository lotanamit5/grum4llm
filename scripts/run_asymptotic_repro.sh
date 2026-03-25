#!/usr/bin/env bash
#SBATCH --job-name=grums-asymptotic-orchestrator
#SBATCH --output=logs/slurm/asymp_orch_%j.out
#SBATCH --error=logs/slurm/asymp_orch_%j.err
#SBATCH --cpus-per-task=128

# Executes the asymptotic configuration iterating over 500 steps (for Figures 2 and 6 bounds).

set -e
set -o pipefail

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "Dispatching Asymptotic Orchestrator pipelines..."
python experiments/run_experiment_orchestration.py --config configs/repro/asymptotic_orchestration.yml
echo "Successfully launched Asymptotic reproduction metrics."
