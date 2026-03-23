#!/usr/bin/env bash
#SBATCH --job-name=grums-elicitation-orchestrator
#SBATCH --output=logs/slurm/elic_orch_%j.out
#SBATCH --error=logs/slurm/elic_orch_%j.err
#SBATCH --cpus-per-task=128

# Executes the robust elicitation configuration evaluating multiple datasets and Criteria over 90 evaluations (Figures 3, 4, and 5).

set -e
set -o pipefail

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "Dispatching Elicitation Orchestrator pipelines..."
python scripts/run_experiment_orchestration.py --config configs/repro/elicitation_orchestration.yml
echo "Successfully launched Elicitation reproduction metrics."
