#!/usr/bin/env bash
#SBATCH --job-name=grums-elicitation-orchestrator
#SBATCH --output=logs/slurm/elic_orch_%j.out
#SBATCH --error=logs/slurm/elic_orch_%j.err
#SBATCH --cpus-per-task=128

# Executes the robust elicitation configuration evaluating multiple datasets and Criteria over 90 evaluations (Figures 3, 4, and 5).

set -e
set -o pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

NODES=""
while getopts "w:" opt; do
  case $opt in
    w) NODES="$OPTARG" ;;
    *) echo "Usage: $0 [-w nodes]"; exit 1 ;;
  esac
done

echo "Dispatching Elicitation Orchestrator pipelines..."
python experiments/run_experiment_orchestration.py --config configs/repro/elicitation_orchestration.yml --nodes "$NODES"
echo "Successfully launched Elicitation reproduction metrics."
