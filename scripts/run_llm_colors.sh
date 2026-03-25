#!/usr/bin/env bash
#SBATCH --job-name=grums-llm-colors-orchestrator
#SBATCH --output=logs/slurm/llm_colors_orch_%j.out
#SBATCH --error=logs/slurm/llm_colors_orch_%j.err
#SBATCH --cpus-per-task=16

# Executes the LLM color preference elicitation orchestration (Qwen2.5-0.5B, K=8, PCA).

set -e
set -o pipefail

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

# Ensure project roots are in PYTHONPATH
export PYTHONPATH=src:experiments

NODES=""
while getopts "w:" opt; do
  case $opt in
    w) NODES="$OPTARG" ;;
    *) echo "Usage: $0 [-w nodes]"; exit 1 ;;
  esac
done

if [ -n "$NODES" ]; then
    NODE_ARG="--nodes $NODES"
else
    NODE_ARG=""
fi

echo "Dispatching LLM Colors Preference Elicitation Orchestrator..."
python experiments/run_experiment_orchestration.py --config configs/llm/llm_colors_orchestration.yml $NODE_ARG

echo "Successfully launched LLM Colors experiment jobs."
