#!/usr/bin/env bash
set -e

echo "Starting Figure 3 orchestrations (Dataset 1 and Dataset 2)."
echo "Ensure you are running this within the 'env' conda environment."

echo "--------------------------------------------------------"
echo "Running Figure 3 (Dataset 1)..."
python scripts/run_experiment_orchestration.py --config configs/repro/figure3_dataset1_orchestration.yml

echo "Running Figure 3 (Dataset 2)..."
python scripts/run_experiment_orchestration.py --config configs/repro/figure3_dataset2_orchestration.yml

echo "--------------------------------------------------------"
echo "All Figure 3 orchestrated experiments have finished!"
