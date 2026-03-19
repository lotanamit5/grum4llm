#!/usr/bin/env bash

set -e
set -o pipefail

echo "Running Figure 4 Dataset 1 Orchestration..."
python scripts/run_experiment_orchestration.py --config configs/repro/figure4_dataset1_orchestration.yml

echo "Running Figure 4 Dataset 2 Orchestration..."
python scripts/run_experiment_orchestration.py --config configs/repro/figure4_dataset2_orchestration.yml

echo "Figure 4 jobs completed."
