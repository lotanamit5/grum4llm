#!/usr/bin/env bash

# Run a minimal GRUM experiment locally via fit_grum_llm.py (Dummy Mode)
set -e

# Add project root to PYTHONPATH
export PYTHONPATH=src:experiments

# Activate conda environment
source "/home/lotanamit/miniconda3/etc/profile.d/conda.sh"
conda activate env

echo "Running minimal smoke test using fit_grum_llm.py (Dummy Mode)..."
python experiments/fit_grum_llm.py --config configs/smoke_test.yml --dummy --output_json results/smoke/smoke_result.json

echo "Smoke test complete. Results are in results/smoke/smoke_result.json"
