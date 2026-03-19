#!/usr/bin/env bash
# Run all figure orchestration jobs for GRUMs paper reproduction.
# Runs Figures 3, 4, 6 (synthetic datasets) and the Sushi Figure 5 experiments.
#
# Usage:
#   bash scripts/run_all_figures.sh
#
# Make sure you are in the project root and have activated the conda environment:
#   conda activate env

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== GRUMs Figure Reproduction Runner ==="
log "Working directory: $(pwd)"
log ""

# --------------------------------------------------------------------------
# Figure 3: Elicitation criteria for social choice (Dataset 1 + 2)
# --------------------------------------------------------------------------
log "--- Figure 3: Social Choice Elicitation ---"

log "Figure 3 / Dataset 1..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure3_dataset1_orchestration.yml
log "Figure 3 / Dataset 1 DONE."

log "Figure 3 / Dataset 2..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure3_dataset2_orchestration.yml
log "Figure 3 / Dataset 2 DONE."

# --------------------------------------------------------------------------
# Figure 4: Elicitation criteria for personalized choice (Dataset 1 + 2)
# --------------------------------------------------------------------------
log "--- Figure 4: Personalized Choice Elicitation ---"

log "Figure 4 / Dataset 1..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure4_dataset1_orchestration.yml
log "Figure 4 / Dataset 1 DONE."

log "Figure 4 / Dataset 2..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure4_dataset2_orchestration.yml
log "Figure 4 / Dataset 2 DONE."

# --------------------------------------------------------------------------
# Figure 6: Asymptotic behavior for personalized choice
# --------------------------------------------------------------------------
log "--- Figure 6: Personalized Choice Asymptotic ---"

log "Figure 6..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure6_orchestration.yml
log "Figure 6 DONE."

log ""
log "=== All figure jobs finished successfully! ==="
log "Open notebooks/paper_reproduction.ipynb to view the results."
