#!/usr/bin/env bash
# Local (non-SLURM) driver for paper figures that use YAML orchestration:
#   Fig. 2, 3, 4, 6 (synthetic).
# Figure 5 (Sushi) is not orchestrated here; use scripts/slurm_figure5_sushi.sh or run
# scripts/run_sushi_experiment.py per criterion (see docs/repro_figure_scripts.md).
#
# Usage:
#   bash scripts/run_all_figures.sh
#
# Prerequisites: project root, conda env (e.g. conda activate env).

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
# Figure 2: Asymptotic social choice (synthetic, Sec. 6.1)
# --------------------------------------------------------------------------
log "--- Figure 2: Social Choice Asymptotic ---"

log "Figure 2..."
python scripts/run_experiment_orchestration.py \
    --config configs/repro/figure2_orchestration.yml
log "Figure 2 DONE."

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
log "=== Orchestrated synthetic figures (2, 3, 4, 6) finished. ==="
log "For Figure 5 (Sushi), run scripts/slurm_figure5_sushi.sh on a cluster or invoke run_sushi_experiment.py."
log "See docs/repro_figure_scripts.md for a full map of scripts and configs."
