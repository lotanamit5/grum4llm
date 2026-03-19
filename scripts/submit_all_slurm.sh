#!/usr/bin/env bash
# Submit all GRUMs figure reproduction jobs to SLURM.
# Run from the repo root: bash scripts/submit_all_slurm.sh
#
# Each figure runs as a separate SLURM job so they can run in parallel.
# Logs land in logs/slurm/ inside the repo root.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

mkdir -p logs/slurm

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Submitting GRUMs figure jobs to SLURM..."

JOB3=$(sbatch --parsable scripts/slurm_figure3.sh)
log "Figure 3 submitted: job $JOB3"

JOB4=$(sbatch --parsable scripts/slurm_figure4.sh)
log "Figure 4 submitted: job $JOB4"

JOB5=$(sbatch --parsable scripts/slurm_figure5_sushi.sh)
log "Figure 5 (Sushi) submitted: job $JOB5"

JOB6=$(sbatch --parsable scripts/slurm_figure6.sh)
log "Figure 6 submitted: job $JOB6"

log ""
log "All jobs submitted. Monitor with:"
log "  squeue -u \$USER"
log "  tail -f logs/slurm/fig3_${JOB3}.out"
