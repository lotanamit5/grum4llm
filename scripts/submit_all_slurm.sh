#!/usr/bin/env bash
# Submit all GRUMs figure reproduction jobs to SLURM.
# Run from the repo root: bash scripts/submit_all_slurm.sh
#
# Each figure runs as a separate SLURM job so they can run in parallel.
# Logs land in logs/slurm/ inside the repo root.
#
# Using two 128-CPU nodes: either let different figures run concurrently (this script), or shard one
# figure by duplicating its orchestration YAML with disjoint sweep.seeds ranges and submitting two
# sbatch jobs. Job arrays are also an option if you add a wrapper that maps SLURM_ARRAY_TASK_ID to a
# seed subrange. Keep orchestration max_parallel_subprocesses <= the job's --cpus-per-task.

set -e

cd "$SLURM_SUBMIT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Submitting GRUMs figure jobs to SLURM..."

JOB2=$(sbatch --parsable scripts/slurm_figure2.sh)
log "Figure 2 submitted: job $JOB2"

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
log "  tail -f logs/slurm/fig2_${JOB2}.out"
log "  tail -f logs/slurm/fig3_${JOB3}.out"
