#!/usr/bin/env bash

# Experiment: llm_colors_hs-20260326-102059
# Total runs: 3

sbatch -A bml -p bml -w plotinus1 -o /home/lotan.amit/grum4llm/results/llm/llm_colors_hs-20260326-102059/logs/%j_run_000_criterion_social.out -e /home/lotan.amit/grum4llm/results/llm/llm_colors_hs-20260326-102059/logs/%j_run_000_criterion_social.err /home/lotan.amit/grum4llm/scripts/worker_llm_slurm.sh /home/lotan.amit/grum4llm/results/llm/llm_colors_hs-20260326-102059/subconfigs/run_000_criterion_social.yml
sbatch -A bml -p bml -w plotinus1 -o /home/lotan.amit/grum4llm/results/llm/llm_colors_hs-20260326-102059/logs/%j_run_001_criterion_personalized.out -e /home/lotan.amit/grum4llm/results/llm/llm_colors_hs-20260326-102059/logs/%j_run_001_criterion_personalized.err /home/lotan.amit/grum4llm/scripts/worker_llm_slurm.sh /home/lotan.amit/grum4llm/results/llm/llm_colors_hs-20260326-102059/subconfigs/run_001_criterion_personalized.yml
sbatch -A bml -p bml -w plotinus1 -o /home/lotan.amit/grum4llm/results/llm/llm_colors_hs-20260326-102059/logs/%j_run_002_criterion_random.out -e /home/lotan.amit/grum4llm/results/llm/llm_colors_hs-20260326-102059/logs/%j_run_002_criterion_random.err /home/lotan.amit/grum4llm/scripts/worker_llm_slurm.sh /home/lotan.amit/grum4llm/results/llm/llm_colors_hs-20260326-102059/subconfigs/run_002_criterion_random.yml
