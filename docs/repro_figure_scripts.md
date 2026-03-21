# Paper figure reproduction: scripts and configs

This repository targets the experiments in the GRUM paper (see `docs/original paper/GRUMs.md`). Below is a concise map of **what each runner does** and which **figure** it supports.

## Python entrypoints

| Script | Role |
|--------|------|
| `scripts/run_experiment_orchestration.py` | Reads a YAML orchestration file, spawns parallel **worker subprocesses** (each runs one worker script with a merged config), writes per-subrun JSON under `results/repro/…`, then aggregates (e.g. `asymptotic.json`, `criteria_curve.json`, `timing.json`). |
| `scripts/run_social_choice_experiment.py` | **Worker** for synthetic **social-choice** runs: asymptotic recovery (Fig. 2 style), single-endpoint criteria score, or **`criteria_curve`** (one adaptive chain, Kendall τ checkpoints for Fig. 3). |
| `scripts/run_personalized_choice_experiment.py` | **Worker** for synthetic **personalized** runs: asymptotic (Fig. 6), criteria summary, or **`criteria_curve`** (Fig. 4). |
| `scripts/run_sushi_experiment.py` | **Fig. 5** only: real Sushi data, adaptive elicitation per criterion; writes one JSON per invocation (used in a loop from `slurm_figure5_sushi.sh`). |

## Orchestration configs (`configs/repro/`)

| Config | Paper figure | Experiment |
|--------|----------------|------------|
| `figure2_orchestration.yml` | **Fig. 2** | Asymptotic social choice: more observed agents → Kendall τ vs. true δ ranking. Sweeps **seeds**; parallelizes across subruns. |
| `figure3_dataset1_orchestration.yml` | **Fig. 3(a)** | `criteria_curve`, Dataset 1, criteria random / D / E / social; 100 elicitation rounds; seed sweep. |
| `figure3_dataset2_orchestration.yml` | **Fig. 3(b)** | Same for Dataset 2. |
| `figure4_dataset1_orchestration.yml` | **Fig. 4(a)** | Personalized `criteria_curve`, Dataset 1; includes **personalized** criterion (Eq. 6-style). YAML comments note Eq. (5) vs (6). |
| `figure4_dataset2_orchestration.yml` | **Fig. 4(b)** | Same for Dataset 2. |
| `figure6_orchestration.yml` | **Fig. 6** | Asymptotic **personalized** recovery (consistency dataset, many agent-counts); seed sweep. |

**Fig. 5** has no orchestration YAML: the SLURM driver calls `run_sushi_experiment.py` repeatedly for each criterion.

## Shell drivers

| Script | Use |
|--------|-----|
| `scripts/run_all_figures.sh` | Local sequential run of orchestrations for **Figs. 2, 3, 4, 6** (no Sushi). |
| `scripts/submit_all_slurm.sh` | Submits **Fig. 2, 3, 4, 5, 6** SLURM jobs (each figure is its own job so they can run on different nodes). |
| `scripts/slurm_figure2.sh` … `scripts/slurm_figure6.sh` | One figure per script; SLURM **cpus/mem** should match `max_parallel_subprocesses` in the corresponding YAML (see comments in each file). |
| `scripts/slurm_figure5_sushi.sh` | Runs five Sushi experiments (criteria loop); CPU count should cover `run_sushi_experiment.py --n-jobs` inside the script. |

## SLURM settings (128-core nodes)

- **Orchestrated figures (2, 3, 4, 6):** `configs/repro/figure*_orchestration.yml` use `max_parallel_subprocesses: 128` with matching `#SBATCH --cpus-per-task=128` and `#SBATCH --mem=256G`. If jobs are **OOM-killed**, lower both (e.g. 64 / 128G) in YAML and SLURM together.
- **Fig. 5:** Uses `--cpus-per-task=16` and `--n-jobs 16` in the sushi loop (not 128-way orchestration). Increase both if you add more intra-run parallelism.
- **Two nodes:** Run different figures concurrently via `submit_all_slurm.sh`, or **shard seeds** by copying a YAML with a smaller `sweep.seeds` range and submitting two jobs.

## Post-processing

Paper plots use **moving-window smoothing** (Fig. 3: 25, Fig. 4: 20, Fig. 5: 10) on the aggregated series; apply when plotting from `aggregates/criteria_curve.json` (or sushi JSONs), not in these runners.
