# grums4llms

A provider-agnostic GRUM package for preference elicitation.

## Installation

### 1. Create Conda Environment
Create a new environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate env
```

### 2. Install Package
[Optional] Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

## Architecture boundary

- `grums.core`: model and inference logic (domain core).
- `grums.providers`: adapters for external preference sources (simulation, human, LLM).
- `grums.contracts`: shared interfaces connecting core and providers.

The core package must never import model-specific provider code.

## Development

Run tests:

```bash
pytest
```

## Quick Start (Smoke Test)

Verify your installation by running a minimal experiment (Dummy Mode):

```bash
bash scripts/run_smoke_test.sh
```

## Experiment Execution

The pipeline supports two execution modes: **Local** (sequential) and **Orchestrated** (SLURM-based parallel).

### 1. Local Execution (Recommended for Testing)
Use `fit_grum_llm.py` with the `--dummy` flag for fast local validation.

```bash
python experiments/fit_grum_llm.py --config configs/smoke_test.yml --dummy
```

### 2. Orchestrated Execution (SLURM)
Use `run_experiment_orchestration.py` (via shell scripts) to dispatch a sweep of trials as individual SLURM jobs.

```bash
# Example: LLM Color Preference experiment
bash scripts/run_llm_colors.sh

# Example: Asymptotic reproduction experiment
bash scripts/run_asymptotic_repro.sh
```

## Paper Reproduction

Reproduction scripts for the paper use orchestration configs under `configs/repro/`:

```bash
bash scripts/run_asymptotic_repro.sh
bash scripts/run_elicitation_repro.sh
```


Notes:

- `n_jobs` controls CPU worker processes for parallel repeat execution in asymptotic and criteria phases.
- For large servers, set `--n-jobs` close to your available CPU count (for example, `256` or `512`) and benchmark throughput.
- Current experiment code path is CPU-based (NumPy/SciPy); GPUs are not used yet.
