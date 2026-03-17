# grums4llms

A provider-agnostic GRUM package for preference elicitation.

## Architecture boundary

- `grums.core`: model and inference logic (domain core).
- `grums.providers`: adapters for external preference sources (simulation, human, LLM).
- `grums.contracts`: shared interfaces connecting core and providers.

The core package must never import model-specific provider code.

## Development

Run tests:

```bash
/home/lotanamit/miniconda3/envs/env/bin/python -m pytest
```

## Quick experiment script

Run a social-choice synthetic experiment end-to-end:

```bash
/home/lotanamit/miniconda3/envs/env/bin/python scripts/run_social_choice_experiment.py \
	--mode both \
	--agent-counts 10,20,30 \
	--rounds 20 \
	--repeats 3 \
	--seed 0 \
	--output-json results_social_choice.json
```
