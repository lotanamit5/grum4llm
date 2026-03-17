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
