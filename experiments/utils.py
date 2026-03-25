import itertools
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from grums.inference import MCEMConfig
from grums.core.parameters import GRUMParameters
from grums.elicitation import (
    RandomCriterion,
    DOptimalityCriterion,
    EOptimalityCriterion,
    SocialChoiceCriterion,
    PersonalizedChoiceCriterion,
)
# TODO: we don't need 2 timestamp function, keep 'get_utc_timestamp' and remove 'get_utc_now_iso'
def get_utc_timestamp(compact: bool = False) -> str:
    """Returns the current UTC time. ISO format by default, compact if requested."""
    now = datetime.now(timezone.utc)
    if compact:
        return now.strftime("%Y%m%d-%H%M%S")
    return now.isoformat(timespec="seconds")

def expand_sweep(sweep_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expands a sweep configuration into a list of individual trial overrides.
    Supports lists, range-style dicts (start, stop, step), and single values.
    """
    if not sweep_cfg:
        return [{}]

    sweep_keys = list(sweep_cfg.keys())
    sweep_values = []
    
    for k in sweep_keys:
        val = sweep_cfg[k]
        if isinstance(val, list):
            sweep_values.append(val)
        elif isinstance(val, dict):
            # Support range-style: {start: 0, stop: 10, step: 2}
            start = val.get("start", 0)
            stop = val.get("stop", val.get("end", 0))
            step = val.get("step", 1)
            sweep_values.append(list(range(start, stop + 1, step)))
        else:
            sweep_values.append([val])
            
    combinations = list(itertools.product(*sweep_values))
    
    overrides_list = []
    for combo in combinations:
        overrides_list.append(dict(zip(sweep_keys, combo)))
        
    return overrides_list

def get_trial_id(index: int, overrides: dict[str, Any]) -> str:
    """Generates a unique, parameter-descriptive trial ID."""
    suffix = "_".join(f"{k}_{v}" for k, v in overrides.items())
    return f"run_{index:03d}_{suffix}" if suffix else f"run_{index:03d}"

def create_trial_config(
    base_cfg: dict[str, Any], 
    overrides: dict[str, Any], 
    trial_id: str, 
    exp_dir: Path, 
    dest_path: Path
) -> None:
    """Merges configurations and writes the trial YAML to disk."""
    trial_cfg = dict(base_cfg)
    trial_cfg.update(overrides)
    trial_cfg["trial_id"] = trial_id
    trial_cfg["exp_dir"] = str(exp_dir)
    
    with open(dest_path, "w") as f:
        yaml.safe_dump(trial_cfg, f, sort_keys=False)

def get_mcem_config(mcem_cfg: dict[str, Any]) -> MCEMConfig:
    """Extracts MCEM configuration from a dictionary."""
    return MCEMConfig(
        n_iterations=mcem_cfg.get("n_iterations", 8),
        n_gibbs_samples=mcem_cfg.get("n_gibbs_samples", 30),
        n_gibbs_burnin=mcem_cfg.get("n_gibbs_burnin", 15),
    )

def get_torch_device(config_device: str = "auto") -> torch.device:
    """
    Standardizes device selection:
    - 'auto': Try CUDA, fallback to CPU.
    - 'cpu': Force CPU.
    - 'cuda': Force CUDA (fail if unavailable).
    """
    if config_device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    elif config_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device_type = "cuda"
    elif config_device == "cpu":
        device_type = "cpu"
    else:
        # Unexpected value, default to CPU or raise?
        # User specified "cude" (typo) should fail.
        device_type = config_device
        # Validating it's a valid torch device string
        try:
            torch.device(device_type)
        except Exception as e:
            raise ValueError(f"Invalid device specification: {config_device}") from e

    device = torch.device(device_type)
    print(f"[INFO] Using device: {device}")
    return device

def get_init_params(m: int, k: int, l: int, device: torch.device) -> GRUMParameters:
    """Initializes zeroed GRUM parameters."""
    return GRUMParameters(
        delta=torch.zeros(m, device=device, dtype=torch.float64),
        interaction=torch.zeros((k, l), device=device, dtype=torch.float64)
    )

def get_criteria_map(
    m: int, 
    k: int, 
    l: int, 
    seed: int, 
    alternative_features: torch.Tensor, 
    population_agents: torch.Tensor
) -> dict[str, Any]:
    """Returns a standard map of elicitation criteria."""
    return {
        "random": RandomCriterion(seed),
        "d_opt": DOptimalityCriterion(),
        "e_opt": EOptimalityCriterion(),
        "social": SocialChoiceCriterion(n_alternatives=m),
        "personalized": PersonalizedChoiceCriterion(
            n_alternatives=m,
            n_agent_features=k,
            n_alternative_features=l,
            alternative_features=alternative_features,
            population_agents=population_agents,
        ),
    }

def save_experiment_result(payload: dict[str, Any], cli_output_path: Path, cfg: dict[str, Any]) -> None:
    """Persists experiment payload to JSON."""
    output_path = cli_output_path
    if not output_path and "trial_id" in cfg and "exp_dir" in cfg:
        output_path = Path(cfg["exp_dir"]) / "outputs" / f"{cfg['trial_id']}.json"

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[SUCCESS] Results saved to: {output_path}")
    else:
        # Standard output fallback
        print(json.dumps(payload, indent=2))
