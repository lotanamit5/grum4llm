import itertools
import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

def get_utc_timestamp() -> str:
    """Returns a formatted UTC timestamp for experiment naming."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

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
