import itertools
from typing import Any

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
