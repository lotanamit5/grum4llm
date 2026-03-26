import json
from pathlib import Path
from typing import Any

def load_experiment_results(output_dir: str | Path) -> dict[str, Any]:
    """
    Loads all JSON experiment results from a directory into a dictionary 
    keyed by trial ID or filename stem.
    """
    output_dir = Path(output_dir)
    results = {}
    if not output_dir.exists() or not output_dir.is_dir():
        return results
        
    for f in sorted(output_dir.glob("*.json")):
        with open(f, "r") as j:
            try:
                res = json.load(j)
                # Use filename as trial ID if not present in payload
                trial_id = res.get("trial_id", f.stem)
                results[trial_id] = res
            except json.JSONDecodeError:
                print(f"[WARNING] Could not parse {f}")
                continue
                
    return results
