import json
from pathlib import Path
import pandas as pd

def load_metrics_dataframe(run_dir: str | Path) -> pd.DataFrame:
    """Loads metrics from outputs directory into a unified unaggregated DataFrame."""
    run_dir = Path(run_dir)
    outputs_dir = run_dir / "outputs"
    if not outputs_dir.exists():
        print(f"Directory {outputs_dir} not found. Outputting blank frame.")
        return pd.DataFrame(columns=["dataset", "step", "seed", "criteria", "social_tau", "person_tau", "raw_tau"])
        
    rows = []
    for json_path in outputs_dir.glob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
            
        dataset = data.get("dataset")
        criterion = data.get("criterion")
        seed = data.get("seed")
        curve = data.get("criteria_curve", [])
        
        for pt in curve:
            row = {
                "dataset": dataset,
                "step": pt.get("n_observations"),
                "seed": seed,
                "criteria": criterion,
                "social_tau": float(pt.get("social_tau", 0.0)) if pt.get("social_tau") is not None else None,
                "person_tau": float(pt.get("mean_person_tau", 0.0)) if pt.get("mean_person_tau") is not None else None,
                "raw_tau": float(pt.get("raw_person_tau", 0.0)) if pt.get("raw_person_tau") is not None else None,
            }
            rows.append(row)
                    
    return pd.DataFrame(rows)
