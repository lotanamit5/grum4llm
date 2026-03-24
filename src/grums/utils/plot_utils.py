import json
from pathlib import Path
import pandas as pd
import numpy as np

def load_metrics_for_datasets(run_dir: str | Path, datasets: list[str] | None = None, pivot_metrics: bool = False, metric_filter: str | None = None) -> pd.DataFrame:
    """Loads and aggregates criteria curves filtering dynamically against target datasets."""
    run_dir = Path(run_dir)
    outputs_dir = run_dir / "outputs"
    if not outputs_dir.exists():
        print(f"Directory {outputs_dir} not found. Outputting blank frame.")
        return pd.DataFrame()
        
    rows = []
    for json_path in outputs_dir.glob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
            
        dataset = data.get("dataset")
        if datasets is not None and dataset not in datasets:
            continue
            
        criterion = data.get("criterion")
        seed = data.get("seed")
        curve = data.get("criteria_curve", [])
        
        if pivot_metrics:
            for pt in curve:
                row = {
                    "step": pt.get("n_observations"),
                    "seed": seed,
                    "criteria": criterion,
                    "dataset": dataset
                }
                for m_key in ["social_tau", "mean_person_tau", "raw_person_tau"]:
                    if m_key in pt and pt[m_key] is not None:
                        row[m_key] = float(pt[m_key])
                rows.append(row)
        else:
            for pt in curve:
                for m_key in ["social_tau", "mean_person_tau", "raw_person_tau"]:
                    if m_key in pt and pt[m_key] is not None:
                        if metric_filter and m_key != metric_filter:
                            continue
                        rows.append({
                            "dataset": dataset,
                            "seed": seed,
                            "criteria": criterion,
                            "step": pt.get("n_observations"),
                            "metric": m_key,
                            "value": float(pt[m_key])
                        })
                    
    df = pd.DataFrame(rows)
    return df
