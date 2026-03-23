import json
from pathlib import Path
import pandas as pd
import numpy as np

def load_metrics_for_datasets(run_dir: str | Path, datasets: list[str] | None = None) -> pd.DataFrame:
    """Loads and aggregates criteria curves filtering dynamically against target datasets dynamically grouping."""
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
        curve = data.get("criteria_curve", [])
        for pt in curve:
            n_obs = pt.get("n_observations")
            for m_key in ["social_tau", "mean_person_tau", "raw_person_tau"]:
                if m_key in pt and pt[m_key] is not None:
                    rows.append({
                        "dataset": dataset,
                        "criterion": criterion,
                        "n_observations": n_obs,
                        "metric": m_key,
                        "score": float(pt[m_key])
                    })
                    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
        
    # Aggregate means and standard deviations combining whatever matching datasets we scraped
    agg_df = df.groupby(["criterion", "n_observations", "metric"])["score"].agg(["mean", "std"]).reset_index()
    agg_df["std"] = agg_df["std"].fillna(0.0)
    return agg_df
    
def smooth_curve(y, window_size: int = 10):
    """Calculates a discrete rolling metric using Pandas, avoiding np dependencies constraints natively."""
    if window_size <= 1:
        return np.array(y)
    return pd.Series(y).rolling(window=window_size, min_periods=1).mean().values
